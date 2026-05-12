# Copyright 2024-2026 The Alibaba Wan Team Authors.
"""xFuser wrapper for Wan2.x diffusers pipeline.

This is the missing `WanXFuserPipeline` — xfuser ships the model-layer adapter
(`xFuserWanAttnProcessor`) and registers it via `xFuserAttentionProcessorRegister`,
but no pipeline-level wrapper exists yet. We provide one here, modeled on
xfuser's `xFuserCogVideoXPipeline` (similar video diffusion pattern).

Supports:
- Sequence parallel (Ulysses + Ring + USP hybrid via the registered attn processor)
- CFG-parallel (cond on rank-group 0, uncond on rank-group 1) for `cfg_parallel=2`
- DataParallel via the base wrapper's decorator

Wan2.2-specific:
- Two transformers (`transformer` for high-noise, `transformer_2` for low-noise),
  switched at `boundary_timestep`. Both must have xfuser attn processors applied.
- `guidance_scale` (high-noise) and `guidance_scale_2` (low-noise) tuple.
"""

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
from diffusers import WanPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_wan import WanPipelineOutput

from xfuser.config import EngineConfig
from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_world_size,
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    is_dp_last_group,
)
from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister


@xFuserPipelineWrapperRegister.register(WanPipeline)
class WanXFuserPipeline(xFuserPipelineBaseWrapper):
    """xfuser pipeline wrapper for diffusers' WanPipeline (Wan2.1/2.2 T2V)."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        **kwargs,
    ):
        pipeline = WanPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(pipeline, engine_config)

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        **kwargs,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        patch_size = (
            self.transformer.config.patch_size
            if self.transformer is not None
            else self.transformer_2.config.patch_size
        )
        h_multiple_of = self.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = self.vae_scale_factor_spatial * patch_size[2]
        height = height // h_multiple_of * h_multiple_of
        width = width // w_multiple_of * w_multiple_of

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        # Set on the wrapped diffusers pipeline (self.module). When we later
        # call self.encode_prompt(...) or read self.do_classifier_free_guidance,
        # the underlying property/method resolves against `module._guidance_scale`,
        # not the wrapper's attribute.
        self.module._guidance_scale = guidance_scale
        self.module._guidance_scale_2 = guidance_scale_2
        self.module._attention_kwargs = attention_kwargs
        self.module._current_timestep = None
        self.module._interrupt = False

        device = self._execution_device

        # 2. batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 2.5 initialize xfuser runtime input state — required before any
        # transformer.forward; otherwise the wrapped transformer raises
        # "Runtime state is not ready, please call set_input_parameters".
        get_runtime_state().set_video_input_parameters(
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

        # 3. encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = (
            self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. latents
        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        self.scheduler.set_begin_index(0)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        cfg_world = get_classifier_free_guidance_world_size()
        cfg_rank_is_uncond = (
            cfg_world == 2
            and torch.distributed.is_initialized()
            and get_cfg_group().rank_in_group == 1
        )

        # Sequence-parallel split of latents (xfuser's USP infra hooks attn
        # processors below; we still need to keep latent shape aligned).
        # For Wan2.x the transformer's sequence axis is along temporal × spatial
        # patches; xfuser's WanAttnProcessor expects the SP split happens INSIDE
        # the attention via all_to_all, so latents stay full-shape here.

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                if cfg_world == 2 and self.do_classifier_free_guidance:
                    # CFG-parallel-2: each rank computes only one branch.
                    if cfg_rank_is_uncond:
                        local_emb = negative_prompt_embeds
                    else:
                        local_emb = prompt_embeds
                    with current_model.cache_context("cond"):
                        local_pred = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=local_emb,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred, noise_uncond = get_cfg_group().all_gather(
                        local_pred, separate_tensors=True
                    )
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)
                else:
                    # Default path: rank does both forwards locally.
                    with current_model.cache_context("cond"):
                        noise_pred = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            noise_uncond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        # VAE decode only on dp-last-group rank 0.
        # After 40 diffusion steps both DiTs are still resident (~77 GB / 80 GB
        # on each H100). VAE decode needs ~2.6 GB scratch. Move DiTs to CPU
        # and free the reserved-but-unallocated pool before decoding so it fits.
        if is_dp_last_group():
            if not output_type == "latent":
                if self.transformer is not None:
                    self.transformer.to("cpu")
                if self.transformer_2 is not None:
                    self.transformer_2.to("cpu")
                torch.cuda.empty_cache()
                latents = latents.to(self.vae.dtype)
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                video = self.vae.decode(latents, return_dict=False)[0]
                video = self.video_processor.postprocess_video(video, output_type=output_type)
            else:
                video = latents
        else:
            video = [None for _ in range(batch_size)]

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
