"""xFuser transformer wrapper for diffusers WanTransformer3DModel.

This is the missing transformer-level wrapper xfuser needs to bench Wan2.x.
xfuser ships the attention-processor (xFuserWanAttnProcessor) but not the
transformer-level wrapper that the pipeline base class wants. We provide
that here, modeled on xFuserCogVideoXTransformer3DWrapper.

Behavior:
- For pure CFG-parallel (sp=1, pp=1, tp=1): base class `_convert_transformer_for_parallel`
  short-circuits and returns the transformer unchanged. Our forward is a
  passthrough.
- For SP > 1 (Ulysses/Ring/USP): base class wraps the attention submodules
  (attn1 = self-attn, attn2 = cross-attn) — replacing their processor with
  `xFuserWanAttnProcessor` which does the seq-parallel all_to_all internally.
  We still passthrough — diffusers' WanTransformer3DModel.forward handles the
  rest unchanged because the attention modules now do SP transparently.

Submodule names verified against diffusers 0.38 WanTransformerBlock: attn1, attn2.
Blocks attribute: transformer.blocks (not "transformer_blocks").
"""

import inspect
from typing import Optional

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanTransformer3DModel,
)

from xfuser.logger import init_logger
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.model_executor.layers import xFuserLayerWrappersRegister
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionBaseWrapper,
    xFuserAttentionProcessorRegister,
)

# Importing this module side-effect-registers xFuserWanAttnProcessor against
# diffusers' WanAttnProcessor, which our layer wrapper below looks up.
import xfuser.model_executor.models.transformers.transformer_wan  # noqa: F401
from xfuser.model_executor.models.transformers.base_transformer import (
    xFuserTransformerBaseWrapper,
)
from xfuser.model_executor.models.transformers.register import (
    xFuserTransformerWrappersRegister,
)

logger = init_logger(__name__)


@xFuserLayerWrappersRegister.register(WanAttention)
class xFuserWanAttentionWrapper(xFuserAttentionBaseWrapper):
    """Layer wrapper for diffusers `WanAttention` modules.

    Mirrors `xFuserAttentionWrapper` for diffusers `Attention` but with
    WanAttention's forward signature, which takes `rotary_emb` as an
    extra arg (positional or kw) and does NOT take attention_mask in the
    same position.

    Looks up the SP-aware processor via `xFuserAttentionProcessorRegister`
    (which has `xFuserWanAttnProcessor` registered against `WanAttnProcessor`).
    """

    def __init__(self, attention: WanAttention, *_, **__):
        super().__init__(attention=attention)
        self.processor = xFuserAttentionProcessorRegister.get_processor(
            attention.processor
        )()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        # The processor's __call__ expects (attn, hidden_states, encoder_hidden_states,
        # attention_mask, rotary_emb). Forward through, dropping unknown kwargs.
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        cross_attention_kwargs = {
            k: v for k, v in cross_attention_kwargs.items() if k in attn_parameters
        }
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            **cross_attention_kwargs,
        )


@xFuserTransformerWrappersRegister.register(WanTransformer3DModel)
class xFuserWanTransformer3DWrapper(xFuserTransformerBaseWrapper):
    """Minimal xfuser wrapper for diffusers' WanTransformer3DModel.

    Targets two configurations we benchmark on 2× H100:
    1. CFG-parallel only (sp=1, cfg=2): no internal changes — pipeline-level
       wrapper handles the cond/uncond split; this wrapper is a passthrough.
    2. Ulysses-2 (sp=2, cfg=1): base class swaps `attn1`/`attn2` processors
       for xFuserWanAttnProcessor (which does Ulysses all_to_all internally).
       Forward stays unchanged because attention is now SP-aware.
    """

    def __init__(self, transformer: WanTransformer3DModel):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[],
            # Wan attention modules: attn1 (self), attn2 (cross). Both should
            # get the xfuser processor swap so SP works inside attention.
            submodule_name_to_wrap=["attn1", "attn2"],
            # Wan calls its main block list `blocks`, not `transformer_blocks`.
            transformer_blocks_name=["blocks"],
        )

    @xFuserBaseWrapper.forward_check_condition
    def forward(self, *args, **kwargs):
        # Pass through to the underlying transformer. When SP is active, the
        # attention modules have already been swapped in-place to xfuser
        # variants, so SP comm happens transparently inside each block.
        return self.module(*args, **kwargs)
