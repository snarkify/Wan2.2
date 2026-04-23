# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import time

import torch
import torch.distributed as dist


def init_distributed_group():
    """r initialize sequence parallel group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def _time_collective(op_name: str, nbytes: int, fn):
    """Run fn() while timing wall + GPU; record via profiling if enabled."""
    from wan.profiling import get_config, record_collective
    if not get_config().enabled:
        return fn()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    t0 = time.perf_counter()
    result = fn()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    end_evt.record()
    torch.cuda.synchronize()
    gpu_ms = start_evt.elapsed_time(end_evt)
    record_collective(op_name, nbytes, wall_ms, gpu_ms)
    return result


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        # Total bytes shipped per rank in all_to_all = input tensor size (each
        # rank sends world_size-1 slices of input/world_size and receives same).
        nbytes = x.element_size() * x.numel()

        def _do():
            dist.all_to_all(outputs, inputs, group=group, **kwargs)

        _time_collective("all_to_all", nbytes, _do)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    nbytes = tensor.element_size() * tensor.numel() * world_size

    def _do():
        torch.distributed.all_gather(tensor_list, tensor)

    _time_collective("all_gather", nbytes, _do)
    return tensor_list


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input)
    return torch.cat(output, dim=dim).contiguous()
