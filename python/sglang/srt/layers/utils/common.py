import logging
import re

import torch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)


def get_layer_id(weight_name):
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None


def narrow_weight_tensor(
    loaded_weight: torch.Tensor, dim: int, start: int, size: int
) -> torch.Tensor:
    """Narrow a loaded weight along ``dim``.

    Handles both :class:`torch.Tensor` (via ``.narrow``) and the
    ``PySafeSlice`` objects returned by ``safetensors.safe_open.get_slice``.
    Indexing a ``PySafeSlice`` triggers I/O for exactly the requested byte
    range, so only the TP-rank's shard is ever read from disk.
    """
    if isinstance(loaded_weight, torch.Tensor):
        return loaded_weight.narrow(dim, start, size)
    ndim = len(loaded_weight.get_shape())
    idx = tuple(
        slice(start, start + size) if i == dim else slice(None)
        for i in range(ndim)
    )
    return loaded_weight[idx]


def get_weight_shape(loaded_weight: torch.Tensor):
    """Return the shape of a weight without materializing it.

    Works for both :class:`torch.Tensor` and ``PySafeSlice`` objects returned
    by ``safetensors.safe_open.get_slice``.  Use this instead of
    ``loaded_weight.shape`` whenever the weight may still be a lazy slice.
    """
    if isinstance(loaded_weight, torch.Tensor):
        return loaded_weight.shape
    return loaded_weight.get_shape()


def materialize_weight(loaded_weight: torch.Tensor) -> torch.Tensor:
    """Materialize a ``PySafeSlice`` into a :class:`torch.Tensor`.

    If *loaded_weight* is already a tensor this is a no-op.  Call this
    whenever downstream code needs to access ``.shape``, ``.dtype``,
    ``.narrow``, etc. and the partial-read optimisation in
    ``narrow_weight_tensor`` is not applicable.
    """
    if not isinstance(loaded_weight, torch.Tensor):
        return loaded_weight[:]
    return loaded_weight


def pad_or_narrow_weight(
    loaded_weight: torch.Tensor, input_dim: int, start_idx: int, shard_size: int
) -> torch.Tensor:
    # Padding with zeros for special case such as qwen2_5_VL's mlp which is not 8-aligned
    valid_size = max(loaded_weight.shape[input_dim] - start_idx, 0)

    if valid_size > 0:
        loaded_slice = loaded_weight.narrow(input_dim, start_idx, valid_size)
        pad_shape = list(loaded_weight.shape)
        pad_shape[input_dim] = shard_size - valid_size
        pad = torch.zeros(
            pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
        )
        return torch.cat([loaded_slice, pad], dim=input_dim)

    # All padding
    pad_shape = list(loaded_weight.shape)
    pad_shape[input_dim] = shard_size
    return torch.zeros(
        pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
    )


def copy_or_rebind_param(
    module: torch.nn.Module, name: str, new_value: torch.Tensor
) -> None:
    """Keep parameter identities stable for CUDA graph reuse and hot reload."""
    new_value = new_value.detach()
    param = getattr(module, name, None)
    if isinstance(param, Parameter):
        if param.data.shape == new_value.shape and param.data.dtype == new_value.dtype:
            param.data.copy_(new_value)
        else:
            param.data = new_value
        param.requires_grad_(False)
    else:
        setattr(module, name, Parameter(new_value, requires_grad=False))


class PPMissingLayer(torch.nn.Identity):
    # Adapted from
    # https://github.com/vllm-project/vllm/blob/18ed3132d2bfe1df9a74729457b69243955221e8/vllm/model_executor/models/utils.py#L468C1-L486C1
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.return_tuple = kwargs.get("return_tuple", False)

    def forward(self, *args, **kwargs):
        """
        Return the first arg from args or the first value from kwargs.

        Wraps the input in a tuple if `self.return_tuple` is True.
        """
        input = args[0] if args else next(iter(kwargs.values()))
        return (input,) if self.return_tuple else input
