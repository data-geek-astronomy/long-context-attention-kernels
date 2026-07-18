"""Python-facing API for the long-context (sliding window + global token)
attention CUDA kernels.

Falls back to a pure-PyTorch reference implementation when the compiled
extension isn't available, so this package is importable (and testable)
even on machines without a CUDA build toolchain.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

try:
    import long_context_attention_kernels as _ext  # compiled CUDA extension
    _HAS_EXT = True
except ImportError:
    _ext = None
    _HAS_EXT = False


def _reference_sliding_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int,
    global_tokens: Sequence[int],
    scale: float,
) -> torch.Tensor:
    """Dense reference implementation: build the Longformer mask explicitly
    and run ordinary softmax attention. O(n^2) memory — for correctness
    testing on small sequences only, and as a CPU/no-GPU fallback.
    """
    batch, heads, seq_len, head_dim = Q.shape
    device = Q.device

    idx = torch.arange(seq_len, device=device)
    window_mask = (idx[:, None] - idx[None, :]).abs() <= window_size

    global_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if len(global_tokens) > 0:
        global_mask[list(global_tokens)] = True

    # A key is visible to a query if: within window, OR key is global, OR query is global.
    mask = window_mask | global_mask[None, :] | global_mask[:, None]

    scores = torch.einsum("bhqd,bhkd->bhqk", Q.float(), K.float()) * scale
    scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", attn, V.float())
    return out


def sliding_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int,
    global_tokens: Sequence[int] = (),
    scale: float | None = None,
    force_reference: bool = False,
) -> torch.Tensor:
    """Longformer-style sparse attention: each query attends to a local
    window plus a small set of global tokens, and global tokens attend
    densely to everything.

    Q, K, V: [batch, heads, seq_len, head_dim], fp16, CUDA.
    window_size: radius (tokens within +/- window_size are visible).
    global_tokens: sequence of token indices with dense (full) attention.
    scale: attention scale, defaults to 1/sqrt(head_dim).
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])

    use_ext = _HAS_EXT and Q.is_cuda and not force_reference
    if use_ext:
        global_idx = torch.as_tensor(list(global_tokens), dtype=torch.int32, device=Q.device)
        return _ext.sliding_window_attention_forward(Q, K, V, window_size, global_idx, scale)

    return _reference_sliding_window_attention(Q, K, V, window_size, global_tokens, scale)


__all__ = ["sliding_window_attention"]
