"""Correctness tests for sliding-window + global-token attention.

Compares the CUDA kernel (when available) against the dense reference
implementation. When no CUDA device is present, this instead checks that
the reference implementation itself is internally consistent (matches
plain full attention when window_size >= seq_len, i.e. nothing is masked).
"""

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from long_context_attention import sliding_window_attention, _reference_sliding_window_attention


def _full_attention(Q, K, V, scale):
    scores = torch.einsum("bhqd,bhkd->bhqk", Q.float(), K.float()) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", attn, V.float())


def test_reference_matches_full_attention_when_unmasked():
    torch.manual_seed(0)
    batch, heads, seq_len, head_dim = 1, 2, 32, 16
    Q = torch.randn(batch, heads, seq_len, head_dim)
    K = torch.randn(batch, heads, seq_len, head_dim)
    V = torch.randn(batch, heads, seq_len, head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    # window_size >= seq_len means every key is within every query's window,
    # so this should degrade to ordinary dense attention regardless of
    # global_tokens.
    out_ref = _reference_sliding_window_attention(Q, K, V, window_size=seq_len, global_tokens=[], scale=scale)
    out_full = _full_attention(Q, K, V, scale)

    torch.testing.assert_close(out_ref, out_full, atol=1e-4, rtol=1e-4)


def test_global_tokens_get_full_context_even_with_small_window():
    torch.manual_seed(1)
    batch, heads, seq_len, head_dim = 1, 1, 64, 16
    Q = torch.randn(batch, heads, seq_len, head_dim)
    K = torch.randn(batch, heads, seq_len, head_dim)
    V = torch.randn(batch, heads, seq_len, head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    out_small_window = _reference_sliding_window_attention(
        Q, K, V, window_size=1, global_tokens=[0], scale=scale
    )
    out_full = _full_attention(Q, K, V, scale)

    # Token 0 is global -> its row should match full dense attention exactly,
    # even though the window is tiny.
    torch.testing.assert_close(out_small_window[:, :, 0, :], out_full[:, :, 0, :], atol=1e-4, rtol=1e-4)

    # A non-global, non-adjacent row should generally NOT match full attention
    # (it's missing most of the context) -- sanity check they differ.
    assert not torch.allclose(out_small_window[:, :, 32, :], out_full[:, :, 32, :], atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU + compiled extension")
def test_cuda_kernel_matches_reference():
    torch.manual_seed(2)
    batch, heads, seq_len, head_dim = 1, 2, 512, 64
    device = "cuda"
    Q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=device)
    K = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=device)
    V = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=device)
    window = 64
    global_tokens = [0, 1, 2]
    scale = 1.0 / math.sqrt(head_dim)

    out_kernel = sliding_window_attention(Q, K, V, window, global_tokens, scale)
    out_ref = _reference_sliding_window_attention(Q, K, V, window, global_tokens, scale)

    torch.testing.assert_close(out_kernel.float(), out_ref, atol=2e-2, rtol=2e-2)
