// combine.cu
//
// Merges the windowed-local attention output (sliding_window_attention.cu,
// which already includes global KEYS inside the window pass) with the
// query-to-global-keys partial state (global_attention.cu) for non-global
// query rows, using log-sum-exp style accumulation so the result is
// identical to running full softmax over the union of "window" and
// "global" keys without ever forming that union explicitly.
//
// Global QUERY rows are simply overwritten by global_query_dense_attention_kernel
// and are skipped here (their attention is already dense/full by definition).

#include <cuda_runtime.h>
#include <math.h>

namespace lca {

__global__ void combine_local_and_global_kernel(
    float* __restrict__ O,              // [B, H, N, D], holds local-window result in, combined result out
    const float* __restrict__ local_m,  // [B, H, N] running max from the windowed pass
    const float* __restrict__ local_l,  // [B, H, N] running sum from the windowed pass
    const float* __restrict__ global_m, // [B, H, N] from query_to_global_keys_kernel
    const float* __restrict__ global_l,
    const float* __restrict__ global_acc, // [B, H, N, D] unnormalized weighted V from global keys
    const bool* __restrict__ is_global_query, // [N] true => skip, already dense
    int seq_len,
    int head_dim)
{
    int q_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_row >= seq_len) return;
    if (is_global_query[q_row]) return;

    int head = blockIdx.y, batch = blockIdx.z;
    long idx = ((long)batch * gridDim.y + head) * seq_len + q_row;

    float m_local = local_m[idx];
    float l_local = local_l[idx];
    float m_glob = global_m[idx];
    float l_glob = global_l[idx];

    float m_new = fmaxf(m_local, m_glob);
    float corr_local = __expf(m_local - m_new);
    float corr_glob = __expf(m_glob - m_new);

    float l_new = l_local * corr_local + l_glob * corr_glob;
    float inv_l = (l_new > 0.f) ? 1.f / l_new : 0.f;

    long out_base = idx * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        // O currently holds the *normalized* local output (local_acc / l_local).
        // Recover the unnormalized local accumulation before merging.
        float local_unnorm = O[out_base + d] * l_local;
        float merged = local_unnorm * corr_local + global_acc[out_base + d] * corr_glob;
        O[out_base + d] = merged * inv_l;
    }
}

void launch_combine_local_and_global(
    float* O,
    const float* local_m, const float* local_l,
    const float* global_m, const float* global_l, const float* global_acc,
    const bool* is_global_query,
    int batch, int heads, int seq_len, int head_dim,
    cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((seq_len + block.x - 1) / block.x, heads, batch);
    combine_local_and_global_kernel<<<grid, block, 0, stream>>>(
        O, local_m, local_l, global_m, global_l, global_acc, is_global_query, seq_len, head_dim);
}

} // namespace lca
