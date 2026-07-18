// global_attention.cu
//
// Handles the "global token" half of Longformer-style attention:
//   - Global tokens attend densely to ALL positions (not just their window).
//   - ALL positions attend densely back to the global tokens.
// Both directions are small relative to full attention because
// num_global_tokens << seq_len, so this is a comparatively cheap dense
// GEMM-style kernel, kept separate from the windowed kernel because the
// memory access pattern (gather over a short global-index list, scatter
// over the full sequence) is fundamentally different from tiled windowed
// access.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif
#ifndef GLOBAL_BLOCK
#define GLOBAL_BLOCK 128
#endif

namespace lca {

// Direction A: every global query token attends to every key in the sequence.
// Grid: (num_global_tokens, heads, batch), one block per global query.
__global__ void global_query_dense_attention_kernel(
    const half* __restrict__ Q,       // [B, H, N, D]
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ O,            // overwritten for global rows only
    const int* __restrict__ global_tokens,
    int seq_len,
    float scale)
{
    const int g = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int q_row = global_tokens[g];

    const long base = ((long)batch * gridDim.y + head) * (long)seq_len * HEAD_DIM;
    const half* Q_ptr = Q + base;
    const half* K_ptr = K + base;
    const half* V_ptr = V + base;
    float* O_ptr = O + base;

    extern __shared__ float shared_scores[]; // [GLOBAL_BLOCK]
    float q_vec[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) q_vec[d] = __half2float(Q_ptr[(long)q_row * HEAD_DIM + d]);

    float m = -FLT_MAX, l = 0.f;
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.f;

    const int tid = threadIdx.x;
    for (int tile_start = 0; tile_start < seq_len; tile_start += GLOBAL_BLOCK) {
        int tile_len = min(GLOBAL_BLOCK, seq_len - tile_start);
        if (tid < tile_len) {
            int k_row = tile_start + tid;
            float score = 0.f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += q_vec[d] * __half2float(K_ptr[(long)k_row * HEAD_DIM + d]);
            }
            shared_scores[tid] = score * scale;
        }
        __syncthreads();

        if (tid == 0) {
            for (int j = 0; j < tile_len; ++j) {
                float score = shared_scores[j];
                int k_row = tile_start + j;
                float m_new = fmaxf(m, score);
                float corr = __expf(m - m_new);
                float p = __expf(score - m_new);
                l = l * corr + p;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    acc[d] = acc[d] * corr + p * __half2float(V_ptr[(long)k_row * HEAD_DIM + d]);
                }
                m = m_new;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float inv_l = (l > 0.f) ? 1.f / l : 0.f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) O_ptr[(long)q_row * HEAD_DIM + d] = acc[d] * inv_l;
    }
}

// Direction B: every position attends to the global tokens as additional
// keys. This is folded into the online-softmax combine step (combine.cu)
// rather than overwriting O directly, since non-global queries already have
// partial windowed-attention state that must be merged, not replaced.
// This kernel just precomputes, for each non-global query, the (max, sum,
// weighted-V) triple restricted to global keys, so combine.cu can merge it.
__global__ void query_to_global_keys_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ partial_m,   // [B, H, N]
    float* __restrict__ partial_l,   // [B, H, N]
    float* __restrict__ partial_acc, // [B, H, N, D]
    const int* __restrict__ global_tokens,
    int num_global_tokens,
    int seq_len,
    float scale)
{
    int q_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_row >= seq_len) return;

    int head = blockIdx.y, batch = blockIdx.z;
    long base = ((long)batch * gridDim.y + head) * (long)seq_len * HEAD_DIM;
    const half* Q_ptr = Q + base;
    const half* K_ptr = K + base;
    const half* V_ptr = V + base;

    float m = -FLT_MAX, l = 0.f;
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.f;

    for (int g = 0; g < num_global_tokens; ++g) {
        int k_row = global_tokens[g];
        float score = 0.f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            score += __half2float(Q_ptr[(long)q_row * HEAD_DIM + d]) * __half2float(K_ptr[(long)k_row * HEAD_DIM + d]);
        }
        score *= scale;
        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * corr + p;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] = acc[d] * corr + p * __half2float(V_ptr[(long)k_row * HEAD_DIM + d]);
        m = m_new;
    }

    long out_idx = ((long)batch * gridDim.y + head) * seq_len + q_row;
    partial_m[out_idx] = m;
    partial_l[out_idx] = l;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) partial_acc[out_idx * HEAD_DIM + d] = acc[d];
}

void launch_global_query_dense_attention(
    const half* Q, const half* K, const half* V, float* O,
    const int* global_tokens, int num_global_tokens,
    int batch, int heads, int seq_len, float scale, cudaStream_t stream)
{
    dim3 grid(num_global_tokens, heads, batch);
    dim3 block(GLOBAL_BLOCK);
    size_t smem = GLOBAL_BLOCK * sizeof(float);
    global_query_dense_attention_kernel<<<grid, block, smem, stream>>>(
        Q, K, V, O, global_tokens, seq_len, scale);
}

void launch_query_to_global_keys(
    const half* Q, const half* K, const half* V,
    float* partial_m, float* partial_l, float* partial_acc,
    const int* global_tokens, int num_global_tokens,
    int batch, int heads, int seq_len, float scale, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((seq_len + block.x - 1) / block.x, heads, batch);
    query_to_global_keys_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, partial_m, partial_l, partial_acc, global_tokens, num_global_tokens, seq_len, scale);
}

} // namespace lca
