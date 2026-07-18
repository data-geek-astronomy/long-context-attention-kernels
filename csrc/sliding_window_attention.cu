// sliding_window_attention.cu
//
// Longformer-style local attention: query i attends only to keys in
// [i - window_size, i + window_size]. Implemented with:
//   - Tiling aligned to the window so out-of-window K/V tiles are never
//     loaded into shared memory (not just masked after the fact).
//   - Online (running) softmax, Flash-Attention style, so the local score
//     matrix for a query tile is never fully materialized.
//
// Layout: [batch, heads, seq_len, head_dim], row-major, fp16 in / fp32 accum.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

#define WARP_SIZE 32

// Tunable tile sizes. BLOCK_M queries per block, BLOCK_N keys per inner tile.
#ifndef BLOCK_M
#define BLOCK_M 64
#endif
#ifndef BLOCK_N
#define BLOCK_N 64
#endif
#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

namespace lca {

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// One CUDA block handles BLOCK_M consecutive queries for one (batch, head).
// Grid: (num_query_tiles, num_heads, batch)
__global__ void sliding_window_attention_fwd_kernel(
    const half* __restrict__ Q,   // [B, H, N, D]
    const half* __restrict__ K,   // [B, H, N, D]
    const half* __restrict__ V,   // [B, H, N, D]
    float* __restrict__ O,        // [B, H, N, D]  (accumulated fp32 out, cast on write)
    const int* __restrict__ global_tokens, // sorted indices, may be nullptr
    int num_global_tokens,
    int seq_len,
    int window_size,
    float scale)
{
    extern __shared__ half smem[];
    half* sQ = smem;                                  // [BLOCK_M, HEAD_DIM]
    half* sK = sQ + BLOCK_M * HEAD_DIM;                // [BLOCK_N, HEAD_DIM]
    half* sV = sK + BLOCK_N * HEAD_DIM;                // [BLOCK_N, HEAD_DIM]

    const int batch = blockIdx.z;
    const int head  = blockIdx.y;
    const int q_tile_start = blockIdx.x * BLOCK_M;

    const int B_stride_head = seq_len * HEAD_DIM;
    const long base_offset = ((long)batch * gridDim.y + head) * (long)B_stride_head;

    const half* Q_ptr = Q + base_offset;
    const half* K_ptr = K + base_offset;
    const half* V_ptr = V + base_offset;
    float* O_ptr = O + base_offset;

    const int tid = threadIdx.x;
    const int rows_per_thread = (BLOCK_M * HEAD_DIM + blockDim.x - 1) / blockDim.x;

    // Load Q tile into shared memory.
    for (int i = 0; i < rows_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < BLOCK_M * HEAD_DIM) {
            int row = idx / HEAD_DIM, col = idx % HEAD_DIM;
            int q_row = q_tile_start + row;
            sQ[idx] = (q_row < seq_len) ? Q_ptr[(long)q_row * HEAD_DIM + col] : __float2half(0.f);
        }
    }
    __syncthreads();

    // Running softmax state per query row, kept in registers by the thread
    // that "owns" that row (thread i handles query row i, i < BLOCK_M).
    float m_i = -FLT_MAX;   // running max
    float l_i = 0.f;        // running normalizer
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.f;

    const int q_row_global = q_tile_start + tid; // valid only if tid < BLOCK_M

    // Local window: only key tiles overlapping [q_row - window, q_row + window]
    // for ANY row in this query tile need to be visited.
    int k_lo = max(0, q_tile_start - window_size);
    int k_hi = min(seq_len, q_tile_start + BLOCK_M + window_size);

    for (int k_tile_start = k_lo; k_tile_start < k_hi; k_tile_start += BLOCK_N) {
        // Cooperative load of K, V tile.
        for (int i = 0; i < rows_per_thread; ++i) {
            int idx = tid + i * blockDim.x;
            if (idx < BLOCK_N * HEAD_DIM) {
                int row = idx / HEAD_DIM, col = idx % HEAD_DIM;
                int k_row = k_tile_start + row;
                half kv_k = (k_row < seq_len) ? K_ptr[(long)k_row * HEAD_DIM + col] : __float2half(0.f);
                half kv_v = (k_row < seq_len) ? V_ptr[(long)k_row * HEAD_DIM + col] : __float2half(0.f);
                sK[idx] = kv_k;
                sV[idx] = kv_v;
            }
        }
        __syncthreads();

        if (tid < BLOCK_M && q_row_global < seq_len) {
            int tile_n = min(BLOCK_N, seq_len - k_tile_start);
            for (int j = 0; j < tile_n; ++j) {
                int k_row_global = k_tile_start + j;
                bool in_window = abs(k_row_global - q_row_global) <= window_size;
                bool is_global_key = false;
                for (int g = 0; g < num_global_tokens; ++g) {
                    if (global_tokens[g] == k_row_global) { is_global_key = true; break; }
                }
                if (!in_window && !is_global_key) continue;

                float score = 0.f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += __half2float(sQ[tid * HEAD_DIM + d]) * __half2float(sK[j * HEAD_DIM + d]);
                }
                score *= scale;

                float m_new = fmaxf(m_i, score);
                float correction = __expf(m_i - m_new);
                float p = __expf(score - m_new);

                l_i = l_i * correction + p;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    acc[d] = acc[d] * correction + p * __half2float(sV[j * HEAD_DIM + d]);
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (tid < BLOCK_M && q_row_global < seq_len) {
        float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_ptr[(long)q_row_global * HEAD_DIM + d] = acc[d] * inv_l;
        }
    }
}

void launch_sliding_window_attention(
    const half* Q, const half* K, const half* V, float* O,
    const int* global_tokens, int num_global_tokens,
    int batch, int heads, int seq_len, int window_size, float scale,
    cudaStream_t stream)
{
    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, heads, batch);
    dim3 block(max(BLOCK_M, BLOCK_N));
    size_t smem_bytes = (BLOCK_M * HEAD_DIM + 2 * BLOCK_N * HEAD_DIM) * sizeof(half);

    sliding_window_attention_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, global_tokens, num_global_tokens, seq_len, window_size, scale);
}

} // namespace lca
