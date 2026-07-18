// bindings.cpp
// PyTorch extension entry point exposing the sliding-window / global-token
// attention kernels as a Python-callable op.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <vector>

namespace lca {

void launch_sliding_window_attention(
    const half* Q, const half* K, const half* V, float* O,
    const int* global_tokens, int num_global_tokens,
    int batch, int heads, int seq_len, int window_size, float scale,
    cudaStream_t stream);

void launch_global_query_dense_attention(
    const half* Q, const half* K, const half* V, float* O,
    const int* global_tokens, int num_global_tokens,
    int batch, int heads, int seq_len, float scale, cudaStream_t stream);

} // namespace lca

static void check_input(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat16, name, " must be fp16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor sliding_window_attention_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int64_t window_size, torch::Tensor global_tokens, double scale)
{
    check_input(Q, "Q"); check_input(K, "K"); check_input(V, "V");
    TORCH_CHECK(Q.dim() == 4, "Q must be [batch, heads, seq_len, head_dim]");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    TORCH_CHECK(head_dim == 64, "This build is compiled for HEAD_DIM=64");

    auto O = torch::empty({batch, heads, seq_len, head_dim},
                           Q.options().dtype(torch::kFloat32));

    auto global_tokens_i32 = global_tokens.to(torch::kInt32).contiguous();

    lca::launch_sliding_window_attention(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        O.data_ptr<float>(),
        global_tokens_i32.data_ptr<int>(),
        static_cast<int>(global_tokens_i32.numel()),
        batch, heads, seq_len, static_cast<int>(window_size),
        static_cast<float>(scale),
        at::cuda::getCurrentCUDAStream());

    // Overwrite the global-token query rows with dense attention (they need
    // full-sequence context, not just their window).
    lca::launch_global_query_dense_attention(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        O.data_ptr<float>(),
        global_tokens_i32.data_ptr<int>(),
        static_cast<int>(global_tokens_i32.numel()),
        batch, heads, seq_len, static_cast<float>(scale),
        at::cuda::getCurrentCUDAStream());

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sliding_window_attention_forward", &sliding_window_attention_forward,
          "Sliding window + global token attention forward (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("window_size"), py::arg("global_tokens"), py::arg("scale"));
}
