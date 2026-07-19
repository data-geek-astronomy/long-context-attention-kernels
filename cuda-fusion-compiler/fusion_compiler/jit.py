"""Optional runtime path: compile generated CUDA source and load it as a
callable PyTorch op, using torch.utils.cpp_extension.load_inline.

This is intentionally isolated from graph/fuser/codegen so those modules
(the actual "compiler" logic this project is about) are testable without a
GPU or CUDA toolchain. This module is only exercised when both are present.
"""

from __future__ import annotations

from typing import Dict, List

from .fuser import FusionGroup
from .codegen import generate_cuda_source

try:
    import torch
    from torch.utils.cpp_extension import load_inline
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _cpp_wrapper(kernel_name: str, num_inputs: int) -> str:
    args = ", ".join(f"torch::Tensor in{i}" for i in range(num_inputs))
    call_args = ", ".join(f"in{i}.data_ptr<float>()" for i in range(num_inputs))
    return f"""
#include <torch/extension.h>

extern "C" void launch_{kernel_name}(const float** ins, float* out, int n);

torch::Tensor {kernel_name}_call({args}) {{
    auto out = torch::empty_like(in0);
    int n = in0.numel();
    const float* ins[{max(num_inputs, 1)}] = {{ {call_args} }};
    launch_{kernel_name}(ins, out.data_ptr<float>(), n);
    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &{kernel_name}_call, "fused kernel");
}}
"""


def compile_group(group: FusionGroup, kernel_name: str = "fused_kernel"):
    """Compile a single FusionGroup into a callable(*tensors) -> tensor.

    Raises RuntimeError if torch / a CUDA toolchain isn't available -- callers
    that only care about inspecting generated source should use
    fusion_compiler.codegen.generate_cuda_source directly instead, which has
    no such dependency.
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch is required to JIT-compile fusion groups; codegen.generate_cuda_source works without it")
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available to JIT-compile fusion groups")

    cuda_source = generate_cuda_source(group, kernel_name=kernel_name)
    num_inputs = len(group.inputs)

    # Minimal C launcher bridging the raw extern "C" kernel to a simple
    # pointer-array ABI the C++ wrapper above can call.
    launcher = f"""
{cuda_source}

extern "C" void launch_{kernel_name}(const float** ins, float* out, int n) {{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    {kernel_name}<<<blocks, threads>>>({', '.join(f'ins[{i}]' for i in range(num_inputs))}{', ' if num_inputs else ''}out, n);
}}
"""

    # NOTE: cpp_sources already contains a hand-written PYBIND11_MODULE
    # block (see _cpp_wrapper above), so we must NOT also pass `functions=`
    # here -- that tells load_inline to auto-generate its own bindings on
    # top of ours, producing a duplicate PYBIND11_MODULE definition and a
    # compile error.
    module = load_inline(
        name=kernel_name,
        cpp_sources=[_cpp_wrapper(kernel_name, max(num_inputs, 1))],
        cuda_sources=[launcher],
        verbose=False,
        use_ninja=False,
    )
    return module.run


def compile_all(groups: List[FusionGroup]) -> Dict[str, object]:
    return {
        f"fused_kernel_{idx}": compile_group(g, kernel_name=f"fused_kernel_{idx}")
        for idx, g in enumerate(groups)
    }
