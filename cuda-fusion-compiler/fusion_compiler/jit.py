"""Optional runtime path: compile generated CUDA source and load it as a
callable PyTorch op, using torch.utils.cpp_extension.load.

This is intentionally isolated from graph/fuser/codegen so those modules
(the actual "compiler" logic this project is about) are testable without a
GPU or CUDA toolchain. This module is only exercised when both are present.

We write the generated source to a real .cu file and build it with
torch.utils.cpp_extension.load() rather than load_inline(...), purely for
the real on-disk build directory, which is easier to debug if compilation
fails. This torch build requires ninja unconditionally for JIT extension
builds (no use_ninja kwarg, no distutils fallback), so a working `ninja`
binary must be available on PATH -- see this project's packages.txt
("ninja-build" via apt), since the pip "ninja" wheel's bundled binary has
been observed failing with exit code 127 in some containers.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List

from .fuser import FusionGroup
from .codegen import generate_cuda_source

try:
    import torch
    from torch.utils.cpp_extension import load
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _use_torch_bundled_cuda_toolchain():
    """Point CUDA_HOME/PATH/LD_LIBRARY_PATH at the CUDA toolchain that ships
    as a pip dependency of the installed torch wheel (nvidia-cuda-nvcc-cu12,
    nvidia-cuda-runtime-cu12, etc.) instead of whatever CUDA toolkit happens
    to be installed system-wide in the container, so compile-time and
    run-time CUDA library versions stay consistent."""
    try:
        import nvidia
    except ImportError:
        return
    import glob

    nvidia_root = os.path.dirname(nvidia.__file__)
    lib_dirs = glob.glob(os.path.join(nvidia_root, "*", "lib"))
    if lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")])

    nvcc_dir = os.path.join(nvidia_root, "cuda_nvcc", "bin")
    if os.path.isdir(nvcc_dir):
        os.environ["PATH"] = nvcc_dir + ":" + os.environ.get("PATH", "")
        os.environ["CUDA_HOME"] = os.path.join(nvidia_root, "cuda_nvcc")


def _full_source(group: FusionGroup, kernel_name: str) -> str:
    """One self-contained .cu file: the generated kernel, a launcher, a
    torch-tensor-facing wrapper function, and its pybind11 bindings --
    everything nvcc needs in a single translation unit."""
    cuda_source = generate_cuda_source(group, kernel_name=kernel_name)
    num_inputs = len(group.inputs)
    args = ", ".join(f"torch::Tensor in{i}" for i in range(num_inputs))
    call_args = ", ".join(f"in{i}.data_ptr<float>()" for i in range(num_inputs))
    kernel_args = ", ".join(f"ins[{i}]" for i in range(num_inputs))

    return f"""
#include <torch/extension.h>

{cuda_source}

extern "C" void launch_{kernel_name}(const float** ins, float* out, int n) {{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    {kernel_name}<<<blocks, threads>>>({kernel_args}{', ' if num_inputs else ''}out, n);
}}

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

    _use_torch_bundled_cuda_toolchain()

    build_dir = os.path.join(tempfile.gettempdir(), f"{kernel_name}_build")
    os.makedirs(build_dir, exist_ok=True)
    src_path = os.path.join(build_dir, f"{kernel_name}.cu")
    with open(src_path, "w") as f:
        f.write(_full_source(group, kernel_name))

    module = load(
        name=kernel_name,
        sources=[src_path],
        verbose=False,
        build_directory=build_dir,
    )
    return module.run


def compile_all(groups: List[FusionGroup]) -> Dict[str, object]:
    return {
        f"fused_kernel_{idx}": compile_group(g, kernel_name=f"fused_kernel_{idx}")
        for idx, g in enumerate(groups)
    }
