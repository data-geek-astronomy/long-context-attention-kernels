from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="long-context-attention",
    version="0.1.0",
    description="Sliding window + global token (Longformer-style) CUDA attention kernels",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=[
        CUDAExtension(
            name="long_context_attention_kernels",
            sources=[
                "csrc/sliding_window_attention.cu",
                "csrc/global_attention.cu",
                "csrc/combine.cu",
                "csrc/bindings.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0", "numpy"],
)
