cmake_minimum_required(VERSION 3.18)
project(long_context_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

add_library(long_context_attention_kernels SHARED
    csrc/sliding_window_attention.cu
    csrc/global_attention.cu
    csrc/combine.cu
    csrc/bindings.cpp
)

target_include_directories(long_context_attention_kernels PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(long_context_attention_kernels PRIVATE ${TORCH_LIBRARIES} pybind11::module)
target_compile_options(long_context_attention_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)
set_target_properties(long_context_attention_kernels PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
