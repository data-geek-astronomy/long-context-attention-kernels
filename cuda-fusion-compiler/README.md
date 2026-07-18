# CUDA Fusion Compiler (toy)

A small, from-scratch demonstration of what "kernel fusion" actually means at
the compiler level — the same idea behind `torch.compile` / TorchInductor and
XLA, shrunk down to something you can read end-to-end in an afternoon.

Companion to [cuda-ml-kernels](https://github.com/data-geek-astronomy/cuda-ml-kernels)
(hand-written fused kernels like LayerNorm+GELU) and
[long-context-attention-kernels](../long-context-attention-kernels) (hand-written
sparse attention). Those repos fuse *specific, hand-picked* op sequences.
This one automates the decision: given an arbitrary small computation graph
of elementwise ops, it decides *which* ops to fuse and *generates* the CUDA
kernel source for each fused group.

## The problem this solves

`y = gelu(x * w + b)` naively runs as three separate CUDA kernels:

```
t1 = mul(x, w)      # kernel launch 1, writes t1 to global memory
t2 = add(t1, b)     # kernel launch 2, reads t1, writes t2
y  = gelu(t2)       # kernel launch 3, reads t2, writes y
```

Every intermediate (`t1`, `t2`) makes a full round trip to GPU global memory,
and every kernel launch has fixed overhead. If you fuse all three into one
kernel, each thread computes `mul -> add -> gelu` for one element while it's
still in a register, and only the final `y` ever touches global memory:

```
y = gelu(x * w + b)   # one kernel launch, one read of x/w/b, one write of y
```

That's what this project automates.

## How it works

1. **`fusion_compiler/graph.py`** — a minimal IR. A `Graph` is a list of
   `Node`s (op + input names + output name), similar in spirit to an FX
   graph or an ONNX graph, but deliberately tiny.

2. **`fusion_compiler/ops.py`** — a registry of elementwise ops
   (`add`, `sub`, `mul`, `div`, `relu`, `sigmoid`, `tanh`, `gelu`, `exp`,
   `neg`, `scalar_mul`, ...), each with a C expression template
   (e.g. `add: "{a} + {b}"`) used by codegen.

3. **`fusion_compiler/fuser.py`** — the fusion pass. Walks the graph and
   greedily groups maximal chains of elementwise ops where each intermediate
   has exactly one consumer (the standard "producer has a single use" fusion
   rule — fusing a value with >1 consumer would mean recomputing it multiple
   times, which is a real tradeoff real compilers make explicitly, not
   silently).

4. **`fusion_compiler/codegen.py`** — for each fused group, emits a single
   CUDA `__global__` kernel: a grid-stride loop that inlines every op in the
   group as one straight-line expression, so intermediates live in registers,
   never global memory.

5. **`fusion_compiler/jit.py`** — optional runtime path: if `torch` and a
   CUDA toolchain are available, compiles the generated source with
   `torch.utils.cpp_extension.load_inline` and returns a callable. If not
   (e.g. this dev machine has no GPU), the compiler still runs end-to-end and
   you can inspect the generated `.cu` source — the whole point is to see
   *what* gets generated, which doesn't require a GPU.

## Example

```bash
python examples/example_fuse.py
```

```python
from fusion_compiler.graph import Graph

g = Graph()
g.add("mul", ["x", "w"], "t1")
g.add("add", ["t1", "b"], "t2")
g.add("gelu", ["t2"], "y")

from fusion_compiler.fuser import fuse
from fusion_compiler.codegen import generate_cuda_source

groups = fuse(g)                     # -> 1 FusionGroup containing all 3 ops
source = generate_cuda_source(groups[0], kernel_name="fused_kernel_0")
print(source)
```

Generated kernel (abridged):

```cuda
extern "C" __global__ void fused_kernel_0(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, float* __restrict__ y, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float t1 = x[i] * w[i];
        float t2 = t1 + b[i];
        float y_val = 0.5f * t2 * (1.0f + tanhf(0.7978845608f * (t2 + 0.044715f * t2*t2*t2)));
        y[i] = y_val;
    }
}
```

Three kernel launches and two global-memory round trips collapse into one of
each.

## What it deliberately does NOT do (toy project scope)

- No fusion across non-elementwise ops (reductions, matmuls, reshapes) —
  that's what makes TorchInductor's fusion pass hard, and out of scope here.
- No cost model / autotuning of block size — fixed 256-thread blocks.
- No multi-output group support beyond the graph's final nodes.
- No broadcasting; all tensors in a fused group are assumed same shape.

These are the natural "next steps" if you want to extend it toward something
closer to a real fusion compiler.

## Tests

```bash
pip install -r requirements.txt   # just pytest, no GPU needed
pytest tests/
```

Tests validate the fusion grouping logic (which ops get merged, and that a
node with multiple consumers correctly blocks fusion) and check the generated
CUDA source contains the expected inlined expression — all without needing a
GPU or CUDA toolchain, since the interesting part is the compiler, not the
execution.

## License

MIT
