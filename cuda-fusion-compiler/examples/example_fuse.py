"""Example: y = gelu(x * w + b)

Demonstrates the full compiler pipeline: build a graph, run the fusion
pass, print the generated CUDA source, and report how many kernel launches
were saved. Runs with no GPU required -- only the optional JIT-compile-and-run
step at the bottom needs CUDA.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fusion_compiler.graph import Graph
from fusion_compiler.fuser import fuse
from fusion_compiler.codegen import generate_all


def build_graph() -> Graph:
    g = Graph()
    g.add("mul", ["x", "w"], "t1")
    g.add("add", ["t1", "b"], "t2")
    g.add("gelu", ["t2"], "y")
    return g


def build_diamond_graph() -> Graph:
    """A case where fusion is BLOCKED: t1 is consumed by two different
    downstream ops, so it can't be silently duplicated into both without a
    cost-model decision. Demonstrates the fuser correctly leaving it as its
    own group.
    """
    g = Graph()
    g.add("mul", ["x", "w"], "t1")
    g.add("relu", ["t1"], "left")
    g.add("sigmoid", ["t1"], "right")
    g.add("add", ["left", "right"], "y")
    return g


def run_example(name: str, graph: Graph) -> None:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(graph)

    groups = fuse(graph)
    print(f"\n{len(graph)} ops fused into {len(groups)} kernel(s):")
    for i, group in enumerate(groups):
        op_names = [n.op for n in group.nodes]
        print(f"  kernel {i}: {op_names}  (inputs={group.inputs}, output={group.output})")

    sources = generate_all(groups)
    for kname, src in sources.items():
        print(f"\n--- {kname}.cu ---")
        print(src)

    saved = len(graph) - len(groups)
    print(f"Kernel launches: {len(graph)} (unfused) -> {len(groups)} (fused). Saved {saved} launch(es).")


if __name__ == "__main__":
    run_example("Linear chain: y = gelu(x * w + b)", build_graph())
    run_example("Diamond (fan-out blocks fusion): y = relu(x*w) + sigmoid(x*w)", build_diamond_graph())

    # Optional: actually compile and run on GPU if available.
    try:
        import torch
        if torch.cuda.is_available():
            from fusion_compiler.jit import compile_group

            g = build_graph()
            groups = fuse(g)
            fn = compile_group(groups[0], kernel_name="demo_fused")

            x = torch.randn(1 << 20, device="cuda")
            w = torch.randn(1 << 20, device="cuda")
            b = torch.randn(1 << 20, device="cuda")
            out = fn(x, w, b)

            ref = torch.nn.functional.gelu(x * w + b, approximate="tanh")
            print("\nGPU run matches reference:", torch.allclose(out, ref, atol=1e-3))
        else:
            print("\n(No CUDA device found -- skipping JIT compile/run step. Source above is what would run.)")
    except ImportError:
        print("\n(torch not installed -- skipping JIT compile/run step.)")
