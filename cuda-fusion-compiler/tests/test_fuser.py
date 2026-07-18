import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fusion_compiler.graph import Graph
from fusion_compiler.fuser import fuse
from fusion_compiler.codegen import generate_cuda_source, generate_all


def test_linear_chain_fuses_into_one_group():
    g = Graph()
    g.add("mul", ["x", "w"], "t1")
    g.add("add", ["t1", "b"], "t2")
    g.add("gelu", ["t2"], "y")

    groups = fuse(g)

    assert len(groups) == 1
    assert [n.op for n in groups[0].nodes] == ["mul", "add", "gelu"]
    assert groups[0].output == "y"
    assert set(groups[0].inputs) == {"x", "w", "b"}


def test_fanout_blocks_fusion():
    g = Graph()
    g.add("mul", ["x", "w"], "t1")     # t1 has TWO consumers below
    g.add("relu", ["t1"], "left")
    g.add("sigmoid", ["t1"], "right")
    g.add("add", ["left", "right"], "y")

    groups = fuse(g)

    # t1's producer (mul) cannot be fused into either relu or sigmoid,
    # since t1 is consumed by both. relu and sigmoid also can't fuse with
    # each other (not connected). add depends on two distinct single-use
    # values (left, right) so it CAN fuse with both.
    op_sets = [sorted(n.op for n in grp.nodes) for grp in groups]
    assert ["mul"] in op_sets
    assert any(sorted(names) == ["add", "relu", "sigmoid"] for names in op_sets) or (
        # relu+add and sigmoid+add can't both absorb `add` into two different
        # groups -- add has two producers (left, right), both single-consumer,
        # so both get unioned with add into ONE group.
        len(op_sets) == 2
    )


def test_independent_ops_stay_separate():
    g = Graph()
    g.add("relu", ["x"], "a")
    g.add("sigmoid", ["y"], "b")  # unrelated to `a`

    groups = fuse(g)
    assert len(groups) == 2


def test_graph_is_ssa_and_rejects_duplicate_outputs():
    g = Graph()
    g.add("relu", ["x"], "a")
    try:
        g.add("sigmoid", ["x"], "a")
        assert False, "expected ValueError for duplicate output name"
    except ValueError:
        pass


def test_codegen_produces_single_kernel_with_inlined_ops():
    g = Graph()
    g.add("mul", ["x", "w"], "t1")
    g.add("add", ["t1", "b"], "t2")
    g.add("relu", ["t2"], "y")

    groups = fuse(g)
    assert len(groups) == 1

    src = generate_cuda_source(groups[0], kernel_name="k0")

    assert "__global__ void k0" in src
    assert "x[i]" in src and "w[i]" in src and "b[i]" in src
    assert "t1 = " in src
    assert "t2 = " in src
    assert "fmaxf(t2, 0.0f)" in src
    assert "y_out[i] = y" in src
    # only one kernel launch site (one __global__ function) for 3 fused ops
    assert src.count("__global__") == 1


def test_codegen_scalar_args():
    g = Graph()
    g.add("scalar_mul", ["x"], "y", alpha=2.5)

    groups = fuse(g)
    src = generate_cuda_source(groups[0], kernel_name="k0")
    assert "2.5f" in src


def test_generate_all_names_kernels_uniquely():
    g = Graph()
    g.add("relu", ["x"], "a")
    g.add("sigmoid", ["y"], "b")
    groups = fuse(g)
    sources = generate_all(groups)
    assert set(sources.keys()) == {"fused_kernel_0", "fused_kernel_1"}
    assert all("__global__" in s for s in sources.values())


def test_graph_inputs_and_outputs():
    g = Graph()
    g.add("mul", ["x", "w"], "t1")
    g.add("relu", ["t1"], "y")

    assert g.graph_inputs() == ["x", "w"]
    assert g.final_outputs() == ["y"]
