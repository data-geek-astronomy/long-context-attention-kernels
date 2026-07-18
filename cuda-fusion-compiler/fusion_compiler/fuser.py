"""The fusion pass.

Rule: two adjacent elementwise nodes A -> B (A's output feeds B) may be
fused into the same group iff:
  1. Both A and B are elementwise ops (see ops.is_elementwise).
  2. A's output has exactly one consumer in the whole graph (namely B).

Condition (2) is the standard real-world fusion constraint: if A's output
were used by two different downstream nodes, fusing A into just one of them
would mean recomputing A's work for the other consumer (or keeping a
separate materialized copy anyway), which is a genuine cost-model decision,
not something to do silently. So by default we keep A un-fused into B in
that case, and A ends up as its own single-node group (or fused with ITS
single-consumer... no, it has two, so it stays separate).

Implementation: union-find over node output names. Group membership is a
maximal weakly-connected component under the "fusable edge" relation above.
Groups are then topologically ordered (in original node order, which is
already topological since the Graph is built via sequential .add() calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .graph import Graph, Node
from .ops import is_elementwise


class _UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


@dataclass
class FusionGroup:
    nodes: List[Node] = field(default_factory=list)

    @property
    def inputs(self) -> List[str]:
        """External inputs to this group: names read but not produced within it."""
        produced = {n.output for n in self.nodes}
        seen = []
        for n in self.nodes:
            for i in n.inputs:
                if i not in produced and i not in seen:
                    seen.append(i)
        return seen

    @property
    def output(self) -> str:
        """The single external-facing output of this group. Assumes groups
        formed by `fuse()` have exactly one node whose output escapes the
        group (true by construction: we only merge single-consumer edges)."""
        internal = {i for n in self.nodes for i in n.inputs}
        externally_visible = [n.output for n in self.nodes if n.output not in internal]
        if len(externally_visible) != 1:
            raise ValueError(
                f"FusionGroup expected exactly one external output, got {externally_visible}"
            )
        return externally_visible[0]

    def __repr__(self) -> str:
        body = "\n    ".join(repr(n) for n in self.nodes)
        return f"FusionGroup(\n    {body}\n  )"


def fuse(graph: Graph) -> List[FusionGroup]:
    consumer_count: Dict[str, int] = {}
    for node in graph:
        for i in node.inputs:
            consumer_count[i] = consumer_count.get(i, 0) + 1

    uf = _UnionFind()
    for node in graph:
        uf.find(node.output)  # ensure registered

    for node in graph:
        if not is_elementwise(node.op):
            continue
        for input_name in node.inputs:
            producer = graph.producer_of(input_name)
            if producer is None:
                continue  # graph input, nothing to fuse with
            if not is_elementwise(producer.op):
                continue
            if consumer_count.get(producer.output, 0) != 1:
                continue  # producer feeds >1 consumer, don't fuse
            uf.union(producer.output, node.output)

    groups_by_root: Dict[str, FusionGroup] = {}
    order: List[str] = []
    for node in graph:  # graph.nodes is already topological order
        root = uf.find(node.output)
        if root not in groups_by_root:
            groups_by_root[root] = FusionGroup()
            order.append(root)
        groups_by_root[root].nodes.append(node)

    return [groups_by_root[r] for r in order]
