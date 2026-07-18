"""Minimal computation-graph IR.

Deliberately tiny: a Graph is just an ordered list of Nodes. Each Node is one
op applied to named inputs, producing one named output. Graph inputs are any
name that's never produced by a node; graph outputs are names never consumed
(or explicitly marked).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Node:
    op: str                 # op name, must exist in ops.OP_REGISTRY
    inputs: List[str]       # names of input values (graph inputs or other nodes' outputs)
    output: str              # name of the value this node produces
    scalar_args: dict = field(default_factory=dict)  # e.g. {"alpha": 0.5} for scalar_mul

    def __repr__(self) -> str:
        args = ", ".join(self.inputs)
        extra = f", {self.scalar_args}" if self.scalar_args else ""
        return f"{self.output} = {self.op}({args}{extra})"


class Graph:
    """Ordered list of Nodes forming a DAG (single static assignment: each
    output name is produced exactly once)."""

    def __init__(self):
        self.nodes: List[Node] = []
        self._outputs = set()

    def add(self, op: str, inputs: List[str], output: str, **scalar_args) -> Node:
        if output in self._outputs:
            raise ValueError(f"output name '{output}' already produced by another node (not SSA)")
        node = Node(op=op, inputs=list(inputs), output=output, scalar_args=scalar_args)
        self.nodes.append(node)
        self._outputs.add(output)
        return node

    def producer_of(self, name: str) -> Optional[Node]:
        """Return the Node that produces `name`, or None if `name` is a graph input."""
        for node in self.nodes:
            if node.output == name:
                return node
        return None

    def consumers_of(self, name: str) -> List[Node]:
        """All nodes that read `name` as an input."""
        return [n for n in self.nodes if name in n.inputs]

    def graph_inputs(self) -> List[str]:
        """Names consumed but never produced within this graph -- external inputs."""
        produced = {n.output for n in self.nodes}
        seen = []
        for n in self.nodes:
            for i in n.inputs:
                if i not in produced and i not in seen:
                    seen.append(i)
        return seen

    def final_outputs(self) -> List[str]:
        """Names produced but never consumed within this graph -- graph outputs."""
        consumed = {i for n in self.nodes for i in n.inputs}
        return [n.output for n in self.nodes if n.output not in consumed]

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __repr__(self) -> str:
        return "Graph(\n  " + "\n  ".join(repr(n) for n in self.nodes) + "\n)"
