"""Registry of supported elementwise ops and their CUDA expression templates.

Every op is "elementwise" in the fusion sense: given already-computed scalar
inputs for a single element, it produces a single scalar output with no
cross-element dependency. That's exactly the property that makes fusion
valid without any data-flow analysis beyond "is every op elementwise".

Each entry maps op name -> (arity, expr_template). expr_template is a Python
str.format() template using placeholder names {a}, {b} (positional inputs)
and any declared scalar args (e.g. {alpha}).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class OpSpec:
    arity: int                 # number of tensor inputs
    template: str               # C expression template, uses {a}, {b}, ... and scalar arg names
    scalar_args: tuple = ()     # names of required scalar args, e.g. ("alpha",)
    reference: Callable = None  # optional pure-python reference impl for testing, fn(*args, **kwargs)


OP_REGISTRY: Dict[str, OpSpec] = {
    "add": OpSpec(2, "({a} + {b})"),
    "sub": OpSpec(2, "({a} - {b})"),
    "mul": OpSpec(2, "({a} * {b})"),
    "div": OpSpec(2, "({a} / {b})"),
    "neg": OpSpec(1, "(-{a})"),
    "relu": OpSpec(1, "fmaxf({a}, 0.0f)"),
    "sigmoid": OpSpec(1, "(1.0f / (1.0f + expf(-{a})))"),
    "tanh": OpSpec(1, "tanhf({a})"),
    "exp": OpSpec(1, "expf({a})"),
    "sqrt": OpSpec(1, "sqrtf({a})"),
    "scalar_mul": OpSpec(1, "({a} * {alpha}f)", scalar_args=("alpha",)),
    "scalar_add": OpSpec(1, "({a} + {alpha}f)", scalar_args=("alpha",)),
    # GELU (tanh approximation), same formula used in the LayerNorm+GELU
    # kernel in cuda-ml-kernels -- expressed here as a single fused op so the
    # fuser can treat it as one node, or you could decompose it into
    # mul/add/tanh nodes and let the fuser re-derive the same kernel.
    "gelu": OpSpec(
        1,
        "(0.5f * {a} * (1.0f + tanhf(0.7978845608f * ({a} + 0.044715f * {a} * {a} * {a}))))",
    ),
}


def is_elementwise(op_name: str) -> bool:
    return op_name in OP_REGISTRY


def op_arity(op_name: str) -> int:
    return OP_REGISTRY[op_name].arity
