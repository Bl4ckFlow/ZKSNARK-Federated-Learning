"""Integer fixed-point arithmetic, bit-identical to the circuit.

The whole point of this module is that there is no back-derivation anywhere.
The circuit constrains a forward computation; this file *is* that forward
computation, in the same integers, so a witness is just the natural byproduct
of training.  The previous design ran training in float64 and then solved the
circuit's equation backwards for a "gradient" witness -- which is exactly why
any target model was provable.

Two properties matter and both are tested:

1. Every division here is Python's ``//`` on a positive divisor, i.e. floor
   division with a non-negative remainder, which is what ``DivConst`` in
   ``circuits/lib/fixedpoint.circom`` constrains.
2. No float ever reaches a value that gets hashed or committed.  Quantization
   uses :class:`fractions.Fraction`, which is exact on IEEE-754 doubles, so two
   machines quantizing the same dataset get the same integers -- unlike a
   float64 reduction, whose result depends on the BLAS build.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence

from zkfl.config import FIELD_PRIME, FixedPointConfig

Vector = list[int]
Matrix = list[list[int]]

# Piecewise-linear sigmoid, mirroring circuits/lib/activation.circom.
# Breakpoints on |z|, the sigmoid there, and the chord slopes -- all scaled by
# 10**6.  Regenerate with scripts/gen_activation_constants.py.
PWL_T = (0, 1_000_000, 2_000_000, 3_000_000, 5_000_000)
PWL_C = (500_000, 731_059, 880_797, 952_574, 993_307)
PWL_M = (231_059, 149_738, 71_777, 20_367, 0)
PWL_SCALE = 10 ** 6
#: worst-case |sigma_pwl(z) - sigma(z)| over z in [-12, 12]; asserted by tests
PWL_MAX_ERROR = 0.0117


def quantize(value: float, scale: int) -> int:
    """Round a real to the scaled integer grid, deterministically.

    ``Fraction(value)`` is the exact rational the double represents, so the
    rounding below involves no floating-point arithmetic at all and is
    reproducible on any machine.  Ties go away from zero.
    """
    exact = Fraction(value) * scale
    if exact >= 0:
        return (exact.numerator * 2 + exact.denominator) // (2 * exact.denominator)
    neg = -exact
    return -((neg.numerator * 2 + neg.denominator) // (2 * neg.denominator))


def quantize_vector(values: Iterable[float], scale: int) -> Vector:
    return [quantize(v, scale) for v in values]


def dequantize(value: int, scale: int) -> float:
    """Back to a float, for reporting only -- never for anything hashed."""
    return value / scale


def dequantize_vector(values: Iterable[int], scale: int) -> list[float]:
    return [v / scale for v in values]


def pwl_sigmoid(z: int, scale: int = PWL_SCALE) -> int:
    """The protocol's activation, scaled.  Matches ``PWLSigmoid`` exactly."""
    if scale != PWL_SCALE:
        raise ValueError(
            f"the PWL constants are fixed for scale {PWL_SCALE}; "
            "regenerate circuits/lib/activation.circom to change it"
        )
    negative = z < 0
    a = -z if negative else z

    piece = 4
    for i in range(4):
        if a < PWL_T[i + 1]:
            piece = i
            break

    # slope * (a - anchor) has scale^2; one floor division brings it back.
    q = (PWL_M[piece] * (a - PWL_T[piece])) // scale
    sig_abs = PWL_C[piece] + q
    return (scale - sig_abs) if negative else sig_abs


def sample_errors(
    w: Sequence[int], b: int, xs: Matrix, ys: Sequence[int], scale: int
) -> Vector:
    """Per-sample ``sigma_pwl(z) - y``, scaled.  Section 3 of the circuit."""
    n = len(w)
    errs: Vector = []
    for x, y in zip(xs, ys):
        if len(x) != n:
            raise ValueError(f"sample has {len(x)} features, model has {n}")
        if y not in (0, 1):
            raise ValueError(f"labels must be binary, got {y}")
        dot = sum(x[j] * w[j] for j in range(n))
        z = dot // scale + b
        errs.append(pwl_sigmoid(z, scale) - y * scale)
    return errs


def training_step(
    w: Sequence[int],
    b: int,
    xs: Matrix,
    ys: Sequence[int],
    eta: int,
    scale: int,
) -> tuple[Vector, int]:
    """One full-batch logistic-regression step, exactly as constrained.

    ``w_new[j] = w[j] - floor(eta * sum_i err_i * x_ij / (k * scale^2))``
    ``b_new    = b    - floor(eta * sum_i err_i      / (k * scale))``
    """
    k = len(xs)
    if k == 0:
        raise ValueError("empty batch")
    n = len(w)
    errs = sample_errors(w, b, xs, ys, scale)

    w_new = [
        w[j] - (eta * sum(errs[i] * xs[i][j] for i in range(k))) // (k * scale * scale)
        for j in range(n)
    ]
    b_new = b - (eta * sum(errs)) // (k * scale)
    return w_new, b_new


def step_within_bound(
    w_old: Sequence[int], b_old: int, w_new: Sequence[int], b_new: int, max_delta: int
) -> bool:
    """The clipping bound the circuit enforces, checkable outside it."""
    if any(abs(a - c) > max_delta for a, c in zip(w_old, w_new)):
        return False
    return abs(b_new - b_old) <= max_delta


def in_signed_range(value: int, bits: int) -> bool:
    return -(1 << bits) <= value < (1 << bits)


def model_in_range(w: Sequence[int], b: int, cfg: FixedPointConfig) -> bool:
    """Whether a model satisfies the circuit's own range checks.

    A proposal outside this cannot be proved, so rejecting it early saves a
    verification and gives a clearer reason than "proof failed".
    """
    return all(in_signed_range(v, cfg.w_bits) for v in w) and in_signed_range(b, cfg.w_bits)


def to_field_str(value: int) -> str:
    """Signed integer to the decimal string circom's witness generator wants."""
    return str(value % FIELD_PRIME)
