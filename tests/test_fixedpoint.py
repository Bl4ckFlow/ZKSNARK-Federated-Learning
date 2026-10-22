"""The Python arithmetic and the circuit must agree exactly.

These are the tests that make "cross-machine determinism" a claim with evidence
rather than an assertion.  The old suite checked that two calls on one machine
produced the same hash, which float64 satisfies trivially and says nothing
about a different BLAS build.
"""
from __future__ import annotations

import math
import random

import pytest

from zkfl.config import FIELD_PRIME, FixedPointConfig
from zkfl.fixedpoint import (
    PWL_C,
    PWL_M,
    PWL_MAX_ERROR,
    PWL_T,
    dequantize,
    model_in_range,
    pwl_sigmoid,
    quantize,
    sample_errors,
    step_within_bound,
    training_step,
)


def test_quantize_is_exact_and_symmetric():
    # Ties away from zero, and no floating-point arithmetic in the path.
    assert quantize(1.5, 1) == 2
    assert quantize(-1.5, 1) == -2
    assert quantize(2.5, 1) == 3
    assert quantize(0.0, 10 ** 6) == 0
    # Ties are decided on the double's *exact* rational value, so use one that
    # is exactly representable: 0.25 * 2 is exactly 0.5.
    assert quantize(0.25, 2) == 1
    assert quantize(-0.25, 2) == -1
    # 5e-07 is not exactly half a unit at scale 10^6; the nearer side wins.
    assert quantize(-0.0000005, 10 ** 6) == 0


def test_quantize_round_trip_within_one_ulp():
    rng = random.Random(0)
    scale = 10 ** 6
    for _ in range(2000):
        v = rng.uniform(-30, 30)
        assert abs(dequantize(quantize(v, scale), scale) - v) <= 0.5 / scale


def test_pwl_sigmoid_matches_declared_accuracy():
    worst = 0.0
    for i in range(-12000, 12001):
        z = i / 1000.0
        worst = max(worst, abs(pwl_sigmoid(round(z * 10 ** 6)) / 10 ** 6
                               - 1 / (1 + math.exp(-z))))
    assert worst <= PWL_MAX_ERROR, f"PWL error {worst} exceeds the documented bound"


def test_pwl_sigmoid_is_monotone_and_bounded():
    scale = 10 ** 6
    previous = -1
    for i in range(-900, 901):
        v = pwl_sigmoid(i * 10_000)
        assert 0 <= v <= scale
        assert v >= previous
        previous = v


def test_pwl_sigmoid_is_continuous_at_every_breakpoint():
    scale = 10 ** 6
    for i in range(1, 4):
        a = PWL_T[i]
        left = PWL_C[i - 1] + (PWL_M[i - 1] * (a - PWL_T[i - 1])) // scale
        assert left == PWL_C[i], f"discontinuity at breakpoint {a}"


def test_pwl_sigmoid_reflects_about_a_half():
    scale = 10 ** 6
    for z in range(0, 8_000_000, 7919):
        assert pwl_sigmoid(z) + pwl_sigmoid(-z) == scale


def test_floor_division_convention_matches_the_circuit():
    # DivConst enforces num = q*d + r with 0 <= r < d, i.e. Python's // and %.
    for num in (-1_000_001, -1, 0, 1, 999_999, 1_000_001):
        d = 10 ** 6
        q, r = num // d, num % d
        assert num == q * d + r
        assert 0 <= r < d


def test_training_step_is_pure_integer_and_reproducible():
    cfg = FixedPointConfig()
    rng = random.Random(11)
    xs = [[rng.randrange(-2_000_000, 2_000_000) for _ in range(cfg.n_features)]
          for _ in range(cfg.batch_size)]
    ys = [rng.randrange(2) for _ in range(cfg.batch_size)]
    w, b = [0] * cfg.n_features, 0
    first = training_step(w, b, xs, ys, 50_000, cfg.scale)
    second = training_step(w, b, xs, ys, 50_000, cfg.scale)
    assert first == second
    assert all(isinstance(v, int) for v in first[0])
    assert isinstance(first[1], int)


def test_training_step_reduces_error_on_a_separable_problem():
    cfg = FixedPointConfig()
    xs = [[1_000_000, 0, 0, 0]] * 4 + [[-1_000_000, 0, 0, 0]] * 4
    ys = [1] * 4 + [0] * 4
    w, b = [0] * 4, 0
    before = sum(abs(e) for e in sample_errors(w, b, xs, ys, cfg.scale))
    for _ in range(50):
        w, b = training_step(w, b, xs, ys, 500_000, cfg.scale)
    after = sum(abs(e) for e in sample_errors(w, b, xs, ys, cfg.scale))
    assert after < before, "fixed-point training does not descend"


def test_labels_must_be_binary():
    cfg = FixedPointConfig()
    with pytest.raises(ValueError):
        sample_errors([0] * 4, 0, [[0] * 4], [2], cfg.scale)


def test_step_bound_and_range_helpers():
    cfg = FixedPointConfig()
    assert step_within_bound([0] * 4, 0, [cfg.max_delta] * 4, 0, cfg.max_delta)
    assert not step_within_bound([0] * 4, 0, [cfg.max_delta + 1] * 4, 0, cfg.max_delta)
    assert model_in_range([0] * 4, 0, cfg)
    assert not model_in_range([1 << cfg.w_bits] * 4, 0, cfg)


def test_config_rejects_sizes_the_circuit_cannot_prove():
    with pytest.raises(AssertionError):
        FixedPointConfig(z_bits=20).check()
    with pytest.raises(AssertionError):
        FixedPointConfig(x_bits=60).check()


def test_z_bits_are_derived_from_the_feature_count():
    """Widening the feature vector must widen z automatically; the old value
    was hand-set and silently too small at 8 features."""
    narrow = FixedPointConfig(n_features=4)
    wide = FixedPointConfig(n_features=8)
    assert wide.z_bits > narrow.z_bits
    for n in (1, 2, 4, 8, 16, 30):
        cfg = FixedPointConfig(n_features=n)
        cfg.check()


def test_negative_values_map_into_the_field_consistently():
    assert (-1) % FIELD_PRIME == FIELD_PRIME - 1


def test_the_sigmoid_table_in_config_matches_the_circuit():
    """`FixedPointConfig.check()` bounds the sigmoid division numerator using
    two constants copied from the circuit.  If the table moves and the copy
    does not, check() certifies a circuit it no longer describes -- and the
    thing it is certifying is exactly what stops DivConst's field equation from
    wrapping.  So the copy is checked.
    """
    import re
    from pathlib import Path

    from zkfl.config import PWL_LAST_BREAKPOINT, PWL_MAX_SLOPE

    src = (Path(__file__).resolve().parent.parent
           / "circuits" / "lib" / "activation.circom").read_text()

    def table(name):
        m = re.search(rf"var {name}\[5\] = \[([^\]]+)\]", src)
        assert m, f"{name}[5] not found in activation.circom"
        return [int(v.strip()) for v in m.group(1).split(",")]

    slopes, anchors = table("M"), table("T")
    assert max(slopes) == PWL_MAX_SLOPE, (
        f"config says the largest slope is {PWL_MAX_SLOPE}, circuit says "
        f"{max(slopes)}"
    )
    assert max(anchors) == PWL_LAST_BREAKPOINT, (
        f"config says the last breakpoint is {PWL_LAST_BREAKPOINT}, circuit "
        f"says {max(anchors)}"
    )
    assert slopes[-1] == 0, (
        "the last piece must be flat, or |a - tSel| is unbounded there and the "
        "numerator bound in config.check() does not hold"
    )
