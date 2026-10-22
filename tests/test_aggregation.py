"""Aggregation: deterministic, and robust in the way it claims to be."""
from __future__ import annotations

import random

import pytest

from zkfl.aggregation import (
    AggregationError,
    aggregate,
    robustness_note,
    trim_count,
)
from zkfl.config import FixedPointConfig
from zkfl.messages import Proposal
from zkfl.model import Model


def proposal(fp, name, w, b=0):
    return Proposal(height=1, prev_hash="p" * 64,
                    model=Model(w=tuple(w), b=b, scale=fp.scale),
                    proof=None, batch_index=0, proposer=name)


@pytest.fixture
def fp():
    return FixedPointConfig()


def test_aggregation_is_independent_of_input_order(fp):
    base = Model.zeros(fp)
    proposals = [proposal(fp, f"n{i}", [1000 + i, 2000 - i, 3000, 4000], b=i)
                 for i in range(8)]
    reference = aggregate(base, proposals, fp)
    rng = random.Random(0)
    for _ in range(20):
        shuffled = proposals[:]
        rng.shuffle(shuffled)
        result = aggregate(base, shuffled, fp)
        assert result.model == reference.model
        assert result.proposal_ids == reference.proposal_ids


def test_aggregation_is_pure_integer_arithmetic(fp):
    base = Model.zeros(fp)
    proposals = [proposal(fp, f"n{i}", [3, 3, 3, 3]) for i in range(3)]
    result = aggregate(base, proposals, fp, rule="fedavg")
    assert all(isinstance(v, int) for v in result.model.w)
    assert isinstance(result.model.b, int)


def test_trimmed_mean_removes_the_extremes(fp):
    base = Model.zeros(fp)
    honest = [proposal(fp, f"h{i}", [1000, 1000, 1000, 1000]) for i in range(6)]
    attackers = [proposal(fp, "atk-hi", [400_000] * 4),
                 proposal(fp, "atk-lo", [-400_000] * 4)]

    robust = aggregate(base, honest + attackers, fp, rule="trimmed_mean", permille=250)
    assert robust.model.w == (1000, 1000, 1000, 1000)

    naive = aggregate(base, honest + attackers, fp, rule="fedavg")
    assert naive.model.w != robust.model.w, "fedavg should be dragged by outliers"


def test_median_matches_trimmed_mean_on_symmetric_attacks(fp):
    base = Model.zeros(fp)
    honest = [proposal(fp, f"h{i}", [1000] * 4) for i in range(5)]
    attackers = [proposal(fp, "a", [400_000] * 4), proposal(fp, "b", [-400_000] * 4)]
    assert (aggregate(base, honest + attackers, fp, rule="median").model.w
            == aggregate(base, honest + attackers, fp,
                         rule="trimmed_mean", permille=300).model.w)


def test_fedavg_weights_by_committed_dataset_size(fp):
    base = Model.zeros(fp)
    proposals = [proposal(fp, "big", [1000] * 4), proposal(fp, "small", [0] * 4)]
    weights = {"big": 900, "small": 100}
    result = aggregate(base, proposals, fp, rule="fedavg", weight_of=weights)
    assert result.model.w == (900, 900, 900, 900)


def test_trim_never_empties_the_set(fp):
    assert trim_count(1, 490) == 0
    assert trim_count(2, 490) == 0
    assert trim_count(3, 490) == 1
    for m in range(1, 40):
        assert 2 * trim_count(m, 490) < m


def test_robustness_note_says_when_the_bound_is_vacuous():
    assert "holds" in robustness_note(8, 2, 250, "trimmed_mean")
    assert "DOES NOT HOLD" in robustness_note(8, 3, 250, "trimmed_mean")
    assert "no robustness guarantee" in robustness_note(8, 1, 250, "fedavg")


def test_an_aggregate_outside_the_clipping_bound_is_refused(fp):
    base = Model.zeros(fp)
    over = [proposal(fp, f"n{i}", [fp.max_delta * 3] * 4) for i in range(3)]
    with pytest.raises(AggregationError, match="clipping bound"):
        aggregate(base, over, fp, rule="fedavg")


def test_the_aggregate_inherits_the_per_proposal_bound(fp):
    """Every rule is a mean or an order statistic, so it cannot exceed the
    largest input -- worth asserting rather than reasoning about."""
    base = Model.zeros(fp)
    rng = random.Random(4)
    for rule in ("fedavg", "trimmed_mean", "median"):
        proposals = [
            proposal(fp, f"n{i}",
                     [rng.randrange(-fp.max_delta, fp.max_delta) for _ in range(4)],
                     b=rng.randrange(-fp.max_delta, fp.max_delta))
            for i in range(9)
        ]
        result = aggregate(base, proposals, fp, rule=rule)
        assert result.max_abs_delta <= fp.max_delta


def test_empty_and_mismatched_inputs_are_errors(fp):
    base = Model.zeros(fp)
    with pytest.raises(AggregationError):
        aggregate(base, [], fp)
    wrong = Proposal(height=1, prev_hash="p", model=Model(w=(1, 2), b=0, scale=fp.scale),
                     proof=None, batch_index=0, proposer="n")
    with pytest.raises(AggregationError):
        aggregate(base, [wrong], fp)
