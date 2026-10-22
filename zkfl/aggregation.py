"""Deterministic, Byzantine-robust aggregation.

This is what turns the protocol back into federated learning.  The old design
committed *one* proposal per round -- whichever arrived first -- so the global
model was a sequence of single-node steps ordered by network latency, and the
best-connected node won every round.  Here every verified proposal in the round
contributes.

Two properties do the work:

* **Determinism.**  ``aggregate`` is a pure function of the base model and the
  verified proposal set, computed in integers, with proposals in a canonical
  order.  Every honest node holding the same set therefore derives the same
  model, byte for byte, which is why a vote can be a commitment to a *set* and
  a conflicting vote is misbehaviour rather than a timing artefact.  It also
  means the aggregate needs no proof of its own -- every node recomputes it.

* **Robustness.**  The circuit bounds what a proposal can be (a real training
  step on committed data, with a bounded norm) but not what data a proposer
  committed to.  A coordinate-wise trimmed mean bounds the influence of the
  ``f`` worst proposals; that, not the proof, is what limits poisoning.

The trimmed-mean guarantee needs ``trim >= f`` on each side.  :func:`aggregate`
returns that count so a caller can assert it rather than assume it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from zkfl.config import FixedPointConfig
from zkfl.messages import Proposal
from zkfl.model import Model

RULES = ("fedavg", "trimmed_mean", "median")


class AggregationError(ValueError):
    pass


@dataclass(frozen=True)
class AggregationResult:
    model: Model
    rule: str
    #: proposal ids in the canonical order used, part of what a vote commits to
    proposal_ids: tuple[str, ...]
    #: values dropped per coordinate on each side (trimmed_mean only)
    trimmed_per_side: int
    #: |aggregate delta|_inf, in scaled units
    max_abs_delta: int


def _floor_div(numerator: int, denominator: int) -> int:
    """Floor division, the same convention as everywhere else in the codebase.

    Using Python's ``//`` rather than rounding keeps this reproducible: there
    is no tie-breaking rule to get subtly wrong on another machine.
    """
    if denominator <= 0:
        raise AggregationError("denominator must be positive")
    return numerator // denominator


def _weighted_mean(values: Sequence[int], weights: Sequence[int]) -> int:
    total = sum(weights)
    if total <= 0:
        raise AggregationError("total weight must be positive")
    return _floor_div(sum(v * w for v, w in zip(values, weights)), total)


def _trimmed_mean(values: Sequence[int], trim: int) -> int:
    ordered = sorted(values)
    kept = ordered[trim:len(ordered) - trim] if trim else ordered
    if not kept:
        raise AggregationError("trimming removed every value")
    return _floor_div(sum(kept), len(kept))


def _median(values: Sequence[int]) -> int:
    ordered = sorted(values)
    m = len(ordered)
    if m % 2:
        return ordered[m // 2]
    return _floor_div(ordered[m // 2 - 1] + ordered[m // 2], 2)


def trim_count(n_proposals: int, permille: int) -> int:
    """How many values to drop per side, as exact integer arithmetic.

    Never enough to empty the set: with two proposals nothing is trimmed, which
    is honest about the fact that robustness needs a crowd.
    """
    if n_proposals <= 2:
        return 0
    return min((permille * n_proposals) // 1000, (n_proposals - 1) // 2)


def aggregate(
    base: Model,
    proposals: Sequence[Proposal],
    cfg: FixedPointConfig,
    *,
    rule: str = "trimmed_mean",
    permille: int = 250,
    weight_of: Callable[[str], int] | Mapping[str, int] | None = None,
) -> AggregationResult:
    """Combine verified proposals into the next global model.

    ``proposals`` must already be verified -- signature, proof, height and
    previous hash.  This function assumes nothing about their content beyond
    the shape of a model, and deliberately does no verification of its own so
    that the ordering of checks is visible at the call site.
    """
    if rule not in RULES:
        raise AggregationError(f"unknown rule {rule!r}; expected one of {RULES}")
    if not proposals:
        raise AggregationError("no proposals to aggregate")

    # Canonical order.  Every node must iterate identically or the floor
    # divisions below can differ in the last unit.
    ordered = sorted(proposals, key=lambda p: p.proposal_id)
    n = len(base.w)
    for p in ordered:
        if len(p.model.w) != n or p.model.scale != base.scale:
            raise AggregationError("proposal model shape does not match the base")

    deltas_w = [[p.model.w[j] - base.w[j] for p in ordered] for j in range(n)]
    deltas_b = [p.model.b - base.b for p in ordered]

    trimmed = 0
    if rule == "fedavg":
        if weight_of is None:
            weights = [1] * len(ordered)
        elif callable(weight_of):
            weights = [max(1, int(weight_of(p.proposer))) for p in ordered]
        else:
            weights = [max(1, int(weight_of.get(p.proposer, 1))) for p in ordered]
        agg_w = [_weighted_mean(deltas_w[j], weights) for j in range(n)]
        agg_b = _weighted_mean(deltas_b, weights)
    elif rule == "trimmed_mean":
        trimmed = trim_count(len(ordered), permille)
        agg_w = [_trimmed_mean(deltas_w[j], trimmed) for j in range(n)]
        agg_b = _trimmed_mean(deltas_b, trimmed)
    else:
        agg_w = [_median(deltas_w[j]) for j in range(n)]
        agg_b = _median(deltas_b)

    # Every proposal satisfies |delta|_inf <= max_delta in circuit, and all
    # three rules are means or order statistics of those values, so the
    # aggregate inherits the bound.  Assert it rather than trust the reasoning.
    max_abs = max([abs(v) for v in agg_w] + [abs(agg_b)])
    if max_abs > cfg.max_delta:
        raise AggregationError(
            f"aggregate step {max_abs} exceeds the clipping bound {cfg.max_delta}; "
            "a proposal was accepted that the circuit should have rejected"
        )

    model = Model(
        w=tuple(base.w[j] + agg_w[j] for j in range(n)),
        b=base.b + agg_b,
        scale=base.scale,
    )
    from zkfl.fixedpoint import model_in_range

    if not model_in_range(model.w, model.b, cfg):
        raise AggregationError(
            "aggregate model falls outside the circuit's weight range; "
            "the next round would be unprovable"
        )

    return AggregationResult(
        model=model,
        rule=rule,
        proposal_ids=tuple(p.proposal_id for p in ordered),
        trimmed_per_side=trimmed,
        max_abs_delta=max_abs,
    )


def robustness_note(n_proposals: int, f_byzantine: int, permille: int, rule: str) -> str:
    """Say plainly whether the configured rule actually tolerates ``f`` here.

    Printed by the simulation and recorded in the audit log, because a robust
    rule that trims fewer values than there are attackers provides no bound and
    should not be reported as though it does.
    """
    if rule == "fedavg":
        return ("fedavg has no robustness guarantee: a single proposal can move "
                "the mean by delta/m")
    if rule == "median":
        ok = 2 * f_byzantine < n_proposals
        return (f"median tolerates f < m/2; m={n_proposals}, f={f_byzantine} -> "
                f"{'holds' if ok else 'DOES NOT HOLD'}")
    t = trim_count(n_proposals, permille)
    ok = t >= f_byzantine
    return (f"trimmed mean drops {t} per side; needs >= f={f_byzantine} -> "
            f"{'holds' if ok else 'DOES NOT HOLD (bound is vacuous)'}")
