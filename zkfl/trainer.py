"""Local training in exact integer fixed point.

Every number the circuit will see is produced here, forward, in the arithmetic
the circuit constrains.  The witness is therefore a transcript of the
computation rather than a solution to the circuit's equation -- which is the
difference between "this update is a training step" and "this prover can do
division".

Floats appear in exactly two places, both outside anything that gets hashed:
loading a dataset (quantized once, then committed) and reporting accuracy and
loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from zkfl.commitment import DatasetCommitment, MerkleProof
from zkfl.config import FixedPointConfig
from zkfl.fixedpoint import (
    pwl_sigmoid,
    quantize,
    sample_errors,
    training_step,
)
from zkfl.model import Model


@dataclass(frozen=True)
class TrainingWitness:
    """Everything the prover needs, and nothing it has to invent."""

    model_before: Model
    model_after: Model
    eta: int
    batch_index: int
    xs: tuple[tuple[int, ...], ...]
    ys: tuple[int, ...]
    merkle: MerkleProof
    data_root: int


class LocalDataset:
    """A node's private, quantized, committed data.

    The commitment is computed at construction: by the time any training
    happens the data is already fixed and its root publishable.
    """

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        cfg: FixedPointConfig,
        *,
        already_scaled: bool = False,
    ):
        cfg.check()
        if len(features) != len(labels):
            raise ValueError("features and labels differ in length")
        self.cfg = cfg

        if already_scaled:
            xs = [[int(v) for v in row] for row in features]
        else:
            xs = [[quantize(v, cfg.scale) for v in row] for row in features]
        ys = [int(v) for v in labels]

        for row in xs:
            if len(row) != cfg.n_features:
                raise ValueError(
                    f"expected {cfg.n_features} features, got {len(row)}"
                )
            for v in row:
                if not -(1 << cfg.x_bits) <= v < (1 << cfg.x_bits):
                    raise ValueError(
                        f"feature {v} is outside the circuit's {cfg.x_bits}-bit "
                        "range; rescale the dataset or widen x_bits"
                    )
        if any(y not in (0, 1) for y in ys):
            raise ValueError("labels must be binary")

        n_batches = len(xs) // cfg.batch_size
        if n_batches == 0:
            raise ValueError(
                f"{len(xs)} samples cannot fill a batch of {cfg.batch_size}"
            )
        #: samples past the last whole batch, reported rather than silently kept
        self.dropped = len(xs) - n_batches * cfg.batch_size

        batches = []
        for i in range(n_batches):
            lo = i * cfg.batch_size
            hi = lo + cfg.batch_size
            batches.append((xs[lo:hi], ys[lo:hi]))

        self.commitment = DatasetCommitment(batches, cfg)
        self.xs = xs[: n_batches * cfg.batch_size]
        self.ys = ys[: n_batches * cfg.batch_size]

    @property
    def root(self) -> int:
        return self.commitment.root

    @property
    def n_batches(self) -> int:
        return len(self.commitment.batches)

    @property
    def n_samples(self) -> int:
        return self.commitment.n_samples


class FixedPointTrainer:
    """One provable gradient step at a time."""

    def __init__(self, cfg: FixedPointConfig, learning_rate: float):
        cfg.check()
        self.cfg = cfg
        self.eta = quantize(learning_rate, cfg.scale)
        if not 0 < self.eta <= cfg.scale:
            raise ValueError(
                f"learning rate {learning_rate} quantizes to {self.eta}; "
                f"it must land in (0, {cfg.scale}]"
            )

    def step(
        self, model: Model, dataset: LocalDataset, batch_index: int
    ) -> tuple[Model, TrainingWitness]:
        """Train one batch and return the new model plus its witness."""
        if model.scale != self.cfg.scale:
            raise ValueError("model scale does not match the circuit's")
        xs, ys = dataset.commitment.batch(batch_index)

        w_new, b_new = training_step(
            model.w, model.b, xs, ys, self.eta, self.cfg.scale
        )
        after = Model(w=tuple(w_new), b=b_new, scale=self.cfg.scale)

        witness = TrainingWitness(
            model_before=model,
            model_after=after,
            eta=self.eta,
            batch_index=batch_index,
            xs=tuple(tuple(row) for row in xs),
            ys=tuple(ys),
            merkle=dataset.commitment.proof(batch_index),
            data_root=dataset.root,
        )
        return after, witness

    def recompute(self, witness: TrainingWitness) -> Model:
        """Replay a witness -- used by tests and by the POISON1 experiment."""
        w_new, b_new = training_step(
            witness.model_before.w,
            witness.model_before.b,
            [list(r) for r in witness.xs],
            list(witness.ys),
            witness.eta,
            self.cfg.scale,
        )
        return Model(w=tuple(w_new), b=b_new, scale=self.cfg.scale)


# ---------------------------------------------------------------------------
# Evaluation.  Reporting only: nothing here feeds a hash, a proof or a vote.
# ---------------------------------------------------------------------------


def predict_scaled(model: Model, xs: Sequence[Sequence[int]]) -> list[int]:
    n = len(model.w)
    out = []
    for x in xs:
        dot = sum(x[j] * model.w[j] for j in range(n))
        out.append(pwl_sigmoid(dot // model.scale + model.b, model.scale))
    return out


def accuracy(model: Model, xs: Sequence[Sequence[int]], ys: Sequence[int]) -> float:
    if not xs:
        return float("nan")
    preds = predict_scaled(model, xs)
    half = model.scale // 2
    correct = sum(1 for p, y in zip(preds, ys) if (1 if p >= half else 0) == y)
    return correct / len(ys)


def log_loss(model: Model, xs: Sequence[Sequence[int]], ys: Sequence[int]) -> float:
    """Cross-entropy under the protocol's own activation.

    Clamped away from 0 and 1 because the PWL curve saturates exactly, unlike
    the logistic function.
    """
    if not xs:
        return float("nan")
    preds = predict_scaled(model, xs)
    eps = 1e-9
    total = 0.0
    for p, y in zip(preds, ys):
        prob = min(max(p / model.scale, eps), 1.0 - eps)
        total -= math.log(prob) if y == 1 else math.log(1.0 - prob)
    return total / len(ys)


def mean_error_magnitude(model: Model, xs, ys, scale: int) -> float:
    """Mean |sigma(z) - y|; a scale-free progress signal for the live run."""
    if not xs:
        return float("nan")
    errs = sample_errors(model.w, model.b, [list(r) for r in xs], list(ys), scale)
    return sum(abs(e) for e in errs) / (len(errs) * scale)
