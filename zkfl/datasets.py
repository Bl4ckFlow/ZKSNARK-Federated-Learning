"""Datasets, partitioning, and the standardisation the circuit's ranges need.

Two things the previous version left undefined and a reviewer will ask about:
what "Dataset_A" and "Dataset_B" actually were, and how a 100-sample dataset is
split across 10 nodes.  Both are explicit here, and the split is reported --
including how many samples fall outside whole batches and are dropped.

Features are standardised on the *training* split only and the statistics are
part of the public protocol parameters, so every node quantizes identically.
Fitting them on the union of everyone's data would be a leak; fitting them
per-node would make the same feature mean different things to different nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from zkfl.config import FixedPointConfig


@dataclass(frozen=True)
class Split:
    """A named train/test split with the standardisation used to build it."""

    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    positive_label: str
    negative_label: str

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.x_train)} train / {len(self.x_test)} test, "
            f"{self.n_features} features, "
            f"positive={self.positive_label} ({int(self.y_train.sum())} train), "
            f"negative={self.negative_label}"
        )


def _standardise(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def iris_binary(test_fraction: float = 0.3, seed: int = 0) -> Split:
    """Iris versicolor vs virginica -- the harder, non-separable pair.

    Setosa is linearly separable from the others, so including it would make
    the task trivial and the accuracy number meaningless.
    """
    from sklearn.datasets import load_iris

    data = load_iris()
    mask = data.target != 0                      # drop setosa
    x = data.data[mask].astype(np.float64)
    y = (data.target[mask] == 2).astype(np.int64)  # 1 = virginica

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    cut = int(len(x) * (1 - test_fraction))
    x_tr, x_te, mean, std = _standardise(x[:cut], x[cut:])
    return Split("iris-versicolor-vs-virginica", x_tr, y[:cut], x_te, y[cut:],
                 mean, std, "virginica", "versicolor")


def breast_cancer_binary(
    test_fraction: float = 0.3, seed: int = 0, n_features: int = 8
) -> Split:
    """Breast Cancer Wisconsin, truncated to the first ``n_features`` columns.

    A second dataset with more samples and more features, so the circuit's cost
    can be measured against feature count rather than asserted to scale.
    """
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    x = data.data[:, :n_features].astype(np.float64)
    y = (data.target == 0).astype(np.int64)      # 1 = malignant

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    cut = int(len(x) * (1 - test_fraction))
    x_tr, x_te, mean, std = _standardise(x[:cut], x[cut:])
    return Split(f"breast-cancer-{n_features}f", x_tr, y[:cut], x_te, y[cut:],
                 mean, std, "malignant", "benign")


def synthetic(
    n_samples: int = 512, n_features: int = 4, seed: int = 0,
    test_fraction: float = 0.3, separation: float = 1.5,
) -> Split:
    """A controlled dataset, for tests that should not depend on scikit-learn."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n_features)
    x = rng.normal(size=(n_samples, n_features))
    logits = x @ w + rng.normal(scale=0.5, size=n_samples)
    y = (logits > 0).astype(np.int64)
    x = x + separation * y[:, None] * w / np.linalg.norm(w)
    cut = int(n_samples * (1 - test_fraction))
    x_tr, x_te, mean, std = _standardise(x[:cut], x[cut:])
    return Split(f"synthetic-{n_features}f", x_tr, y[:cut], x_te, y[cut:],
                 mean, std, "class-1", "class-0")


@dataclass(frozen=True)
class Partition:
    """One node's share, plus what the split cost in dropped samples."""

    node_index: int
    x: np.ndarray
    y: np.ndarray
    dropped: int


def partition(
    split: Split, n_nodes: int, cfg: FixedPointConfig, *, iid: bool = True,
    seed: int = 0, dirichlet_alpha: float = 0.5,
) -> list[Partition]:
    """Split the training set across nodes.

    ``iid=False`` uses a Dirichlet label split, the standard way to produce
    non-IID federated shards, so the aggregation rule is exercised on shards
    that actually disagree.
    """
    rng = np.random.default_rng(seed)
    n = len(split.x_train)
    if iid:
        order = rng.permutation(n)
        shards = np.array_split(order, n_nodes)
    else:
        shards = [[] for _ in range(n_nodes)]
        for label in (0, 1):
            idx = np.where(split.y_train == label)[0]
            rng.shuffle(idx)
            proportions = rng.dirichlet([dirichlet_alpha] * n_nodes)
            cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                shards[i].extend(chunk.tolist())
        shards = [np.array(sorted(s), dtype=int) for s in shards]

    out: list[Partition] = []
    for i, shard in enumerate(shards):
        shard = np.asarray(shard, dtype=int)
        usable = (len(shard) // cfg.batch_size) * cfg.batch_size
        out.append(Partition(
            node_index=i,
            x=split.x_train[shard[:usable]],
            y=split.y_train[shard[:usable]],
            dropped=len(shard) - usable,
        ))
    return out


def partition_report(parts: Sequence[Partition], cfg: FixedPointConfig) -> str:
    sizes = [len(p.x) for p in parts]
    dropped = sum(p.dropped for p in parts)
    pos = [int(p.y.sum()) for p in parts]
    return (
        f"{len(parts)} nodes | samples/node min={min(sizes)} max={max(sizes)} "
        f"| batches/node {min(sizes) // cfg.batch_size}-{max(sizes) // cfg.batch_size} "
        f"| positives/node {min(pos)}-{max(pos)} | {dropped} samples dropped "
        f"(not a whole batch of {cfg.batch_size})"
    )
