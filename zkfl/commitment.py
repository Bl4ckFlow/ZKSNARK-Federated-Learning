"""Dataset commitments -- what stops a proof from being about invented data.

A node quantizes its dataset once, cuts it into fixed-size batches, and
publishes a Poseidon-Merkle root over the batch digests when it joins the
network.  The root is signed and lands in the genesis state, so every verifier
holds its own copy.  When the node later proves a training step it supplies a
membership path for the batch it used, and the verifier feeds the root in from
its registry rather than from the proposal.

The digest layout mirrors ``circuits/lib/merkle.circom`` exactly:

    leaf_i        = Poseidon(x_i[0], ..., x_i[n-1], y_i)
    batch_digest  = fold(Poseidon(acc, leaf_i)), acc starting at 0
    tree          = binary Poseidon, padded to 2^depth with the zero leaf

Padding with zero is safe: occupying a padded slot means exhibiting a batch
whose Poseidon chain digest is exactly 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from zkfl.config import FixedPointConfig
from zkfl.poseidon import poseidon, poseidon2, to_field

Batch = tuple[list[list[int]], list[int]]


def sample_digest(x: Sequence[int], y: int) -> int:
    return poseidon([to_field(v) for v in x] + [to_field(y)])


def batch_digest(xs: Sequence[Sequence[int]], ys: Sequence[int]) -> int:
    acc = 0
    for x, y in zip(xs, ys):
        acc = poseidon2(acc, sample_digest(x, y))
    return acc


@dataclass(frozen=True)
class MerkleProof:
    leaf_index: int
    elements: tuple[int, ...]
    indices: tuple[int, ...]


class DatasetCommitment:
    """Merkle commitment over one node's quantized, batched dataset."""

    def __init__(self, batches: Sequence[Batch], cfg: FixedPointConfig):
        cfg.check()
        if len(batches) > cfg.max_batches:
            raise ValueError(
                f"{len(batches)} batches exceeds the tree width {cfg.max_batches}; "
                f"raise merkle_depth above {cfg.merkle_depth}"
            )
        for xs, ys in batches:
            if len(xs) != cfg.batch_size or len(ys) != cfg.batch_size:
                raise ValueError(
                    f"every batch must hold exactly {cfg.batch_size} samples"
                )
        self.cfg = cfg
        self.batches: tuple[Batch, ...] = tuple(
            ([list(x) for x in xs], list(ys)) for xs, ys in batches
        )
        self.leaves: tuple[int, ...] = tuple(
            batch_digest(xs, ys) for xs, ys in self.batches
        )
        self._levels = self._build()

    # -- tree ---------------------------------------------------------------

    def _build(self) -> list[list[int]]:
        width = self.cfg.max_batches
        level = list(self.leaves) + [0] * (width - len(self.leaves))
        levels = [level]
        while len(level) > 1:
            level = [poseidon2(level[i], level[i + 1]) for i in range(0, len(level), 2)]
            levels.append(level)
        return levels

    @property
    def root(self) -> int:
        return self._levels[-1][0]

    @property
    def n_samples(self) -> int:
        return len(self.batches) * self.cfg.batch_size

    def proof(self, batch_index: int) -> MerkleProof:
        if not 0 <= batch_index < len(self.batches):
            raise IndexError(
                f"batch {batch_index} out of range (have {len(self.batches)})"
            )
        elements, indices = [], []
        idx = batch_index
        for level in self._levels[:-1]:
            sibling = level[idx ^ 1]
            elements.append(sibling)
            # 0 means "the running node is the left child", matching the mux in
            # MerkleInclusion.
            indices.append(idx & 1)
            idx >>= 1
        return MerkleProof(batch_index, tuple(elements), tuple(indices))

    def batch(self, batch_index: int) -> Batch:
        xs, ys = self.batches[batch_index]
        return [list(x) for x in xs], list(ys)


def verify_merkle_proof(leaf: int, proof: MerkleProof, root: int) -> bool:
    """Recompute a path -- the same check the circuit performs."""
    cur = leaf
    for sibling, index in zip(proof.elements, proof.indices):
        cur = poseidon2(sibling, cur) if index else poseidon2(cur, sibling)
    return cur == root
