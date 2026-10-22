"""What the circuit does and does not prove.

The headline is POISON1: the same attack run against v1 and v2.  It succeeds
against v1 -- not as an exotic exploit but via v1's own documented witness
recipe -- and is rejected by v2.  Everything else here pins down the individual
constraints that make that difference.
"""
from __future__ import annotations

import random

import pytest

from zkfl import legacy_v1
from zkfl.commitment import DatasetCommitment, batch_digest, verify_merkle_proof
from zkfl.config import FixedPointConfig
from zkfl.fixedpoint import pwl_sigmoid, training_step
from zkfl.model import Model
from zkfl.trainer import FixedPointTrainer, LocalDataset, TrainingWitness
from zkfl.zk import ProofError, PublicInputs, witness_input

pytestmark = pytest.mark.circuit


@pytest.fixture(scope="module")
def trained(request):
    """One honest step, proof included, shared across the module."""
    fp = FixedPointConfig()
    from zkfl.datasets import synthetic

    split = synthetic(n_samples=64, n_features=fp.n_features, seed=5)
    dataset = LocalDataset(split.x_train.tolist(), split.y_train.tolist(), fp)
    trainer = FixedPointTrainer(fp, 0.05)
    before = Model.zeros(fp)
    after, witness = trainer.step(before, dataset, 0)
    return fp, dataset, trainer, before, after, witness


def test_honest_step_proves_and_verifies(groth16, trained):
    fp, dataset, _, before, after, witness = trained
    proof = groth16.prove(witness)
    expected = PublicInputs.from_models(before, after, witness.eta, dataset.root)
    assert groth16.verify(proof, expected)


def test_poison1_succeeds_against_v1(trained):
    """v1's statement is satisfied by any (w_old, w_new) pair."""
    if not legacy_v1.available():
        pytest.skip("v1 ablation artifacts absent")
    target = [9_999_999, -9_999_999, 4_242_424]
    forged = legacy_v1.forge_witness([0, 0, 0], target, 10_000)
    proof = legacy_v1.prove(forged)
    assert proof is not None, "v1 could not even be proved; the ablation is broken"
    assert legacy_v1.verify(proof), (
        "v1 was expected to accept an arbitrary target -- that is the point of "
        "the ablation"
    )


def test_poison1_fails_against_v2(groth16, trained):
    """The same idea against v2: pick a target, try to find a witness.

    There is nothing to solve for.  The private inputs are the data, the data
    is pinned by a commitment, and the update equation is checked forward.
    """
    fp, dataset, _, before, _, witness = trained
    target = Model(w=(9_999_999, -9_999_999, 4_242_424, 1_000_000), b=0,
                   scale=fp.scale)
    forged = TrainingWitness(
        model_before=before, model_after=target, eta=witness.eta,
        batch_index=0, xs=witness.xs, ys=witness.ys,
        merkle=witness.merkle, data_root=dataset.root,
    )
    with pytest.raises(ProofError):
        groth16.prove(forged)


def test_uncommitted_data_cannot_be_used(groth16, trained):
    """Input binding: the batch must be in the published Merkle tree."""
    fp, dataset, _, before, _, witness = trained
    forged_x = tuple(tuple([15_000_000] * fp.n_features)
                     for _ in range(fp.batch_size))
    forged_y = tuple([0] * fp.batch_size)
    w_new, b_new = training_step(before.w, before.b,
                                 [list(r) for r in forged_x], list(forged_y),
                                 witness.eta, fp.scale)
    forged = TrainingWitness(
        model_before=before,
        model_after=Model(w=tuple(w_new), b=b_new, scale=fp.scale),
        eta=witness.eta, batch_index=0, xs=forged_x, ys=forged_y,
        merkle=witness.merkle, data_root=dataset.root,
    )
    with pytest.raises(ProofError):
        groth16.prove(forged)


def test_out_of_range_features_are_rejected(groth16, trained):
    """A feature beyond x_bits would let the dot product wrap the field."""
    fp, dataset, _, before, _, witness = trained
    huge = 1 << (fp.x_bits + 3)
    xs = [list(r) for r in witness.xs]
    xs[0][0] = huge
    w_new, b_new = training_step(before.w, before.b, xs, list(witness.ys),
                                 witness.eta, fp.scale)
    forged = TrainingWitness(
        model_before=before,
        model_after=Model(w=tuple(w_new), b=b_new, scale=fp.scale),
        eta=witness.eta, batch_index=0,
        xs=tuple(tuple(r) for r in xs), ys=witness.ys,
        merkle=witness.merkle, data_root=dataset.root,
    )
    with pytest.raises(ProofError):
        groth16.prove(forged)


def test_non_binary_labels_are_rejected(groth16, trained):
    """y must satisfy y*(y-1)=0; a label of 2 doubles that sample's influence.

    The Python trainer refuses such a label outright, so the witness is built
    here with the guard inlined -- otherwise the test would only prove that our
    own validation works, not that the circuit constrains it.
    """
    fp, dataset, _, before, _, witness = trained
    ys = list(witness.ys)
    ys[0] = 2
    xs = [list(r) for r in witness.xs]
    errs = [
        pwl_sigmoid(sum(x[j] * before.w[j] for j in range(fp.n_features))
                    // fp.scale + before.b, fp.scale) - y * fp.scale
        for x, y in zip(xs, ys)
    ]
    k = fp.batch_size
    w_new = [
        before.w[j]
        - (witness.eta * sum(errs[i] * xs[i][j] for i in range(k)))
        // (k * fp.scale * fp.scale)
        for j in range(fp.n_features)
    ]
    b_new = before.b - (witness.eta * sum(errs)) // (k * fp.scale)
    forged = TrainingWitness(
        model_before=before,
        model_after=Model(w=tuple(w_new), b=b_new, scale=fp.scale),
        eta=witness.eta, batch_index=0, xs=witness.xs, ys=tuple(ys),
        merkle=witness.merkle, data_root=dataset.root,
    )
    with pytest.raises(ProofError):
        groth16.prove(forged)


def test_step_beyond_the_clipping_bound_is_unprovable(groth16):
    """|w_new - w_old|_inf <= max_delta is what the aggregation rule leans on,
    so it has to be a constraint rather than a Python check.

    Built on purpose-made data with large features so the step is guaranteed to
    exceed the bound; a step that happened to stay inside it would prove
    nothing.
    """
    fp = FixedPointConfig()
    xs = [[3 * fp.scale] * fp.n_features for _ in range(fp.batch_size)]
    ys = [0] * fp.batch_size
    dataset = LocalDataset(xs, ys, fp, already_scaled=True)
    before = Model.zeros(fp)
    trainer = FixedPointTrainer(fp, 0.9)
    after, witness = trainer.step(before, dataset, 0)

    delta = max(abs(a - b) for a, b in zip(after.w, before.w))
    assert delta > fp.max_delta, (
        f"test setup failed: step of {delta} is inside the bound {fp.max_delta}"
    )
    with pytest.raises(ProofError):
        groth16.prove(witness)


def test_public_inputs_are_bound_to_the_verifier_state(groth16, trained):
    """A valid proof must not verify against a statement the prover chose."""
    fp, dataset, _, before, after, witness = trained
    proof = groth16.prove(witness)

    other_base = Model(w=(1, 2, 3, 4), b=5, scale=fp.scale)
    assert not groth16.verify(
        proof, PublicInputs.from_models(other_base, after, witness.eta, dataset.root)
    )
    assert not groth16.verify(
        proof, PublicInputs.from_models(before, after, witness.eta, dataset.root + 1)
    )
    assert not groth16.verify(
        proof, PublicInputs.from_models(before, after, witness.eta + 1, dataset.root)
    )
    target = Model(w=(0, 0, 0, 0), b=999, scale=fp.scale)
    assert not groth16.verify(
        proof, PublicInputs.from_models(before, target, witness.eta, dataset.root)
    )


def test_string_and_null_proofs_are_always_rejected(groth16, trained):
    fp, dataset, _, before, after, witness = trained
    expected = PublicInputs.from_models(before, after, witness.eta, dataset.root)
    assert not groth16.verify("zkproof_deadbeef", expected)
    assert not groth16.verify(None, expected)
    assert not groth16.verify({}, expected)
    assert not groth16.verify({"scheme": "groth16"}, expected)
    assert not groth16.verify({"scheme": "null", "tag": "x"}, expected)


def test_swapping_the_public_signals_inside_a_proof_is_rejected(groth16, trained):
    fp, dataset, _, before, after, witness = trained
    proof = groth16.prove(witness)
    target = Model(w=(7, 7, 7, 7), b=7, scale=fp.scale)
    bad = PublicInputs.from_models(before, target, witness.eta, dataset.root)
    tampered = {**proof, "public_signals": bad.to_signals()}
    assert not groth16.verify(tampered, bad)


def test_witness_input_shape_matches_the_circuit(trained):
    fp, _, _, _, _, witness = trained
    payload = witness_input(witness)
    assert len(payload["wOld"]) == fp.n_features
    assert len(payload["x"]) == fp.batch_size
    assert len(payload["x"][0]) == fp.n_features
    assert len(payload["pathElements"]) == fp.merkle_depth
    assert all(isinstance(v, str) for v in payload["y"])


def test_merkle_commitment_round_trips():
    fp = FixedPointConfig()
    rng = random.Random(2)
    batches = [
        ([[rng.randrange(-10 ** 6, 10 ** 6) for _ in range(fp.n_features)]
          for _ in range(fp.batch_size)],
         [rng.randrange(2) for _ in range(fp.batch_size)])
        for _ in range(6)
    ]
    commitment = DatasetCommitment(batches, fp)
    for i in range(6):
        proof = commitment.proof(i)
        assert verify_merkle_proof(commitment.leaves[i], proof, commitment.root)
        assert not verify_merkle_proof(commitment.leaves[i] + 1, proof, commitment.root)


def test_batch_digest_binds_sample_order():
    xs = [[1, 2, 3, 4], [5, 6, 7, 8]]
    ys = [0, 1]
    assert batch_digest(xs, ys) != batch_digest(xs[::-1], ys[::-1])
