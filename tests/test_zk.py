"""The proof backends' own guarantees, independent of any circuit.

Two things are checked here that no circuit test can check: that the stand-in
backend cannot be mistaken for a real one, and that the verifier -- not the
prover -- decides what statement is being verified.
"""
from __future__ import annotations

import os

import pytest

from zkfl.config import FIELD_PRIME, FixedPointConfig
from zkfl.model import Model
from zkfl.zk import (
    NullProofBackend,
    PublicInputs,
    assert_signal_order,
    make_backend,
)


def witness_for(before, after, eta=50_000, data_root=7):
    from zkfl.commitment import MerkleProof
    from zkfl.trainer import TrainingWitness

    return TrainingWitness(
        model_before=before, model_after=after, eta=eta, batch_index=0,
        xs=(), ys=(), merkle=MerkleProof(0, (), ()), data_root=data_root,
    )


@pytest.fixture
def models(fp):
    before = Model.zeros(fp)
    after = Model(w=(1000, -2000, 3000, -4000), b=500, scale=fp.scale)
    return before, after


# ---------------------------------------------------------------------------
# The stand-in must never pass for the real thing
# ---------------------------------------------------------------------------


def test_the_null_backend_will_not_be_constructed_by_accident():
    """A backend that proves nothing has to be asked for by name."""
    with pytest.raises(RuntimeError, match="proves nothing"):
        NullProofBackend()


def test_the_null_backend_refuses_when_the_environment_forbids_it(monkeypatch):
    """A deployment can make "no real proofs" a hard failure rather than a note."""
    monkeypatch.setenv("ZKFL_FORBID_NULL_BACKEND", "1")
    with pytest.raises(RuntimeError, match="ZKFL_FORBID_NULL_BACKEND"):
        NullProofBackend(i_understand_this_is_not_a_proof=True)
    with pytest.raises(RuntimeError, match="ZKFL_FORBID_NULL_BACKEND"):
        make_backend(FixedPointConfig(), strict=False)


def test_every_null_proof_is_tagged_as_unverified(fp, null_backend, models):
    """An audit log full of these must never read as a verified history."""
    before, after = models
    proof = null_backend.prove(witness_for(before, after))
    assert proof["scheme"] == "null"
    assert proof["unverified"] is True
    assert proof["scheme"] != "groth16"


# ---------------------------------------------------------------------------
# The statement is the verifier's, not the prover's
# ---------------------------------------------------------------------------


def test_the_null_backend_still_binds_the_statement(fp, null_backend, models):
    """Weak as it is, it must not accept a proof for a different transition."""
    before, after = models
    proof = null_backend.prove(witness_for(before, after))
    good = PublicInputs.from_models(before, after, 50_000, 7)
    assert null_backend.verify(proof, good)

    other = Model(w=(9, 9, 9, 9), b=9, scale=fp.scale)
    for wrong in (
        PublicInputs.from_models(other, after, 50_000, 7),      # base model
        PublicInputs.from_models(before, other, 50_000, 7),     # target
        PublicInputs.from_models(before, after, 50_001, 7),     # learning rate
        PublicInputs.from_models(before, after, 50_000, 8),     # data root
    ):
        assert not null_backend.verify(proof, wrong)


def test_a_foreign_scheme_is_never_accepted(null_backend, models):
    before, after = models
    expected = PublicInputs.from_models(before, after, 50_000, 7)
    tag = NullProofBackend._tag(expected)
    assert not null_backend.verify({"scheme": "groth16", "tag": tag}, expected)
    assert not null_backend.verify({"tag": tag}, expected)
    assert not null_backend.verify("proof", expected)
    assert not null_backend.verify(None, expected)


# ---------------------------------------------------------------------------
# Public input ordering
# ---------------------------------------------------------------------------


def test_public_signal_order_is_fixed_and_total(fp, models):
    """w_old..., b_old, w_new..., b_new, eta, dataRoot -- and nothing else.

    The Python side and the circuit agree on this ordering by convention, not
    by construction, so it is pinned here and re-checked against the compiled
    .sym at setup time.
    """
    def field(v):
        return str(v % FIELD_PRIME)

    before, after = models
    signals = PublicInputs.from_models(before, after, 50_000, 7).to_signals()
    n = fp.n_features
    assert len(signals) == 2 * n + 4
    assert signals[:n] == [field(v) for v in before.w]
    assert signals[n] == field(before.b)
    assert signals[n + 1:2 * n + 1] == [field(v) for v in after.w]
    assert signals[2 * n + 1] == field(after.b)
    assert signals[2 * n + 2] == "50000"
    assert signals[2 * n + 3] == "7"


def test_negative_values_are_encoded_in_the_field_not_as_minus_signs(fp, models):
    """A negative weight must reach snarkjs as a field element.

    Passing "-1" would either be rejected or silently reinterpreted, and a
    silently reinterpreted public input is a verifier checking the wrong claim.
    """
    before = Model(w=(-1, -2, -3, -4), b=-5, scale=fp.scale)
    _, after = models
    signals = PublicInputs.from_models(before, after, 50_000, 7).to_signals()
    assert not any(s.startswith("-") for s in signals), signals
    assert all(s.isdigit() for s in signals), signals


@pytest.mark.circuit
def test_the_compiled_circuit_agrees_with_that_order(fp, circuit_available):
    """The convention above, checked against the artefact rather than assumed."""
    if not circuit_available:
        pytest.skip("circuit artifacts absent")
    from zkfl.zk import CircuitArtifacts

    assert_signal_order(CircuitArtifacts.for_config(fp), fp)
