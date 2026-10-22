"""Signing, canonical encoding, and the peer registry."""
from __future__ import annotations

import pytest

from zkfl.crypto import (
    Identity,
    PeerRegistry,
    canonical_json,
    load_public_key,
    signing_bytes,
    verify,
)


def test_a_signature_verifies_only_for_its_own_signer_kind_and_payload():
    a, b = Identity.from_seed(b"\x01" * 32), Identity.from_seed(b"\x02" * 32)
    payload = {"height": 1, "model_hash": "abc"}
    signature = a.sign("vote", payload)

    assert verify(a.node_id, "vote", payload, signature)
    assert not verify(b.node_id, "vote", payload, signature)
    assert not verify(a.node_id, "proposal", payload, signature)
    assert not verify(a.node_id, "vote", {**payload, "height": 2}, signature)


def test_missing_or_malformed_credentials_never_verify():
    """The old ``verify_signature`` returned True when handed no key."""
    a = Identity.from_seed(b"\x01" * 32)
    payload = {"x": 1}
    signature = a.sign("vote", payload)
    assert not verify("", "vote", payload, signature)
    assert not verify(a.node_id, "vote", payload, "")
    assert not verify(a.node_id, "vote", payload, "not-hex")
    assert not verify("not-hex", "vote", payload, signature)
    assert not verify("ab" * 16, "vote", payload, signature)     # wrong length


def test_the_message_kind_is_inside_the_signed_bytes():
    payload = {"x": 1}
    assert signing_bytes("vote", payload) != signing_bytes("proposal", payload)


def test_canonical_json_is_order_independent_and_float_free():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    with pytest.raises(TypeError, match="float"):
        canonical_json({"a": 1.5})
    with pytest.raises(TypeError, match="float"):
        canonical_json({"a": {"b": [1, 2.0]}})


def test_deterministic_identities_are_reproducible():
    assert (Identity.from_seed(b"\x09" * 32).node_id
            == Identity.from_seed(b"\x09" * 32).node_id)
    with pytest.raises(ValueError):
        Identity.from_seed(b"short")


def test_node_id_is_the_public_key():
    identity = Identity.from_seed(b"\x03" * 32)
    assert load_public_key(identity.node_id) is not None


def test_registration_is_signed_and_immutable():
    identity = Identity.from_seed(b"\x04" * 32)
    registry = PeerRegistry()
    payload = PeerRegistry.registration_payload(identity.node_id, 12345, 40)
    assert registry.register(identity.node_id, 12345, 40,
                             identity.sign("registration", payload))
    assert registry.root_of(identity.node_id) == 12345
    assert registry.weight_of(identity.node_id) == 40

    # Re-committing after seeing the global model must not be possible.
    other = PeerRegistry.registration_payload(identity.node_id, 999, 40)
    assert not registry.register(identity.node_id, 999, 40,
                                 identity.sign("registration", other))
    assert registry.root_of(identity.node_id) == 12345

    # Idempotent re-registration with the same content is fine.
    assert registry.register(identity.node_id, 12345, 40,
                             identity.sign("registration", payload))


def test_a_registration_cannot_be_forged_for_someone_else():
    victim = Identity.from_seed(b"\x05" * 32)
    attacker = Identity.from_seed(b"\x06" * 32)
    registry = PeerRegistry()
    payload = PeerRegistry.registration_payload(victim.node_id, 7, 8)
    assert not registry.register(victim.node_id, 7, 8,
                                 attacker.sign("registration", payload))
    assert victim.node_id not in registry


def test_registry_digest_is_stable_and_content_dependent():
    ids = [Identity.from_seed(bytes([i]) * 32) for i in range(3)]
    def build(roots):
        registry = PeerRegistry()
        for identity, root in zip(ids, roots):
            payload = PeerRegistry.registration_payload(identity.node_id, root, 10)
            registry.register(identity.node_id, root, 10,
                              identity.sign("registration", payload))
        return registry
    assert build([1, 2, 3]).digest() == build([1, 2, 3]).digest()
    assert build([1, 2, 3]).digest() != build([1, 2, 4]).digest()
