"""Identity and message authentication.

The network is permissioned: a node's identity *is* its Ed25519 public key,
and the peer set is a configured list of those keys.  That is stated as an
assumption rather than left implicit -- Sybil resistance is out of scope, and
nothing here pretends otherwise.

The previous implementation had two holes this closes.  ``verify_signature``
returned ``True`` when handed no key, so an unsigned message authenticated
itself; and votes were never signed at all, which meant any peer could assert
"j voted B" and get an honest j blacklisted forever.  Here every message is
signed, verification takes no shortcuts, and a signature over a canonical
encoding is the only thing that makes a message count.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

#: Domain separator.  Bumping this invalidates every signature from an older
#: protocol version, which is what we want across a breaking change.
DOMAIN = b"zkfl/v2"


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """The one encoding used for signing and hashing.

    Sorted keys, no whitespace, integers as integers.  Anything that cannot be
    represented exactly -- a float, say -- is a bug rather than something to
    round, so this refuses instead of silently formatting one.
    """

    def check(value: Any, path: str) -> None:
        if isinstance(value, float):
            raise TypeError(
                f"float at {path}: signed payloads must be exact. Quantize first."
            )
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(k, str):
                    raise TypeError(f"non-string key at {path}")
                check(v, f"{path}.{k}")
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                check(v, f"{path}[{i}]")

    check(payload, "$")
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def signing_bytes(kind: str, payload: Mapping[str, Any]) -> bytes:
    """Domain-separated bytes for one message kind.

    The kind is inside the signed material, so a signature over a vote can
    never be replayed as a signature over a proposal.
    """
    return DOMAIN + b"|" + kind.encode() + b"|" + canonical_json(payload)


@dataclass(frozen=True)
class Identity:
    """A node's key pair.  ``node_id`` is the hex public key."""

    _private: Ed25519PrivateKey
    node_id: str

    @classmethod
    def generate(cls) -> "Identity":
        sk = Ed25519PrivateKey.generate()
        return cls(_private=sk, node_id=public_key_hex(sk.public_key()))

    @classmethod
    def from_seed(cls, seed: bytes) -> "Identity":
        """Deterministic identity, for reproducible experiments only."""
        if len(seed) != 32:
            raise ValueError("Ed25519 seeds are 32 bytes")
        sk = Ed25519PrivateKey.from_private_bytes(seed)
        return cls(_private=sk, node_id=public_key_hex(sk.public_key()))

    def sign(self, kind: str, payload: Mapping[str, Any]) -> str:
        return self._private.sign(signing_bytes(kind, payload)).hex()

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._private.public_key()


def public_key_hex(key: Ed25519PublicKey) -> str:
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def load_public_key(node_id: str) -> Ed25519PublicKey:
    """Recover the verifying key from a node id, or raise."""
    try:
        raw = bytes.fromhex(node_id)
    except ValueError as exc:
        raise ValueError(f"node id is not hex: {node_id!r}") from exc
    if len(raw) != 32:
        raise ValueError(f"node id must be 32 bytes, got {len(raw)}")
    return Ed25519PublicKey.from_public_bytes(raw)


def verify(node_id: str, kind: str, payload: Mapping[str, Any], signature: str) -> bool:
    """Check a signature.  No key, no signature, no benefit of the doubt."""
    if not node_id or not signature:
        return False
    try:
        key = load_public_key(node_id)
        key.verify(bytes.fromhex(signature), signing_bytes(kind, payload))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


class PeerRegistry:
    """The permissioned peer set, plus each node's committed data root.

    A verifier reads ``data_root`` from here -- never from the proposal it is
    checking -- which is what ties a proof to data the proposer published
    before it trained.
    """

    def __init__(self) -> None:
        self._roots: dict[str, int] = {}
        self._weights: dict[str, int] = {}
        self._registration_sigs: dict[str, str] = {}

    def register(
        self, node_id: str, data_root: int, n_samples: int, signature: str
    ) -> bool:
        """Admit a node's signed commitment to its dataset.

        Re-registering with different content is rejected: the root is fixed at
        join time, so changing it later would let a node re-commit after seeing
        the global model.
        """
        payload = {"data_root": str(data_root), "n_samples": n_samples,
                   "node_id": node_id}
        if not verify(node_id, "registration", payload, signature):
            return False
        if node_id in self._roots:
            return (self._roots[node_id] == data_root
                    and self._weights[node_id] == n_samples)
        self._roots[node_id] = data_root
        self._weights[node_id] = n_samples
        self._registration_sigs[node_id] = signature
        return True

    @staticmethod
    def registration_payload(node_id: str, data_root: int, n_samples: int) -> dict:
        return {"data_root": str(data_root), "n_samples": n_samples,
                "node_id": node_id}

    def root_of(self, node_id: str) -> int | None:
        return self._roots.get(node_id)

    def weight_of(self, node_id: str) -> int:
        return self._weights.get(node_id, 0)

    def members(self) -> frozenset[str]:
        return frozenset(self._roots)

    def __len__(self) -> int:
        return len(self._roots)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._roots

    def digest(self) -> str:
        """A hash of the whole registry -- part of the genesis state."""
        import hashlib

        members = [
            {"node_id": n, "data_root": str(self._roots[n]),
             "n_samples": self._weights[n]}
            for n in sorted(self._roots)
        ]
        return hashlib.sha256(canonical_json({"members": members})).hexdigest()
