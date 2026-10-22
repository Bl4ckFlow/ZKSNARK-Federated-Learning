"""Wire messages.  Every one of them is signed.

The protocol's accountability rests entirely on this: a message is only worth
acting on if it carries a signature over its own canonical encoding, so
"node j voted for B" is a statement j made, not a claim a peer makes about j.
That is what makes equivocation evidence *transferable* -- any node can check
it for itself and reach the same verdict, which is why blacklists converge
instead of forking.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from zkfl.crypto import Identity, canonical_json, verify
from zkfl.model import Model

PROTOCOL_VERSION = 2


class MalformedMessage(ValueError):
    """A message that does not parse; drop it and note the peer."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise MalformedMessage(msg)


def _as_int(value: Any, name: str, *, lo: int | None = None,
            hi: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{name} must be an integer")
    if lo is not None:
        _require(value >= lo, f"{name} must be >= {lo}")
    if hi is not None:
        _require(value <= hi, f"{name} must be <= {hi}")
    return value


def _as_str(value: Any, name: str, *, max_len: int = 4096) -> str:
    _require(isinstance(value, str), f"{name} must be a string")
    _require(len(value) <= max_len, f"{name} exceeds {max_len} characters")
    return value


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """A candidate model for height h, with its proof of training."""

    height: int
    prev_hash: str
    model: Model
    proof: Any
    batch_index: int
    proposer: str
    signature: str = ""

    KIND = "proposal"

    def payload(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "height": self.height,
            "prev_hash": self.prev_hash,
            "model": self.model.to_dict(),
            "proof": self.proof,
            "batch_index": self.batch_index,
            "proposer": self.proposer,
        }

    def signed(self, identity: Identity) -> "Proposal":
        _require(identity.node_id == self.proposer, "proposer must sign its own proposal")
        return Proposal(**{**self.__dict__, "signature": identity.sign(self.KIND, self.payload())})

    def verify_signature(self) -> bool:
        return verify(self.proposer, self.KIND, self.payload(), self.signature)

    @property
    def proposal_id(self) -> str:
        """Identifies a proposal, not just its parameters.

        The old ``update_id`` was SHA-256 of the parameters alone, so identical
        weights at different heights or from different proposers collided and
        the audit log could not say who produced what.  Height, previous hash
        and proposer are all inside this digest.
        """
        return hashlib.sha256(canonical_json(self.payload())).hexdigest()

    @property
    def model_hash(self) -> str:
        return self.model.digest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Proposal":
        _require(isinstance(data, Mapping), "proposal must be an object")
        _require(data.get("version") == PROTOCOL_VERSION, "unsupported protocol version")
        model = data.get("model")
        _require(isinstance(model, Mapping), "model must be an object")
        try:
            parsed = Model.from_dict(dict(model))
        except (KeyError, ValueError, TypeError) as exc:
            raise MalformedMessage(f"bad model: {exc}") from exc
        return cls(
            height=_as_int(data.get("height"), "height", lo=1),
            prev_hash=_as_str(data.get("prev_hash"), "prev_hash", max_len=64),
            model=parsed,
            proof=data.get("proof"),
            batch_index=_as_int(data.get("batch_index"), "batch_index", lo=0),
            proposer=_as_str(data.get("proposer"), "proposer", max_len=64),
            signature=_as_str(data.get("signature", ""), "signature", max_len=256),
        )


#: The empty model hash, used by a precommit that endorses nothing.  A node
#: that saw no quorum in the prevote phase still has to say so, or the round
#: cannot tell "not yet" from "never".
NIL = ""

PREVOTE = "prevote"
PRECOMMIT = "precommit"
PHASES = (PREVOTE, PRECOMMIT)


@dataclass(frozen=True)
class Vote:
    """One node's vote in one phase of one sub-round.

    A vote names the model hash it endorses and the exact proposal set that
    produced it.  Because the aggregate is a deterministic function of that set
    (:mod:`zkfl.aggregation`), two honest nodes holding the same set vote
    identically -- so a conflicting vote in the same slot is misbehaviour
    rather than a timing artefact.

    Two phases, and the distinction is load-bearing.  A *prevote* says "this is
    what my proposal set aggregates to".  A quorum of matching prevotes is a
    **polka**: proof that a quorum could see this value, which is what licenses
    a node to lock on it.  A *precommit* says "I am locked on this".  A quorum
    of matching precommits is a certificate, and commits the height.

    Locking on a polka rather than on one's own prevote is the whole reason the
    protocol is live: when proposal sets differ, prevotes split, nobody locks,
    and the next sub-round starts with a longer window and no stale locks.
    Locking on one's own vote deadlocks the moment two nodes close their
    windows with different sets, which on any real network is immediately.
    """

    height: int
    sub_round: int
    phase: str
    model_hash: str
    proposal_ids: tuple[str, ...]
    voter: str
    signature: str = ""

    KIND = "vote"

    def payload(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "height": self.height,
            "sub_round": self.sub_round,
            "phase": self.phase,
            "model_hash": self.model_hash,
            "proposal_ids": list(self.proposal_ids),
            "voter": self.voter,
        }

    def signed(self, identity: Identity) -> "Vote":
        _require(identity.node_id == self.voter, "voter must sign its own vote")
        return Vote(**{**self.__dict__, "signature": identity.sign(self.KIND, self.payload())})

    def verify_signature(self) -> bool:
        return verify(self.voter, self.KIND, self.payload(), self.signature)

    @property
    def slot(self) -> tuple[str, int, int, str]:
        """The slot a voter may occupy at most once; a second one is evidence."""
        return (self.voter, self.height, self.sub_round, self.phase)

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Vote":
        _require(isinstance(data, Mapping), "vote must be an object")
        _require(data.get("version") == PROTOCOL_VERSION, "unsupported protocol version")
        ids = data.get("proposal_ids")
        _require(isinstance(ids, (list, tuple)), "proposal_ids must be a list")
        _require(len(ids) <= 1024, "proposal_ids too long")
        _require(data.get("phase") in PHASES, "phase must be prevote or precommit")
        return cls(
            height=_as_int(data.get("height"), "height", lo=1),
            sub_round=_as_int(data.get("sub_round"), "sub_round", lo=0, hi=1024),
            phase=str(data["phase"]),
            model_hash=_as_str(data.get("model_hash"), "model_hash", max_len=64),
            proposal_ids=tuple(_as_str(i, "proposal_id", max_len=64) for i in ids),
            voter=_as_str(data.get("voter"), "voter", max_len=64),
            signature=_as_str(data.get("signature", ""), "signature", max_len=256),
        )


@dataclass(frozen=True)
class Certificate:
    """``q`` signed votes of one phase for the same (height, sub_round, hash).

    With ``phase == PRECOMMIT`` this is the commit rule: holding one is
    sufficient and the local voting window is irrelevant, which is what removes
    the old design's dependence on every node's timer expiring in agreement.

    With ``phase == PREVOTE`` it is a **polka**, and making that transferable is
    not a convenience.  A node locks only on a polka it has observed, and under
    loss only some nodes observe any given one.  The rest keep prevoting
    ``Agg(P)``, the locked ones keep prevoting their lock, and if neither group
    reaches ``q`` the split is permanent -- nothing can move a lock except a
    polka, and no polka can form.  Gossiping the polka itself is what lets the
    unlocked nodes adopt the lock and the height finish.  Tendermint gets this
    from a proposer re-proposing with a proof-of-lock; leaderless, the evidence
    has to travel on its own.
    """

    height: int
    sub_round: int
    model_hash: str
    votes: tuple[Vote, ...]
    phase: str = PRECOMMIT

    def voters(self) -> frozenset[str]:
        return frozenset(v.voter for v in self.votes)

    @property
    def is_polka(self) -> bool:
        return self.phase == PREVOTE

    def proposal_ids(self) -> tuple[str, ...]:
        """The proposal set the voters agreed on, if they recorded one."""
        return self.votes[0].proposal_ids if self.votes else ()

    def is_valid(self, quorum: int, members: frozenset[str] | None = None) -> bool:
        if self.phase not in PHASES:
            return False
        if len(self.votes) < quorum:
            return False
        if self.model_hash == NIL:
            return False          # a quorum of "nothing" certifies nothing
        seen: set[str] = set()
        for vote in self.votes:
            if (vote.height != self.height or vote.sub_round != self.sub_round
                    or vote.model_hash != self.model_hash
                    or vote.phase != self.phase):
                return False
            if vote.voter in seen:
                return False
            if members is not None and vote.voter not in members:
                return False
            if not vote.verify_signature():
                return False
            seen.add(vote.voter)
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "height": self.height,
            "sub_round": self.sub_round,
            "model_hash": self.model_hash,
            "phase": self.phase,
            "votes": [v.to_dict() for v in self.votes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Certificate":
        _require(isinstance(data, Mapping), "certificate must be an object")
        _require(data.get("version") == PROTOCOL_VERSION, "unsupported protocol version")
        votes = data.get("votes")
        _require(isinstance(votes, (list, tuple)), "votes must be a list")
        _require(len(votes) <= 4096, "certificate too large")
        return cls(
            height=_as_int(data.get("height"), "height", lo=1),
            sub_round=_as_int(data.get("sub_round"), "sub_round", lo=0, hi=1024),
            model_hash=_as_str(data.get("model_hash"), "model_hash", max_len=64),
            votes=tuple(Vote.from_dict(v) for v in votes),
            # Absent means precommit: that is what every certificate was before
            # polkas became transferable, and defaulting keeps an old log
            # readable rather than silently reinterpreting it as a polka.
            phase=_as_str(data.get("phase", PRECOMMIT), "phase", max_len=16),
        )


@dataclass(frozen=True)
class Evidence:
    """Two conflicting signed messages from one key in one slot.

    Anyone can check this: two signatures, one sender, same slot, different
    content.  No trust in the reporter is required, so a Byzantine node cannot
    frame an honest one -- it would have to forge a signature.
    """

    kind: str          # "vote" | "proposal"
    offender: str
    height: int
    sub_round: int
    a: dict[str, Any]
    b: dict[str, Any]

    def is_valid(self) -> bool:
        if self.kind == "vote":
            try:
                va, vb = Vote.from_dict(self.a), Vote.from_dict(self.b)
            except MalformedMessage:
                return False
            if va.voter != self.offender or vb.voter != self.offender:
                return False
            if va.slot != vb.slot:
                return False
            if va.payload() == vb.payload():
                return False
            return va.verify_signature() and vb.verify_signature()

        if self.kind == "proposal":
            try:
                pa, pb = Proposal.from_dict(self.a), Proposal.from_dict(self.b)
            except MalformedMessage:
                return False
            if pa.proposer != self.offender or pb.proposer != self.offender:
                return False
            if pa.height != pb.height or pa.height != self.height:
                return False
            if pa.payload() == pb.payload():
                return False
            return pa.verify_signature() and pb.verify_signature()

        return False

    @property
    def evidence_id(self) -> str:
        pair = sorted([canonical_json(self.a).hex(), canonical_json(self.b).hex()])
        return hashlib.sha256(
            f"{self.kind}|{self.offender}|{pair[0]}|{pair[1]}".encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "kind": self.kind,
            "offender": self.offender,
            "height": self.height,
            "sub_round": self.sub_round,
            "a": self.a,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Evidence":
        _require(isinstance(data, Mapping), "evidence must be an object")
        _require(data.get("version") == PROTOCOL_VERSION, "unsupported protocol version")
        _require(data.get("kind") in ("vote", "proposal"), "unknown evidence kind")
        _require(isinstance(data.get("a"), Mapping), "evidence.a must be an object")
        _require(isinstance(data.get("b"), Mapping), "evidence.b must be an object")
        return cls(
            kind=str(data["kind"]),
            offender=_as_str(data.get("offender"), "offender", max_len=64),
            height=_as_int(data.get("height"), "height", lo=0),
            sub_round=_as_int(data.get("sub_round"), "sub_round", lo=0, hi=1024),
            a=dict(data["a"]),
            b=dict(data["b"]),
        )


@dataclass(frozen=True)
class SyncRequest:
    height: int
    requester: str

    def to_dict(self) -> dict[str, Any]:
        return {"version": PROTOCOL_VERSION, "height": self.height,
                "requester": self.requester}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SyncRequest":
        _require(isinstance(data, Mapping), "sync request must be an object")
        _require(data.get("version") == PROTOCOL_VERSION, "unsupported protocol version")
        return cls(
            height=_as_int(data.get("height"), "height", lo=1),
            requester=_as_str(data.get("requester"), "requester", max_len=64),
        )


@dataclass(frozen=True)
class Registration:
    """A node's signed commitment to its dataset, published at join time."""

    node_id: str
    data_root: str
    n_samples: int
    signature: str = ""

    KIND = "registration"

    def payload(self) -> dict[str, Any]:
        return {"data_root": self.data_root, "n_samples": self.n_samples,
                "node_id": self.node_id}

    def signed(self, identity: Identity) -> "Registration":
        return Registration(**{**self.__dict__,
                               "signature": identity.sign(self.KIND, self.payload())})

    def to_dict(self) -> dict[str, Any]:
        return {"version": PROTOCOL_VERSION, **self.payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Registration":
        _require(isinstance(data, Mapping), "registration must be an object")
        _require(data.get("version") == PROTOCOL_VERSION, "unsupported protocol version")
        root = _as_str(data.get("data_root"), "data_root", max_len=96)
        _require(root.isdigit(), "data_root must be a decimal field element")
        return cls(
            node_id=_as_str(data.get("node_id"), "node_id", max_len=64),
            data_root=root,
            n_samples=_as_int(data.get("n_samples"), "n_samples", lo=1, hi=10 ** 9),
            signature=_as_str(data.get("signature", ""), "signature", max_len=256),
        )


@dataclass(frozen=True)
class Envelope:
    """What actually travels: a kind, a body, and per-peer flow-control keys."""

    kind: str
    body: dict[str, Any]
    sender: str
    height: int
    seq: int

    #: dedup key.  Keyed by (sender, height, seq, content) rather than by
    #: content alone, so flooding one peer's slots cannot evict another's.
    @property
    def dedup_key(self) -> str:
        digest = hashlib.sha256(canonical_json(self.body)).hexdigest()[:32]
        return f"{self.sender}:{self.height}:{self.seq}:{self.kind}:{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {"version": PROTOCOL_VERSION, "kind": self.kind, "body": self.body,
                "sender": self.sender, "height": self.height, "seq": self.seq}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Envelope":
        _require(isinstance(data, Mapping), "envelope must be an object")
        _require(data.get("version") == PROTOCOL_VERSION, "unsupported protocol version")
        _require(isinstance(data.get("body"), Mapping), "body must be an object")
        return cls(
            kind=_as_str(data.get("kind"), "kind", max_len=32),
            body=dict(data["body"]),
            sender=_as_str(data.get("sender"), "sender", max_len=64),
            height=_as_int(data.get("height"), "height", lo=0),
            seq=_as_int(data.get("seq"), "seq", lo=0),
        )
