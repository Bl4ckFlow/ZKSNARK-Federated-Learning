"""Accountable equivocation detection.

The property implemented here is *accountability*, not fault tolerance -- they
are separate claims and the paper should make both separately.  Fault tolerance
comes from the quorum size; accountability means that when a node does
equivocate, every honest node ends up holding the same transferable proof of it
and reaches the same verdict.

The previous design got this backwards.  Votes were unsigned, and a peer's
claim inside a ``vote_set`` that "j voted B" was enough to blacklist j
permanently -- so a single Byzantine node could evict every honest node in the
network.  Here the only thing that blacklists anyone is
:class:`~zkfl.messages.Evidence`: two messages carrying j's own signature, in
the same slot, with different content.  A reporter is never trusted; it is only
a courier.

Consequences worth stating:

* Blacklists converge, because every honest node evaluates the same evidence
  with the same deterministic rule.
* Framing is impossible without forging Ed25519.
* Evidence arriving after the height commits is still checkable, because signed
  votes are retained for the last ``retention`` heights rather than pruned the
  moment a height closes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from zkfl.messages import Evidence, Proposal, Vote


@dataclass
class EquivocationRecord:
    offender: str
    kind: str
    height: int
    sub_round: int
    evidence_id: str
    detected_by: str
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "offender": self.offender,
            "kind": self.kind,
            "height": self.height,
            "sub_round": self.sub_round,
            "evidence_id": self.evidence_id,
            "detected_by": self.detected_by,
            "detected_at": self.detected_at,
        }


class AccountabilityLog:
    """Tracks signed messages by slot and converts conflicts into evidence."""

    def __init__(self, node_id: str, retention: int = 8):
        self.node_id = node_id
        self.retention = max(1, retention)
        #: (voter, height, sub_round) -> the first vote seen in that slot
        self._votes: dict[tuple[str, int, int], Vote] = {}
        #: (proposer, height) -> the first proposal seen at that height
        self._proposals: dict[tuple[str, int], Proposal] = {}
        self._blacklist: set[str] = set()
        self._evidence: dict[str, Evidence] = {}
        self.records: list[EquivocationRecord] = []

    # -- observation --------------------------------------------------------

    def observe_vote(self, vote: Vote) -> Optional[Evidence]:
        """Record a *signature-checked* vote; return evidence if it conflicts.

        The caller must have verified the signature already -- otherwise this
        would manufacture evidence out of an unsigned forgery.
        """
        key = vote.slot
        first = self._votes.get(key)
        if first is None:
            self._votes[key] = vote
            return None
        if first.payload() == vote.payload():
            return None
        return Evidence(
            kind="vote",
            offender=vote.voter,
            height=vote.height,
            sub_round=vote.sub_round,
            a=first.to_dict(),
            b=vote.to_dict(),
        )

    def observe_proposal(self, proposal: Proposal) -> Optional[Evidence]:
        """One proposal per (proposer, height); a second distinct one is evidence.

        This closes the gap that vote-set cross-checking never could: a
        Byzantine *proposer* sending A to half the network and B to the other
        half used to split the vote every round with no consequence.
        """
        key = (proposal.proposer, proposal.height)
        first = self._proposals.get(key)
        if first is None:
            self._proposals[key] = proposal
            return None
        if first.payload() == proposal.payload():
            return None
        return Evidence(
            kind="proposal",
            offender=proposal.proposer,
            height=proposal.height,
            sub_round=0,
            a=first.to_dict(),
            b=proposal.to_dict(),
        )

    # -- evidence -----------------------------------------------------------

    def accept(self, evidence: Evidence, *, from_peer: str | None = None) -> bool:
        """Verify evidence and blacklist the offender.

        Returns True only when this is new, valid evidence -- the caller uses
        that to decide whether to gossip it onward, so replaying old evidence
        does not amplify.
        """
        if not evidence.is_valid():
            return False
        eid = evidence.evidence_id
        if eid in self._evidence:
            return False
        self._evidence[eid] = evidence
        self._blacklist.add(evidence.offender)
        self.records.append(EquivocationRecord(
            offender=evidence.offender,
            kind=evidence.kind,
            height=evidence.height,
            sub_round=evidence.sub_round,
            evidence_id=eid,
            detected_by=from_peer or self.node_id,
        ))
        return True

    def is_blacklisted(self, node_id: str) -> bool:
        return node_id in self._blacklist

    @property
    def blacklist(self) -> frozenset[str]:
        return frozenset(self._blacklist)

    def all_evidence(self) -> list[Evidence]:
        return list(self._evidence.values())

    # -- retention ----------------------------------------------------------

    def prune(self, committed_height: int) -> None:
        """Drop slot state older than the retention window.

        Not on commit: keeping the last few heights is what lets evidence that
        arrives late still be checked, which the old design made impossible by
        pruning vote state the instant a height closed.
        """
        floor = committed_height - self.retention
        if floor < 0:
            return
        self._votes = {k: v for k, v in self._votes.items() if k[1] > floor}
        self._proposals = {k: v for k, v in self._proposals.items() if k[1] > floor}

    def stats(self) -> dict:
        return {
            "blacklisted": sorted(self._blacklist),
            "evidence_count": len(self._evidence),
            "tracked_vote_slots": len(self._votes),
            "tracked_proposal_slots": len(self._proposals),
        }
