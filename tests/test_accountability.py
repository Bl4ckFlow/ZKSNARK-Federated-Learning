"""Accountability: evidence is transferable, and framing is impossible.

The property the old design claimed -- "detects and excludes equivocating
nodes" -- was implemented by trusting a peer's unsigned assertion about a third
party, which made the mechanism a weapon rather than a defence.  These tests
pin down the replacement.
"""
from __future__ import annotations

import pytest

from zkfl.accountability import AccountabilityLog
from zkfl.crypto import Identity
from zkfl.messages import PREVOTE, Evidence, Proposal, Vote
from zkfl.model import Model
from zkfl.config import FixedPointConfig


@pytest.fixture
def parties():
    return (Identity.from_seed(b"\x01" * 32),   # byzantine
            Identity.from_seed(b"\x02" * 32),   # honest
            Identity.from_seed(b"\x03" * 32))   # observer


def test_conflicting_votes_produce_valid_evidence(parties):
    byz, _, observer = parties
    log = AccountabilityLog(observer.node_id)
    a = Vote(1, 0, PREVOTE, "a" * 64, ("p1",), byz.node_id).signed(byz)
    b = Vote(1, 0, PREVOTE, "b" * 64, ("p2",), byz.node_id).signed(byz)

    assert log.observe_vote(a) is None
    evidence = log.observe_vote(b)
    assert evidence is not None and evidence.is_valid()
    assert log.accept(evidence)
    assert log.is_blacklisted(byz.node_id)


def test_evidence_is_idempotent(parties):
    byz, _, observer = parties
    log = AccountabilityLog(observer.node_id)
    a = Vote(1, 0, PREVOTE, "a" * 64, ("p1",), byz.node_id).signed(byz)
    b = Vote(1, 0, PREVOTE, "b" * 64, ("p2",), byz.node_id).signed(byz)
    log.observe_vote(a)
    evidence = log.observe_vote(b)
    assert log.accept(evidence)
    assert not log.accept(evidence), "replayed evidence must not re-amplify"
    assert len(log.records) == 1


def test_the_same_vote_twice_is_not_equivocation(parties):
    byz, _, observer = parties
    log = AccountabilityLog(observer.node_id)
    vote = Vote(1, 0, PREVOTE, "a" * 64, ("p1",), byz.node_id).signed(byz)
    log.observe_vote(vote)
    assert log.observe_vote(vote) is None
    assert not log.is_blacklisted(byz.node_id)


def test_votes_in_different_sub_rounds_are_not_equivocation(parties):
    """Changing a vote across sub-rounds is legal; the lock governs when."""
    byz, _, observer = parties
    log = AccountabilityLog(observer.node_id)
    log.observe_vote(Vote(1, 0, PREVOTE, "a" * 64, ("p1",), byz.node_id).signed(byz))
    assert log.observe_vote(
        Vote(1, 1, PREVOTE, "b" * 64, ("p2",), byz.node_id).signed(byz)
    ) is None


def test_an_honest_node_cannot_be_framed_with_an_unsigned_message(parties):
    _, honest, observer = parties
    log = AccountabilityLog(observer.node_id)
    real = Vote(1, 0, PREVOTE, "real", ("p1",), honest.node_id).signed(honest)
    invented = Vote(1, 0, PREVOTE, "invented", ("p9",), honest.node_id)      # no signature
    frame = Evidence(kind="vote", offender=honest.node_id, height=1, sub_round=0,
                     a=invented.to_dict(), b=real.to_dict())
    assert not frame.is_valid()
    assert not log.accept(frame)
    assert not log.is_blacklisted(honest.node_id)


def test_an_honest_node_cannot_be_framed_with_someone_elses_signature(parties):
    byz, honest, observer = parties
    log = AccountabilityLog(observer.node_id)
    real = Vote(1, 0, PREVOTE, "real", ("p1",), honest.node_id).signed(honest)
    invented = Vote(1, 0, PREVOTE, "invented", ("p9",), honest.node_id)
    cross_signed = {**invented.to_dict(),
                    "signature": byz.sign("vote", invented.payload())}
    frame = Evidence(kind="vote", offender=honest.node_id, height=1, sub_round=0,
                     a=cross_signed, b=real.to_dict())
    assert not frame.is_valid()
    assert not log.accept(frame)


def test_evidence_from_a_different_slot_is_invalid(parties):
    byz, _, observer = parties
    log = AccountabilityLog(observer.node_id)
    a = Vote(1, 0, PREVOTE, "a", ("p1",), byz.node_id).signed(byz)
    b = Vote(2, 0, PREVOTE, "b", ("p2",), byz.node_id).signed(byz)
    bogus = Evidence("vote", byz.node_id, 1, 0, a.to_dict(), b.to_dict())
    assert not bogus.is_valid()
    assert not log.accept(bogus)


def test_proposal_equivocation_is_detected(parties):
    """The attack vote-set cross-checking could never see."""
    byz, _, observer = parties
    fp = FixedPointConfig()
    log = AccountabilityLog(observer.node_id)
    base = dict(height=1, prev_hash="p" * 64, proof=None, batch_index=0,
                proposer=byz.node_id)
    a = Proposal(model=Model(w=(1, 0, 0, 0), b=0, scale=fp.scale), **base).signed(byz)
    b = Proposal(model=Model(w=(2, 0, 0, 0), b=0, scale=fp.scale), **base).signed(byz)

    assert log.observe_proposal(a) is None
    evidence = log.observe_proposal(b)
    assert evidence is not None and evidence.is_valid()
    assert log.accept(evidence)
    assert log.is_blacklisted(byz.node_id)


def test_all_honest_nodes_reach_the_same_verdict(parties):
    """Convergence: the same evidence, checked independently, blacklists the
    same key everywhere.  Nothing depends on who reported it."""
    byz, _, _ = parties
    a = Vote(1, 0, PREVOTE, "a" * 64, ("p1",), byz.node_id).signed(byz)
    b = Vote(1, 0, PREVOTE, "b" * 64, ("p2",), byz.node_id).signed(byz)
    evidence = Evidence("vote", byz.node_id, 1, 0, a.to_dict(), b.to_dict())

    logs = [AccountabilityLog(f"observer-{i}") for i in range(20)]
    for log in logs:
        assert log.accept(evidence, from_peer="anyone-at-all")
    assert all(log.blacklist == frozenset({byz.node_id}) for log in logs)


def test_late_evidence_is_still_checkable_within_the_retention_window(parties):
    byz, _, observer = parties
    log = AccountabilityLog(observer.node_id, retention=4)
    log.observe_vote(Vote(5, 0, PREVOTE, "a", ("p1",), byz.node_id).signed(byz))

    log.prune(committed_height=6)             # inside the window
    assert log.observe_vote(
        Vote(5, 0, PREVOTE, "b", ("p2",), byz.node_id).signed(byz)
    ) is not None

    log.prune(committed_height=20)            # far outside it
    assert log.observe_vote(
        Vote(5, 0, PREVOTE, "c", ("p3",), byz.node_id).signed(byz)
    ) is None
