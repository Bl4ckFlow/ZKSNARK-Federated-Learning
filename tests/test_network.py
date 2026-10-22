"""Flow control: dedup that cannot be evicted, and limits that actually bind."""
from __future__ import annotations

import json
import random
import string

import pytest

from zkfl.messages import (
    Certificate,
    Envelope,
    Evidence,
    MalformedMessage,
    Proposal,
    Registration,
    SyncRequest,
    Vote,
    PREVOTE,
)
from zkfl.network import MessageGate


def envelope(sender="a" * 64, height=1, seq=0, kind="vote", body=None):
    return Envelope(kind=kind, body=body or {"n": seq}, sender=sender,
                    height=height, seq=seq)


def test_duplicates_are_dropped():
    gate = MessageGate()
    e = envelope()
    assert gate.admit(e)
    assert not gate.admit(e)
    assert gate.stats.duplicate == 1


def test_one_peer_cannot_evict_another_peers_dedup_entries():
    """The old global 10,000-entry window made this the primary amplification
    vector: flood it, and honest messages get reprocessed and rebroadcast."""
    gate = MessageGate(peer_rate_limit=10_000)
    honest = envelope(sender="h" * 64, height=1, seq=1)
    assert gate.admit(honest)

    flooder = "f" * 64
    for i in range(50_000):
        gate.admit(envelope(sender=flooder, height=1, seq=i))

    assert not gate.admit(honest), "honest message was evicted by a flood"


def test_per_peer_rate_limit_binds_per_height():
    gate = MessageGate(peer_rate_limit=8)
    peer = "p" * 64
    admitted = sum(gate.admit(envelope(sender=peer, height=1, seq=i))
                   for i in range(40))
    assert admitted == 8
    assert gate.stats.rate_limited == 32
    # A different height gets its own budget, so a burst at one height cannot
    # silence the peer forever.
    assert gate.admit(envelope(sender=peer, height=2, seq=0))


def test_unknown_senders_are_refused_before_parsing():
    gate = MessageGate()
    members = frozenset({"a" * 64})
    assert gate.admit(envelope(sender="a" * 64), members)
    assert not gate.admit(envelope(sender="z" * 64), members)
    assert gate.stats.from_stranger == 1


def test_state_is_released_when_a_height_commits():
    gate = MessageGate()
    e = envelope(height=3, seq=1)
    assert gate.admit(e)
    gate.release(3)
    assert gate.admit(e), "state for a committed height should be reclaimed"


def test_dedup_key_covers_sender_height_seq_and_content():
    a = envelope(sender="a" * 64, height=1, seq=1, body={"x": 1})
    assert a.dedup_key != envelope(sender="b" * 64, height=1, seq=1, body={"x": 1}).dedup_key
    assert a.dedup_key != envelope(sender="a" * 64, height=2, seq=1, body={"x": 1}).dedup_key
    assert a.dedup_key != envelope(sender="a" * 64, height=1, seq=2, body={"x": 1}).dedup_key
    assert a.dedup_key != envelope(sender="a" * 64, height=1, seq=1, body={"x": 2}).dedup_key


# ---------------------------------------------------------------------------
# Parser hardening
# ---------------------------------------------------------------------------

PARSERS = [Proposal, Vote, Certificate, Evidence, SyncRequest, Registration, Envelope]


@pytest.mark.parametrize("parser", PARSERS)
def test_parsers_reject_garbage_without_crashing(parser):
    junk = [None, 42, "string", [], {}, {"version": 1}, {"version": 2},
            {"version": 2, "height": "not-a-number"},
            {"version": 2, "height": -5}, {"version": 2, "height": 1.5}]
    for value in junk:
        with pytest.raises((MalformedMessage, TypeError, AttributeError)):
            parser.from_dict(value)


@pytest.mark.parametrize("parser", PARSERS)
def test_parsers_survive_fuzzed_input(parser):
    """A crude fuzzer over the message shapes.

    The bar is not that every input parses -- it is that nothing raises
    something the node does not catch, which is what would take a peer offline.
    """
    rng = random.Random(1234)
    alphabet = string.printable
    for _ in range(300):
        body = {
            rng.choice(["version", "height", "kind", "body", "votes", "model",
                        "proposal_ids", "sub_round", "signature", "a", "b",
                        "n_samples", "data_root", "sender", "seq"]):
            rng.choice([
                rng.randint(-10 ** 9, 10 ** 9),
                "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30))),
                [rng.randint(0, 5) for _ in range(rng.randint(0, 4))],
                {"nested": rng.randint(0, 5)},
                None, True, 1.5,
            ])
            for _ in range(rng.randint(1, 6))
        }
        try:
            parser.from_dict(body)
        except (MalformedMessage, TypeError, ValueError, AttributeError, KeyError):
            pass


def test_oversized_fields_are_refused():
    with pytest.raises(MalformedMessage):
        Vote.from_dict({"version": 2, "height": 1, "sub_round": 0,
                        "model_hash": "x" * 5000, "proposal_ids": [],
                        "voter": "v", "signature": ""})
    with pytest.raises(MalformedMessage):
        Vote.from_dict({"version": 2, "height": 1, "sub_round": 0,
                        "model_hash": "x", "proposal_ids": ["a"] * 5000,
                        "voter": "v", "signature": ""})


def test_a_protocol_version_mismatch_is_refused():
    with pytest.raises(MalformedMessage):
        Vote.from_dict({"version": 1, "height": 1, "sub_round": 0,
                        "model_hash": "x", "proposal_ids": [], "voter": "v"})


def test_messages_round_trip_through_json():
    from zkfl.crypto import Identity

    identity = Identity.from_seed(b"\x07" * 32)
    vote = Vote(3, 1, PREVOTE, "h" * 64, ("a", "b"), identity.node_id).signed(identity)
    revived = Vote.from_dict(json.loads(json.dumps(vote.to_dict())))
    assert revived == vote
    assert revived.verify_signature()
