"""One vote per slot, across a crash.

Both cases of the agreement proof rest on ``a correct node signs one vote per
slot``.  Crash-restart is in the fault model, so without a write-ahead log that
hypothesis is simply false for a correct node: it restarts having forgotten that
it voted, signs a second vote in the same slot, and its peers blacklist it on
evidence that is entirely genuine.
"""
from __future__ import annotations

import json

import pytest

from zkfl.config import FixedPointConfig, ProtocolConfig
from zkfl.consensus import Broadcast, ConsensusEngine
from zkfl.crypto import Identity, PeerRegistry
from zkfl.messages import PREVOTE, Vote
from zkfl.model import Model
from zkfl.trainer import FixedPointTrainer, LocalDataset
from zkfl.votelog import DurableVoteLog, InMemoryVoteLog, slot_of
from zkfl.zk import NullProofBackend


def _fixtures():
    fp = FixedPointConfig()
    protocol = ProtocolConfig(n_nodes=10, f_byzantine=3, strict_zk=False,
                              collect_window=0.01)
    ids = [Identity.from_seed(bytes([i]) * 32) for i in range(10)]
    from zkfl.datasets import synthetic
    split = synthetic(n_samples=64, n_features=fp.n_features, seed=9)
    dataset = LocalDataset(split.x_train.tolist(), split.y_train.tolist(), fp)
    registry = PeerRegistry()
    for identity in ids:
        payload = PeerRegistry.registration_payload(
            identity.node_id, dataset.root, dataset.n_samples)
        registry.register(identity.node_id, dataset.root, dataset.n_samples,
                          identity.sign("registration", payload))
    return fp, protocol, ids, registry, dataset, Model.zeros(fp)


def _engine(identity, fp, protocol, registry, genesis, vote_log):
    return ConsensusEngine(
        identity=identity, fp=fp, protocol=protocol, registry=registry,
        backend=NullProofBackend(i_understand_this_is_not_a_proof=True),
        genesis=genesis, vote_log=vote_log,
    )


def _propose(engine, dataset, fp, protocol):
    trainer = FixedPointTrainer(fp, protocol.learning_rate)
    after, witness = trainer.step(engine.model, dataset, 0)
    return engine.create_proposal(after, engine.backend.prove(witness), 0)


def _own_votes(engine, effects):
    return [Vote.from_dict(b.body) for b in effects
            if isinstance(b, Broadcast) and b.kind == "vote"
            and Vote.from_dict(b.body).voter == engine.node_id]


def test_a_restarted_node_resends_its_vote_instead_of_signing_a_new_one(tmp_path):
    fp, protocol, ids, registry, dataset, genesis = _fixtures()
    path = tmp_path / "votes.log"

    before = _engine(ids[0], fp, protocol, registry, genesis,
                     DurableVoteLog(path))
    before.handle_proposal(_propose(before, dataset, fp, protocol))
    first = _own_votes(before, before.on_timeout())
    assert first, "the node must have prevoted"
    original = first[0]
    before.vote_log.close()

    # The process dies here, after the vote was recorded and broadcast.
    # A fresh engine comes up on the same log with no in-memory state at all.
    after = _engine(ids[0], fp, protocol, registry, genesis,
                    DurableVoteLog(path))
    assert after.vote_log.recall(slot_of(original)) is not None, \
        "the vote did not survive the restart"

    after.handle_proposal(_propose(after, dataset, fp, protocol))
    again = _own_votes(after, after.on_timeout())
    assert again, "the restarted node must still vote"
    assert again[0].model_hash == original.model_hash, (
        "the restarted node signed a *different* value in a slot it had already "
        "voted in -- that is self-equivocation, and its peers will blacklist it"
    )
    assert again[0].signature == original.signature, \
        "it should re-send the recorded vote, byte for byte"


def test_the_log_survives_a_torn_final_record(tmp_path):
    """A crash mid-write leaves a partial line; everything before it is intact."""
    fp, protocol, ids, registry, dataset, genesis = _fixtures()
    path = tmp_path / "votes.log"

    log = DurableVoteLog(path)
    engine = _engine(ids[0], fp, protocol, registry, genesis, log)
    engine.handle_proposal(_propose(engine, dataset, fp, protocol))
    kept = _own_votes(engine, engine.on_timeout())[0]
    log.close()

    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"height": 1, "sub_round": 9, "phase": "prev')  # torn

    recovered = DurableVoteLog(path)
    assert recovered.recall(slot_of(kept)) is not None, \
        "a torn tail must not lose the records written before it"
    assert len(recovered) == 1


def test_a_vote_is_on_disk_before_it_is_returned_for_broadcast(tmp_path):
    """Ordering is the whole point: durable first, sendable second."""
    fp, protocol, ids, registry, dataset, genesis = _fixtures()
    path = tmp_path / "votes.log"
    log = DurableVoteLog(path)

    seen: list[int] = []
    real_record = log.record

    def watching_record(vote):
        real_record(vote)
        # At the moment record() returns, the bytes must already be readable.
        seen.append(len(path.read_text().splitlines()))

    log.record = watching_record
    engine = _engine(ids[0], fp, protocol, registry, genesis, log)
    engine.handle_proposal(_propose(engine, dataset, fp, protocol))
    engine.on_timeout()

    assert seen and all(n >= 1 for n in seen), \
        "record() returned before the vote reached the file"


def test_the_in_memory_log_does_not_claim_to_be_durable():
    assert InMemoryVoteLog().durable is False
    assert DurableVoteLog.durable is True
