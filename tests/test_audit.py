"""The audit log: hash-chained, persistent, and re-verifiable offline."""
from __future__ import annotations

import pytest

from zkfl.audit import GENESIS_CHAIN_HASH, AuditLog, chain_hash


def entry(height, prev, model_hash, ids=("p1",)):
    return {
        "height": height,
        "prev_model_hash": prev,
        "model_hash": model_hash,
        "model": {"w": [1, 2, 3, 4], "b": 0, "scale": 10 ** 6},
        "proposal_ids": list(ids),
        "proposals": [],
        "roots": {},
        "eta": 50_000,
        "certificate": {"version": 2, "height": height, "sub_round": 0,
                        "model_hash": model_hash, "votes": []},
        "aggregation": {"rule": "trimmed_mean", "trim_permille": 250},
        "committed_at_ms": 1_700_000_000_000 + height,
    }


def test_chain_links_and_verifies():
    log = AuditLog()
    a = entry(1, "0" * 64, "aa")
    b = entry(2, "aa", "bb")
    log.append(a)
    log.append(b)
    ok, reason = log.verify_chain()
    assert ok, reason
    assert log.height == 2
    assert log.head != GENESIS_CHAIN_HASH


def test_a_gap_is_refused():
    log = AuditLog()
    log.append(entry(1, "0" * 64, "aa"))
    with pytest.raises(ValueError, match="height"):
        log.append(entry(3, "aa", "cc"))


def test_a_broken_parent_link_is_refused():
    log = AuditLog()
    log.append(entry(1, "0" * 64, "aa"))
    with pytest.raises(ValueError, match="extend"):
        log.append(entry(2, "not-aa", "bb"))


def test_tampering_with_an_intermediate_entry_breaks_the_chain():
    """The property the paper claims for the log, tested rather than asserted."""
    log = AuditLog()
    log.append(entry(1, "0" * 64, "aa"))
    log.append(entry(2, "aa", "bb"))
    log.append(entry(3, "bb", "cc"))
    assert log.verify_chain()[0]

    log._entries[1]["model_hash"] = "forged"
    ok, reason = log.verify_chain()
    assert not ok and "height 2" in reason


def test_chain_hash_covers_the_predecessor():
    a = entry(1, "0" * 64, "aa")
    assert chain_hash("x" * 64, a) != chain_hash("y" * 64, a)


def test_persistence_survives_a_restart(tmp_path):
    path = tmp_path / "audit.sqlite"
    log = AuditLog(path)
    log.append(entry(1, "0" * 64, "aa"))
    log.append(entry(2, "aa", "bb"))
    head = log.head
    log.close()

    revived = AuditLog(path)
    assert revived.height == 2, "a crashed node must resume at its committed height"
    assert revived.head == head
    assert revived.verify_chain()[0]
    revived.close()


def test_evidence_is_persisted(tmp_path):
    path = tmp_path / "audit.sqlite"
    log = AuditLog(path)
    log.record_evidence("e1", "offender", "vote", 3, {"a": 1})
    log.record_evidence("e1", "offender", "vote", 3, {"a": 1})   # idempotent
    rows = log._db.execute("SELECT COUNT(*) FROM evidence").fetchone()
    assert rows[0] == 1
    log.close()


# ---------------------------------------------------------------------------
# Offline re-verification -- what an outside auditor can actually check
# ---------------------------------------------------------------------------


def _committed_log(rounds=2, nodes=7):
    """A real run, so the entries are the ones the protocol actually writes."""
    import asyncio

    from zkfl.config import FixedPointConfig, ProtocolConfig
    from zkfl.datasets import synthetic
    from zkfl.simulation import Simulation

    fp = FixedPointConfig()
    protocol = ProtocolConfig(collect_window=0.15, strict_zk=False,
                              aggregation_rule="trimmed_mean").with_nodes(nodes)
    split = synthetic(n_samples=64 * nodes, n_features=fp.n_features, seed=23)
    sim = Simulation(split=split, fp=fp, protocol=protocol, n_nodes=nodes,
                     strict_zk=False, latency=0.003, seed=23)

    async def go():
        await sim.run(rounds=rounds, proposers_per_round=3, timeout=30,
                      evaluate=False)
        entries = sim.nodes[0].audit.entries()
        await sim.shutdown()
        return entries

    return sim, asyncio.run(go())


@pytest.fixture(scope="module")
def committed():
    """One real run, shared by every tampering test in this section."""
    return _committed_log()


def _audit(entries, sim):
    from zkfl.audit import verify_offline
    from zkfl.zk import make_backend

    return verify_offline(
        entries=entries, genesis_model=sim.genesis.to_dict(),
        quorum=sim.protocol.effective_quorum(), members=sim.registry.members(),
        fp=sim.fp, protocol=sim.protocol, backend=make_backend(sim.fp, False),
    )


def test_a_real_log_re_verifies_from_scratch(committed):
    sim, entries = committed
    assert entries, "the run committed nothing to audit"
    ok, reason = _audit(entries, sim)
    assert ok, reason


def test_a_model_edited_without_fixing_its_hash_is_caught(committed):
    """The entry states the model twice; a reader consumes the readable copy.

    Checking only the hash-derived path left that copy free to say anything,
    which is a forged model that passes verification.
    """
    import copy

    sim, entries = committed
    forged = copy.deepcopy(entries)
    forged[-1]["model"]["w"] = [v + 1 for v in forged[-1]["model"]["w"]]
    ok, reason = _audit(forged, sim)
    assert not ok and "hash" in reason, reason


def test_a_model_edited_with_a_matching_hash_is_still_caught(committed):
    """Because the aggregate is recomputed, not taken on trust."""
    import copy

    from zkfl.model import Model

    sim, entries = committed
    forged = copy.deepcopy(entries)
    swapped = Model(w=tuple(v + 1 for v in forged[-1]["model"]["w"]),
                    b=forged[-1]["model"]["b"], scale=sim.fp.scale)
    forged[-1]["model"] = swapped.to_dict()
    forged[-1]["model_hash"] = swapped.digest()
    ok, reason = _audit(forged, sim)
    assert not ok and "aggregate" in reason, reason


def test_a_certificate_short_of_a_quorum_is_caught(committed):
    import copy

    sim, entries = committed
    forged = copy.deepcopy(entries)
    forged[-1]["certificate"]["votes"] = forged[-1]["certificate"]["votes"][:-1]
    ok, reason = _audit(forged, sim)
    assert not ok and "quorum" in reason, reason


def test_a_polka_passed_off_as_a_decision_is_caught(committed):
    """q prevotes are proof a quorum *could* commit, not that it did."""
    import copy

    sim, entries = committed
    forged = copy.deepcopy(entries)
    for vote in forged[-1]["certificate"]["votes"]:
        vote["phase"] = "prevote"
    ok, reason = _audit(forged, sim)
    assert not ok and "quorum" in reason, reason


def test_dropping_a_proposal_from_the_set_is_caught(committed):
    """Censoring a contributor after the fact changes the aggregate."""
    import copy

    sim, entries = committed
    forged = copy.deepcopy(entries)
    target = next((i for i, e in enumerate(forged) if len(e["proposals"]) > 1), None)
    if target is None:
        pytest.skip("no height aggregated more than one proposal")
    forged[target]["proposals"] = forged[target]["proposals"][:-1]
    ok, reason = _audit(forged, sim)
    assert not ok, "a censored contributor went unnoticed"
