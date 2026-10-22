"""Live network tests: honest operation, then each adversary in turn.

These run the real node code over a real (in-memory) mesh with injected
latency, so the timing is not a localhost artefact.  ZK is off by default here
because a strict run costs seconds per proof; ``test_live_round_with_real_proofs``
covers the strict path and is marked ``circuit`` and ``slow``.
"""
from __future__ import annotations

import pytest

from zkfl.config import FixedPointConfig, ProtocolConfig
from zkfl.datasets import synthetic
from zkfl.node import EquivocatingProposer, EquivocatingVoter, Flooder, Node
from zkfl.simulation import Simulation

pytestmark = pytest.mark.asyncio


def make(n=7, byzantine=None, rounds_cfg=None, **kw):
    fp = FixedPointConfig()
    protocol = rounds_cfg or ProtocolConfig(
        n_nodes=n, f_byzantine=(n - 1) // 3, collect_window=0.15,
        strict_zk=False, aggregation_rule="trimmed_mean",
    )
    protocol.check()
    split = synthetic(n_samples=64 * n, n_features=fp.n_features, seed=11)
    return Simulation(
        split=split, fp=fp, protocol=protocol, n_nodes=n, strict_zk=False,
        byzantine=byzantine, latency=0.005, jitter=0.005, seed=5, **kw
    )


async def test_honest_network_agrees_every_round():
    sim = make(n=7)
    result = await sim.run(rounds=3, proposers_per_round=3, timeout=15)
    assert all(r.agreed for r in result.rounds), [r.heights for r in result.rounds]
    for name, ok, detail in sim.check_invariants():
        assert ok, f"{name}: {detail}"


async def test_all_nodes_hold_bitwise_identical_models():
    sim = make(n=7)
    await sim.run(rounds=2, proposers_per_round=3, timeout=15)
    models = {node.model.w for node in sim.nodes}
    biases = {node.model.b for node in sim.nodes}
    assert len(models) == 1 and len(biases) == 1


async def test_every_committed_model_is_the_aggregate_of_its_proposals():
    """The audit log is checkable: recompute each round from its own entry."""
    from zkfl.aggregation import aggregate
    from zkfl.messages import Proposal
    from zkfl.model import Model

    sim = make(n=7)
    await sim.run(rounds=2, proposers_per_round=3, timeout=15)
    node = sim.nodes[0]
    model = sim.genesis
    for entry in node.audit:
        proposals = [Proposal.from_dict(p) for p in entry["proposals"]]
        assert entry["prev_model_hash"] == model.digest()
        result = aggregate(model, proposals, sim.fp,
                           rule=entry["aggregation"]["rule"],
                           permille=entry["aggregation"]["trim_permille"])
        assert result.model.digest() == entry["model_hash"]
        model = Model.from_dict(entry["model"])
        assert model.digest() == entry["model_hash"]


async def test_a_certificate_carries_a_quorum_at_every_height():
    from zkfl.messages import Certificate

    sim = make(n=7)
    await sim.run(rounds=2, proposers_per_round=3, timeout=15)
    q = sim.protocol.effective_quorum()
    for entry in sim.nodes[0].audit:
        certificate = Certificate.from_dict(entry["certificate"])
        assert certificate.is_valid(q, sim.registry.members())


async def test_an_equivocating_voter_is_caught_and_the_round_still_commits():
    sim = make(n=10, byzantine={9: EquivocatingVoter})
    result = await sim.run(rounds=2, proposers_per_round=3, timeout=20)
    offender = sim.nodes[9].node_id
    honest = [n for n in sim.nodes if type(n) is Node]
    assert all(n.engine.accountability.is_blacklisted(offender) for n in honest), (
        "the equivocator was not blacklisted everywhere"
    )
    assert result.rounds[-1].committed >= len(honest)


async def test_an_equivocating_proposer_is_caught():
    """The attack vote-set cross-checking never covered."""
    sim = make(n=10, byzantine={0: EquivocatingProposer})
    await sim.run(rounds=2, proposers_per_round=3, timeout=20)
    offender = sim.nodes[0].node_id
    honest = [n for n in sim.nodes if type(n) is Node]
    caught = sum(1 for n in honest
                 if n.engine.accountability.is_blacklisted(offender))
    assert caught == len(honest), f"only {caught}/{len(honest)} detected it"


async def test_an_equivocating_proposer_cannot_stall_the_height():
    """Detecting the equivocation is not enough; the proposal must be retracted.

    Its first message is accepted before its second exposes it, and honest
    nodes see the two in different orders -- so they accept *different* firsts.
    If the node that catches it locally keeps what it accepted while the nodes
    that learn by gossip drop it, the proposal sets never converge: prevotes
    split 5/4 against q=7 at every sub-round and the height never decides.
    """
    sim = make(n=10, byzantine={0: EquivocatingProposer})
    result = await sim.run(rounds=3, proposers_per_round=4, timeout=25)
    honest = [n for n in sim.nodes if type(n) is Node]
    offender = sim.nodes[0].node_id

    assert all(r.agreed for r in result.rounds), (
        f"an equivocating proposer stalled the network: "
        f"{[r.heights for r in result.rounds]}"
    )
    for node in honest:
        extra = set(node.engine.accountability.blacklist) - {offender}
        assert not extra, f"{node.label} blacklisted an honest node: {extra}"
    for name, ok, detail in sim.check_invariants():
        assert ok, f"{name}: {detail}"


async def test_flooding_does_not_stall_the_network():
    sim = make(n=10, byzantine={0: Flooder})
    result = await sim.run(rounds=2, proposers_per_round=3, timeout=25)
    honest = [n for n in sim.nodes if type(n) is Node]
    assert all(n.height >= 1 for n in honest), [n.height for n in honest]
    assert any(n.gate.stats.rate_limited > 0 for n in honest), (
        "the rate limiter never engaged under a flood"
    )


async def test_a_node_that_misses_a_round_catches_up_by_sync():
    """A node whose window expired short of a quorum must not be stuck.

    The old design recorded the missed height in ``_missed_heights`` and never
    acted on it, so such a node stayed at h forever.  Catching up must also
    survive a lost answer: asking once and giving up is the same bug.
    """
    sim = make(n=7)
    lagging = sim.nodes[-1]

    sim.silence(len(sim.nodes) - 1)
    await sim.run(rounds=1, proposers_per_round=3, timeout=12, stop_after=False)
    assert lagging.height == 0, "the isolated node should have missed the round"

    sim.unsilence(len(sim.nodes) - 1)
    try:
        assert await sim.sync_until(lagging, 1, timeout=5), "catch-up never completed"
    finally:
        await sim.shutdown()

    assert lagging.model.digest() == sim.nodes[0].model.digest()


async def test_message_cost_is_quadratic_not_cubic():
    """Relaying every vote would make this grow with n^3."""
    costs = {}
    for n in (5, 10):
        sim = make(n=n)
        result = await sim.run(rounds=1, proposers_per_round=2, timeout=20)
        costs[n] = result.rounds[0].messages
    growth = costs[10] / max(costs[5], 1)
    assert growth < 6.0, f"messages grew {growth:.1f}x when n doubled: {costs}"


@pytest.mark.circuit
@pytest.mark.slow
async def test_live_round_with_real_proofs(circuit_available):
    """The claim the paper needs: a live round in strict mode, end to end."""
    if not circuit_available:
        pytest.skip("circuit artifacts absent")
    fp = FixedPointConfig()
    protocol = ProtocolConfig(n_nodes=4, f_byzantine=1, collect_window=0.5,
                              strict_zk=True, aggregation_rule="trimmed_mean")
    split = synthetic(n_samples=256, n_features=fp.n_features, seed=13)
    sim = Simulation(split=split, fp=fp, protocol=protocol, n_nodes=4,
                     strict_zk=True, latency=0.005, seed=2)
    result = await sim.run(rounds=1, proposers_per_round=2, timeout=180)
    assert result.rounds[0].agreed, result.rounds[0].heights
    entry = sim.nodes[0].audit.entry(1)
    assert entry is not None and entry["proposals"], "no proofs recorded"
    for proposal in entry["proposals"]:
        assert isinstance(proposal["proof"], dict)
        assert proposal["proof"]["scheme"] == "groth16"
