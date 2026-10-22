"""Consensus safety and liveness, exercised on the state machine directly.

The engine returns effects instead of performing I/O, so these are real
protocol tests rather than timing tests: no sockets, no sleeps, and a
deliberately hostile message order where it matters.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from zkfl.config import FixedPointConfig, ProtocolConfig
from zkfl.consensus import (
    Broadcast,
    Committed,
    ConsensusEngine,
    NeedSync,
    OpenWindow,
    Rejected,
)
from zkfl.crypto import Identity, PeerRegistry
from zkfl.messages import NIL, PRECOMMIT, PREVOTE, Certificate, Proposal, Vote
from zkfl.model import Model
from zkfl.trainer import FixedPointTrainer, LocalDataset
from zkfl.zk import NullProofBackend


def build(n=10, f=3, rule="trimmed_mean"):
    fp = FixedPointConfig()
    protocol = ProtocolConfig(n_nodes=n, f_byzantine=f, strict_zk=False,
                              aggregation_rule=rule, collect_window=0.01)
    protocol.check()
    ids = [Identity.from_seed(bytes([i]) * 32) for i in range(n)]

    from zkfl.datasets import synthetic
    split = synthetic(n_samples=64, n_features=fp.n_features, seed=9)
    dataset = LocalDataset(split.x_train.tolist(), split.y_train.tolist(), fp)

    registry = PeerRegistry()
    for identity in ids:
        payload = PeerRegistry.registration_payload(
            identity.node_id, dataset.root, dataset.n_samples)
        registry.register(identity.node_id, dataset.root, dataset.n_samples,
                          identity.sign("registration", payload))

    genesis = Model.zeros(fp)
    engines = [
        ConsensusEngine(
            identity=identity, fp=fp, protocol=protocol, registry=registry,
            backend=NullProofBackend(i_understand_this_is_not_a_proof=True),
            genesis=genesis,
        )
        for identity in ids
    ]
    return fp, protocol, ids, registry, dataset, genesis, engines


def make_proposal(engine, dataset, fp, protocol, batch=0):
    trainer = FixedPointTrainer(fp, protocol.learning_rate)
    after, witness = trainer.step(engine.model, dataset, batch)
    proof = engine.backend.prove(witness)
    return engine.create_proposal(after, proof, batch)


def broadcasts(effects, kind):
    return [e for e in effects if isinstance(e, Broadcast) and e.kind == kind]


def votes_out(engine, effects, phase):
    """The engine's *own* votes of one phase, ignoring relays of others'."""
    out = [Vote.from_dict(b.body) for b in broadcasts(effects, "vote")]
    return [v for v in out if v.phase == phase and v.voter == engine.node_id]


def prevote(engine):
    """Expire the prevote window and return what the engine prevoted."""
    cast = votes_out(engine, engine.on_timeout(), PREVOTE)
    assert cast, "a node holding proposals must prevote"
    return cast[0]


def run_sub_round(engines, deliver_to=None):
    """Drive one full sub-round and return every precommit produced.

    Each engine prevotes, every prevote is delivered to every engine, and the
    polka that follows is what licenses each engine to precommit.  This is the
    protocol as the network runs it, not a shortcut around it.
    """
    deliver_to = engines if deliver_to is None else deliver_to
    prevotes = [prevote(e) for e in engines]
    out = []
    for engine in deliver_to:
        for vote in prevotes:
            if vote.voter == engine.node_id:
                continue
            out.extend(votes_out(engine, engine.handle_vote(vote), PRECOMMIT))
    return prevotes, out


# ---------------------------------------------------------------------------
# Quorum arithmetic -- the fault model itself
# ---------------------------------------------------------------------------


def test_quorum_must_exceed_half_of_n_plus_f():
    ProtocolConfig(n_nodes=10, f_byzantine=3).check()          # q = 7
    with pytest.raises(AssertionError, match="unsafe quorum"):
        ProtocolConfig(n_nodes=10, f_byzantine=3, quorum=6).check()
    with pytest.raises(AssertionError, match="no liveness"):
        ProtocolConfig(n_nodes=10, f_byzantine=3, quorum=9).check()


def test_the_papers_original_parameters_only_tolerate_one_fault():
    """n=10, q=6 was reported as Byzantine-resilient; it tolerates f=1."""
    ProtocolConfig(n_nodes=10, f_byzantine=1, quorum=6).check()
    with pytest.raises(AssertionError):
        ProtocolConfig(n_nodes=10, f_byzantine=2, quorum=6).check()


def test_derived_quorum_scales_with_n():
    for n in (4, 7, 10, 20, 50):
        cfg = ProtocolConfig().with_nodes(n)
        assert 2 * cfg.effective_quorum() > n + cfg.f_byzantine
        assert n - cfg.f_byzantine >= cfg.effective_quorum()


# ---------------------------------------------------------------------------
# Deterministic decision
# ---------------------------------------------------------------------------


def test_same_proposal_set_yields_the_same_vote_on_every_node():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    proposals = [make_proposal(engines[i], dataset, fp, protocol, batch=i % dataset.n_batches)
                 for i in range(3)]

    votes = []
    for engine in engines:
        for proposal in proposals:
            engine.handle_proposal(proposal)
        votes.append(prevote(engine))

    assert len({v.model_hash for v in votes}) == 1, (
        "honest nodes with identical proposal sets disagreed"
    )
    assert len({v.proposal_ids for v in votes}) == 1


def test_proposal_order_does_not_change_the_decision():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    proposals = [make_proposal(engines[i], dataset, fp, protocol, batch=i % dataset.n_batches)
                 for i in range(3)]

    a, b = engines[0], engines[1]
    for p in proposals:
        a.handle_proposal(p)
    for p in reversed(proposals):
        b.handle_proposal(p)
    assert prevote(a).model_hash == prevote(b).model_hash


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------


def test_commit_requires_a_full_quorum():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    proposals = [make_proposal(engines[0], dataset, fp, protocol)]

    for engine in engines:
        engine.handle_proposal(proposals[0])

    # Every engine but the target runs the sub-round, so the target holds only
    # its own precommit and must wait for q-1 more.
    target = engines[0]
    _, precommits = run_sub_round(engines[1:])
    assert precommits, "a polka must produce precommits"
    prevote(target)
    for vote in [v for v in _ if v.voter != target.node_id][:q - 1]:
        target.handle_vote(vote)          # reach the polka, precommit locally

    partial = [effect
               for vote in precommits[:q - 2]
               for effect in target.handle_vote(vote)]
    assert not any(isinstance(e, Committed) for e in partial), (
        f"committed on fewer than q={q} precommits"
    )
    effects = target.handle_vote(precommits[q - 2])
    assert any(isinstance(e, Committed) for e in effects), "q precommits did not commit"
    assert target.height == 1


def test_certificate_alone_commits_a_node_that_never_voted():
    """Commit is certificate-based: a node's own window is irrelevant."""
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    proposal = make_proposal(engines[0], dataset, fp, protocol)

    voters = engines[:q]
    for engine in voters:
        engine.handle_proposal(proposal)
    _, votes = run_sub_round(voters)
    certificate = Certificate(height=1, sub_round=0,
                              model_hash=votes[0].model_hash, votes=tuple(votes[:q]))

    late = engines[-1]
    late.handle_proposal(proposal)           # holds the proposal, never voted
    effects = late.handle_certificate(certificate)
    assert any(isinstance(e, Committed) for e in effects)
    assert late.height == 1
    assert late.model.digest() == certificate.model_hash


def test_certificate_for_an_unknown_model_triggers_sync_not_a_commit():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    proposal = make_proposal(engines[0], dataset, fp, protocol)
    for engine in engines[:q]:
        engine.handle_proposal(proposal)
    _, votes = run_sub_round(engines[:q])
    certificate = Certificate(height=1, sub_round=0,
                              model_hash=votes[0].model_hash, votes=tuple(votes[:q]))

    ignorant = engines[-1]                    # never saw the proposal
    effects = ignorant.handle_certificate(certificate)
    assert not any(isinstance(e, Committed) for e in effects)
    assert any(isinstance(e, NeedSync) for e in effects)
    assert ignorant.height == 0


def test_certificate_with_duplicate_or_foreign_voters_is_rejected():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    proposal = make_proposal(engines[0], dataset, fp, protocol)
    for engine in engines[:q]:
        engine.handle_proposal(proposal)
    _, votes = run_sub_round(engines[:q])
    votes = list(votes[:q])

    padded = Certificate(1, 0, votes[0].model_hash, tuple(votes[:-1] + [votes[0]]))
    assert not padded.is_valid(q, registry.members())

    outsider = Identity.from_seed(b"\xff" * 32)
    fake = Vote(1, 0, PRECOMMIT, votes[0].model_hash, votes[0].proposal_ids,
                outsider.node_id).signed(outsider)
    foreign = Certificate(1, 0, votes[0].model_hash, tuple(votes[:-1] + [fake]))
    assert not foreign.is_valid(q, registry.members())

    # A quorum of *prevotes* is a polka, not a decision, and must never be
    # accepted as a certificate: only precommits carry a lock.
    prevoted = [Vote(1, 0, PREVOTE, v.model_hash, v.proposal_ids, v.voter)
                for v in votes]
    signed = [v.signed(next(i for i in ids if i.node_id == v.voter)) for v in prevoted]
    assert not Certificate(1, 0, votes[0].model_hash, tuple(signed)).is_valid(
        q, registry.members())


def test_unsigned_votes_never_count():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    proposal = make_proposal(engines[0], dataset, fp, protocol)
    engines[0].handle_proposal(proposal)
    unsigned = Vote(1, 0, PREVOTE, "a" * 64, (proposal.proposal_id,), ids[1].node_id)
    effects = engines[0].handle_vote(unsigned)
    assert any(isinstance(e, Rejected) and "signature" in e.reason for e in effects)
    assert engines[0].metrics["votes_accepted"] == 0


def test_two_conflicting_certificates_need_more_than_f_equivocators():
    """Safety, arithmetically: any two quorums overlap in > f voters."""
    for n, f in ((10, 3), (4, 1), (7, 2), (22, 7)):
        cfg = ProtocolConfig(n_nodes=n, f_byzantine=f)
        cfg.check()
        q = cfg.effective_quorum()
        assert 2 * q - n > f, (
            f"n={n} f={f} q={q}: two quorums share only {2 * q - n} nodes"
        )


# ---------------------------------------------------------------------------
# Sub-rounds and the lock
# ---------------------------------------------------------------------------


def test_a_polka_locks_the_node_and_the_lock_survives_new_proposals():
    """A node that saw q prevotes for X keeps prevoting X.

    Without this a later sub-round could certify a different model while X was
    already committed somewhere, which is exactly the safety violation the lock
    exists to prevent.
    """
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    engine = engines[0]
    first = make_proposal(engine, dataset, fp, protocol, batch=0)
    engine.handle_proposal(first)
    v0 = prevote(engine)

    for identity in ids[1:q]:                       # q-1 peers echo our value
        engine.handle_vote(
            Vote(1, 0, PREVOTE, v0.model_hash, v0.proposal_ids,
                 identity.node_id).signed(identity)
        )
    assert engine.current_round().locked_value == v0.model_hash, "no polka, no lock"

    # A new proposal arrives.  Unlocked, the node would now prefer a different
    # aggregate; locked, it must not.
    other = make_proposal(engines[1], dataset, fp, protocol, batch=1 % dataset.n_batches)
    engine.handle_proposal(other)
    engine.current_round().sub_round = 1
    engine.current_round().step = PREVOTE
    v1 = prevote(engine)

    assert v1.model_hash == v0.model_hash, "node changed its vote while locked"
    assert v1.sub_round == 1


def test_no_polka_means_no_lock_and_the_next_sub_round_is_free():
    """The liveness half of the same rule.

    Two honest nodes whose collect windows closed on different proposal sets
    prevote differently.  Under the old design each locked on its own vote and
    the height could never be decided.  Now neither locks, and the next
    sub-round -- by which time the sets have converged -- agrees.
    """
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    a, b = engines[0], engines[1]
    p0 = make_proposal(engines[0], dataset, fp, protocol, batch=0)
    p1 = make_proposal(engines[1], dataset, fp, protocol, batch=1 % dataset.n_batches)

    a.handle_proposal(p0)                  # a saw one proposal
    b.handle_proposal(p0)                  # b saw both
    b.handle_proposal(p1)
    va, vb = prevote(a), prevote(b)
    assert va.model_hash != vb.model_hash, "the test needs a genuine split"

    a.handle_vote(vb)
    b.handle_vote(va)
    assert a.current_round().locked_value is None
    assert b.current_round().locked_value is None

    # The straggler catches up on the missing proposal; the next sub-round
    # starts clean and both nodes now compute the same aggregate.
    a.handle_proposal(p1)
    for engine in (a, b):
        engine.on_timeout()                # precommit nil, advance
    assert a.current_round().sub_round >= 1 and b.current_round().sub_round >= 1
    assert prevote(a).model_hash == prevote(b).model_hash


def test_a_quorum_of_nil_precommits_starts_the_next_sub_round():
    """Do not sit out a timer whose outcome a quorum has already settled.

    Any two quorums share an honest node, so once q nodes precommit nil no
    value can be certified in that sub-round.  Waiting for the local timer
    would just add latency to every failed sub-round.
    """
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    engine = engines[0]
    engine.handle_proposal(make_proposal(engine, dataset, fp, protocol))
    prevote(engine)                        # waiting in the precommit phase
    assert engine.current_round().sub_round == 0
    for identity in ids[1:q + 1]:
        engine.handle_vote(
            Vote(1, 0, PRECOMMIT, NIL, (), identity.node_id).signed(identity)
        )
    assert engine.current_round().sub_round == 1
    assert engine.current_round().step == PREVOTE


def test_a_node_that_sees_a_polka_does_not_then_prevote_in_that_sub_round():
    """Invariant (I) of the safety proof, as a test.

    The Agreement proof needs a correct node's prevote at sub-round s to
    reflect a lock formed strictly *before* s.  If a node could observe a polka
    at s and still prevote at s, the induction step has a gap: the lock could
    move to a value at the same sub-round it is being reasoned about, and the
    induction hypothesis -- which covers only earlier sub-rounds -- would not
    apply.  The state machine closes that gap by leaving the PREVOTE step the
    moment a polka lands, and this test is what keeps it closed.
    """
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    engine = engines[0]

    # The node holds a proposal but its window has not expired, so it has not
    # prevoted yet.  A polka assembled by its peers arrives first.
    proposal = make_proposal(engines[1], dataset, fp, protocol, batch=0)
    engine.handle_proposal(proposal)
    # Peers aggregate the same set deterministically, so this is their value
    # too -- computing it does not make the node prevote.
    value = engine._prevote_value(engine.current_round())[0]
    assert value is not None

    effects = []
    for identity in ids[1:q + 1]:
        effects.extend(engine.handle_vote(
            Vote(1, 0, PREVOTE, value, (proposal.proposal_id,),
                 identity.node_id).signed(identity)))

    assert engine.current_round().locked_value == value, "q prevotes are a polka"
    assert not votes_out(engine, effects, PREVOTE), \
        "locking must not emit a prevote for the sub-round just locked"
    assert votes_out(engine, effects, PRECOMMIT), "a lock is precommitted"

    # Now the window expires.  The node must not double back and prevote at
    # sub-round 0; it has already precommitted there.
    late = engine.on_timeout()
    stragglers = [v for v in votes_out(engine, late, PREVOTE) if v.sub_round == 0]
    assert not stragglers, f"prevoted at 0 after a polka at 0: {stragglers}"


def test_a_polka_at_a_later_sub_round_moves_the_lock():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    engine = engines[0]
    mine = make_proposal(engine, dataset, fp, protocol, batch=0)
    engine.handle_proposal(mine)
    locked = prevote(engine)
    for identity in ids[1:q]:
        engine.handle_vote(
            Vote(1, 0, PREVOTE, locked.model_hash, locked.proposal_ids,
                 identity.node_id).signed(identity)
        )
    assert engine.current_round().locked_value == locked.model_hash

    # q peers polka on a different value at sub-round 1: proof the old value
    # cannot already have been decided, so the lock must move.
    other_hash = "b" * 64
    for identity in ids[1:1 + q]:
        engine.handle_vote(
            Vote(1, 1, PREVOTE, other_hash, ("x",), identity.node_id).signed(identity)
        )
    assert engine.current_round().locked_value == other_hash

    engine.current_round().sub_round = 2
    engine.current_round().step = PREVOTE
    freed = prevote(engine)
    assert freed.model_hash == other_hash
    assert freed.model_hash != locked.model_hash


def test_a_gossiped_polka_breaks_a_locked_unlocked_split():
    """Without transferable polkas the height can wedge for good.

    A lock moves only on an observed polka.  Under loss only some nodes observe
    any given one, so the rest keep prevoting Agg(P) while the locked ones keep
    prevoting their lock.  If neither side reaches q, no new polka can form --
    and nothing else can move a lock.  The split is permanent, and no amount of
    extra sub-rounds helps, because every sub-round reproduces it.

    Tendermint escapes this by having the proposer re-propose with a
    proof-of-lock from the highest round.  Leaderless, the polka itself has to
    be transferable, which is what this test pins down.
    """
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    members = frozenset(i.node_id for i in ids)

    first = make_proposal(engines[1], dataset, fp, protocol, batch=0)
    for engine in engines:
        engine.handle_proposal(first)
    prevotes = [prevote(engine) for engine in engines]
    locked_value = prevotes[0].model_hash

    # Half the network sees the quorum form; the other half loses the prevotes.
    polka_body = None
    for engine in engines[:5]:
        for vote in prevotes:
            if vote.voter == engine.node_id:
                continue
            for effect in engine.handle_vote(vote):
                if (isinstance(effect, Broadcast) and effect.kind == "polka"
                        and polka_body is None):
                    polka_body = effect.body
    assert sum(1 for e in engines if e.current_round().locked_value) == 5

    # A second proposal reaches everyone, so the unlocked half now aggregates
    # to a different value than the locked half is pinned to.
    second = make_proposal(engines[2], dataset, fp, protocol,
                           batch=1 % dataset.n_batches)
    for engine in engines:
        engine.handle_proposal(second)
        rnd = engine.current_round()
        rnd.sub_round = 1
        rnd.step = PREVOTE

    split: dict[str, int] = {}
    for engine in engines:
        cast = prevote(engine)
        split[cast.model_hash] = split.get(cast.model_hash, 0) + 1
    assert len(split) == 2, "expected the two camps that make the split"
    assert all(count < q for count in split.values()), (
        "this test is meaningless unless neither camp can form a polka"
    )

    # The evidence that resolves it.
    assert polka_body is not None, "a node that locked must publish the polka"
    polka = Certificate.from_dict(polka_body)
    assert polka.is_polka and polka.is_valid(q, members)

    for engine in engines[5:]:
        engine.handle_polka(polka)
    agreed = sum(1 for e in engines
                 if e.current_round().locked_value == locked_value)
    assert agreed == len(engines), (
        f"only {agreed}/{len(engines)} adopted the polka; the split survives"
    )


def test_a_polka_is_only_adopted_when_it_really_is_one():
    """The sender is never trusted -- only q verified member signatures are."""
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    engine, victim = engines[0], engines[9]

    proposal = make_proposal(engines[1], dataset, fp, protocol, batch=0)
    engine.handle_proposal(proposal)
    value = engine._prevote_value(engine.current_round())[0]

    good = [Vote(1, 0, PREVOTE, value, (proposal.proposal_id,),
                 identity.node_id).signed(identity) for identity in ids[:q]]

    one_short = Certificate(1, 0, value, tuple(good[:q - 1]), PREVOTE)
    victim.handle_polka(one_short)
    assert victim.current_round().locked_value is None, "locked below quorum"

    precommits = Certificate(1, 0, value, tuple(
        Vote(1, 0, PRECOMMIT, value, (proposal.proposal_id,),
             identity.node_id).signed(identity) for identity in ids[:q]), PREVOTE)
    victim.handle_polka(precommits)
    assert victim.current_round().locked_value is None, \
        "precommits relabelled as a polka must not lock a node"

    stranger = Identity.from_seed(b"\xfe" * 32)
    forged = list(good[:q - 1]) + [
        Vote(1, 0, PREVOTE, value, (proposal.proposal_id,),
             stranger.node_id).signed(stranger)]
    victim.handle_polka(Certificate(1, 0, value, tuple(forged), PREVOTE))
    assert victim.current_round().locked_value is None, "non-member counted"

    victim.handle_polka(Certificate(1, 0, value, tuple(good), PREVOTE))
    assert victim.current_round().locked_value == value, \
        "a genuine polka must be adopted"


def test_a_second_vote_in_one_sub_round_is_equivocation():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    engine = engines[0]
    byz = ids[5]
    a = Vote(1, 0, PREVOTE, "a" * 64, ("p1",), byz.node_id).signed(byz)
    b = Vote(1, 0, PREVOTE, "b" * 64, ("p2",), byz.node_id).signed(byz)
    engine.handle_vote(a)
    effects = engine.handle_vote(b)
    assert any(isinstance(e, Broadcast) and e.kind == "evidence" for e in effects)
    assert engine.accountability.is_blacklisted(byz.node_id)


# ---------------------------------------------------------------------------
# Proposal handling
# ---------------------------------------------------------------------------


def test_exhausting_the_sub_round_budget_asks_for_sync_without_stopping():
    """A node past `max_sub_rounds` must keep its own clock running.

    Sync is only answerable by a node that is *ahead*.  The situation that
    exhausts the budget is the one where no quorum formed, so nobody is ahead
    and nothing can answer -- and because every node hits the budget at roughly
    the same time, a node that merely asked and then went quiet would take the
    whole network down with it.  This was a real deadlock: `_advance` returned
    NeedSync *instead of* OpenWindow, no further timeout ever fired, and a run
    at 5% packet loss committed three heights and then nothing, forever.
    """
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    protocol = replace(protocol, max_sub_rounds=3)
    for engine in engines:
        engine.protocol = protocol

    engine = engines[0]
    engine.handle_proposal(make_proposal(engines[1], dataset, fp, protocol,
                                         batch=0))

    # Nobody else ever votes, so every sub-round dies on its own timer.
    asked = None
    for _ in range(12):
        effects = engine.on_timeout()
        if any(isinstance(e, NeedSync) for e in effects):
            asked = effects
            break
    assert asked is not None, "the budget was never exhausted"
    assert any(isinstance(e, OpenWindow) for e in asked), (
        "asked for sync and stopped: nothing will ever fire again, and no peer "
        "is ahead to answer"
    )

    # And it must go on asking, not ask once.
    later = engine.on_timeout()
    for _ in range(3):
        if any(isinstance(e, NeedSync) for e in later):
            break
        later = engine.on_timeout()
    assert any(isinstance(e, NeedSync) for e in later), \
        "one sync request is not a retry"


def test_a_certificate_stays_valid_after_one_of_its_voters_is_blacklisted():
    """Validity is a property of the signatures, not of the checker's evidence.

    Purging a proven equivocator retroactively changes who counts.  If that
    also unmade certificates, two correct nodes holding different evidence
    would disagree about whether a height was decided -- and the safety
    argument models no such thing.  It buys nothing either: 2q > n + f already
    budgets for every faulty node signing both sides, so an equivocator inside
    a quorum is the case the inequality is sized for.  Evidence governs future
    tallies only.
    """
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    engine = engines[0]

    proposal = make_proposal(engines[1], dataset, fp, protocol, batch=0)
    engine.handle_proposal(proposal)
    cast = prevote(engine)          # so the engine can rebuild the model later
    votes = tuple(
        Vote(1, 0, PRECOMMIT, cast.model_hash, cast.proposal_ids,
             identity.node_id).signed(identity) for identity in ids[:q])
    certificate = Certificate(1, 0, cast.model_hash, votes, PRECOMMIT)

    # Blacklist one of its voters, on a slot unrelated to the one above.
    byz = ids[1]
    a = Vote(1, 5, PREVOTE, "aa", (), byz.node_id).signed(byz)
    b = Vote(1, 5, PREVOTE, "bb", (), byz.node_id).signed(byz)
    engine.handle_vote(a)
    engine.handle_vote(b)
    assert engine.accountability.is_blacklisted(byz.node_id), "setup failed"

    effects = engine.handle_certificate(certificate)
    assert not any(isinstance(e, Rejected) for e in effects), (
        "a certificate was unmade by evidence that arrived after it formed"
    )
    assert any(isinstance(e, Committed) for e in effects), \
        "the height should still commit on a certificate that was validly formed"


def test_a_proposal_from_a_stranger_is_refused():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    outsider = Identity.from_seed(b"\xee" * 32)
    trainer = FixedPointTrainer(fp, protocol.learning_rate)
    after, witness = trainer.step(genesis, dataset, 0)
    proof = engines[0].backend.prove(witness)
    proposal = Proposal(height=1, prev_hash=genesis.digest(), model=after,
                        proof=proof, batch_index=0,
                        proposer=outsider.node_id).signed(outsider)
    effects = engines[0].handle_proposal(proposal)
    assert any(isinstance(e, Rejected) and "member" in e.reason for e in effects)


def test_a_proposal_that_does_not_extend_our_model_is_refused():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    proposal = make_proposal(engines[0], dataset, fp, protocol)
    forged = Proposal(height=1, prev_hash="0" * 64, model=proposal.model,
                      proof=proposal.proof, batch_index=0,
                      proposer=proposal.proposer).signed(ids[0])
    effects = engines[1].handle_proposal(forged)
    assert any(isinstance(e, Rejected) and "descend" in e.reason for e in effects)


def test_future_proposals_are_buffered_within_a_bounded_window():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    engine = engines[0]
    proposal = make_proposal(engines[1], dataset, fp, protocol)

    near = Proposal(height=3, prev_hash=proposal.prev_hash, model=proposal.model,
                    proof=proposal.proof, batch_index=0,
                    proposer=proposal.proposer).signed(ids[1])
    effects = engine.handle_proposal(near)
    assert any(isinstance(e, NeedSync) for e in effects)
    assert 3 in engine.future

    far = Proposal(height=500, prev_hash=proposal.prev_hash, model=proposal.model,
                   proof=proposal.proof, batch_index=0,
                   proposer=proposal.proposer).signed(ids[1])
    effects = engine.handle_proposal(far)
    assert any(isinstance(e, Rejected) and "future window" in e.reason for e in effects)
    assert 500 not in engine.future


def test_nothing_unverified_is_relayed():
    """Every proposal broadcast must be marked as a post-verification relay."""
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    proposal = make_proposal(engines[0], dataset, fp, protocol)
    tampered = Proposal(height=1, prev_hash=proposal.prev_hash,
                        model=Model(w=(1, 2, 3, 4), b=0, scale=fp.scale),
                        proof=proposal.proof, batch_index=0,
                        proposer=proposal.proposer).signed(ids[0])
    effects = engines[1].handle_proposal(tampered)
    assert not broadcasts(effects, "proposal")
    assert any(isinstance(e, Rejected) for e in effects)


def test_stale_and_duplicate_proposals_are_dropped():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    engine = engines[1]
    proposal = make_proposal(engines[0], dataset, fp, protocol)
    assert broadcasts(engine.handle_proposal(proposal), "proposal")
    assert not broadcasts(engine.handle_proposal(proposal), "proposal")
    assert len(engine.current_round().proposals) == 1


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def test_a_lagging_node_catches_up_from_an_audit_entry():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    proposal = make_proposal(engines[0], dataset, fp, protocol)
    for engine in engines[:q]:
        engine.handle_proposal(proposal)
    _, votes = run_sub_round(engines[:q])
    for vote in votes:
        engines[0].handle_vote(vote)
    assert engines[0].height == 1

    entry = engines[0].audit_entry(1)
    lagging = engines[-1]
    assert lagging.height == 0
    effects = lagging.handle_sync_entry(entry)
    assert any(isinstance(e, Committed) for e in effects)
    assert lagging.height == 1
    assert lagging.model.digest() == engines[0].model.digest()


def test_a_sync_entry_with_a_forged_model_is_refused():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    proposal = make_proposal(engines[0], dataset, fp, protocol)
    for engine in engines[:q]:
        engine.handle_proposal(proposal)
    _, votes = run_sub_round(engines[:q])
    for vote in votes:
        engines[0].handle_vote(vote)

    entry = dict(engines[0].audit_entry(1))
    entry["model"] = Model(w=(9, 9, 9, 9), b=9, scale=fp.scale).to_dict()
    effects = engines[-1].handle_sync_entry(entry)
    assert any(isinstance(e, Rejected) for e in effects)
    assert engines[-1].height == 0


def test_a_sync_entry_without_a_quorum_is_refused():
    fp, protocol, ids, registry, dataset, genesis, engines = build()
    q = protocol.effective_quorum()
    proposal = make_proposal(engines[0], dataset, fp, protocol)
    for engine in engines[:q]:
        engine.handle_proposal(proposal)
    _, votes = run_sub_round(engines[:q])
    for vote in votes:
        engines[0].handle_vote(vote)

    entry = dict(engines[0].audit_entry(1))
    certificate = dict(entry["certificate"])
    certificate["votes"] = certificate["votes"][:q - 1]
    entry["certificate"] = certificate
    effects = engines[-1].handle_sync_entry(entry)
    assert any(isinstance(e, Rejected) and "quorum" in e.reason for e in effects)
