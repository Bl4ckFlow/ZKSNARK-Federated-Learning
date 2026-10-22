"""Leaderless consensus: certificate-based commit over an aggregated round.

What changed, and why each change was necessary
-----------------------------------------------

*Aggregate, do not select.*  Nodes used to vote for the first valid update they
saw, and exactly one proposal was committed per round, so the global model was
a chain of single-node steps ordered by network latency.  Here a round's
decision is ``Aggregate(P)`` over the whole verified proposal set.

*Decide deterministically.*  Because the aggregate is a pure function of the
proposal set (:mod:`zkfl.aggregation`), two honest nodes holding the same set
vote identically.  A conflicting vote is therefore misbehaviour, not a timing
artefact -- which is what makes equivocation a meaningful accusation.

*Commit on a certificate, not on a local timer.*  A certificate is ``q`` signed
votes for the same (height, sub-round, model hash).  Any node holding one
commits immediately; its own window is irrelevant.  The old rule -- commit if
the leading candidate had more than half of *the votes this node happened to
see* -- contributed nothing to agreement, because the denominator differed
across nodes.

*Fault model, stated.*  ``n`` nodes, at most ``f`` Byzantine, permissioned
Ed25519 identities, partial synchrony (messages eventually arrive; windows grow
until they do).  Sybil resistance is out of scope.  Safety needs
``q > (n+f)/2`` so any two quorums share an honest node; liveness needs
``n - f >= q``; together, ``n >= 3f+1``.  :meth:`ProtocolConfig.check` refuses
anything weaker rather than letting a run claim a guarantee it lacks.

*Two phases per sub-round.*  A prevote says "my proposal set aggregates to
this".  A quorum of matching prevotes -- a *polka* -- is proof that a quorum
could see that value, and only then does a node lock on it and precommit.  A
quorum of matching precommits is a certificate, and commits the height.

The two phases are not ceremony.  A single-phase protocol has to lock a node on
its own vote, and honest nodes close their collect windows with different
proposal sets on any real network, so they lock on different values in
sub-round 0 and no quorum ever forms again.  Locking on a polka instead means
split prevotes simply produce no lock: everyone precommits nil, moves to the
next sub-round with a longer window, and by then the proposal sets have
converged.  This is Tendermint's rule, and the safety argument is Tendermint's;
see docs/PROTOCOL.md for the proposition and proof sketch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from zkfl.accountability import AccountabilityLog
from zkfl.aggregation import AggregationError, aggregate
from zkfl.config import FixedPointConfig, ProtocolConfig
from zkfl.crypto import Identity, PeerRegistry
from zkfl.fixedpoint import model_in_range, quantize, step_within_bound
from zkfl.messages import (
    Certificate,
    Evidence,
    MalformedMessage,
    NIL,
    PRECOMMIT,
    PREVOTE,
    Proposal,
    Vote,
)
from zkfl.model import Model
from zkfl.votelog import InMemoryVoteLog, VoteLog
from zkfl.zk import ProofBackend, PublicInputs

# ---------------------------------------------------------------------------
# Effects.  The engine is a pure state machine: it never touches the network,
# it returns what should happen.  That is what makes the safety properties
# testable without sockets or sleeps.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Effect:
    pass


@dataclass(frozen=True)
class Broadcast(Effect):
    kind: str
    body: dict[str, Any]
    #: True when this is a re-gossip of something we received rather than
    #: something we authored.  On a full mesh the originator already reached
    #: every peer, so relaying turns O(n^2) messages into O(n^3) -- the node
    #: suppresses relays unless the topology needs them.
    relay: bool = False


@dataclass(frozen=True)
class Committed(Effect):
    height: int
    model: Model
    certificate: Certificate
    proposal_ids: tuple[str, ...]


@dataclass(frozen=True)
class NeedSync(Effect):
    height: int
    reason: str


@dataclass(frozen=True)
class OpenWindow(Effect):
    height: int
    sub_round: int
    seconds: float
    #: which phase the timer is for: "prevote" or "precommit"
    step: str = PREVOTE


@dataclass(frozen=True)
class Rejected(Effect):
    """A message we declined, with the reason -- surfaced in tests and metrics."""

    kind: str
    sender: str
    reason: str


# ---------------------------------------------------------------------------


@dataclass
class RoundState:
    height: int
    sub_round: int = 0
    #: which phase this node is in for the current sub-round
    step: str = PREVOTE
    proposals: dict[str, Proposal] = field(default_factory=dict)
    #: phase -> (sub_round, model_hash) -> voter -> vote
    votes: dict[str, dict[tuple[int, str], dict[str, Vote]]] = field(
        default_factory=lambda: {PREVOTE: {}, PRECOMMIT: {}}
    )
    #: models we derived, keyed by hash, so a certificate can be honoured
    derived: dict[str, tuple[Model, tuple[str, ...]]] = field(default_factory=dict)
    #: the value this node is locked on, and the sub-round whose polka set it
    locked_value: Optional[str] = None
    locked_round: int = -1
    locked_ids: tuple[str, ...] = ()
    #: sub-rounds in which this node has already prevoted / precommitted
    prevoted: set[int] = field(default_factory=set)
    precommitted: set[int] = field(default_factory=set)
    pending_certificate: Optional[Certificate] = None
    #: highest sub-round whose polka we have already gossiped (-1 = none)
    published_polka: int = -1
    opened_at: float = field(default_factory=time.monotonic)

    def bucket(self, phase: str, sub_round: int, model_hash: str) -> dict[str, Vote]:
        return self.votes[phase].setdefault((sub_round, model_hash), {})

    def tally(self, phase: str, sub_round: int, model_hash: str) -> int:
        return len(self.votes[phase].get((sub_round, model_hash), {}))


class ConsensusEngine:
    """One node's view of the protocol.  Transport-agnostic by construction."""

    def __init__(
        self,
        identity: Identity,
        fp: FixedPointConfig,
        protocol: ProtocolConfig,
        registry: PeerRegistry,
        backend: ProofBackend,
        genesis: Model,
        accountability: AccountabilityLog | None = None,
        vote_log: VoteLog | None = None,
    ):
        fp.check()
        protocol.check()
        if not protocol.strict_zk and backend.strict:
            pass  # a strict backend under a lenient config is fine
        if protocol.strict_zk and not backend.strict:
            raise RuntimeError(
                "protocol requires strict ZK but a non-proving backend was supplied; "
                "set strict_zk=False explicitly if this is a test"
            )

        self.identity = identity
        self.node_id = identity.node_id
        self.fp = fp
        self.protocol = protocol
        self.registry = registry
        self.backend = backend
        self.accountability = accountability or AccountabilityLog(
            identity.node_id, retention=protocol.evidence_retention
        )
        #: One vote per slot, across restarts.  In-memory by default because a
        #: simulation never restarts; a deployment passes a DurableVoteLog and
        #: `vote_log.durable` says which it got.
        # `or` would be wrong here: both logs define __len__, so an empty
        # DurableVoteLog is falsy and would be silently swapped for an
        # in-memory one -- a node that believed it was crash-safe and was not.
        self.vote_log: VoteLog = (
            InMemoryVoteLog() if vote_log is None else vote_log)

        self.eta = quantize(protocol.learning_rate, fp.scale)
        self.height = 0
        self.model = genesis
        self.rounds: dict[int, RoundState] = {}
        self.audit: list[dict[str, Any]] = []
        #: proposals for heights above h+1, bounded by protocol.future_window
        self.future: dict[int, dict[str, Proposal]] = {}
        self.metrics: dict[str, int] = {
            "proposals_accepted": 0,
            "proposals_rejected": 0,
            "votes_accepted": 0,
            "votes_rejected": 0,
            "certificates_seen": 0,
            "polkas_seen": 0,
            "sub_rounds": 0,
        }

    # -- helpers ------------------------------------------------------------

    @property
    def quorum(self) -> int:
        return self.protocol.effective_quorum()

    @property
    def next_height(self) -> int:
        return self.height + 1

    def _round(self, height: int) -> RoundState:
        return self.rounds.setdefault(height, RoundState(height=height))

    def current_round(self) -> RoundState:
        return self._round(self.next_height)

    def _members(self) -> frozenset[str]:
        return self.registry.members()

    # -- proposing ----------------------------------------------------------

    def create_proposal(
        self, model_after: Model, proof: Any, batch_index: int
    ) -> Proposal:
        """Build this node's signed proposal for the next height."""
        return Proposal(
            height=self.next_height,
            prev_hash=self.model.digest(),
            model=model_after,
            proof=proof,
            batch_index=batch_index,
            proposer=self.node_id,
        ).signed(self.identity)

    # -- inbound: proposals -------------------------------------------------

    def handle_proposal(self, proposal: Proposal) -> list[Effect]:
        effects: list[Effect] = []

        if proposal.proposer not in self._members():
            self.metrics["proposals_rejected"] += 1
            return [Rejected("proposal", proposal.proposer, "not a registered member")]
        if self.accountability.is_blacklisted(proposal.proposer):
            self.metrics["proposals_rejected"] += 1
            return [Rejected("proposal", proposal.proposer, "blacklisted")]
        if not proposal.verify_signature():
            self.metrics["proposals_rejected"] += 1
            return [Rejected("proposal", proposal.proposer, "bad signature")]

        # Equivocation is checked on the *signed* message, before anything
        # expensive, so a proposer flooding conflicting proposals is caught
        # rather than paid for in proof verifications.
        evidence = self.accountability.observe_proposal(proposal)
        if evidence is not None and self.accountability.accept(evidence):
            effects.extend(self._purge_offender(proposal.proposer))
            effects.append(Broadcast("evidence", evidence.to_dict()))
            effects.append(Rejected("proposal", proposal.proposer, "equivocating proposer"))
            self.metrics["proposals_rejected"] += 1
            return effects

        if proposal.height <= self.height:
            self.metrics["proposals_rejected"] += 1
            return [*effects, Rejected("proposal", proposal.proposer, "stale height")]

        if proposal.height > self.next_height:
            # Cannot be verified yet: the public inputs for that height are not
            # known.  Buffer within a bounded window and never rebroadcast --
            # an unverified store that grows without limit is a memory DoS.
            if proposal.height > self.height + self.protocol.future_window:
                self.metrics["proposals_rejected"] += 1
                return [*effects,
                        Rejected("proposal", proposal.proposer, "beyond future window")]
            bucket = self.future.setdefault(proposal.height, {})
            if len(bucket) >= self.protocol.peer_rate_limit:
                self.metrics["proposals_rejected"] += 1
                return [*effects,
                        Rejected("proposal", proposal.proposer, "future buffer full")]
            bucket[proposal.proposal_id] = proposal
            return [*effects, NeedSync(self.next_height, "gap detected")]

        ok, reason = self._verify_for_current_height(proposal)
        if not ok:
            self.metrics["proposals_rejected"] += 1
            return [*effects, Rejected("proposal", proposal.proposer, reason)]

        rnd = self.current_round()
        if proposal.proposal_id in rnd.proposals:
            return effects
        rnd.proposals[proposal.proposal_id] = proposal
        self.metrics["proposals_accepted"] += 1
        # Only now is it safe to relay: nothing unverified leaves this node.
        effects.append(Broadcast("proposal", proposal.to_dict(), relay=True))
        return effects

    def _verify_for_current_height(self, proposal: Proposal) -> tuple[bool, str]:
        if proposal.prev_hash != self.model.digest():
            return False, "does not descend from our committed model"
        if len(proposal.model.w) != self.fp.n_features:
            return False, "wrong feature count"
        if proposal.model.scale != self.fp.scale:
            return False, "wrong fixed-point scale"
        if not 0 <= proposal.batch_index < self.fp.max_batches:
            return False, "batch index out of range"
        if not model_in_range(proposal.model.w, proposal.model.b, self.fp):
            return False, "model outside the circuit's weight range"
        if not step_within_bound(
            self.model.w, self.model.b,
            proposal.model.w, proposal.model.b, self.fp.max_delta
        ):
            return False, "step exceeds the clipping bound"

        root = self.registry.root_of(proposal.proposer)
        if root is None:
            return False, "proposer has no registered data root"

        # The statement is built from *our* state: our committed model, the
        # protocol's learning rate, and the root this proposer published when it
        # joined.  Nothing but (w_new, b_new) comes from the message.
        expected = PublicInputs.from_models(
            before=self.model,
            after=proposal.model,
            eta=self.eta,
            data_root=root,
        )
        try:
            if not self.backend.verify(proposal.proof, expected):
                return False, "zk proof rejected"
        except Exception as exc:  # a broken prover must not take the node down
            return False, f"proof verification error: {exc}"
        return True, "ok"

    # -- the round clock: prevote, then precommit --------------------------

    def on_timeout(self) -> list[Effect]:
        """The collect window for the current phase expired."""
        rnd = self._round(self.next_height)
        if rnd.step == PREVOTE:
            return self._do_prevote(rnd)
        if rnd.sub_round in rnd.precommitted:
            # We precommitted and no certificate formed before the timer.  Go
            # again; the lock, if we hold one, survives -- so this costs a
            # round trip and never costs safety.
            return self._advance(rnd, rnd.sub_round + 1, "precommit timeout")
        return self._precommit(rnd, rnd.sub_round, NIL, ())

    def _do_prevote(self, rnd: RoundState) -> list[Effect]:
        """Say what our proposal set aggregates to -- or repeat our lock."""
        if rnd.sub_round in rnd.prevoted:
            return self._open(rnd, PRECOMMIT)

        value, ids, model = self._prevote_value(rnd)
        if value is None:
            # Nothing to say this sub-round.  Move on rather than spin; past
            # max_sub_rounds _advance also asks a peer for the height, but it
            # keeps opening windows either way -- see the note there on why
            # asking *instead* of opening was a deadlock.
            return self._advance(rnd, rnd.sub_round + 1, "no proposals to aggregate")

        if model is not None:
            rnd.derived[value] = (model, ids)
        rnd.prevoted.add(rnd.sub_round)

        vote = self._sign_vote(rnd.height, rnd.sub_round, PREVOTE, value, ids)
        effects = list(self.handle_vote(vote))
        effects.append(Broadcast("vote", vote.to_dict()))
        effects.extend(self._open(rnd, PRECOMMIT))
        return effects

    def _sign_vote(self, height: int, sub_round: int, phase: str,
                   value: str, ids: tuple[str, ...]) -> Vote:
        """Our vote for this slot, made durable before anyone can see it.

        The agreement proof assumes a correct node signs one vote per slot, and
        a crash-restart would otherwise break that assumption without the node
        being Byzantine: the restarted node, having lost the fact that it voted,
        would sign a second, different vote in the same slot and be blacklisted
        on genuine evidence.  So a slot already in the log re-sends exactly what
        it recorded, and a new one is written and fsynced *before* it is
        returned for broadcast.
        """
        slot = (height, sub_round, phase)
        prior = self.vote_log.recall(slot)
        if prior is not None:
            return prior
        vote = Vote(height=height, sub_round=sub_round, phase=phase,
                    model_hash=value, proposal_ids=ids,
                    voter=self.node_id).signed(self.identity)
        self.vote_log.record(vote)
        return vote

    def _prevote_value(
        self, rnd: RoundState
    ) -> tuple[Optional[str], tuple[str, ...], Optional[Model]]:
        """The locked value if we hold one, else the aggregate of what we have."""
        if rnd.locked_value is not None:
            return (rnd.locked_value, rnd.locked_ids,
                    rnd.derived.get(rnd.locked_value, (None, ()))[0])
        if not rnd.proposals:
            return None, (), None
        try:
            result = aggregate(
                self.model,
                list(rnd.proposals.values()),
                self.fp,
                rule=self.protocol.aggregation_rule,
                permille=self.protocol.trim_permille,
                weight_of=self.registry.weight_of,
            )
        except AggregationError:
            return None, (), None
        return result.model.digest(), result.proposal_ids, result.model

    def _on_polka(self, rnd: RoundState, vote: Vote) -> list[Effect]:
        """A quorum prevoted one value: lock on it, then precommit it.

        Only a polka licenses a lock.  It is proof that a quorum was willing to
        commit the value, so no *earlier* sub-round can have decided anything
        else -- which is precisely what the safety argument needs -- while a
        node's own prevote proves nothing about anyone else and locking on it
        would deadlock the height the moment two windows closed differently.
        """
        if vote.sub_round < rnd.locked_round:
            return []                       # stale news; our lock is newer
        rnd.locked_value = vote.model_hash
        rnd.locked_round = vote.sub_round
        rnd.locked_ids = vote.proposal_ids
        if vote.sub_round > rnd.sub_round:
            # A quorum is ahead of us.  Catch up instead of timing out alone;
            # _precommit sets the step from here.
            rnd.sub_round = vote.sub_round
        return self._precommit(rnd, vote.sub_round, vote.model_hash, vote.proposal_ids)

    def _precommit(self, rnd: RoundState, sub_round: int,
                   value: str, ids: tuple[str, ...]) -> list[Effect]:
        """Precommit once per sub-round: the locked value, or nil."""
        if sub_round in rnd.precommitted:
            return []
        rnd.precommitted.add(sub_round)
        if sub_round >= rnd.sub_round:
            rnd.sub_round = sub_round
            rnd.step = PRECOMMIT

        vote = self._sign_vote(rnd.height, sub_round, PRECOMMIT, value, ids)
        effects = list(self.handle_vote(vote))
        effects.append(Broadcast("vote", vote.to_dict()))
        if sub_round < rnd.sub_round:
            # A late polka for a sub-round we have already left.  Broadcasting
            # the precommit is still useful -- it can complete a certificate at
            # a node that never moved on -- but the clock belongs to the
            # sub-round we are actually in, so leave it alone.
            return effects
        if value == NIL:
            effects.extend(self._advance(rnd, sub_round + 1, "no polka"))
        else:
            # Restart the clock: the precommit phase begins now, not whenever
            # the prevote timer happened to have been set.
            effects.extend(self._open(rnd, PRECOMMIT))
        return effects

    def _advance(self, rnd: RoundState, to: int, reason: str) -> list[Effect]:
        """Start sub-round `to` with a longer window."""
        if to <= rnd.sub_round:
            return []
        self.metrics["sub_rounds"] += to - rnd.sub_round
        rnd.sub_round = to
        rnd.step = PREVOTE
        window = OpenWindow(rnd.height, to, self._window_seconds(rnd), PREVOTE)
        if to >= self.protocol.max_sub_rounds:
            # Past the budget we also ask a peer for the height -- but we do not
            # stop.  Sync can only be answered by a node that is *ahead*, and
            # the case that exhausts the budget is precisely the one where no
            # quorum formed, so nobody is ahead and nothing can answer.  A node
            # that only asked would idle forever, and since every node reaches
            # this point together, so would the network.  Keeping the clock
            # running costs one window per attempt and is what lets a height
            # recover once the loss burst passes.
            return [NeedSync(rnd.height, f"{reason} after {to} sub-rounds"),
                    window]
        return [window]

    def _open(self, rnd: RoundState, step: str) -> list[Effect]:
        rnd.step = step
        return [OpenWindow(rnd.height, rnd.sub_round,
                           self._window_seconds(rnd), step)]

    def _window_seconds(self, rnd: RoundState) -> float:
        """Grow the window with each sub-round.

        Under partial synchrony a fixed window can starve forever; doubling
        guarantees it eventually exceeds the real message delay, which is
        exactly what the liveness argument needs.
        """
        base = self.protocol.collect_window
        if not self.protocol.adaptive_window:
            return base
        return base * (2 ** min(rnd.sub_round, 6))

    def handle_vote(self, vote: Vote, relay: bool = True) -> list[Effect]:
        effects: list[Effect] = []

        if vote.phase not in (PREVOTE, PRECOMMIT):
            return [Rejected("vote", vote.voter, "unknown phase")]
        if vote.voter not in self._members():
            self.metrics["votes_rejected"] += 1
            return [Rejected("vote", vote.voter, "not a registered member")]
        if self.accountability.is_blacklisted(vote.voter):
            self.metrics["votes_rejected"] += 1
            return [Rejected("vote", vote.voter, "blacklisted")]
        if not vote.verify_signature():
            self.metrics["votes_rejected"] += 1
            return [Rejected("vote", vote.voter, "bad signature")]
        if vote.height <= self.height:
            # Still worth observing: late evidence about a committed height is
            # exactly what the retention window exists for.
            evidence = self.accountability.observe_vote(vote)
            if evidence is not None and self.accountability.accept(evidence):
                return [Broadcast("evidence", evidence.to_dict())]
            return [Rejected("vote", vote.voter, "stale height")]
        if vote.height > self.next_height:
            return [NeedSync(self.next_height, "vote from a future height")]

        evidence = self.accountability.observe_vote(vote)
        if evidence is not None:
            if self.accountability.accept(evidence):
                effects.append(Broadcast("evidence", evidence.to_dict()))
                effects.extend(self._purge_offender(vote.voter))
            self.metrics["votes_rejected"] += 1
            effects.append(Rejected("vote", vote.voter, "equivocating voter"))
            return effects

        rnd = self._round(vote.height)
        bucket = rnd.bucket(vote.phase, vote.sub_round, vote.model_hash)
        if vote.voter in bucket:
            return effects
        bucket[vote.voter] = vote
        self.metrics["votes_accepted"] += 1
        if relay and vote.voter != self.node_id:
            effects.append(Broadcast("vote", vote.to_dict(), relay=True))

        if len(bucket) < self.quorum:
            return effects

        if vote.phase == PREVOTE and vote.model_hash != NIL:
            polka = Certificate(
                height=vote.height, sub_round=vote.sub_round,
                model_hash=vote.model_hash, votes=tuple(bucket.values()),
                phase=PREVOTE,
            )
            effects.extend(self._on_polka(rnd, vote))
            # Publish it.  Only a node that *observed* q prevotes can lock, and
            # under loss that is not everyone; the nodes that missed them keep
            # prevoting Agg(P) while the locked ones keep prevoting their lock,
            # and if neither side reaches q the height never finishes.  The
            # polka is the evidence that breaks the tie, so it has to travel.
            if rnd.published_polka < vote.sub_round or rnd.published_polka < 0:
                rnd.published_polka = vote.sub_round
                effects.append(Broadcast("polka", polka.to_dict()))
        elif vote.phase == PRECOMMIT and vote.model_hash != NIL:
            certificate = Certificate(
                height=vote.height, sub_round=vote.sub_round,
                model_hash=vote.model_hash, votes=tuple(bucket.values()),
            )
            effects.extend(self.handle_certificate(certificate))
        elif vote.phase == PRECOMMIT:
            # A quorum precommitted nil.  Since any two quorums share an honest
            # node, no value can be certified in this sub-round; go again at
            # once rather than waiting out a timer we know will expire.
            effects.extend(self._advance(rnd, vote.sub_round + 1,
                                         "quorum of nil precommits"))
        return effects

    def _purge_offender(self, offender: str) -> list[Effect]:
        """Erase a proven equivocator from every open round.

        Both halves matter, and dropping either one is a liveness bug.

        *Votes*, because a tally that still counts an excluded node is not the
        tally the rest of the network is computing.

        *Proposals*, because an equivocating proposer's first message is
        accepted before its second one exposes it -- and honest nodes see the
        two in different orders, so they accept *different* firsts.  If the
        node that caught the equivocation locally keeps what it accepted while
        the nodes that learned by gossip drop it, the proposal sets never
        converge and the height never decides.  Observed: prevotes split 5/4
        against q=7 at every sub-round, forever.

        The purge is a deterministic function of the evidence, and the evidence
        is self-verifying and gossiped, so every honest node performs exactly
        the same one.  Locks are deliberately *not* released here: a lock
        implies a polka, so the nodes that formed it purge too and the next
        polka moves the lock by the ordinary rule.  Releasing it by hand would
        be the one thing Proposition 1 forbids.
        """
        for rnd in self.rounds.values():
            for phase in rnd.votes.values():
                for bucket in phase.values():
                    bucket.pop(offender, None)
            for pid, proposal in list(rnd.proposals.items()):
                if proposal.proposer == offender:
                    rnd.proposals.pop(pid, None)
        for bucket in self.future.values():
            for pid, proposal in list(bucket.items()):
                if proposal.proposer == offender:
                    bucket.pop(pid, None)
        return []

    # -- inbound: certificates ---------------------------------------------

    def handle_polka(self, polka: Certificate) -> list[Effect]:
        """Adopt a quorum's prevote that we did not see form ourselves.

        This is the leaderless stand-in for Tendermint's proposer re-proposing
        with a proof-of-lock.  Without it, a node that missed the prevotes keeps
        offering ``Agg(P)`` while locked nodes keep offering their lock, and
        when neither side reaches ``q`` the height is stuck for good: a lock
        moves only on a polka, and no new polka can form out of a permanent
        split.  Verified here in full -- ``q`` distinct member signatures on the
        same prevote -- so the sender is never trusted, only the evidence.
        """
        members = self._members()
        if not polka.is_polka:
            return [Rejected("polka", "?", "not a prevote certificate")]
        if not polka.is_valid(self.quorum, members):
            return [Rejected("polka", "?", "invalid polka")]
        # As with commit certificates, a polka is valid once formed: see
        # handle_certificate for why blacklists must not enter this decision.

        self.metrics["polkas_seen"] += 1

        if polka.height <= self.height:
            return []                       # the height is already decided
        if polka.height > self.next_height:
            return [NeedSync(self.next_height, "polka from a future height")]

        rnd = self._round(polka.height)
        if polka.sub_round < rnd.locked_round:
            return []                       # our lock is already newer

        # Feed the votes through the ordinary path so equivocation detection and
        # the vote tallies see them, then apply the lock exactly as if we had
        # watched the quorum form.
        effects: list[Effect] = []
        for vote in polka.votes:
            if vote.voter == self.node_id:
                continue
            effects.extend(self.handle_vote(vote, relay=False))

        if rnd.locked_round >= polka.sub_round and rnd.locked_value == polka.model_hash:
            return effects                  # handle_vote already locked us

        witness = polka.votes[0]
        effects.extend(self._on_polka(rnd, witness))
        return effects

    def handle_certificate(self, certificate: Certificate) -> list[Effect]:
        members = self._members()
        if not certificate.is_valid(self.quorum, members):
            return [Rejected("certificate", "?", "invalid certificate")]
        # A certificate is valid *once formed*, and validity is a property of
        # the signatures alone -- never of who the checker happens to have
        # blacklisted.  We used to reject certificates containing a proven
        # equivocator, which made validity view-dependent: two correct nodes
        # holding different evidence disagreed about whether the same
        # certificate decided the height, and the safety argument models no such
        # thing.  It also bought nothing.  Safety rests on 2q > n + f, and that
        # inequality already budgets for all f faulty nodes signing both sides;
        # an equivocator's signature inside a quorum is exactly the case it is
        # sized for.  Evidence therefore governs *future* tallies only: a
        # blacklisted node's new votes are refused (see handle_vote) and its
        # pending votes and proposals are purged from open rounds, but nothing
        # already certified is retroactively unmade.

        self.metrics["certificates_seen"] += 1

        if certificate.height <= self.height:
            return []
        if certificate.height > self.next_height:
            return [NeedSync(self.next_height, "certificate from a future height")]

        rnd = self._round(certificate.height)
        resolved = self._resolve(rnd, certificate)
        if resolved is None:
            rnd.pending_certificate = certificate
            return [
                Broadcast("certificate", certificate.to_dict()),
                NeedSync(certificate.height, "certificate for a model we cannot rebuild"),
            ]

        model, ids = resolved
        return self._commit(certificate, model, ids)

    def _resolve(
        self, rnd: RoundState, certificate: Certificate
    ) -> Optional[tuple[Model, tuple[str, ...]]]:
        """Rebuild the model a certificate names, from proposals we hold.

        Aggregation is a public deterministic function of public inputs, so
        there is nothing to trust here: we recompute and check the hash.
        """
        cached = rnd.derived.get(certificate.model_hash)
        if cached is not None:
            return cached

        for vote in certificate.votes:
            ids = vote.proposal_ids
            if not ids or any(i not in rnd.proposals for i in ids):
                continue
            try:
                result = aggregate(
                    self.model,
                    [rnd.proposals[i] for i in ids],
                    self.fp,
                    rule=self.protocol.aggregation_rule,
                    permille=self.protocol.trim_permille,
                    weight_of=self.registry.weight_of,
                )
            except AggregationError:
                continue
            if result.model.digest() == certificate.model_hash:
                rnd.derived[certificate.model_hash] = (result.model, result.proposal_ids)
                return result.model, result.proposal_ids
        return None

    def _commit(
        self, certificate: Certificate, model: Model, ids: tuple[str, ...]
    ) -> list[Effect]:
        height = certificate.height
        rnd = self._round(height)
        prev_hash = self.model.digest()

        entry = {
            "height": height,
            "prev_model_hash": prev_hash,
            "model_hash": model.digest(),
            "model": model.to_dict(),
            "proposal_ids": list(ids),
            "proposals": [rnd.proposals[i].to_dict() for i in ids if i in rnd.proposals],
            # The roots the proofs were checked against, so an outside auditor
            # can rebuild the exact public inputs without our registry.
            "roots": {
                rnd.proposals[i].proposer: str(self.registry.root_of(rnd.proposals[i].proposer))
                for i in ids if i in rnd.proposals
            },
            "eta": self.eta,
            "certificate": certificate.to_dict(),
            "aggregation": {
                "rule": self.protocol.aggregation_rule,
                "trim_permille": self.protocol.trim_permille,
            },
            # Milliseconds as an integer: the entry is hash-chained and
            # gossiped, and a float would encode differently across platforms.
            "committed_at_ms": int(time.time() * 1000),
        }
        self.audit.append(entry)

        self.model = model
        self.height = height
        self.accountability.prune(height)

        # Keep the round for the retention window so a late certificate or late
        # evidence about it can still be evaluated.
        for h in list(self.rounds):
            if h < height - self.protocol.evidence_retention:
                self.rounds.pop(h, None)
        for h in list(self.future):
            if h <= height:
                self.future.pop(h, None)

        effects: list[Effect] = [
            Committed(height, model, certificate, ids),
            Broadcast("certificate", certificate.to_dict()),
        ]
        effects.extend(self._replay_future())
        return effects

    def _replay_future(self) -> list[Effect]:
        """Re-examine buffered proposals now that their height is verifiable."""
        buffered = self.future.pop(self.next_height, {})
        effects: list[Effect] = []
        for proposal in buffered.values():
            effects.extend(self.handle_proposal(proposal))
        return effects

    # -- inbound: evidence --------------------------------------------------

    def handle_evidence(self, evidence: Evidence, from_peer: str | None = None) -> list[Effect]:
        if not self.accountability.accept(evidence, from_peer=from_peer):
            return []
        effects: list[Effect] = list(self._purge_offender(evidence.offender))
        effects.append(Broadcast("evidence", evidence.to_dict()))
        return effects

    # -- sync ---------------------------------------------------------------

    def audit_entry(self, height: int) -> Optional[dict[str, Any]]:
        for entry in self.audit:
            if entry["height"] == height:
                return entry
        return None

    def handle_sync_entry(self, entry: dict[str, Any]) -> list[Effect]:
        """Apply an audit entry fetched from a peer, verifying it end to end.

        Nothing is taken on faith: the certificate's signatures, the parent
        link, every proposal's proof against *our* model, and the aggregate
        recomputed from those proposals.
        """
        try:
            height = int(entry["height"])
            certificate = Certificate.from_dict(entry["certificate"])
            proposals = [Proposal.from_dict(p) for p in entry.get("proposals", [])]
            model = Model.from_dict(entry["model"])
        except (KeyError, TypeError, ValueError, MalformedMessage) as exc:
            return [Rejected("sync", "?", f"malformed entry: {exc}")]

        if height != self.next_height:
            return [Rejected("sync", "?", f"entry for height {height}, need {self.next_height}")]
        if entry.get("prev_model_hash") != self.model.digest():
            return [Rejected("sync", "?", "entry does not extend our chain")]
        if not certificate.is_valid(self.quorum, self._members()):
            return [Rejected("sync", "?", "certificate does not carry a quorum")]
        if certificate.height != height:
            return [Rejected("sync", "?", "certificate height mismatch")]

        rnd = self._round(height)
        for proposal in proposals:
            ok, reason = self._verify_for_current_height(proposal)
            if not ok or not proposal.verify_signature():
                return [Rejected("sync", proposal.proposer, f"bad proposal: {reason}")]
            rnd.proposals[proposal.proposal_id] = proposal

        resolved = self._resolve(rnd, certificate)
        if resolved is None:
            return [Rejected("sync", "?", "cannot rebuild the certified model")]
        rebuilt, ids = resolved
        if rebuilt.digest() != model.digest():
            return [Rejected("sync", "?", "entry model disagrees with the aggregate")]
        return self._commit(certificate, rebuilt, ids)

    # -- introspection ------------------------------------------------------

    def state(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id[:8],
            "height": self.height,
            "model_hash": self.model.digest()[:12],
            "quorum": self.quorum,
            "blacklisted": len(self.accountability.blacklist),
            "audit_entries": len(self.audit),
            **self.metrics,
        }
