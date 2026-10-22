"""A node: training, proving, consensus and the network, wired together.

The node is deliberately thin.  All the protocol logic lives in
:class:`~zkfl.consensus.ConsensusEngine`, which is a pure state machine
returning effects; this class turns effects into I/O and drives the clock.
That split is what lets the safety and accountability properties be tested
directly, without sockets or sleeps, and it is why the adversarial variants
below are small: a Byzantine node is the same node with one method replaced.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from zkfl.accountability import AccountabilityLog
from zkfl.audit import AuditLog
from zkfl.config import FixedPointConfig, ProtocolConfig
from zkfl.consensus import (
    Broadcast,
    Committed,
    ConsensusEngine,
    Effect,
    NeedSync,
    OpenWindow,
    Rejected,
)
from zkfl.crypto import Identity, PeerRegistry
from zkfl.messages import (
    Certificate,
    Envelope,
    Evidence,
    MalformedMessage,
    Proposal,
    SyncRequest,
    Vote,
)
from zkfl.model import Model
from zkfl.votelog import DurableVoteLog
from zkfl.network import MessageGate, Transport
from zkfl.trainer import FixedPointTrainer, LocalDataset
from zkfl.zk import ProofBackend, ProofError


class Node:
    """One participant."""

    def __init__(
        self,
        identity: Identity,
        transport: Transport,
        registry: PeerRegistry,
        backend: ProofBackend,
        genesis: Model,
        fp: FixedPointConfig,
        protocol: ProtocolConfig,
        dataset: Optional[LocalDataset] = None,
        audit_path: Optional[Path | str] = None,
        vote_log_path: Optional[Path | str] = None,
        label: str = "",
    ):
        self.identity = identity
        self.node_id = identity.node_id
        self.label = label or identity.node_id[:8]
        self.transport = transport
        self.fp = fp
        self.protocol = protocol
        self.dataset = dataset

        self.engine = ConsensusEngine(
            identity=identity,
            fp=fp,
            protocol=protocol,
            registry=registry,
            backend=backend,
            genesis=genesis,
            accountability=AccountabilityLog(
                identity.node_id, retention=protocol.evidence_retention
            ),
            # A deployment that allows crash-restart must give a path here, or
            # a restarted node forgets it voted and signs a second vote in the
            # same slot -- self-equivocation on genuine evidence.  Simulations
            # never restart, so they leave it unset and get the in-memory log.
            vote_log=(DurableVoteLog(vote_log_path)
                      if vote_log_path is not None else None),
        )
        self.trainer = FixedPointTrainer(fp, protocol.learning_rate)
        self.audit = AuditLog(audit_path)
        self.gate = MessageGate(peer_rate_limit=protocol.peer_rate_limit)

        self._seq = 0
        self._window_task: Optional[asyncio.Task] = None
        self._running = False
        self.rejections: list[Rejected] = []
        self.commit_times: list[float] = []
        self._round_started: Optional[float] = None
        #: height -> when we last asked.  A set would mean a node that asks
        #: once and is never answered never asks again, which is exactly how a
        #: node that missed one round stays stuck forever.
        self._sync_inflight: dict[int, float] = {}

        transport.set_handler(self._on_envelope)

    # -- properties ---------------------------------------------------------

    @property
    def height(self) -> int:
        return self.engine.height

    @property
    def model(self) -> Model:
        return self.engine.model

    def state(self) -> dict[str, Any]:
        return {**self.engine.state(), "label": self.label,
                "gate": self.gate.stats.as_dict()}

    # -- outbound -----------------------------------------------------------

    def _envelope(self, kind: str, body: dict[str, Any], height: int) -> Envelope:
        self._seq += 1
        return Envelope(kind=kind, body=body, sender=self.node_id,
                        height=height, seq=self._seq)

    async def _apply(self, effects: Sequence[Effect]) -> None:
        for effect in effects:
            if isinstance(effect, Broadcast):
                if effect.relay and not self._should_relay(effect.kind):
                    continue
                height = int(effect.body.get("height", self.engine.next_height))
                await self.transport.broadcast(
                    self._envelope(effect.kind, effect.body, height)
                )
            elif isinstance(effect, Committed):
                await self._on_commit(effect)
            elif isinstance(effect, OpenWindow):
                self._schedule_window(effect.seconds)
            elif isinstance(effect, NeedSync):
                await self._request_sync(effect.height)
            elif isinstance(effect, Rejected):
                self.rejections.append(effect)

    def _should_relay(self, kind: str) -> bool:
        if kind == "vote":
            return self.protocol.relay_votes
        if kind == "proposal":
            return self.protocol.relay_proposals
        # Certificates, polkas and evidence are the messages whose spread we
        # *want*: all are self-verifying, all are terminal (dedup stops the
        # echo), and each is what pulls a stuck or lagging node forward -- the
        # polka specifically because a lock can only move on one, so a node that
        # never sees it can never leave a split.
        return True

    async def _on_commit(self, effect: Committed) -> None:
        entry = self.engine.audit_entry(effect.height)
        if entry is not None:
            try:
                self.audit.append(entry)
            except ValueError:
                pass
        self.gate.release(effect.height)
        if self._round_started is not None:
            self.commit_times.append(time.monotonic() - self._round_started)
            self._round_started = None
        self._sync_inflight.pop(effect.height, None)
        for record in self.engine.accountability.records:
            self.audit.record_evidence(
                record.evidence_id, record.offender, record.kind,
                record.height, record.to_dict()
            )
        self._cancel_window()

    # -- inbound ------------------------------------------------------------

    async def _on_envelope(self, envelope: Envelope) -> None:
        members = self.engine.registry.members()
        if not self.gate.admit(envelope, members):
            return
        if self.engine.accountability.is_blacklisted(envelope.sender):
            return

        try:
            effects = self._dispatch(envelope)
        except MalformedMessage as exc:
            self.gate.stats.malformed += 1
            self.rejections.append(Rejected(envelope.kind, envelope.sender, str(exc)))
            return
        except Exception as exc:  # never let one peer's message kill the node
            self.rejections.append(
                Rejected(envelope.kind, envelope.sender, f"handler error: {exc}")
            )
            return
        await self._apply(effects)

    def _dispatch(self, envelope: Envelope) -> list[Effect]:
        kind = envelope.kind
        if kind == "proposal":
            return self.engine.handle_proposal(Proposal.from_dict(envelope.body))
        if kind == "vote":
            return self.engine.handle_vote(Vote.from_dict(envelope.body))
        if kind == "certificate":
            return self.engine.handle_certificate(Certificate.from_dict(envelope.body))
        if kind == "polka":
            return self.engine.handle_polka(Certificate.from_dict(envelope.body))
        if kind == "evidence":
            return self.engine.handle_evidence(
                Evidence.from_dict(envelope.body), from_peer=envelope.sender
            )
        if kind == "sync_request":
            return self._serve_sync(SyncRequest.from_dict(envelope.body))
        if kind == "sync_entry":
            return self.engine.handle_sync_entry(envelope.body)
        return [Rejected(kind, envelope.sender, "unknown message kind")]

    def _serve_sync(self, request: SyncRequest) -> list[Effect]:
        entry = self.engine.audit_entry(request.height) or self.audit.entry(request.height)
        if entry is None:
            return []
        return [Broadcast("sync_entry", entry)]

    #: how long to wait before re-asking for the same height
    SYNC_RETRY_SECONDS = 1.0

    async def _request_sync(self, height: int) -> None:
        if height <= self.engine.height:
            return
        now = time.monotonic()
        last = self._sync_inflight.get(height)
        if last is not None and now - last < self.SYNC_RETRY_SECONDS:
            return
        self._sync_inflight[height] = now
        body = SyncRequest(height=height, requester=self.node_id).to_dict()
        await self.transport.broadcast(self._envelope("sync_request", body, height))

    # -- the round clock ----------------------------------------------------

    def _schedule_window(self, seconds: float) -> None:
        self._cancel_window()
        self._window_task = asyncio.create_task(self._window(seconds))

    def _cancel_window(self) -> None:
        if self._window_task is not None and not self._window_task.done():
            self._window_task.cancel()
        self._window_task = None

    async def _window(self, seconds: float) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        await self._apply(self.engine.on_timeout())

    async def open_round(self) -> None:
        """Start collecting for the next height."""
        self._round_started = time.monotonic()
        rnd = self.engine.current_round()
        self._schedule_window(self.engine._window_seconds(rnd))

    # -- training -----------------------------------------------------------

    async def train_and_propose(self, batch_index: int = 0) -> Optional[Proposal]:
        """Train one batch, prove it, and broadcast the proposal.

        Proving runs in a thread so the event loop keeps serving messages -- a
        node that stops answering while it proves would look partitioned.
        """
        if self.dataset is None:
            return None
        batch_index %= max(1, self.dataset.n_batches)

        before = self.engine.model
        after, witness = self.trainer.step(before, self.dataset, batch_index)

        loop = asyncio.get_running_loop()
        try:
            proof = await loop.run_in_executor(
                None, self.engine.backend.prove, witness
            )
        except ProofError as exc:
            self.rejections.append(Rejected("proposal", self.node_id, f"prove: {exc}"))
            return None

        proposal = self.engine.create_proposal(after, proof, batch_index)
        await self._apply(self.engine.handle_proposal(proposal))
        await self.transport.broadcast(
            self._envelope("proposal", proposal.to_dict(), proposal.height)
        )
        return proposal

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        await self.transport.start()
        self._running = True

    async def stop(self) -> None:
        self._running = False
        self._cancel_window()
        if self._window_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._window_task
        await self.transport.stop()
        self.audit.close()


# ---------------------------------------------------------------------------
# Adversaries.  Kept in the library, not the tests, because the live
# adversarial experiments instantiate them -- the paper's claim is that these
# attacks are defeated on a running network, not in a unit test.
# ---------------------------------------------------------------------------


class EquivocatingVoter(Node):
    """Votes for two different models in the same sub-round.

    Splits the vote in the old protocol.  Here the two signed votes *are* the
    evidence, so the first honest node to see both broadcasts a proof that
    every other node checks for itself.
    """

    async def _apply(self, effects: Sequence[Effect]) -> None:
        extra: list[Effect] = []
        for effect in effects:
            if isinstance(effect, Broadcast) and effect.kind == "vote" and not effect.relay:
                forged = dict(effect.body)
                forged["model_hash"] = "0" * 64
                vote = Vote.from_dict(forged)
                signed = Vote(
                    height=vote.height, sub_round=vote.sub_round,
                    phase=vote.phase, model_hash=vote.model_hash,
                    proposal_ids=vote.proposal_ids, voter=self.node_id,
                ).signed(self.identity)
                extra.append(Broadcast("vote", signed.to_dict()))
        await super()._apply([*effects, *extra])


class EquivocatingProposer(Node):
    """Sends two different proposals for the same height.

    Vote-set cross-checking could never catch this -- it only looked at voters.
    """

    async def train_and_propose(self, batch_index: int = 0) -> Optional[Proposal]:
        first = await super().train_and_propose(batch_index)
        if first is None or self.dataset is None:
            return first
        other = (batch_index + 1) % max(1, self.dataset.n_batches)
        if other == batch_index:
            return first
        after, witness = self.trainer.step(self.engine.model, self.dataset, other)
        loop = asyncio.get_running_loop()
        try:
            proof = await loop.run_in_executor(None, self.engine.backend.prove, witness)
        except ProofError:
            return first
        second = self.engine.create_proposal(after, proof, other)
        await self.transport.broadcast(
            self._envelope("proposal", second.to_dict(), second.height)
        )
        return first


class Flooder(Node):
    """Repeats one message far past the per-height budget.

    Exercises the rate limiter and, more importantly, checks that flooding
    cannot evict another peer's dedup entries the way the old global
    10,000-entry window allowed.
    """

    def __init__(self, *args, burst: int = 500, **kwargs):
        super().__init__(*args, **kwargs)
        self.burst = burst

    async def train_and_propose(self, batch_index: int = 0) -> Optional[Proposal]:
        proposal = await super().train_and_propose(batch_index)
        if proposal is None:
            return None
        for _ in range(self.burst):
            await self.transport.broadcast(
                self._envelope("proposal", proposal.to_dict(), proposal.height)
            )
        return proposal
