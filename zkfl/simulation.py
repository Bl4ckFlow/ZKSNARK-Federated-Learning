"""Run an n-node network in one process, over a real (in-memory) mesh.

The old simulation ran 10 WebSocket servers on localhost, which made
propagation delay effectively zero -- so "8 of 10 votes" and "consistent
convergence" were measured on a network with no network in it.  This harness
keeps the same node code but puts a bus underneath that can inject latency,
jitter, loss and partitions, and can be scaled to n = 50 without 50 servers.

``scripts/run_simulation.py --transport websocket`` runs the same thing over
real sockets when that is the point.
"""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Type

from zkfl.aggregation import robustness_note
from zkfl.config import FixedPointConfig, ProtocolConfig
from zkfl.crypto import Identity, PeerRegistry
from zkfl.datasets import Partition, Split, partition, partition_report
from zkfl.model import Model
from zkfl.network import InProcessTransport, MessageBus
from zkfl.node import Node
from zkfl.trainer import LocalDataset, accuracy, log_loss
from zkfl.zk import ProofBackend, make_backend


@dataclass
class RoundResult:
    height: int
    committed: int
    heights: list[int]
    model_hashes: set[str]
    seconds: float
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    proposals: int = 0
    messages: int = 0
    bytes_sent: int = 0
    blacklisted: list[str] = field(default_factory=list)

    @property
    def agreed(self) -> bool:
        return len(self.model_hashes) == 1 and self.committed == len(self.heights)


@dataclass
class SimulationResult:
    rounds: list[RoundResult]
    nodes: list[Node]
    bus: Optional[MessageBus]
    notes: list[str]

    def summary(self) -> dict[str, Any]:
        agreed = [r for r in self.rounds if r.agreed]
        return {
            "rounds": len(self.rounds),
            "rounds_agreed": len(agreed),
            "final_height": max((n.height for n in self.nodes), default=0),
            "heights": sorted({n.height for n in self.nodes}),
            "median_round_seconds": (
                round(statistics.median([r.seconds for r in self.rounds]), 3)
                if self.rounds else None
            ),
            "final_accuracy": self.rounds[-1].accuracy if self.rounds else None,
            "final_loss": self.rounds[-1].loss if self.rounds else None,
            "blacklisted": sorted({b for r in self.rounds for b in r.blacklisted}),
        }


class Simulation:
    """Wires up n nodes, their data commitments, and the shared bus."""

    def __init__(
        self,
        *,
        split: Split,
        fp: FixedPointConfig,
        protocol: ProtocolConfig,
        n_nodes: int,
        strict_zk: bool = True,
        byzantine: dict[int, Type[Node]] | None = None,
        latency: float = 0.0,
        jitter: float = 0.0,
        loss: float = 0.0,
        seed: int = 0,
        iid: bool = True,
        shared_backend: bool = True,
    ):
        fp.check()
        protocol.check()
        self.fp = fp
        self.protocol = protocol
        self.split = split
        self.n_nodes = n_nodes
        self.byzantine = byzantine or {}
        self.notes: list[str] = []
        self._started = False
        self._silenced: dict[int, Any] = {}

        self.bus = MessageBus(latency=latency, jitter=jitter, loss=loss, seed=seed)
        self.registry = PeerRegistry()
        self.genesis = Model.zeros(fp)

        parts: list[Partition] = partition(split, n_nodes, fp, iid=iid, seed=seed)
        self.notes.append(partition_report(parts, fp))

        self._backends: list[ProofBackend] = []
        shared: Optional[ProofBackend] = (
            make_backend(fp, strict_zk) if shared_backend else None
        )

        self.identities = [Identity.from_seed(bytes([i % 251, 7, i // 251]) + b"\x00" * 29)
                           for i in range(n_nodes)]
        self.datasets: list[Optional[LocalDataset]] = []
        for part in parts:
            if len(part.x) < fp.batch_size:
                self.datasets.append(None)
            else:
                self.datasets.append(
                    LocalDataset(part.x.tolist(), part.y.tolist(), fp)
                )

        for identity, dataset in zip(self.identities, self.datasets):
            root = dataset.root if dataset else 0
            n_samples = dataset.n_samples if dataset else 1
            payload = PeerRegistry.registration_payload(identity.node_id, root, n_samples)
            self.registry.register(identity.node_id, root, n_samples,
                                   identity.sign("registration", payload))

        self.nodes: list[Node] = []
        for i, (identity, dataset) in enumerate(zip(self.identities, self.datasets)):
            backend = shared or make_backend(fp, strict_zk)
            if shared is None:
                self._backends.append(backend)
            node_class = self.byzantine.get(i, Node)
            transport = InProcessTransport(identity.node_id, self.bus)
            self.nodes.append(node_class(
                identity=identity, transport=transport, registry=self.registry,
                backend=backend, genesis=self.genesis, fp=fp, protocol=protocol,
                dataset=dataset, label=f"n{i}" + (
                    f"/{node_class.__name__}" if node_class is not Node else ""),
            ))
        if shared is not None:
            self._backends.append(shared)

        self.notes.append(
            f"n={n_nodes} f={protocol.f_byzantine} q={protocol.effective_quorum()} "
            f"rule={protocol.aggregation_rule} strict_zk={strict_zk}"
        )

    # -- running ------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        for node in self.nodes:
            await node.start()
        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return
        for node in self.nodes:
            await node.stop()
        for backend in self._backends:
            backend.close()
        self._started = False

    def silence(self, index: int) -> None:
        """Cut a node off from the network, both directions.

        Used to build the "missed a round" scenario without tearing the
        simulation down, which is what the catch-up test needs.
        """
        transport = self.nodes[index].transport

        async def deaf(_envelope):
            return None

        self._silenced[index] = transport.deliver
        transport.deliver = deaf

    def unsilence(self, index: int) -> None:
        original = self._silenced.pop(index, None)
        if original is not None:
            self.nodes[index].transport.deliver = original

    async def sync_until(self, node: Node, height: int, timeout: float = 10.0) -> bool:
        """Drive a lagging node's catch-up until it reaches ``height``."""
        node.SYNC_RETRY_SECONDS = 0.0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if node.height >= height:
                return True
            await node._request_sync(node.height + 1)
            await asyncio.sleep(0.02)
        return node.height >= height

    async def run(
        self,
        rounds: int,
        *,
        proposers_per_round: int = 3,
        timeout: float = 30.0,
        evaluate: bool = True,
        stop_after: bool = True,
    ) -> SimulationResult:
        await self.start()

        results: list[RoundResult] = []
        x_test = [[int(v) for v in row] for row in self._quantized_test()[0]]
        y_test = self._quantized_test()[1]

        try:
            for r in range(rounds):
                results.append(await self._round(
                    r, proposers_per_round, timeout, x_test, y_test, evaluate
                ))
        finally:
            if stop_after:
                await self.shutdown()

        n_proposals = results[-1].proposals if results else 0
        self.notes.append(robustness_note(
            max(n_proposals, 1), self.protocol.f_byzantine,
            self.protocol.trim_permille, self.protocol.aggregation_rule,
        ))
        return SimulationResult(results, self.nodes, self.bus, self.notes)

    async def _round(self, index, proposers_per_round, timeout, x_test, y_test,
                     evaluate) -> RoundResult:
        target = max(node.height for node in self.nodes) + 1
        started = time.monotonic()
        messages_before = self.bus.stats()["messages_out"]
        bytes_before = self.bus.stats()["bytes_out"]

        for node in self.nodes:
            await node.open_round()

        eligible = [n for n in self.nodes if n.dataset is not None]
        chosen = [eligible[(index * proposers_per_round + i) % len(eligible)]
                  for i in range(min(proposers_per_round, len(eligible)))]
        proposals = await asyncio.gather(
            *(node.train_and_propose(batch_index=index) for node in chosen),
            return_exceptions=True,
        )
        n_proposals = sum(1 for p in proposals if p is not None
                          and not isinstance(p, BaseException))

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.02)
            if all(node.height >= target for node in self.nodes):
                break
        await self.bus.drain(timeout=1.0)

        heights = [node.height for node in self.nodes]
        committed = sum(1 for h in heights if h >= target)
        hashes = {node.model.digest() for node in self.nodes if node.height >= target}

        acc = loss = None
        if evaluate and hashes:
            leader = max(self.nodes, key=lambda n: n.height)
            acc = accuracy(leader.model, x_test, y_test)
            loss = log_loss(leader.model, x_test, y_test)

        stats = self.bus.stats()
        blacklisted = sorted({b for node in self.nodes
                              for b in node.engine.accountability.blacklist})
        return RoundResult(
            height=target, committed=committed, heights=heights,
            model_hashes=hashes, seconds=time.monotonic() - started,
            accuracy=acc, loss=loss, proposals=n_proposals,
            messages=stats["messages_out"] - messages_before,
            bytes_sent=stats["bytes_out"] - bytes_before,
            blacklisted=blacklisted,
        )

    def _quantized_test(self):
        from zkfl.fixedpoint import quantize

        xs = [[quantize(v, self.fp.scale) for v in row]
              for row in self.split.x_test]
        ys = [int(v) for v in self.split.y_test]
        return xs, ys

    # -- invariants ---------------------------------------------------------

    def check_invariants(self) -> list[tuple[str, bool, str]]:
        """Post-run checks, reported as (name, ok, detail)."""
        checks: list[tuple[str, bool, str]] = []
        honest = [n for n in self.nodes if type(n) is Node]

        heights = {n.height for n in honest}
        checks.append(("all honest nodes at the same height", len(heights) == 1,
                       f"heights={sorted(heights)}"))

        hashes = {n.model.digest() for n in honest}
        checks.append(("all honest nodes hold the same model", len(hashes) == 1,
                       f"{len(hashes)} distinct model hash(es)"))

        models = {n.model.w for n in honest}
        checks.append(("weight vectors are bitwise identical", len(models) == 1,
                       f"{len(models)} distinct weight vector(s)"))

        chain_ok = all(n.audit.verify_chain()[0] for n in honest)
        checks.append(("audit hash chains verify", chain_ok, ""))

        entries = {len(n.audit) for n in honest}
        checks.append(("audit length equals height",
                       all(len(n.audit) == n.height for n in honest),
                       f"entries={sorted(entries)}"))

        # Rounds below the committed height are kept on purpose, for
        # evidence_retention heights, so evidence arriving late can still be
        # checked.  The invariant is that the pool stays bounded, not empty.
        window = self.protocol.evidence_retention
        overrun = [h for n in honest for h in n.engine.rounds
                   if h < n.height - window]
        checks.append((f"candidate pool bounded by the {window}-height "
                       "retention window", not overrun,
                       f"{len(overrun)} round(s) past the window"))

        leftover = [h for n in honest for h in n.engine.future if h <= n.height]
        checks.append(("no buffered proposals for committed heights",
                       not leftover, f"{len(leftover)} left"))

        expected = {n.node_id for i, n in enumerate(self.nodes)
                    if type(n) is not Node}
        detected = {b for n in honest for b in n.engine.accountability.blacklist}
        if expected:
            checks.append(("every adversary was blacklisted",
                           expected <= detected,
                           f"{len(detected & expected)}/{len(expected)}"))
            converged = all(
                n.engine.accountability.blacklist == detected for n in honest
            )
            checks.append(("blacklists converged", converged, ""))
        else:
            checks.append(("no honest node was blacklisted", not detected,
                           f"{sorted(detected)}"))
        return checks
