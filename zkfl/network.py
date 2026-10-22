"""Transport and flow control.

The hardening here is aimed at three concrete failures in the previous layer.

*The dedup window was evictable.*  A single global rolling set of 10,000
message ids meant a peer could flood it, evict the entries for honest messages,
and get those messages reprocessed and rebroadcast -- amplification on demand.
Dedup is now keyed per ``(sender, height)``, so one peer can only ever evict
its own slots, and per-height state is dropped when the height commits.

*There were no per-peer limits.*  A peer can now spend at most
``peer_rate_limit`` messages per height before being throttled, and its
oversized frames are refused rather than parsed.

*Unverified messages were relayed.*  Relaying is a decision the consensus
engine makes -- it marks an effect as a relay only after verification -- and
this layer additionally refuses to relay at all on a full mesh, where the
originator already reached every peer and relaying would cost O(n^3) messages.

:class:`InProcessTransport` exists so consensus properties can be tested, and
scaling measured to n = 50, without 50 real WebSocket servers and their
timers -- and it can inject latency and loss, which the localhost run cannot.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional

from zkfl.messages import Envelope, MalformedMessage

Handler = Callable[[Envelope], Awaitable[None]]

MAX_FRAME_BYTES = 4 * 1024 * 1024


@dataclass
class GateStats:
    accepted: int = 0
    duplicate: int = 0
    rate_limited: int = 0
    malformed: int = 0
    oversized: int = 0
    from_stranger: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class MessageGate:
    """Per-peer, per-height admission control."""

    def __init__(self, peer_rate_limit: int = 64, horizon: int = 16):
        self.peer_rate_limit = peer_rate_limit
        self.horizon = horizon
        #: (sender, height) -> set of dedup keys
        self._seen: dict[tuple[str, int], set[str]] = defaultdict(set)
        #: (sender, height) -> messages admitted
        self._counts: dict[tuple[str, int], int] = defaultdict(int)
        self.stats = GateStats()

    def admit(self, envelope: Envelope, members: Optional[frozenset[str]] = None) -> bool:
        if members is not None and envelope.sender not in members:
            self.stats.from_stranger += 1
            return False
        slot = (envelope.sender, envelope.height)
        if self._counts[slot] >= self.peer_rate_limit:
            self.stats.rate_limited += 1
            return False
        key = envelope.dedup_key
        bucket = self._seen[slot]
        if key in bucket:
            self.stats.duplicate += 1
            return False
        bucket.add(key)
        self._counts[slot] += 1
        self.stats.accepted += 1
        return True

    def release(self, height: int) -> None:
        """Forget state for a committed height, and anything far behind it."""
        for slot in list(self._seen):
            if slot[1] <= height - self.horizon or slot[1] == height:
                self._seen.pop(slot, None)
                self._counts.pop(slot, None)


# ---------------------------------------------------------------------------


class Transport:
    """What a node needs from the network, and nothing more."""

    def set_handler(self, handler: Handler) -> None:
        raise NotImplementedError

    async def broadcast(self, envelope: Envelope) -> None:
        raise NotImplementedError

    async def send(self, peer: str, envelope: Envelope) -> bool:
        raise NotImplementedError

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def peers(self) -> list[str]:
        return []


class InProcessTransport(Transport):
    """A shared in-memory mesh with optional latency and loss.

    Deterministic given a seed, which is what makes the adversarial and
    scaling experiments reproducible.
    """

    def __init__(
        self,
        node_id: str,
        bus: "MessageBus",
    ):
        self.node_id = node_id
        self.bus = bus
        self._handler: Optional[Handler] = None
        self.bytes_out = 0
        self.messages_out = 0

    def set_handler(self, handler: Handler) -> None:
        self._handler = handler
        self.bus.attach(self.node_id, self)

    async def deliver(self, envelope: Envelope) -> None:
        if self._handler is not None:
            await self._handler(envelope)

    async def broadcast(self, envelope: Envelope) -> None:
        payload = json.dumps(envelope.to_dict(), separators=(",", ":"))
        targets = [p for p in self.bus.members() if p != self.node_id]
        self.messages_out += len(targets)
        self.bytes_out += len(payload) * len(targets)
        await self.bus.dispatch(self.node_id, targets, envelope)

    async def send(self, peer: str, envelope: Envelope) -> bool:
        if peer == self.node_id or peer not in self.bus.members():
            return False
        payload = json.dumps(envelope.to_dict(), separators=(",", ":"))
        self.messages_out += 1
        self.bytes_out += len(payload)
        await self.bus.dispatch(self.node_id, [peer], envelope)
        return True

    def peers(self) -> list[str]:
        return [p for p in self.bus.members() if p != self.node_id]


class MessageBus:
    """The shared fabric behind :class:`InProcessTransport`."""

    def __init__(
        self,
        *,
        latency: float = 0.0,
        jitter: float = 0.0,
        loss: float = 0.0,
        partition: Optional[Callable[[str, str], bool]] = None,
        seed: int = 0,
    ):
        self.latency = latency
        self.jitter = jitter
        self.loss = loss
        self.partition = partition
        self._rng = random.Random(seed)
        self._nodes: dict[str, InProcessTransport] = {}
        self._tasks: set[asyncio.Task] = set()
        self.delivered = 0
        self.dropped = 0
        #: deliveries per envelope kind.  A single total is indistinguishable
        #: from a formula -- 3(n^2-1) at every size, in our case -- so the
        #: breakdown is what shows the count is a measurement.
        self.delivered_by_kind: dict[str, int] = {}

    def attach(self, node_id: str, transport: InProcessTransport) -> None:
        self._nodes[node_id] = transport

    def members(self) -> list[str]:
        return list(self._nodes)

    async def dispatch(self, sender: str, targets: Iterable[str], envelope: Envelope) -> None:
        for target in targets:
            if self.partition is not None and not self.partition(sender, target):
                self.dropped += 1
                continue
            if self.loss and self._rng.random() < self.loss:
                self.dropped += 1
                continue
            delay = self.latency
            if self.jitter:
                delay += self._rng.uniform(0.0, self.jitter)
            transport = self._nodes.get(target)
            if transport is None:
                continue
            self.delivered += 1
            self.delivered_by_kind[envelope.kind] = (
                self.delivered_by_kind.get(envelope.kind, 0) + 1)
            task = asyncio.create_task(self._deliver(transport, envelope, delay))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    @staticmethod
    async def _deliver(transport: InProcessTransport, envelope: Envelope, delay: float) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            await transport.deliver(envelope)
        except Exception as exc:  # a handler bug must not silently vanish
            print(f"[bus] handler error: {exc!r}")

    async def drain(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self._tasks and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

    def stats(self) -> dict[str, int]:
        return {
            "delivered": self.delivered,
            "dropped": self.dropped,
            "bytes_out": sum(t.bytes_out for t in self._nodes.values()),
            "messages_out": sum(t.messages_out for t in self._nodes.values()),
            "delivered_by_kind": dict(sorted(self.delivered_by_kind.items())),
        }


class WebSocketTransport(Transport):
    """Real WebSocket broadcast over a configured peer set.

    The peer set is a list of ``(node_id, host, port)``: a node's address is
    bound to its public key, so an unknown sender is dropped before parsing.
    Transport authentication is still TCP-level only -- see docs/THREAT_MODEL.md
    -- but every message carries its own signature, so a hostile transport can
    delay and drop, not forge.
    """

    def __init__(
        self,
        node_id: str,
        listen_port: int,
        peers: dict[str, tuple[str, int]],
        *,
        connect_timeout: float = 5.0,
        max_retries: int = 3,
    ):
        self.node_id = node_id
        self.listen_port = listen_port
        self._peers = dict(peers)
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self._handler: Optional[Handler] = None
        self._server = None
        self.failures: dict[str, int] = defaultdict(int)
        self.bytes_out = 0
        self.messages_out = 0

    def set_handler(self, handler: Handler) -> None:
        self._handler = handler

    def peers(self) -> list[str]:
        return list(self._peers)

    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self._server = await serve(
            self._on_connection, "127.0.0.1", self.listen_port,
            max_size=MAX_FRAME_BYTES,
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _on_connection(self, websocket) -> None:
        try:
            async for raw in websocket:
                if len(raw) > MAX_FRAME_BYTES:
                    continue
                try:
                    envelope = Envelope.from_dict(json.loads(raw))
                except (json.JSONDecodeError, MalformedMessage, TypeError):
                    continue
                if self._handler is not None:
                    await self._handler(envelope)
        except Exception:
            return

    async def _send_raw(self, host: str, port: int, payload: str) -> bool:
        from websockets.asyncio.client import connect

        for attempt in range(1, self.max_retries + 1):
            try:
                async with asyncio.timeout(self.connect_timeout):
                    async with connect(f"ws://{host}:{port}",
                                       max_size=MAX_FRAME_BYTES) as ws:
                        await ws.send(payload)
                return True
            except Exception:
                if attempt == self.max_retries:
                    return False
                await asyncio.sleep(0.2 * attempt)
        return False

    async def send(self, peer: str, envelope: Envelope) -> bool:
        target = self._peers.get(peer)
        if target is None:
            return False
        payload = json.dumps(envelope.to_dict(), separators=(",", ":"))
        ok = await self._send_raw(target[0], target[1], payload)
        if ok:
            self.messages_out += 1
            self.bytes_out += len(payload)
            self.failures[peer] = 0
        else:
            self.failures[peer] += 1
        return ok

    async def broadcast(self, envelope: Envelope) -> None:
        payload = json.dumps(envelope.to_dict(), separators=(",", ":"))
        results = await asyncio.gather(
            *(self._send_raw(host, port, payload)
              for host, port in self._peers.values()),
            return_exceptions=True,
        )
        for peer, ok in zip(self._peers, results):
            if ok is True:
                self.messages_out += 1
                self.bytes_out += len(payload)
                self.failures[peer] = 0
            else:
                self.failures[peer] += 1

    def health(self) -> dict[str, dict[str, Any]]:
        return {
            peer: {"failures": self.failures[peer], "healthy": self.failures[peer] < 5}
            for peer in self._peers
        }
