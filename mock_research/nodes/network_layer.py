# nodes/network_layer.py
import json
import asyncio
from typing import Callable, Awaitable, Optional
from nodes.fl_types import UpdateType, VoteMessage, VoteSetBroadcast

import websockets
from websockets import connect, serve
from websockets.exceptions import ConnectionClosed


class NetworkLayer:
    """
    Gossip-based WebSocket networking layer.
    Handles model updates, individual votes, and full vote-set broadcasts.
    """

    def __init__(self, node_id: str, peers: list[tuple[str, int]],
                 max_retry_attempts: int = 3, retry_delay: float = 1.0):
        self.node_id = node_id
        self.peers = peers
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay

        self._callback: Optional[Callable[[dict], Awaitable[None]]] = None
        self.received_messages: set[str] = set()
        self._lock = asyncio.Lock()
        self._peer_failures: dict[tuple[str, int], int] = {}

    # ── Low-level send ──────────────────────────────────────────────────

    async def _send_to_peer(self, ip: str, port: int, msg: str,
                             attempt: int = 1) -> bool:
        peer = (ip, port)
        try:
            async with asyncio.timeout(5.0):
                async with connect(f"ws://{ip}:{port}") as ws:
                    await ws.send(msg)
            if peer in self._peer_failures:
                self._peer_failures[peer] = 0
            return True
        except asyncio.TimeoutError:
            print(f"[Node {self.node_id}] Timeout sending to {ip}:{port}")
        except ConnectionRefusedError:
            print(f"[Node {self.node_id}] Connection refused by {ip}:{port}")
        except Exception as e:
            print(f"[Node {self.node_id}] Failed to send to {ip}:{port} -> {e}")

        self._peer_failures[peer] = self._peer_failures.get(peer, 0) + 1
        if attempt < self.max_retry_attempts:
            await asyncio.sleep(self.retry_delay * attempt)
            return await self._send_to_peer(ip, port, msg, attempt + 1)
        return False

    async def _broadcast(self, message: dict, label: str) -> dict[str, bool]:
        msg = json.dumps(message, default=str)
        tasks = [self._send_to_peer(ip, port, msg) for ip, port in self.peers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        status: dict[str, bool] = {}
        for peer, result in zip(self.peers, results):
            status[str(peer)] = not isinstance(result, Exception) and bool(result)
        success = sum(1 for v in status.values() if v)
        print(f"[Node {self.node_id}] Broadcast {label}: {success}/{len(self.peers)} peers reached")
        return status

    # ── Public broadcast methods ────────────────────────────────────────

    async def broadcast_update(self, update: UpdateType) -> dict[str, bool]:
        return await self._broadcast(
            {"type": "update", "data": update.to_dict()}, "update"
        )

    async def broadcast_vote(self, vote: VoteMessage) -> dict[str, bool]:
        return await self._broadcast(
            {"type": "vote", "data": vote.to_dict()}, "vote"
        )

    async def broadcast_vote_set(self, ballot: VoteSetBroadcast) -> dict[str, bool]:
        """
        Broadcasts the node's full accumulated vote set for a height.
        Peers use this to cross-check for equivocation.
        """
        return await self._broadcast(
            {"type": "vote_set", "data": ballot.to_dict()}, "vote_set"
        )

    # ── Incoming message handling ───────────────────────────────────────

    async def handle_connection(self, websocket):
        if not self._callback:
            return

        remote_addr = websocket.remote_address
        print(f"[Node {self.node_id}] Connection from {remote_addr}")

        try:
            async for msg in websocket:
                try:
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    msg_data = data.get("data")

                    if not msg_type or not msg_data:
                        print(f"[Node {self.node_id}] Malformed message: missing type or data")
                        continue

                    # Build a deduplication key per message type
                    if msg_type == "update":
                        msg_id = msg_data.get("update_id")
                    elif msg_type == "vote":
                        msg_id = (
                            f"{msg_data.get('voter_node_id')}:"
                            f"{msg_data.get('target_height')}:"
                            f"{msg_data.get('voted_update_id')}"
                        )
                    elif msg_type == "vote_set":
                        # The fingerprint encodes sender + height + content hash
                        # so a re-broadcast with new votes gets a different ID
                        sender   = msg_data.get("sender_node_id", "")
                        height   = msg_data.get("target_height", 0)
                        vote_ids = "|".join(sorted(
                            f"{v.get('voter_node_id')}:{v.get('voted_update_id')}"
                            for v in msg_data.get("votes", [])
                        ))
                        import hashlib
                        fp = hashlib.sha256(
                            f"{sender}:{height}:{vote_ids}".encode()
                        ).hexdigest()[:16]
                        msg_id = f"voteset:{sender}:{height}:{fp}"
                    else:
                        msg_id = None

                    if msg_id is None:
                        print(f"[Node {self.node_id}] Message without ID")
                        continue

                    async with self._lock:
                        if msg_id in self.received_messages:
                            continue
                        self.received_messages.add(msg_id)
                        if len(self.received_messages) > 10000:
                            to_remove = list(self.received_messages)[:5000]
                            self.received_messages.difference_update(to_remove)

                    await self._callback(data)

                except json.JSONDecodeError as e:
                    print(f"[Node {self.node_id}] Invalid JSON: {e}")
                except (TypeError, KeyError) as e:
                    print(f"[Node {self.node_id}] Malformed message: {e}")
                except Exception as e:
                    print(f"[Node {self.node_id}] Error processing message: {e}")

        except ConnectionClosed:
            print(f"[Node {self.node_id}] Connection closed by {remote_addr}")
        except Exception as e:
            print(f"[Node {self.node_id}] Connection error: {e}")

    async def gossip_loop(self, callback: Callable[[dict], Awaitable[None]],
                          listen_port: int):
        self._callback = callback
        async with serve(self.handle_connection, "localhost", listen_port,
                         max_size=10 * 1024 * 1024):
            print(f"[Node {self.node_id}] Listening on port {listen_port}...")
            await asyncio.Future()

    def get_peer_health(self) -> dict:
        return {
            f"{ip}:{port}": {
                "failures": self._peer_failures.get((ip, port), 0),
                "healthy": self._peer_failures.get((ip, port), 0) < 5
            }
            for ip, port in self.peers
        }