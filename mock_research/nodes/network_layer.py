# nodes/network_layer.py
import json
import asyncio
from typing import Callable, Awaitable, Optional
from nodes.types import UpdateType, VoteMessage

# ✅ CORRECTION: Imports explicites pour satisfaire Pylance et éviter les erreurs d'attributs
from websockets.client import connect
from websockets.server import serve
from websockets.exceptions import ConnectionClosed

class NetworkLayer:
    """
    Gossip-based WebSocket networking layer.
    Handles both model updates and vote messages.
    """

    def __init__(self, node_id: str, peers: list[tuple[str, int]], max_retry_attempts: int = 3, retry_delay: float = 1.0):
        self.node_id = node_id
        self.peers = peers
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay

        self._callback: Optional[Callable[[dict], Awaitable[None]]] = None
        self.received_messages: set[str] = set()
        self._lock = asyncio.Lock()

        self._peer_failures: dict[tuple[str, int], int] = {}

    async def _send_to_peer(self, ip: str, port: int, msg: str, attempt: int = 1) -> bool:
        """
        Sends message to a peer with retry logic.
        Returns True if successful, False otherwise.
        """
        peer = (ip, port)
        
        try:
            async with asyncio.timeout(5.0): 
                # ✅ Utilisation directe de connect() importé
                async with connect(f"ws://{ip}:{port}") as ws:
                    await ws.send(msg)
                    
            # Reset failure count on success
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

    async def broadcast_update(self, update: UpdateType) -> dict[str, bool]:
        """
        Sends an update to all peers in parallel.
        Returns dict mapping peer to success status.
        """
        update_dict = update.to_dict()
        message = {
            "type": "update",
            "data": update_dict
        }
        msg = json.dumps(message, default=str)
        
        tasks = [
            self._send_to_peer(ip, port, msg) 
            for ip, port in self.peers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        status = {}
        for peer, result in zip(self.peers, results):
            if isinstance(result, Exception):
                status[peer] = False
            else:
                status[peer] = result
        
        success_count = sum(1 for v in status.values() if v)
        print(f"[Node {self.node_id}] Broadcast update: {success_count}/{len(self.peers)} peers reached")
        
        return status

    async def broadcast_vote(self, vote: VoteMessage) -> dict[str, bool]:
        """
        Broadcasts a vote to all peers.
        Returns dict mapping peer to success status.
        """
        vote_dict = vote.to_dict()
        message = {
            "type": "vote",
            "data": vote_dict
        }
        msg = json.dumps(message, default=str)
        
        tasks = [
            self._send_to_peer(ip, port, msg) 
            for ip, port in self.peers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        status = {}
        for peer, result in zip(self.peers, results):
            if isinstance(result, Exception):
                status[peer] = False
            else:
                status[peer] = result
        
        success_count = sum(1 for v in status.values() if v)
        print(f"[Node {self.node_id}] Broadcast vote: {success_count}/{len(self.peers)} peers reached")
        
        return status

    async def handle_connection(self, websocket):
        """
        Handles incoming WebSocket messages.
        Supports both update and vote messages.
        """
        if not self._callback:
            return

        remote_addr = websocket.remote_address
        print(f"[Node {self.node_id}] Connection from {remote_addr}")

        try:
            async for msg in websocket:
                try:
                    # Parse JSON message
                    data = json.loads(msg)
                    
                    # Extract message type
                    msg_type = data.get("type")
                    msg_data = data.get("data")
                    
                    if not msg_type or not msg_data:
                        print(f"[Node {self.node_id}] Malformed message: missing type or data")
                        continue
                    
                    # Create message ID for deduplication
                    if msg_type == "update":
                        msg_id = msg_data.get("update_id")
                    elif msg_type == "vote":
                        # Vote ID is combination of voter + height + voted_update
                        msg_id = f"{msg_data.get('voter_node_id')}:{msg_data.get('target_height')}:{msg_data.get('voted_update_id')}"
                    else:
                        msg_id = None
                    
                    if msg_id is None:
                        print(f"[Node {self.node_id}] Message without ID")
                        continue

                    # Thread-safe deduplication
                    async with self._lock:
                        if msg_id in self.received_messages:
                            continue
                        self.received_messages.add(msg_id)
                        
                        # Bound the deduplication set
                        if len(self.received_messages) > 10000:
                            to_remove = list(self.received_messages)[:5000]
                            self.received_messages.difference_update(to_remove)

                    # Route to callback (node's message handler)
                    await self._callback(data)

                except json.JSONDecodeError as e:
                    print(f"[Node {self.node_id}] Invalid JSON: {e}")
                    continue
                except (TypeError, KeyError) as e:
                    print(f"[Node {self.node_id}] Malformed message: {e}")
                    continue
                except Exception as e:
                    print(f"[Node {self.node_id}] Error processing message: {e}")
                    continue
        
        # ✅ Utilisation directe de ConnectionClosed importé
        except ConnectionClosed:
            print(f"[Node {self.node_id}] Connection closed by {remote_addr}")
        except Exception as e:
            print(f"[Node {self.node_id}] Connection error: {e}")

    async def gossip_loop(self, callback: Callable[[dict], Awaitable[None]], listen_port: int):
        """
        Start listening for incoming connections.
        Callback receives message dict with 'type' and 'data' fields.
        """
        self._callback = callback

        # ✅ Utilisation directe de serve() importé
        async with serve(self.handle_connection, "localhost", listen_port, max_size=10 * 1024 * 1024):
            print(f"[Node {self.node_id}] Listening on port {listen_port}...")
            await asyncio.Future()

    def get_peer_health(self) -> dict:
        """
        Returns health status of all peers.
        """
        return {
            f"{ip}:{port}": {
                "failures": self._peer_failures.get((ip, port), 0),
                "healthy": self._peer_failures.get((ip, port), 0) < 5
            }
            for ip, port in self.peers
        }