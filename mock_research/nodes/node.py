# nodes/node.py
import asyncio
import numpy as np
import hashlib
from typing import Any, Optional, List
from collections import deque, Counter

from nodes.training_engine import TrainingEngine
from nodes.update_manager import UpdateManager
from nodes.network_layer import NetworkLayer
from nodes.fl_types import UpdateLogEntry, UpdateType, VoteMessage, VoteSetBroadcast
from utils.serialization import serialize_model


class Node:
    """
    A federated learning node with:
    - private local data
    - interruptible training
    - gossip-based update propagation
    - model versioning (model_height)
    - Byzantine-resilient distributed voting with full vote-set gossip
    - equivocation detection and exclusion
    - buffering of future updates
    """

    def __init__(self,
                 node_id: str,
                 peers: List[tuple],
                 voting_window: float = 60.0,
                 min_votes: int = 2,
                 max_seen_updates: int = 10000,
                 enable_strict_zk: bool = False,
                 learning_rate: float = 0.01):

        self.node_id        = node_id
        self.peers          = peers
        self.voting_window  = voting_window
        self.min_votes      = min_votes
        self.enable_strict_zk = enable_strict_zk
        self.learning_rate  = learning_rate

        # Private model state
        self.__local_data:   Optional[Any] = None
        self.__local_model:  Optional[Any] = None
        self.__prev_model:   Optional[Any] = None
        self.local_model_hash: Optional[str] = None
        self.model_height: int = 0

        # Sync primitives
        self._data_ready_event = asyncio.Event()
        self._training_task:  Optional[asyncio.Task] = None
        self._consensus_lock  = asyncio.Lock()

        # Sub-modules
        self.training_engine = TrainingEngine(learning_rate=learning_rate)
        self.update_manager  = UpdateManager(
            private_key=None,
            enable_strict_zk=enable_strict_zk,
            learning_rate=learning_rate,
        )
        self.network_layer = NetworkLayer(node_id, peers)

        # Update tracking / audit
        self.seen_updates: deque = deque(maxlen=max_seen_updates)
        self.update_log:   list[UpdateLogEntry] = []

        # Active voting tasks indexed by height
        self._voting_tasks: dict[int, asyncio.Task] = {}

        # Heights we know are missing (gap recovery)
        self._missed_heights: set[int] = set()

        # Vote state
        self._votes_by_height: dict[int, list[VoteMessage]] = {}
        self._vote_lock       = asyncio.Lock()
        self._voted_heights:  set[int] = set()

        # ── Byzantine / equivocation tracking ──────────────────────────
        # Node IDs confirmed to have equivocated (voted differently at same height)
        self._equivocators: set[str] = set()
        # Human-readable log of detected equivocations for audit / tests
        self.equivocation_log: list[dict] = []

    # ===========================
    # ========= SETTERS =========
    # ===========================

    def set_local_data(self, dataset: Any) -> None:
        if self.__local_data is not None:
            raise RuntimeError("Local data already set and not yet consumed.")
        self.__local_data = dataset
        self._data_ready_event.set()

    def set_model(self, model: Any) -> None:
        self.__local_model = model
        self.local_model_hash = hashlib.sha256(serialize_model(model)).hexdigest()

    def get_model(self) -> Optional[dict]:
        return self.__local_model

    # ===========================
    # ==== NETWORK CALLBACKS ====
    # ===========================

    async def on_receive_message(self, message: dict) -> None:
        msg_type = message.get("type")
        data     = message.get("data")

        if msg_type == "update":
            if not isinstance(data, dict):
                return
            await self.on_receive_update(UpdateType.from_dict(data))

        elif msg_type == "vote":
            if not isinstance(data, dict):
                return
            await self.on_receive_vote(VoteMessage.from_dict(data))

        elif msg_type == "vote_set":
            if not isinstance(data, dict):
                return
            await self.on_receive_vote_set(VoteSetBroadcast.from_dict(data))

        else:
            print(f"[Node {self.node_id}] Unknown message type: {msg_type}")

    # ── Update handling ────────────────────────────────────────────────

    async def on_receive_update(self, update: UpdateType) -> None:
        assert update.update_id is not None

        if update.update_id in self.seen_updates:
            return

        if not self.update_manager.verify_update(update, prev_model=self.__local_model):
            print(f"[Node {self.node_id}] Invalid update rejected.")
            return

        self.seen_updates.append(update.update_id)

        if update.model_height <= self.model_height:
            return

        if update.model_height > self.model_height + 1:
            gap_heights = range(self.model_height + 1, update.model_height)
            self._missed_heights.update(gap_heights)
            print(f"[Node {self.node_id}] Detected gap: missing heights {list(gap_heights)}")
            self.update_manager.add_candidate(update)
            print(f"[Node {self.node_id}] Buffered future update H={update.model_height}")
            return

        assert update.model_height == self.model_height + 1
        self.update_manager.add_candidate(update)

        vote_to_broadcast: Optional[VoteMessage] = None
        height = update.model_height

        async with self._vote_lock:
            if height not in self._voted_heights:
                self._voted_heights.add(height)
                my_vote = VoteMessage(
                    voter_node_id=self.node_id,
                    target_height=height,
                    voted_update_id=update.update_id,
                )
                self._votes_by_height.setdefault(height, []).append(my_vote)
                print(f"[Node {self.node_id}] Voting for update {update.update_id[:8]} at H={height}")
                vote_to_broadcast = my_vote

        if vote_to_broadcast is not None:
            await self.network_layer.broadcast_vote(vote_to_broadcast)
            # Also broadcast the full vote set so peers can cross-check
            await self._broadcast_vote_set(height)

        await self._ensure_voting_task(height)

    # ── Vote handling ──────────────────────────────────────────────────

    async def on_receive_vote(self, vote: VoteMessage) -> None:
        """Record an individual vote from a peer."""
        new_vote = False
        async with self._vote_lock:
            bucket = self._votes_by_height.setdefault(vote.target_height, [])
            existing = [v for v in bucket if v.voter_node_id == vote.voter_node_id]
            if not existing and vote.voter_node_id not in self._equivocators:
                bucket.append(vote)
                new_vote = True
                print(f"[Node {self.node_id}] Recorded vote from {vote.voter_node_id} "
                      f"for update {vote.voted_update_id[:8]} at H={vote.target_height}")

        # Gossip the full vote set when we learn something new
        if new_vote:
            await self._broadcast_vote_set(vote.target_height)

    # ── Vote-set (Byzantine-resilient gossip) ─────────────────────────

    async def on_receive_vote_set(self, broadcast: VoteSetBroadcast) -> None:
        """
        Process a full vote set broadcast from a peer.

        For each vote inside:
          - If we already know that voter chose a DIFFERENT update at this height
            → equivocation detected; voter is blacklisted.
          - If it's a new vote from an honest voter → add it and re-gossip.
        """
        height    = broadcast.target_height
        new_info  = False

        async with self._vote_lock:
            for inc in broadcast.votes:
                voter = inc.voter_node_id

                # Skip already-known equivocators
                if voter in self._equivocators:
                    continue

                existing = [v for v in self._votes_by_height.get(height, [])
                            if v.voter_node_id == voter]

                if existing:
                    if existing[0].voted_update_id != inc.voted_update_id:
                        # ── Equivocation detected ──────────────────────
                        self._equivocators.add(voter)
                        record = {
                            "height":      height,
                            "equivocator": voter,
                            "vote_a":      existing[0].voted_update_id,
                            "vote_b":      inc.voted_update_id,
                            "detected_by": self.node_id,
                        }
                        self.equivocation_log.append(record)
                        # Remove the tainted vote from our bucket
                        self._votes_by_height[height] = [
                            v for v in self._votes_by_height[height]
                            if v.voter_node_id != voter
                        ]
                        print(
                            f"[Node {self.node_id}] ⚠️  EQUIVOCATION: Node {voter} "
                            f"voted {existing[0].voted_update_id[:8]} ↔ "
                            f"{inc.voted_update_id[:8]} at H={height}"
                        )
                else:
                    # Brand-new vote — add it
                    self._votes_by_height.setdefault(height, []).append(inc)
                    new_info = True

        # Propagate new information onward (one-hop gossip amplification)
        if new_info:
            await self._broadcast_vote_set(height)

    async def _broadcast_vote_set(self, height: int) -> None:
        """Snapshot the current vote set and broadcast it to all peers."""
        async with self._vote_lock:
            votes = list(self._votes_by_height.get(height, []))
        if not votes:
            return
        ballot = VoteSetBroadcast(
            sender_node_id=self.node_id,
            target_height=height,
            votes=votes,
        )
        await self.network_layer.broadcast_vote_set(ballot)

    # ── Consensus plumbing ─────────────────────────────────────────────

    async def _ensure_voting_task(self, height: int):
        async with self._consensus_lock:
            if height in self._voting_tasks:
                if not self._voting_tasks[height].done():
                    return
            task = asyncio.create_task(self._process_consensus(height))
            self._voting_tasks[height] = task

    async def _process_consensus(self, target_height: int):
        try:
            await asyncio.sleep(self.voting_window)

            if self.model_height >= target_height:
                print(f"[Node {self.node_id}] Already at H={self.model_height}, "
                      f"skipping consensus for H={target_height}")
                return

            async with self._vote_lock:
                all_votes = list(self._votes_by_height.get(target_height, []))

            if not all_votes:
                print(f"[Node {self.node_id}] No votes for H={target_height}")
                return

            # Exclude confirmed equivocators from the tally
            honest_votes = [v for v in all_votes
                            if v.voter_node_id not in self._equivocators]
            vote_counts  = Counter(v.voted_update_id for v in honest_votes)
            total_votes  = len(honest_votes)

            if self._equivocators:
                excluded = len(all_votes) - total_votes
                print(f"[Node {self.node_id}] Excluded {excluded} equivocator vote(s) "
                      f"at H={target_height}")

            print(f"[Node {self.node_id}] Vote tally for H={target_height}: "
                  f"{dict(vote_counts)} (from {total_votes} honest votes)")

            winning_update_id: Optional[str] = None

            if len(vote_counts) == 0:
                print(f"[Node {self.node_id}] No honest votes for H={target_height}")
                return

            elif len(vote_counts) == 1:
                # Single candidate — still must satisfy min_votes
                winning_update_id = list(vote_counts.keys())[0]
                count = vote_counts[winning_update_id]
                if count >= self.min_votes:
                    print(f"[Node {self.node_id}] Single candidate accepted: "
                          f"{count}/{total_votes} for {winning_update_id[:8]}")
                else:
                    print(f"[Node {self.node_id}] Single candidate H={target_height}: "
                          f"only {count} vote(s), need {self.min_votes}")
                    return

            else:
                winning_update_id, count = vote_counts.most_common(1)[0]
                if count > total_votes / 2 and count >= self.min_votes:
                    print(f"[Node {self.node_id}] Majority: {count}/{total_votes} "
                          f"for {winning_update_id[:8]}")
                else:
                    print(f"[Node {self.node_id}] No majority at H={target_height}: "
                          f"best {count}/{total_votes} (need >{total_votes/2} and "
                          f">={self.min_votes})")
                    return

            chosen = self.update_manager.get_update_by_id(winning_update_id)
            if chosen:
                await self._apply_update(chosen)
            else:
                print(f"[Node {self.node_id}] Winning update not in candidate pool")

            # Prune vote state for this height
            async with self._vote_lock:
                self._votes_by_height.pop(target_height, None)
                self._voted_heights.discard(target_height)

        except Exception as e:
            print(f"[Node {self.node_id}] Error in consensus: {e}")
        finally:
            async with self._consensus_lock:
                self._voting_tasks.pop(target_height, None)

    async def _apply_update(self, update: UpdateType):
        if update.model_height != self.model_height + 1:
            print(f"[Node {self.node_id}] Skipping update H={update.model_height} "
                  f"— current height is {self.model_height}")
            return

        print(f"[Node {self.node_id}] Applying update H={update.model_height} "
              f"(hash={update.params_hash[:8]}...)")

        # Interrupt training if running
        if self._training_task and not self._training_task.done():
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass

        self.__prev_model     = self.__local_model.copy() if self.__local_model else None
        self.__local_model    = update.new_params
        self.local_model_hash = update.params_hash
        self.model_height     = update.model_height
        self._missed_heights.discard(update.model_height)

        self.update_log.append(UpdateLogEntry(
            source_node=self.node_id,
            previous_model_hash=update.prev_model_hash,
            params_hash=update.params_hash,
            proof=update.zk_proof,
        ))

        self.update_manager.clear_candidates_for_height(update.model_height)
        await self._check_buffered_updates()

    async def _check_buffered_updates(self):
        next_height = self.model_height + 1
        candidates  = self.update_manager.get_candidates_for_height(next_height)
        if not candidates:
            return

        print(f"[Node {self.node_id}] Found {len(candidates)} buffered update(s) "
              f"for H={next_height}")

        vote_to_broadcast: Optional[VoteMessage] = None

        for update in candidates:
            if update.update_id is None:
                continue
            async with self._vote_lock:
                if next_height not in self._voted_heights:
                    self._voted_heights.add(next_height)
                    my_vote = VoteMessage(
                        voter_node_id=self.node_id,
                        target_height=next_height,
                        voted_update_id=update.update_id,
                    )
                    self._votes_by_height.setdefault(next_height, []).append(my_vote)
                    print(f"[Node {self.node_id}] Voting for buffered update "
                          f"{update.update_id[:8]} at H={next_height}")
                    vote_to_broadcast = my_vote
                    break

        if vote_to_broadcast is not None:
            await self.network_layer.broadcast_vote(vote_to_broadcast)
            await self._broadcast_vote_set(next_height)

        await self._ensure_voting_task(next_height)

    # ===========================
    # ====== MAIN ENTRY =========
    # ===========================

    async def run(self, listen_port: int = 20000) -> None:
        await asyncio.gather(
            self.network_layer.gossip_loop(
                callback=self.on_receive_message,
                listen_port=listen_port,
            ),
            self._training_manager(),
        )

    # ===========================
    # ====== TRAINING ============
    # ===========================

    async def _training_manager(self) -> None:
        while True:
            await self._data_ready_event.wait()
            self._training_task = asyncio.create_task(self._training_worker())
            try:
                await self._training_task
            except asyncio.CancelledError:
                print(f"[Node {self.node_id}] Training cancelled by newer update.")
            finally:
                self._training_task = None

    async def _training_worker(self) -> None:
        assert self.__local_data is not None
        assert self.__local_model is not None
        assert self.local_model_hash is not None

        loop = asyncio.get_running_loop()
        prev_model_copy = self.__local_model.copy()

        new_params = await loop.run_in_executor(
            None,
            self.training_engine.train_until_converged,
            self.__local_data,
            self.__local_model,
        )

        update = self.update_manager.build_update(
            prev_model_hash=self.local_model_hash,
            new_params=new_params,
            model_height=self.model_height + 1,
            prev_model=prev_model_copy,
        )

        assert update.update_id is not None
        print(f"*** Node {self.node_id} CREATED Update: {update.update_id} ***")

        # Add to own candidate pool so _apply_update can find it after consensus
        self.update_manager.add_candidate(update)

        # Self-vote (outside lock, then broadcast outside lock — consistent with FIX 2)
        target_height     = self.model_height + 1
        vote_to_broadcast: Optional[VoteMessage] = None

        async with self._vote_lock:
            if target_height not in self._voted_heights:
                self._voted_heights.add(target_height)
                my_vote = VoteMessage(
                    voter_node_id=self.node_id,
                    target_height=target_height,
                    voted_update_id=update.update_id,
                )
                self._votes_by_height.setdefault(target_height, []).append(my_vote)
                print(f"[Node {self.node_id}] Self-voting for own update "
                      f"{update.update_id[:8]} at H={target_height}")
                vote_to_broadcast = my_vote

        if vote_to_broadcast is not None:
            await self.network_layer.broadcast_vote(vote_to_broadcast)
            # Broadcast full vote set so every peer can start cross-checking
            await self._broadcast_vote_set(target_height)

        # Start our own consensus window
        await self._ensure_voting_task(target_height)

        # Broadcast the update payload to peers
        await self.network_layer.broadcast_update(update)

        # Release the data slot for the next round
        self.__local_data = None
        self._data_ready_event.clear()