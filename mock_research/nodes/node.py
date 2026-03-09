# nodes/node.py
import asyncio
import numpy as np
from typing import Any, Optional, List
from collections import deque, Counter

from nodes.training_engine import TrainingEngine
from nodes.update_manager import UpdateManager
from nodes.network_layer import NetworkLayer
import hashlib
from nodes.fl_types import UpdateLogEntry, UpdateType, VoteMessage
from utils.serialization import serialize_model


# ===========================
# ========== NODE ===========
# ===========================

class Node:
    """
    A federated learning node with:
    - private local data
    - interruptible training
    - gossip-based update propagation
    - model versioning (model_height)
    - distributed first-seen voting
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
        
        self.node_id = node_id
        self.peers = peers
        self.voting_window = voting_window
        self.min_votes = min_votes
        self.enable_strict_zk = enable_strict_zk
        self.learning_rate = learning_rate

        # ---------------------------
        # Private state
        # ---------------------------
        self.__local_data: Optional[Any] = None
        self.__local_model: Optional[Any] = None
        self.__prev_model: Optional[Any] = None  
        self.local_model_hash: Optional[str] = None
        self.model_height: int = 0

        # ---------------------------
        # Sync primitives
        # ---------------------------
        self._data_ready_event = asyncio.Event()
        self._training_task: Optional[asyncio.Task] = None
        self._consensus_lock = asyncio.Lock()

        # ---------------------------
        # Modules
        # ---------------------------
        self.training_engine = TrainingEngine(learning_rate=learning_rate)
        self.update_manager = UpdateManager(
            private_key=None, 
            enable_strict_zk=enable_strict_zk, 
            learning_rate=learning_rate
        )
        self.network_layer = NetworkLayer(node_id, peers)

        # ---------------------------
        # Update tracking / audit
        # ---------------------------
        self.seen_updates: deque = deque(maxlen=max_seen_updates)
        self.update_log: list[UpdateLogEntry] = []
        
        # Track active voting tasks by height
        self._voting_tasks: dict[int, asyncio.Task] = {}
        
        # Track missed heights for recovery
        self._missed_heights: set[int] = set()
        
        # ---------------------------
        # Voting state
        # ---------------------------
        self._votes_by_height: dict[int, list[VoteMessage]] = {}
        self._vote_lock = asyncio.Lock()
        
        # Track which height we've voted for
        self._voted_heights: set[int] = set()

    # ===========================
    # ========= SETTERS =========
    # ===========================

    def set_local_data(self, dataset: Any) -> None:
        """
        Injects a new dataset into the node.
        Triggers exactly ONE training round.
        """
        if self.__local_data is not None:
            raise RuntimeError("Local data already set and not yet consumed.")

        self.__local_data = dataset
        self._data_ready_event.set()

    def set_model(self, model: Any) -> None:
        """
        Initialize or replace the local model.
        Used for genesis or forced resync.
        """
        self.__local_model = model
        self.local_model_hash = hashlib.sha256(serialize_model(model)).hexdigest()

    def get_model(self) -> Optional[dict]:
        """Returns the current local model."""
        return self.__local_model

    # ===========================
    # ==== NETWORK CALLBACKS ====
    # ===========================

    async def on_receive_message(self, message: dict) -> None:
        """
        Router for different message types.
        """
        msg_type = message.get("type")
        data = message.get("data")
        
        if msg_type == "update":
            if not isinstance(data, dict):
                print(f"[Node {self.node_id}] Received invalid update data format")
                return
            update = UpdateType.from_dict(data)
            await self.on_receive_update(update)
            
        elif msg_type == "vote":
            if not isinstance(data, dict):
                print(f"[Node {self.node_id}] Received invalid vote data format")
                return
            vote = VoteMessage.from_dict(data)
            await self.on_receive_vote(vote)
            
        else:
            print(f"[Node {self.node_id}] Unknown message type: {msg_type}")

    async def on_receive_update(self, update: UpdateType) -> None:
        """
        When receiving an update:
        1. Validate it
        2. If it's the first update for this height we've seen, vote for it
        3. Buffer it for consensus (even if future height)
        """
        assert update.update_id is not None

        if update.update_id in self.seen_updates:
            return

        # Pass the previous model if available for ZK verification
        if not self.update_manager.verify_update(update, prev_model=self.__local_model):
            print(f"[Node {self.node_id}] Invalid update rejected.")
            return

        self.seen_updates.append(update.update_id)

        # Ignore old updates
        if update.model_height <= self.model_height:
            return
        
        # Handle future updates - BUFFER THEM (don't discard!)
        if update.model_height > self.model_height + 1:
            gap_heights = range(self.model_height + 1, update.model_height)
            self._missed_heights.update(gap_heights)
            print(f"[Node {self.node_id}] Detected gap: missing heights {list(gap_heights)}")
            
            # BUFFER the update instead of discarding
            self.update_manager.add_candidate(update)
            print(f"[Node {self.node_id}] Buffered future update H={update.model_height}")
            return
        
        # Only process if height = current + 1
        assert update.model_height == self.model_height + 1

        # Add to candidate pool
        self.update_manager.add_candidate(update)

        # FIX 2: Determine whether to vote outside the lock, then broadcast after releasing it.
        vote_to_broadcast: Optional[VoteMessage] = None

        async with self._vote_lock:
            if update.model_height not in self._voted_heights:
                self._voted_heights.add(update.model_height)
                
                my_vote = VoteMessage(
                    voter_node_id=self.node_id,
                    target_height=update.model_height,
                    voted_update_id=update.update_id
                )
                
                if update.model_height not in self._votes_by_height:
                    self._votes_by_height[update.model_height] = []
                self._votes_by_height[update.model_height].append(my_vote)
                
                print(f"[Node {self.node_id}] Voting for update {update.update_id[:8]} at H={update.model_height}")
                
                # Stage the vote for broadcasting — do NOT await inside the lock
                vote_to_broadcast = my_vote

        # FIX 2: Broadcast outside the lock to prevent deadlock / blocking other vote recording
        if vote_to_broadcast is not None:
            await self.network_layer.broadcast_vote(vote_to_broadcast)

        # Start voting window for this height (only once)
        await self._ensure_voting_task(update.model_height)

    async def on_receive_vote(self, vote: VoteMessage) -> None:
        """
        Record votes from other nodes.
        """
        async with self._vote_lock:
            if vote.target_height not in self._votes_by_height:
                self._votes_by_height[vote.target_height] = []
            
            # Check if this node already voted (prevent duplicates)
            existing = [v for v in self._votes_by_height[vote.target_height] 
                       if v.voter_node_id == vote.voter_node_id]
            
            if not existing:
                self._votes_by_height[vote.target_height].append(vote)
                print(f"[Node {self.node_id}] Recorded vote from {vote.voter_node_id} for update {vote.voted_update_id[:8]} at H={vote.target_height}")

    async def _ensure_voting_task(self, height: int):
        """
        Ensures only ONE voting task runs per height.
        Thread-safe with lock.
        """
        async with self._consensus_lock:
            if height in self._voting_tasks:
                task = self._voting_tasks[height]
                if not task.done():
                    return  
            
            task = asyncio.create_task(self._process_consensus(height))
            self._voting_tasks[height] = task

    async def _process_consensus(self, target_height: int):
        """
        Wait for voting window, then count votes.
        The update with majority votes wins.
        """
        try:
            # Wait for voting window to collect votes
            await asyncio.sleep(self.voting_window)
            
            # Simple check: Are we still waiting for this height?
            if self.model_height >= target_height:
                print(f"[Node {self.node_id}] Already at H={self.model_height}, skipping consensus for H={target_height}")
                return
            
            async with self._vote_lock:
                votes = list(self._votes_by_height.get(target_height, []))
            
            if not votes:
                print(f"[Node {self.node_id}] No votes collected for H={target_height}")
                return
            
            # Count votes by update_id
            vote_counts = Counter(v.voted_update_id for v in votes)
            total_votes = len(votes)
            
            print(f"[Node {self.node_id}] Vote tally for H={target_height}: {dict(vote_counts)}")
            
            winning_update_id: Optional[str] = None
            
            # Case 1: Only one update candidate → accept immediately
            if len(vote_counts) == 1:
                winning_update_id = list(vote_counts.keys())[0]
                if winning_update_id:
                    print(f"[Node {self.node_id}] Single candidate for H={target_height}, accepting update {winning_update_id[:8]}")
                else:
                    return
            
            # Case 2: Multiple candidates 
            elif len(vote_counts) > 1:
                winning_update_id, count = vote_counts.most_common(1)[0]
                
                if count > total_votes / 2 and count >= self.min_votes:
                    if winning_update_id:
                        print(f"[Node {self.node_id}] Majority reached: {count}/{total_votes} votes for update {winning_update_id[:8]}")
                else:
                    print(f"[Node {self.node_id}] No majority for H={target_height}: best was {count}/{total_votes} votes (need >{total_votes/2})")
                    return
            else:
                return
            
            # Find and apply the winning update
            if winning_update_id:
                chosen = self.update_manager.get_update_by_id(winning_update_id)
                
                if chosen:
                    await self._apply_update(chosen)
                else:
                    print(f"[Node {self.node_id}] Winning update {winning_update_id[:8]} not found in candidate pool")
            
            # FIX 1: Actually delete votes for this height instead of returning early.
            # FIX 7: Also prune _voted_heights to prevent unbounded growth.
            async with self._vote_lock:
                self._votes_by_height.pop(target_height, None)
                self._voted_heights.discard(target_height)
                    
        except Exception as e:
            print(f"[Node {self.node_id}] Error in consensus: {e}")
        finally:
            async with self._consensus_lock:
                if target_height in self._voting_tasks:
                    del self._voting_tasks[target_height]

    async def _apply_update(self, update: UpdateType):
        """
        Apply a verified update to local state.
        Interrupts training if necessary.
        Checks for buffered next updates after applying.
        """
        # ATOMIC CHECK: Only apply if we're at the right height
        if update.model_height != self.model_height + 1:
            print(f"[Node {self.node_id}] Skipping update H={update.model_height} - current height is {self.model_height}")
            return
        
        print(f"[Node {self.node_id}] Applying update H={update.model_height} (hash={update.params_hash[:8]}...)")
        
        # Interrupt training if running
        if self._training_task and not self._training_task.done():
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass

        # Update local state
        # 1. Save current as previous (for ZK proofs in next round)
        self.__prev_model = self.__local_model.copy() if self.__local_model else None
        # 2. Apply new model
        self.__local_model = update.new_params
        self.local_model_hash = update.params_hash
        self.model_height = update.model_height
        
        # Remove this height from missed set if it was there
        self._missed_heights.discard(update.model_height)
        
        # Add to audit log
        entry = UpdateLogEntry(
            source_node=self.node_id,
            previous_model_hash=update.prev_model_hash,
            params_hash=update.params_hash,
            proof=update.zk_proof
        )
        self.update_log.append(entry)

        # FIX 5: Free memory for candidates at this height after they are no longer needed.
        self.update_manager.clear_candidates_for_height(update.model_height)

        await self._check_buffered_updates()

    async def _check_buffered_updates(self):
        """
        After applying an update, check if we have the next height buffered.
        This enables cascade processing of out-of-order updates.
        """
        next_height = self.model_height + 1
        candidates = self.update_manager.get_candidates_for_height(next_height)
        
        if candidates:
            print(f"[Node {self.node_id}] Found {len(candidates)} buffered update(s) for H={next_height}, processing...")
            
            vote_to_broadcast: Optional[VoteMessage] = None

            for update in candidates:
                if update.update_id is None:
                    continue

                # FIX 2: Prepare vote inside lock, broadcast outside
                async with self._vote_lock:
                    if next_height not in self._voted_heights:
                        self._voted_heights.add(next_height)
                        
                        my_vote = VoteMessage(
                            voter_node_id=self.node_id,
                            target_height=next_height,
                            voted_update_id=update.update_id
                        )
                        
                        if next_height not in self._votes_by_height:
                            self._votes_by_height[next_height] = []
                        self._votes_by_height[next_height].append(my_vote)
                        
                        print(f"[Node {self.node_id}] Voting for buffered update {update.update_id[:8]} at H={next_height}")
                        vote_to_broadcast = my_vote
                        break  # Only vote once for this height

            # FIX 2: Broadcast after releasing the lock
            if vote_to_broadcast is not None:
                await self.network_layer.broadcast_vote(vote_to_broadcast)
            
            await self._ensure_voting_task(next_height)

    # ===========================
    # ====== MAIN ENTRY =========
    # ===========================

    async def run(self, listen_port: int = 20000) -> None:
        """
        Starts the node.
        Runs networking and training manager concurrently.
        """
        await asyncio.gather(
            self.network_layer.gossip_loop(
                callback=self.on_receive_message,
                listen_port=listen_port
            ),
            self._training_manager()
        )

    # ===========================
    # ====== TRAINING CONTROL ===
    # ===========================

    async def _training_manager(self) -> None:
        """
        Waits for data injection and launches training exactly once.
        Training can be cancelled by incoming updates.
        """
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
        """
        Performs:
        - local training
        - update construction (with ZK)
        - add to own candidate pool
        - self-vote + broadcast vote
        - start consensus task (so originator goes through _apply_update like peers)
        - broadcast update to peers
        - reset data

        BUG A FIX: The old code committed locally with hash_object(new_params) while
        the update carried params_hash = sha256(serialize_model(...)). These two hash
        functions produce different digests for the same weights, so the originator
        always ended up with a different local_model_hash than peers — even when its
        own update won consensus.

        BUG B FIX: The old code incremented model_height before consensus. When
        _apply_update later ran with the winning update at the same height, the guard
        `update.model_height != self.model_height + 1` fired and silently skipped it.
        Originators whose update LOST consensus (e.g. node 7 in the 3-way race) kept
        their own locally-trained model forever instead of switching to the agreed one.

        Solution: remove the premature local commit entirely. The originator adds its
        update to the candidate pool, self-votes, starts a voting task, and then lets
        _apply_update handle the state transition — identical to how every peer does it.
        """

        assert self.__local_data is not None
        assert self.__local_model is not None
        assert self.local_model_hash is not None

        loop = asyncio.get_running_loop()

        # Capture copy of model BEFORE training for ZK proof generation
        prev_model_copy = self.__local_model.copy()

        # 1. Train (offloaded to thread pool)
        new_params = await loop.run_in_executor(
            None,
            self.training_engine.train_until_converged,
            self.__local_data,
            self.__local_model
        )

        # 2. Build update (Generate ZK-SNARK)
        update = self.update_manager.build_update(
            prev_model_hash=self.local_model_hash,
            new_params=new_params,
            model_height=self.model_height + 1,
            prev_model=prev_model_copy
        )

        assert update.update_id is not None
        print(f"*** Node {self.node_id} CREATED Update: {update.update_id} ***")

        # 3. Add to own candidate pool so _apply_update can find it after consensus.
        #    (Peers add it when they receive the broadcast; we add it directly here.)
        self.update_manager.add_candidate(update)

        # 4. Self-vote and start our own consensus task.
        #    BUG B FIX: do NOT commit locally here — let _apply_update do it so the
        #    originator always ends up with the network-agreed model and hash.
        target_height = self.model_height + 1
        vote_to_broadcast: Optional[VoteMessage] = None

        async with self._vote_lock:
            if target_height not in self._voted_heights:
                self._voted_heights.add(target_height)

                my_vote = VoteMessage(
                    voter_node_id=self.node_id,
                    target_height=target_height,
                    voted_update_id=update.update_id
                )

                if target_height not in self._votes_by_height:
                    self._votes_by_height[target_height] = []
                self._votes_by_height[target_height].append(my_vote)

                print(f"[Node {self.node_id}] Self-voting for own update {update.update_id[:8]} at H={target_height}")
                vote_to_broadcast = my_vote

        # Broadcast vote outside the lock (FIX 2 consistency)
        if vote_to_broadcast is not None:
            await self.network_layer.broadcast_vote(vote_to_broadcast)

        # Ensure this node's consensus window is running
        await self._ensure_voting_task(target_height)

        # 5. Broadcast update to peers
        await self.network_layer.broadcast_update(update)

        # 6. Reset data so set_local_data can be called again next round
        self.__local_data = None
        self._data_ready_event.clear()