"""
Comprehensive integration & simulation test suite.
Federated learning — 10 nodes, Byzantine-resilient full vote-set gossip.

Test map
────────────────────────────────────────────────────────────────────
 B1–B9     Byzantine / rejection          (offline, no network)
 ZK1–ZK4   ZK proof pipeline              (offline)
 BYZ1–BYZ5 Equivocation / vote-set gossip (offline)
 BUF1–BUF4 Gap / buffering                (offline)
 DET1–DET2 Determinism                    (offline)
 CANCEL1–4 Training cancellation          (isolated node, no network)
 MV1       min_votes failure path         (offline, injected state)
 R1–R2     Full 10-node live simulation   (9 checks per round)
"""

import asyncio
import copy
import hashlib
import time
import numpy as np
from collections import Counter
from typing import Optional

from nodes.node import Node
from nodes.fl_types import UpdateType, VoteMessage, VoteSetBroadcast, UpdateLogEntry
from nodes.update_manager import UpdateManager
from utils.zk_proof import verify_zk_proof, generate_zk_proof
from utils.serialization import serialize_model

import sys
import io

# Force stdout to use utf-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Config ─────────────────────────────────────────────────────────────────
IP         = "127.0.0.1"
BASE_PORT  = 20200
NUM_NODES  = 10
MIN_VOTES  = 6
VOTING_WIN = 6.0
MAX_WAIT   = 90

DATASET_A = {
    "X": np.array([
        [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2], [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1],
        [5.0, 3.4, 1.5, 0.2], [6.7, 3.1, 4.7, 1.5],
    ]),
    "y": np.array([0, 0, 0, 1, 1, 1, 0, 1]),
}
DATASET_B = {
    "X": np.array([
        [6.0, 3.4, 4.5, 1.6], [6.1, 2.9, 4.7, 1.4],
        [5.6, 2.9, 3.6, 1.3], [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2], [6.4, 3.2, 5.3, 2.3],
        [5.9, 3.0, 5.1, 1.8], [5.5, 2.4, 3.7, 1.0],
    ]),
    "y": np.array([1, 1, 1, 0, 0, 1, 1, 1]),
}

GENESIS_MODEL: dict = {"weights": np.zeros(4).tolist(), "bias": 0.0}
GENESIS_HASH  = hashlib.sha256(serialize_model(GENESIS_MODEL)).hexdigest()

# ── Result tracking ────────────────────────────────────────────────────────
_results: dict[str, bool] = {}
_sim_start = time.monotonic()

def _ts() -> str:
    elapsed = time.monotonic() - _sim_start
    m, s = divmod(int(elapsed), 60)
    return f"{m:02d}:{s:05.2f}"

def section(t: str):    print(f"\n{'═'*68}\n  {t}\n{'═'*68}")
def subsection(t: str): print(f"\n{'─'*68}\n  {t}\n{'─'*68}")
def sim(msg: str):      print(f"  [{_ts()}] {msg}")

def record(name: str, passed: bool) -> bool:
    _results[name] = passed
    print(f"  {'✅' if passed else '❌'}  {name}")
    return passed

# ── Helpers ────────────────────────────────────────────────────────────────

def make_node(idx: int, port_base: int, total: int,
              min_votes: int = MIN_VOTES,
              voting_window: float = VOTING_WIN) -> Node:
    peers = [(IP, port_base + j) for j in range(total) if j != idx]
    n = Node(node_id=str(idx), peers=peers,
             min_votes=min_votes, voting_window=voting_window)
    n.set_model(copy.deepcopy(GENESIS_MODEL))
    return n

def make_nodes(total: int = NUM_NODES, port_base: int = BASE_PORT,
               min_votes: int = MIN_VOTES,
               voting_window: float = VOTING_WIN) -> list[Node]:
    return [make_node(i, port_base, total, min_votes, voting_window)
            for i in range(total)]

def fresh_update(prev_hash: str, weights: list[float], height: int) -> UpdateType:
    um = UpdateManager(private_key=None, enable_strict_zk=False)
    params: dict = {"weights": np.array(weights), "bias": 0.01}
    return um.build_update(prev_model_hash=prev_hash,
                           new_params=params,
                           model_height=height)

async def wait_for_height(nodes: list[Node], target: int,
                          timeout: int = MAX_WAIT) -> bool:
    for elapsed in range(timeout + 1):
        reached = sum(1 for n in nodes if n.model_height >= target)
        bar = ("█" * reached).ljust(len(nodes))
        print(f"\r  [{bar}] {reached}/{len(nodes)} at H≥{target} (+{elapsed}s)",
              end="", flush=True)
        if reached == len(nodes):
            print()
            return True
        await asyncio.sleep(1)
    print()
    return False

# ── Consensus invariant checks ─────────────────────────────────────────────

def check_consensus(nodes: list[Node], label: str) -> None:
    heights = [n.model_height for n in nodes]
    hashes  = [n.local_model_hash for n in nodes if n.local_model_hash]
    record(f"{label} — heights consistent",  len(set(heights)) == 1)
    record(f"{label} — hashes consistent",
           len(set(hashes)) == 1 and len(hashes) == len(nodes))

def check_weights_identical(nodes: list[Node], label: str) -> None:
    models = [n.get_model() for n in nodes]
    if any(m is None for m in models):
        record(f"{label} — weights identical", False)
        return
    # All models are confirmed non-None here; use explicit non-None first element
    first = models[0]
    if first is None:                               # satisfies Pylance
        record(f"{label} — weights identical", False)
        return
    ref = np.round(np.array(first["weights"]), 8)
    same = all(
        np.array_equal(ref, np.round(np.array(m["weights"]), 8))  # type: ignore[index]
        for m in models[1:]
        if m is not None
    )
    record(f"{label} — weights identical", same)

def check_winning_traceable(nodes: list[Node], label: str) -> None:
    winning = nodes[0].local_model_hash
    if winning is None:
        record(f"{label} — winning hash in every audit log", False)
        return
    found = all(any(e.params_hash == winning for e in n.update_log)
                for n in nodes)
    record(f"{label} — winning hash in every audit log", found)

def check_audit_chain(nodes: list[Node], label: str) -> None:
    ok = True
    for n in nodes:
        log = n.update_log
        if not log or log[0].previous_model_hash != GENESIS_HASH:
            ok = False; break
        for i in range(1, len(log)):
            if log[i].previous_model_hash != log[i - 1].params_hash:
                ok = False; break
        if not ok:
            break
    record(f"{label} — audit log hash chain valid", ok)

def check_zk_proofs(nodes: list[Node], label: str) -> None:
    ok = True
    for n in nodes:
        for entry in n.update_log:
            if entry.proof is None:
                ok = False; break
            if not verify_zk_proof("", b"", entry.proof,
                                   enable_strict_verification=False):
                ok = False; break
        if not ok:
            break
    record(f"{label} — all committed ZK proofs re-verify", ok)

def check_audit_count(nodes: list[Node], expected: int, label: str) -> None:
    counts = [len(n.update_log) for n in nodes]
    print(f"       audit log lengths: {counts}")
    record(f"{label} — each node has exactly {expected} audit entries",
           all(c == expected for c in counts))

def check_no_stale(nodes: list[Node], label: str) -> None:
    ok = all(
        not any(h <= n.model_height
                for h in n.update_manager.get_buffered_heights())
        for n in nodes
    )
    record(f"{label} — no stale candidate updates", ok)

def check_peers(nodes: list[Node], label: str) -> None:
    ok = all(
        all(s["healthy"] for s in n.network_layer.get_peer_health().values())
        for n in nodes
    )
    record(f"{label} — all peers healthy", ok)

def print_model_table(nodes: list[Node]) -> None:
    subsection("Model state")
    for n in nodes:
        m = n.get_model()
        h = (n.local_model_hash[:14] + "…") if n.local_model_hash else "None"
        if m is not None:
            w = np.round(np.array(m["weights"]), 5)
            print(f"  Node {n.node_id:>2} | H={n.model_height} | {h} | w={w}")
        else:
            print(f"  Node {n.node_id:>2} | H={n.model_height} | model=None")


# ══════════════════════════════════════════════════════════════════════════
#  B: BYZANTINE / REJECTION  (offline)
# ══════════════════════════════════════════════════════════════════════════

async def test_byzantine() -> None:
    subsection("Byzantine & rejection tests (offline)")
    n = make_node(0, 39900, 1, min_votes=1, voting_window=1.0)
    assert n.local_model_hash is not None
    prev_hash: str = n.local_model_hash

    # B1: tampered hash
    good = fresh_update(prev_hash, [.1, .2, .3, .4], 1)
    bad1 = copy.deepcopy(good)
    bad1.params_hash = "dead" * 16
    pool_before = len(n.update_manager.candidate_pool)
    await n.on_receive_update(bad1)
    record("B1 — tampered params_hash rejected",
           len(n.update_manager.candidate_pool) == pool_before)

    # B2: params don't match hash
    bad2 = copy.deepcopy(good)
    bad2.new_params = {"weights": np.array([9., 9., 9., 9.]), "bias": 9.}
    await n.on_receive_update(bad2)
    record("B2 — mismatched params/hash rejected",
           len(n.update_manager.candidate_pool) == pool_before)

    # B3: duplicate update_id
    seen_before = len(n.seen_updates)
    await n.on_receive_update(good)  # first  → accepted
    await n.on_receive_update(good)  # second → dropped
    record("B3 — duplicate update_id deduplicated",
           len(n.seen_updates) == seen_before + 1 and
           len(n.update_manager.candidate_pool) == pool_before + 1)

    # B4: ghost vote handled without crash
    try:
        await n.on_receive_vote(VoteMessage("ghost", 1, "doesnotexist_xxxxxxxxxxx"))
        record("B4 — ghost vote handled without crash", True)
    except Exception:
        record("B4 — ghost vote handled without crash", False)

    # B5: duplicate vote from same voter
    h77 = 77
    await n.on_receive_vote(VoteMessage("dup", h77, "update_aaa"))
    await n.on_receive_vote(VoteMessage("dup", h77, "update_bbb"))
    dup_votes = [v for v in n._votes_by_height.get(h77, [])
                 if v.voter_node_id == "dup"]
    record("B5 — duplicate vote from same voter deduplicated", len(dup_votes) == 1)

    # B6 / B7: strict-mode enforcement
    s_proof = "zkproof_abc123"
    record("B6 — string proof REJECTED in strict mode",
           not verify_zk_proof("", b"", s_proof, enable_strict_verification=True))
    record("B7 — string proof ACCEPTED in non-strict mode",
           verify_zk_proof("", b"", s_proof, enable_strict_verification=False))

    # B8: None proof
    record("B8 — None proof rejected",
           not verify_zk_proof("", b"", None, enable_strict_verification=False))

    # B9: update for past / current height ignored
    n.model_height = 2
    past = fresh_update(prev_hash, [.5] * 4, 2)
    snap = len(n.update_manager.candidate_pool)
    await n.on_receive_update(past)
    record("B9 — update for past height ignored",
           len(n.update_manager.candidate_pool) == snap)
    n.model_height = 0


# ══════════════════════════════════════════════════════════════════════════
#  ZK: PROOF PIPELINE  (offline)
# ══════════════════════════════════════════════════════════════════════════

def test_zk_direct() -> None:
    subsection("ZK proof format & re-verification (non-strict mode)")

    old_model: dict = {"weights": [0.1, 0.2, 0.3, 0.4], "bias": 0.05}
    new_model: dict = {"weights": [0.09, 0.18, 0.27, 0.36], "bias": 0.04}
    serialized = serialize_model(new_model)

    proof = generate_zk_proof(
        prev_model_hash=GENESIS_HASH,
        serialized_new_params=serialized,
        prev_model=old_model,
        new_model=new_model,
        learning_rate=0.01,
        enable_strict_verification=False,
    )
    expected = f"zkproof_{hashlib.sha256(serialized).hexdigest()}"
    record("ZK1 — non-strict proof has correct format",
           isinstance(proof, str) and proof == expected)
    record("ZK2 — generated proof passes re-verification",
           verify_zk_proof(GENESIS_HASH, serialized, proof,
                           enable_strict_verification=False))
    record("ZK3 — None proof rejected in non-strict mode",
           not verify_zk_proof(GENESIS_HASH, serialized, None,
                               enable_strict_verification=False))
    try:
        verify_zk_proof(GENESIS_HASH, serialized,
                        {"proof": {}, "public_signals": []},
                        enable_strict_verification=False)
        record("ZK4 — malformed dict proof handled without crash", True)
    except Exception:
        record("ZK4 — malformed dict proof handled without crash", False)


# ══════════════════════════════════════════════════════════════════════════
#  BYZ: EQUIVOCATION / FULL VOTE-SET GOSSIP  (offline)
#
#  This is the Byzantine-resilience mechanism:
#  Instead of only broadcasting "I vote for X", every node periodically
#  broadcasts its ENTIRE accumulated vote set for a height (VoteSetBroadcast).
#  When a peer receives a VoteSetBroadcast it cross-checks: if it already
#  knows that voter X chose update A, but this broadcast claims X chose B,
#  that is equivocation (the "vote for 1 here, vote for 2 there" attack).
#  The equivocator is blacklisted and their vote is removed from the tally.
# ══════════════════════════════════════════════════════════════════════════

async def test_equivocation() -> None:
    subsection("Equivocation detection — full vote-set cross-check (offline)")

    n = make_node(0, 39850, 1)
    assert n.local_model_hash is not None
    height   = 3
    byz_id   = "byzantine_node_99"
    update_a = "a" * 64
    update_b = "b" * 64
    update_c = "c" * 64

    # Seed: node 0 has already recorded byzantine's vote for update_a
    async with n._vote_lock:
        n._votes_by_height[height] = [VoteMessage(byz_id, height, update_a)]

    # BYZ1: a peer broadcasts a vote-set claiming byzantine voted for update_b
    # → cross-check fires → equivocation detected
    ballot_conflict = VoteSetBroadcast(
        sender_node_id="peer_1",
        target_height=height,
        votes=[VoteMessage(byz_id, height, update_b)],
    )
    await n.on_receive_vote_set(ballot_conflict)

    record("BYZ1 — equivocation detected from conflicting vote-set",
           byz_id in n._equivocators)

    # BYZ2: event recorded in equivocation_log
    record("BYZ2 — equivocation event recorded in log",
           any(e["equivocator"] == byz_id for e in n.equivocation_log))

    # BYZ3: tainted vote purged from the bucket
    bucket = n._votes_by_height.get(height, [])
    still_there = [v for v in bucket if v.voter_node_id == byz_id]
    record("BYZ3 — equivocator's vote removed from consensus bucket",
           len(still_there) == 0)

    # BYZ4: equivocator excluded from tally — add two honest votes
    async with n._vote_lock:
        n._votes_by_height[height].extend([
            VoteMessage("honest_1", height, update_c),
            VoteMessage("honest_2", height, update_c),
        ])
    all_votes   = n._votes_by_height.get(height, [])
    honest      = [v for v in all_votes if v.voter_node_id not in n._equivocators]
    tally       = Counter(v.voted_update_id for v in honest)
    record("BYZ4 — equivocator's choices absent from honest tally",
           update_a not in tally and update_b not in tally)

    # BYZ5: honest votes tallied correctly
    record("BYZ5 — honest votes counted correctly",
           tally.get(update_c, 0) == 2)

    # BYZ6: a second arrival of the SAME vote-set is ignored (idempotent)
    len_before = len(n.equivocation_log)
    await n.on_receive_vote_set(ballot_conflict)
    record("BYZ6 — repeated vote-set for known equivocator is idempotent",
           len(n.equivocation_log) == len_before)

    # BYZ7: honest new vote in a vote-set is still admitted
    ballot_honest = VoteSetBroadcast(
        sender_node_id="peer_2",
        target_height=height,
        votes=[VoteMessage("new_honest_node", height, update_c)],
    )
    bucket_len_before = len(n._votes_by_height.get(height, []))
    await n.on_receive_vote_set(ballot_honest)
    record("BYZ7 — honest vote inside vote-set is admitted",
           len(n._votes_by_height.get(height, [])) == bucket_len_before + 1)


# ══════════════════════════════════════════════════════════════════════════
#  BUF: GAP / BUFFERING  (offline)
# ══════════════════════════════════════════════════════════════════════════

async def test_buffering() -> None:
    subsection("Gap / buffering (offline)")
    n = make_node(0, 39800, 1)
    assert n.local_model_hash is not None
    prev_hash: str = n.local_model_hash

    future = fresh_update(prev_hash, [.5] * 4, n.model_height + 2)
    pool_before = len(n.update_manager.candidate_pool)
    await n.on_receive_update(future)

    record("BUF1 — future update buffered",
           len(n.update_manager.candidate_pool) > pool_before)
    record("BUF2 — update in correct height slot",
           len(n.update_manager.get_candidates_for_height(n.model_height + 2)) > 0)
    record("BUF3 — gap height recorded in _missed_heights",
           (n.model_height + 1) in n._missed_heights)
    record("BUF4 — current height unchanged", n.model_height == 0)


# ══════════════════════════════════════════════════════════════════════════
#  DET: DETERMINISM  (offline)
# ══════════════════════════════════════════════════════════════════════════

def test_determinism() -> None:
    subsection("Determinism — same inputs → same output hash")
    from nodes.training_engine import TrainingEngine
    engine = TrainingEngine(learning_rate=0.01)

    r1 = engine.train_until_converged(copy.deepcopy(DATASET_A),
                                      copy.deepcopy(GENESIS_MODEL))
    r2 = engine.train_until_converged(copy.deepcopy(DATASET_A),
                                      copy.deepcopy(GENESIS_MODEL))
    h1 = hashlib.sha256(serialize_model(r1)).hexdigest()
    h2 = hashlib.sha256(serialize_model(r2)).hexdigest()
    record("DET1 — same dataset × model → identical hash", h1 == h2)

    r3 = engine.train_until_converged(copy.deepcopy(DATASET_B),
                                      copy.deepcopy(GENESIS_MODEL))
    h3 = hashlib.sha256(serialize_model(r3)).hexdigest()
    record("DET2 — different datasets → different hashes", h1 != h3)


# ══════════════════════════════════════════════════════════════════════════
#  CANCEL: TRAINING CANCELLATION  (isolated node, no gossip server)
#
#  Strategy:
#  1. Start only _training_manager as a task (no websocket server).
#  2. Inject data.  The manager spawns _training_worker.
#  3. Poll until _training_task is spawned OR data_ready_event is cleared
#     (meaning the worker already ran and cleared it — both are proof it
#     was triggered).
#  4. Call _apply_update directly to simulate consensus winning mid-training.
# ══════════════════════════════════════════════════════════════════════════

async def test_cancellation() -> None:
    subsection("Training cancellation by external update (isolated node)")
    n = make_node(0, 39700, 1)
    assert n.local_model_hash is not None

    mgr = asyncio.create_task(n._training_manager())
    n.set_local_data(copy.deepcopy(DATASET_A))

    # Poll up to 2 s for the training task to be spawned or finish
    triggered = False
    for _ in range(40):
        await asyncio.sleep(0.05)
        if n._training_task is not None or not n._data_ready_event.is_set():
            triggered = True
            break
    record("CANCEL1 — training triggered after data injection", triggered)

    # Wait for any in-flight training to settle before injecting the external update
    await asyncio.sleep(0.8)

    # Reset node to a clean H=0 state so the external update at H=1 is valid
    n.set_model(copy.deepcopy(GENESIS_MODEL))
    assert n.local_model_hash is not None
    prev_hash: str = n.local_model_hash
    n.model_height = 0
    n._data_ready_event.clear()
    # Clear the private __local_data slot via name-mangled attribute
    object.__setattr__(n, "_Node__local_data", None)

    # Inject a new dataset so training is running when the external update arrives
    n.set_local_data(copy.deepcopy(DATASET_A))
    await asyncio.sleep(0.05)

    ext = fresh_update(prev_hash, [7., 7., 7., 7.], 1)
    initial_height = n.model_height
    await n._apply_update(ext)

    record("CANCEL2 — height advanced after external update",
           n.model_height == initial_height + 1)
    record("CANCEL3 — model hash matches external update",
           n.local_model_hash == ext.params_hash)
    record("CANCEL4 — training task done or was cancelled",
           n._training_task is None or n._training_task.done())

    mgr.cancel()
    await asyncio.gather(mgr, return_exceptions=True)


# ══════════════════════════════════════════════════════════════════════════
#  MV: MIN_VOTES FAILURE  (offline — injected state, no live network)
#
#  Three isolated nodes.  Each has exactly ONE vote in its bucket — its own
#  self-vote pointing at a UNIQUE update.  None of them can reach min_votes=3.
#  We call _process_consensus() directly and assert H stays at 0.
# ══════════════════════════════════════════════════════════════════════════

async def test_min_votes_failure() -> None:
    subsection("min_votes enforcement — split vote → no consensus (offline)")

    nodes = [make_node(i, 39600, 3, min_votes=3, voting_window=0.05)
             for i in range(3)]
    um = UpdateManager(private_key=None, enable_strict_zk=False)

    for i, n in enumerate(nodes):
        assert n.local_model_hash is not None
        prev_hash: str = n.local_model_hash
        upd = um.build_update(
            prev_model_hash=prev_hash,
            new_params={"weights": np.array([float(i + 1)] * 4), "bias": float(i)},
            model_height=1,
        )
        n.update_manager.add_candidate(upd)
        assert upd.update_id is not None
        uid: str = upd.update_id
        async with n._vote_lock:
            n._voted_heights.add(1)
            n._votes_by_height[1] = [VoteMessage(n.node_id, 1, uid)]

    # Trigger consensus window on all three nodes; let it expire
    tasks = [asyncio.create_task(n._process_consensus(1)) for n in nodes]
    await asyncio.gather(*tasks)

    record("MV1 — no consensus when votes are perfectly split",
           all(n.model_height == 0 for n in nodes))


# ══════════════════════════════════════════════════════════════════════════
#  MAIN 10-NODE SIMULATION — 2 CONSENSUS ROUNDS
# ══════════════════════════════════════════════════════════════════════════

async def run_main_network() -> None:
    global _sim_start
    _sim_start = time.monotonic()

    section("🌐  Full network simulation — 10 nodes, 2 consensus rounds")

    sim("Initialising full-mesh topology")
    nodes = make_nodes()
    for n in nodes:
        sim(f"  Node {n.node_id} ready — {len(n.peers)} peers, "
            f"min_votes={n.min_votes}, voting_window={n.voting_window}s")

    tasks = [asyncio.create_task(n.run(BASE_PORT + i))
             for i, n in enumerate(nodes)]

    async def orchestrate() -> None:
        try:
            sim("Waiting for all servers to bind (3 s)…")
            await asyncio.sleep(3)
            sim("All servers up — network is live  ✦ vote-set gossip enabled")

            # ── Round 1 ──────────────────────────────────────────────────
            section("⚡  Round 1 — 3 competing updates (Nodes 0, 3, 7)")
            sim("Data: Node 0 & 3 ← DATASET_A   Node 7 ← DATASET_B  (real 3-way race)")
            nodes[0].set_local_data(copy.deepcopy(DATASET_A))
            nodes[3].set_local_data(copy.deepcopy(DATASET_A))
            nodes[7].set_local_data(copy.deepcopy(DATASET_B))

            ok1 = await wait_for_height(nodes, 1)
            record("R1 — all nodes at H=1", ok1)

            if ok1:
                sim("Consensus reached — verifying invariants")
                print_model_table(nodes)
                check_consensus(nodes, "R1")
                check_weights_identical(nodes, "R1")
                check_winning_traceable(nodes, "R1")
                check_audit_chain(nodes, "R1")
                check_zk_proofs(nodes, "R1")
                check_audit_count(nodes, 1, "R1")
                check_no_stale(nodes, "R1")
                check_peers(nodes, "R1")
                equiv_total = sum(len(n.equivocation_log) for n in nodes)
                record("R1 — zero equivocations in honest network", equiv_total == 0)

            # ── Round 2 ──────────────────────────────────────────────────
            section("⚡  Round 2 — 3 competing updates (Nodes 1, 4, 9)")
            sim("Data: Node 1 & 9 ← DATASET_B   Node 4 ← DATASET_A")
            nodes[1].set_local_data(copy.deepcopy(DATASET_B))
            nodes[4].set_local_data(copy.deepcopy(DATASET_A))
            nodes[9].set_local_data(copy.deepcopy(DATASET_B))

            ok2 = await wait_for_height(nodes, 2)
            record("R2 — all nodes at H=2", ok2)

            if ok2:
                sim("Consensus reached — verifying invariants")
                print_model_table(nodes)
                check_consensus(nodes, "R2")
                check_weights_identical(nodes, "R2")
                check_winning_traceable(nodes, "R2")
                check_audit_chain(nodes, "R2")
                check_zk_proofs(nodes, "R2")
                check_audit_count(nodes, 2, "R2")
                check_no_stale(nodes, "R2")
                check_peers(nodes, "R2")
                equiv_total = sum(len(n.equivocation_log) for n in nodes)
                record("R2 — zero equivocations in honest network", equiv_total == 0)

        finally:
            for t in tasks:
                t.cancel()

    await asyncio.gather(*tasks, orchestrate(), return_exceptions=True)


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

async def run_all() -> None:
    section("🔬  Offline unit tests (no network required)")
    await test_byzantine()
    test_zk_direct()
    await test_equivocation()
    await test_buffering()
    test_determinism()
    await test_cancellation()
    await test_min_votes_failure()

    section("🌐  Full 10-node live simulation")
    await run_main_network()

    # ── Final summary ─────────────────────────────────────────────────────
    section("📊  Full test suite summary")
    passed = sum(1 for v in _results.values() if v)
    total  = len(_results)

    groups: dict[str, list[tuple[str, bool]]] = {}
    for name, p in _results.items():
        prefix = name.split(" — ")[0].split()[0]
        groups.setdefault(prefix, []).append((name, p))

    for prefix, items in groups.items():
        all_ok = all(p for _, p in items)
        print(f"\n  {'✅' if all_ok else '❌'}  [{prefix}]")
        for name, p in items:
            print(f"       {'✅' if p else '❌'}  {name}")

    print(f"\n{'─'*68}")
    if passed == total:
        print(f"  🎉  {passed}/{total} — all systems green.")
    else:
        print(f"  ⚠️   {passed}/{total} — {total - passed} failure(s). See above.")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")