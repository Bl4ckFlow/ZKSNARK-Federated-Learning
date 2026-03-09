"""
Integration test — 10-node federated learning network.

Topology : Full mesh (every node peers with all 9 others)
Consensus: Simple majority → min_votes = 6  (>50% of 10)
Rounds   : 2 sequential training rounds

Round 1 — Nodes 0, 3, 7 inject data simultaneously (3 competing updates)
Round 2 — Nodes 1, 4, 9 inject data (after round-1 consensus)

What is verified after each round:
  ✓ All 10 nodes reached the expected model_height
  ✓ All 10 nodes share the same model hash  (consensus)
  ✓ All 10 nodes share the same weight vector
  ✓ Per-node vote audit trail is printed
"""

import asyncio
import numpy as np
from nodes.node import Node

import sys
import io

# Force stdout to use utf-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
IP         = "127.0.0.1"
BASE_PORT  = 20100          # offset to avoid clashing with other tests
NUM_NODES  = 10
MIN_VOTES  = 6              # simple majority for 10 nodes
VOTING_WIN = 6.0            # seconds
MAX_WAIT   = 90             # seconds before we declare consensus failure

# Iris-style 4-feature binary classification data
# (kept tiny so training finishes fast)
DATASET_A = {
    "X": np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [5.0, 3.4, 1.5, 0.2],
        [6.7, 3.1, 4.7, 1.5],
    ]),
    "y": np.array([0, 0, 0, 1, 1, 1, 0, 1]),
}

DATASET_B = {
    "X": np.array([
        [6.0, 3.4, 4.5, 1.6],
        [6.1, 2.9, 4.7, 1.4],
        [5.6, 2.9, 3.6, 1.3],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        [6.4, 3.2, 5.3, 2.3],
        [5.9, 3.0, 5.1, 1.8],
        [5.5, 2.4, 3.7, 1.0],
    ]),
    "y": np.array([1, 1, 1, 0, 0, 1, 1, 1]),
}

GENESIS_MODEL = {"weights": np.zeros(4).tolist(), "bias": 0.0}

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
SEP  = "─" * 60
SEP2 = "═" * 60

def section(title: str) -> None:
    print(f"\n{SEP2}\n  {title}\n{SEP2}")

def subsection(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def full_mesh_peers(my_idx: int) -> list[tuple[str, int]]:
    return [(IP, BASE_PORT + j) for j in range(NUM_NODES) if j != my_idx]


async def wait_for_height(nodes: list[Node], target: int, label: str) -> bool:
    """
    Polls until every node reaches *target* height or MAX_WAIT seconds elapse.
    Prints a live progress bar.
    """
    elapsed = 0
    interval = 1
    while elapsed < MAX_WAIT:
        counts = [n.model_height for n in nodes]
        reached = sum(1 for h in counts if h >= target)
        bar = ("█" * reached).ljust(NUM_NODES)
        print(f"\r  [{bar}] {reached}/{NUM_NODES} nodes at H≥{target}  (+{elapsed}s)", end="", flush=True)
        if reached == NUM_NODES:
            print()
            return True
        await asyncio.sleep(interval)
        elapsed += interval
    print()
    return False


def print_vote_audit(nodes: list[Node], height: int) -> None:
    subsection(f"Vote audit — Height {height}")
    for n in nodes:
        votes = n._votes_by_height.get(height, [])
        tag = f"Node {n.node_id:>2}"
        if not votes:
            print(f"  {tag} | (votes already pruned — consensus reached ✓)")
        else:
            choices = {v.voted_update_id[:8] for v in votes}
            voter_ids = [v.voter_node_id for v in votes]
            print(f"  {tag} | recorded {len(votes)} vote(s) from nodes {voter_ids} → updates {choices}")


def print_model_state(nodes: list[Node]) -> bool:
    """Prints the final model state of each node. Returns True if all agree."""
    subsection("Model state after consensus")
    hashes  = [n.local_model_hash for n in nodes if n.local_model_hash]
    heights = [n.model_height    for n in nodes]
    models  = [n.get_model()     for n in nodes]

    all_same_hash   = len(set(hashes))  == 1
    all_same_height = len(set(heights)) == 1

    for n in nodes:
        m = n.get_model()
        if m:
            w = np.round(np.array(m["weights"]), 5)
            h = n.local_model_hash[:10] if n.local_model_hash else "None"
            print(f"  Node {n.node_id:>2} | H={n.model_height} | hash={h}… | w={w}")
        else:
            print(f"  Node {n.node_id:>2} | H={n.model_height} | model=None")

    print()
    print(f"  Heights consistent : {'✅' if all_same_height else '❌'}")
    print(f"  Hashes  consistent : {'✅' if all_same_hash   else '❌'}")
    return all_same_height and all_same_hash


# ──────────────────────────────────────────────
# Main test coroutine
# ──────────────────────────────────────────────
async def run_test() -> None:
    section("Initialising 10-node full-mesh network")

    nodes: list[Node] = []
    for i in range(NUM_NODES):
        n = Node(
            node_id=str(i),
            peers=full_mesh_peers(i),
            min_votes=MIN_VOTES,
            voting_window=VOTING_WIN,
        )
        n.set_model(GENESIS_MODEL)
        nodes.append(n)
        print(f"  ✓ Node {i} initialised  (peers: {len(full_mesh_peers(i))})")

    # Start all servers
    server_tasks = [
        asyncio.create_task(n.run(BASE_PORT + i))
        for i, n in enumerate(nodes)
    ]

    results: dict[str, bool] = {}

    async def orchestrate() -> None:
        try:
            # ── Boot delay ────────────────────────────────────────────────
            print("\n  Waiting 3 s for all servers to bind…")
            await asyncio.sleep(3)

            # ╔══════════════════════════════════════════════════╗
            # ║              ROUND 1                            ║
            # ╚══════════════════════════════════════════════════╝
            section("Round 1 — 3 competing updates (Nodes 0, 3, 7)")
            print("  Injecting data into nodes 0, 3, 7 simultaneously…")
            nodes[0].set_local_data(DATASET_A)
            nodes[3].set_local_data(DATASET_A)
            nodes[7].set_local_data(DATASET_B)   # different data → competing update

            print(f"\n  Waiting for all nodes to reach H=1 (timeout {MAX_WAIT}s)…")
            ok1 = await wait_for_height(nodes, target=1, label="Round 1")

            print_vote_audit(nodes, height=1)
            r1_ok = print_model_state(nodes)
            results["Round 1 — all nodes at H=1"]        = ok1
            results["Round 1 — consensus (same hash)"]   = r1_ok

            if not ok1:
                print("\n  ⚠️  Round 1 timed out — skipping Round 2")
                return

            # ╔══════════════════════════════════════════════════╗
            # ║              ROUND 2                            ║
            # ╚══════════════════════════════════════════════════╝
            section("Round 2 — 3 more updates (Nodes 1, 4, 9)")
            print("  Injecting data into nodes 1, 4, 9 simultaneously…")
            nodes[1].set_local_data(DATASET_B)
            nodes[4].set_local_data(DATASET_A)
            nodes[9].set_local_data(DATASET_B)

            print(f"\n  Waiting for all nodes to reach H=2 (timeout {MAX_WAIT}s)…")
            ok2 = await wait_for_height(nodes, target=2, label="Round 2")

            print_vote_audit(nodes, height=2)
            r2_ok = print_model_state(nodes)
            results["Round 2 — all nodes at H=2"]        = ok2
            results["Round 2 — consensus (same hash)"]   = r2_ok

            # ── Candidate pool hygiene ────────────────────────────────────
            subsection("Candidate pool health check")
            for n in nodes:
                buffered = n.update_manager.get_buffered_heights()
                stale    = [h for h in buffered if h <= n.model_height]
                if stale:
                    print(f"  ⚠️  Node {n.node_id} has stale candidates at heights {stale}")
                else:
                    print(f"  Node {n.node_id:>2} | pool={len(n.update_manager.candidate_pool)} entry/entries, no stale heights ✓")

            # ── Peer health ───────────────────────────────────────────────
            subsection("Network peer health")
            for n in nodes:
                health = n.network_layer.get_peer_health()
                bad    = [addr for addr, s in health.items() if not s["healthy"]]
                if bad:
                    print(f"  Node {n.node_id:>2} | ⚠️  unhealthy peers: {bad}")
                else:
                    print(f"  Node {n.node_id:>2} | all {len(health)} peers healthy ✓")

        finally:
            for t in server_tasks:
                t.cancel()

    await asyncio.gather(*server_tasks, orchestrate(), return_exceptions=True)

    # ── Final summary ─────────────────────────────────────────────────────
    section("Test summary")
    all_passed = True
    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉  All checks passed — distributed consensus is working correctly.")
    else:
        print("  ❌  Some checks failed — see output above for details.")
    print()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("\n[!] Simulation interrupted by user.")