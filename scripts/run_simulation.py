#!/usr/bin/env python3
"""Run the protocol live: n nodes, real messages, real (optional) proofs.

Two transports, and the difference matters for what a run is evidence of:

  ``--transport memory``
      Every node in one process over an in-memory bus with injected one-way
      latency, jitter and loss.  Cheap enough to sweep n and to run in CI, and
      the injected delay is what stops a localhost run from silently assuming
      a zero-latency network -- the flaw in the original evaluation, where a
      fixed 6-second window always sufficed because propagation was free.

  ``--transport websocket``
      One asyncio task per node, each with its own listening socket, talking
      over real TCP with real serialisation.  Slower and bounded by file
      descriptors, so it is the honesty check on the in-memory numbers rather
      than the way to reach n = 50.

Examples::

    python scripts/run_simulation.py --nodes 10 --rounds 5
    python scripts/run_simulation.py --nodes 7 --byzantine equivocating-voter:6
    python scripts/run_simulation.py --nodes 4 --rounds 1 --strict     # real proofs
    python scripts/run_simulation.py --nodes 5 --transport websocket
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zkfl.config import FixedPointConfig, ProtocolConfig      # noqa: E402
from zkfl.datasets import breast_cancer_binary, iris_binary, synthetic  # noqa: E402
from zkfl.node import EquivocatingProposer, EquivocatingVoter, Flooder, Node  # noqa: E402
from zkfl.simulation import Simulation                        # noqa: E402

ADVERSARIES = {
    "equivocating-voter": EquivocatingVoter,
    "equivocating-proposer": EquivocatingProposer,
    "flooder": Flooder,
    "honest": Node,
}


def dataset(name: str, n_nodes: int, n_features: int, seed: int):
    if name == "iris":
        return iris_binary(seed=seed)
    if name == "breast-cancer":
        return breast_cancer_binary(seed=seed, n_features=n_features)
    return synthetic(n_samples=64 * n_nodes, n_features=n_features, seed=seed)


def parse_byzantine(specs: list[str], n_nodes: int) -> dict[int, type]:
    """``--byzantine flooder:0 equivocating-voter:9`` -> {0: Flooder, ...}."""
    out: dict[int, type] = {}
    for spec in specs:
        kind, _, index = spec.partition(":")
        if kind not in ADVERSARIES:
            raise SystemExit(f"unknown adversary {kind!r}; "
                             f"choose from {', '.join(sorted(ADVERSARIES))}")
        if not index.isdigit() or not 0 <= int(index) < n_nodes:
            raise SystemExit(f"{spec!r}: index must be 0..{n_nodes - 1}")
        out[int(index)] = ADVERSARIES[kind]
    return out


async def run_memory(args) -> int:
    fp = FixedPointConfig(n_features=args.features)
    protocol = ProtocolConfig(
        collect_window=args.window, strict_zk=args.strict,
        aggregation_rule=args.rule, learning_rate=args.lr,
    ).with_nodes(args.nodes)
    protocol.check()

    byz = parse_byzantine(args.byzantine, args.nodes)
    if len(byz) > protocol.f_byzantine:
        print(f"note: {len(byz)} adversaries exceeds f={protocol.f_byzantine}; "
              f"no guarantee holds and none is claimed")

    sim = Simulation(
        split=dataset(args.dataset, args.nodes, args.features, args.seed),
        fp=fp, protocol=protocol, n_nodes=args.nodes, strict_zk=args.strict,
        byzantine=byz, latency=args.latency, jitter=args.jitter, loss=args.loss,
        seed=args.seed, iid=not args.non_iid,
    )
    for note in sim.notes:
        print(note)

    started = time.perf_counter()
    result = await sim.run(rounds=args.rounds, proposers_per_round=args.proposers,
                           timeout=args.timeout, evaluate=not args.no_eval)
    elapsed = time.perf_counter() - started

    print()
    header = f"{'h':>3}  {'agreed':>6}  {'props':>5}  {'msgs':>6}  {'bytes':>8}  {'s':>6}"
    if not args.no_eval:
        header += f"  {'acc':>6}  {'loss':>7}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(result.rounds, start=1):
        line = (f"{i:>3}  {'yes' if r.agreed else 'NO':>6}  {r.proposals:>5}  "
                f"{r.messages:>6}  {r.bytes_sent:>8}  {r.seconds:>6.2f}")
        if not args.no_eval:
            line += f"  {r.accuracy:>6.3f}  {r.loss:>7.4f}"
        print(line)

    summary = result.summary()
    print()
    for note in sim.notes[len(sim.notes) - 1:]:
        print(note)
    print(json.dumps(summary, indent=2))

    failures = [name for name, ok, _ in sim.check_invariants() if not ok]
    for name, ok, detail in sim.check_invariants():
        if not ok:
            print(f"INVARIANT VIOLATED  {name}: {detail}")
    print(f"\n{args.rounds} heights in {elapsed:.2f}s "
          f"({elapsed / max(args.rounds, 1):.2f}s per height)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"summary": summary, "notes": sim.notes,
             "rounds": [r.__dict__ for r in result.rounds]}, indent=2, default=str))
        print(f"wrote {args.out}")

    agreed = sum(1 for r in result.rounds if r.agreed)
    return 0 if agreed == args.rounds and not failures else 1


async def run_websocket(args) -> int:
    """The same protocol over real sockets.

    Built by hand rather than through :class:`Simulation` because the point is
    to exercise the transport the simulator deliberately replaces.
    """
    from zkfl.crypto import Identity, PeerRegistry
    from zkfl.datasets import partition
    from zkfl.model import Model
    from zkfl.network import WebSocketTransport
    from zkfl.trainer import LocalDataset
    from zkfl.zk import make_backend

    fp = FixedPointConfig(n_features=args.features)
    protocol = ProtocolConfig(
        collect_window=args.window, strict_zk=args.strict,
        aggregation_rule=args.rule, learning_rate=args.lr,
    ).with_nodes(args.nodes)
    protocol.check()

    split = dataset(args.dataset, args.nodes, args.features, args.seed)
    parts = partition(split, args.nodes, fp, iid=not args.non_iid, seed=args.seed)
    identities = [Identity.from_seed(bytes([i % 251, 7, i // 251]) + b"\x00" * 29)
                  for i in range(args.nodes)]
    ports = [args.base_port + i for i in range(args.nodes)]

    registry = PeerRegistry()
    datasets = []
    for identity, part in zip(identities, parts):
        ds = (LocalDataset(part.x.tolist(), part.y.tolist(), fp)
              if len(part.x) >= fp.batch_size else None)
        datasets.append(ds)
        root, n = (ds.root, ds.n_samples) if ds else (0, 1)
        registry.register(identity.node_id, root, n, identity.sign(
            "registration", PeerRegistry.registration_payload(identity.node_id, root, n)))

    genesis = Model.zeros(fp)
    backend = make_backend(fp, args.strict)
    nodes = []
    for i, (identity, ds) in enumerate(zip(identities, datasets)):
        peers = {other.node_id: ("127.0.0.1", ports[j])
                 for j, other in enumerate(identities) if j != i}
        # The websocket runner is the closest thing here to a deployment, so it
        # gets the durable vote log: a node restarted after a crash must not
        # sign a second vote in a slot it already voted in.
        nodes.append(Node(
            identity=identity, transport=WebSocketTransport(identity.node_id, ports[i], peers),
            registry=registry, backend=backend, genesis=genesis, fp=fp,
            protocol=protocol, dataset=ds, label=f"n{i}",
            vote_log_path=(Path(args.state_dir) / f"n{i}" / "votes.log"
                           if args.state_dir else None),
        ))

    for node in nodes:
        await node.start()
    print(f"{args.nodes} nodes listening on ports "
          f"{ports[0]}..{ports[-1]}; q={protocol.effective_quorum()}")

    try:
        for height in range(1, args.rounds + 1):
            started = time.perf_counter()
            for node in nodes:
                await node.open_round()
            proposers = [n for n in nodes if n.dataset][:args.proposers]
            await asyncio.gather(*(n.train_and_propose(height - 1) for n in proposers))

            deadline = time.perf_counter() + args.timeout
            while time.perf_counter() < deadline:
                if all(n.height >= height for n in nodes):
                    break
                await asyncio.sleep(0.05)

            reached = sum(1 for n in nodes if n.height >= height)
            digests = {n.model.digest() for n in nodes if n.height >= height}
            print(f"h={height}  committed {reached}/{args.nodes}  "
                  f"distinct models {len(digests)}  "
                  f"{time.perf_counter() - started:.2f}s")
            if len(digests) > 1:
                print("  SAFETY VIOLATION: nodes committed different models")
                return 1
    finally:
        for node in nodes:
            await node.stop()

    return 0 if all(n.height >= args.rounds for n in nodes) else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--proposers", type=int, default=3,
                    help="how many nodes train and propose each height")
    ap.add_argument("--transport", choices=("memory", "websocket"), default="memory")
    ap.add_argument("--state-dir", default=None,
                    help="write each node's votes to a durable write-ahead log "
                         "under this directory, so a restarted node cannot sign "
                         "a second vote in a slot it already voted in")
    ap.add_argument("--dataset", choices=("synthetic", "iris", "breast-cancer"),
                    default="synthetic")
    ap.add_argument("--features", type=int, default=4)
    ap.add_argument("--strict", action="store_true",
                    help="real Groth16 proofs; requires artifacts/ (see the Makefile)")
    ap.add_argument("--rule", default="trimmed_mean",
                    choices=("trimmed_mean", "median", "mean"))
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--window", type=float, default=0.3,
                    help="base collect window in seconds; it doubles per sub-round")
    ap.add_argument("--latency", type=float, default=0.005,
                    help="injected one-way delay, seconds (memory transport)")
    ap.add_argument("--jitter", type=float, default=0.005)
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--non-iid", action="store_true",
                    help="label-skewed partition instead of iid")
    ap.add_argument("--byzantine", nargs="*", default=[], metavar="KIND:INDEX",
                    help=f"e.g. flooder:0; kinds: {', '.join(sorted(ADVERSARIES))}")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base-port", type=int, default=9800)
    ap.add_argument("--no-eval", action="store_true",
                    help="skip test-set evaluation between heights")
    ap.add_argument("--out", help="write the run summary to this JSON file")
    args = ap.parse_args()

    runner = run_websocket if args.transport == "websocket" else run_memory
    return asyncio.run(runner(args))


if __name__ == "__main__":
    raise SystemExit(main())
