#!/usr/bin/env python3
"""E3 -- what consensus costs, and how it behaves on a network that isn't ideal.

Two sweeps:

  * **scale** -- round latency, messages and bytes at n in {5, 10, 20, 50}.
    The old design re-broadcast an O(n)-sized vote set on every new vote, which
    is O(n^3) bytes per round; individual signed votes plus one certificate is
    O(n^2).  The measured growth is the evidence.

  * **adversity** -- injected one-way delay and loss at fixed n.  The original
    live run was 10 processes on loopback, where propagation delay is
    effectively zero, so a fixed 6-second window always sufficed.  Here the
    window is adaptive and the round-failure rate under delay and loss is
    reported rather than assumed away.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
from dataclasses import replace

from experiments.common import save, table
from zkfl.config import FixedPointConfig, ProtocolConfig
from zkfl.datasets import synthetic
from zkfl.simulation import Simulation


async def one(n_nodes, rounds, latency, jitter, loss, seed, proposers,
              timeout=60, max_sub_rounds=None):
    fp = FixedPointConfig()
    protocol = ProtocolConfig(collect_window=0.2, strict_zk=False,
                              aggregation_rule="trimmed_mean",
                              learning_rate=0.5).with_nodes(n_nodes)
    if max_sub_rounds is not None:
        protocol = replace(protocol, max_sub_rounds=max_sub_rounds)
    split = synthetic(n_samples=64 * n_nodes, n_features=fp.n_features, seed=seed)
    sim = Simulation(split=split, fp=fp, protocol=protocol, n_nodes=n_nodes,
                     strict_zk=False, latency=latency, jitter=jitter, loss=loss,
                     seed=seed)
    result = await sim.run(rounds=rounds,
                           proposers_per_round=min(proposers, n_nodes),
                           timeout=timeout, evaluate=False)
    agreed = [r for r in result.rounds if r.agreed]
    return {
        "n": n_nodes,
        "f": protocol.f_byzantine,
        "q": protocol.effective_quorum(),
        "latency_ms": round(latency * 1000, 1),
        "loss": loss,
        "rounds": rounds,
        "rounds_agreed": len(agreed),
        "round_failure_rate": round(1 - len(agreed) / rounds, 3),
        "median_round_s": round(statistics.median(r.seconds for r in result.rounds), 3),
        # Two totals, and the difference matters.  ``messages_per_round`` is the
        # *median* height; ``deliveries_per_round`` is the mean, and it is the
        # one the decomposition below is on.  Mixing them is how a table stops
        # adding up: at zero loss and n <= 20 every height is identical so the
        # two agree, but at n=50 a retried sub-round in one height pulls the
        # mean above the median, and a median total beside mean components is
        # a decomposition that does not sum.  Everything reported per kind, and
        # every total quoted next to it, is the mean.
        "messages_per_round": int(statistics.median(r.messages for r in result.rounds)),
        "deliveries_per_round": round(sim.bus.stats()["delivered"] / rounds, 1),
        "bytes_per_round": int(statistics.median(r.bytes_sent for r in result.rounds)),
        # A single total is indistinguishable from a formula: at zero loss this
        # one lands exactly on (n-1)(p + 4n) at every size, because every node
        # votes once per phase, gossips one polka and one certificate, and
        # nothing is retried.  The breakdown is what makes it a measurement
        # rather than an assertion, and under loss it stops matching.
        "deliveries_per_round_by_kind": {
            kind: round(count / rounds, 1)
            for kind, count in sim.bus.stats()["delivered_by_kind"].items()
        },
        "deliveries_total": sim.bus.stats()["delivered"],
        "deliveries_dropped": sim.bus.stats()["dropped"],
        # (n-1)(p + 4n): p proposals, a prevote and a precommit per node, one
        # gossiped polka per node, one gossiped certificate per node.
        "closed_form": (n_nodes - 1) * (proposers + 4 * n_nodes),
        "bus": sim.bus.stats(),
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=5)
    # More samples for the adversity sweep: it runs at n=10, so it is cheap,
    # and a failure rate estimated from five rounds moves by 0.2 whenever one
    # round goes the other way.  Five samples is not a rate.
    ap.add_argument("--adversity-rounds", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3,
                    help="independent seeds per adversity configuration")
    ap.add_argument("--adversity-timeout", type=float, default=15.0,
                    help="per-height cap in the adversity sweep, seconds")
    ap.add_argument("--sub-round-sweep", action="store_true",
                    help="also sweep max_sub_rounds under loss")
    ap.add_argument("--sub-round-budgets", type=int, nargs="*",
                    default=[4, 8, 16])
    ap.add_argument("--sub-round-losses", type=float, nargs="*",
                    default=[0.05, 0.20],
                    help="loss rates to sweep the budget at.  The 20%% rows "
                         "cost an hour and say what the adversity sweep "
                         "already says -- nothing commits at any budget -- so "
                         "a re-measurement can drop them with `0.05`")
    ap.add_argument("--sizes", type=int, nargs="*", default=[5, 10, 20, 50])
    ap.add_argument("--proposers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=== scale sweep (5 ms one-way delay, no loss) ===")
    scale = []
    for n in args.sizes:
        record = await one(n, args.rounds, 0.005, 0.002, 0.0, args.seed,
                           args.proposers)
        scale.append(record)
        print(f"  n={n:<3} committed {record['rounds_agreed']}/{args.rounds} "
              f"median {record['median_round_s']:.2f}s "
              f"{record['messages_per_round']} msgs "
              f"{record['bytes_per_round'] / 1024:.0f} KB")

    base = scale[0]
    rows = [[r["n"], r["q"], r["rounds_agreed"], r["median_round_s"],
             r["deliveries_per_round"],
             f"{r['deliveries_per_round'] / max(base['deliveries_per_round'], 1):.1f}x",
             f"{r['bytes_per_round'] / 1024:.0f}",
             f"{r['deliveries_per_round'] / (r['n'] ** 2):.2f}"]
            for r in scale]
    print("\n" + table(
        ["n", "q", "agreed", "median s", "deliveries", "vs n=%d" % base["n"],
         "KB", "deliv/n^2"],
        rows,
    ))
    print("A flat deliv/n^2 column is the point: cost is quadratic, not cubic.")

    print("\n" + table(
        ["n", "proposals", "votes", "polkas", "certificates", "total/height",
         "(n-1)(p+4n)"],
        [[r["n"],
          r["deliveries_per_round_by_kind"].get("proposal", 0),
          r["deliveries_per_round_by_kind"].get("vote", 0),
          r["deliveries_per_round_by_kind"].get("polka", 0),
          r["deliveries_per_round_by_kind"].get("certificate", 0),
          r["deliveries_per_round"], r["closed_form"]]
         for r in scale],
    ))
    print("Deliveries, not broadcasts, and per height, not per run.  With p\n"
          "proposers the count is (n-1)(p + 4n): each of p proposals reaches\n"
          "n-1 peers; each node sends a prevote and a precommit, gossips the\n"
          "polka it observed, and gossips the certificate.  The polka column is\n"
          "the price of the liveness fix in section 5 -- it is n(n-1) per\n"
          "height, a quarter of the total, and buys termination out of a\n"
          "locked/unlocked split that no number of sub-rounds could resolve.\n"
          "Every column is a mean over heights, so the row sums to the total\n"
          "-- a median total beside mean components would not.  The total\n"
          "matches the closed form exactly at zero loss because nothing is\n"
          "retried; under loss, and at n=50 where a window expires, it does\n"
          "not.")

    print(f"\n=== adversity sweep (n=10, {args.adversity_rounds} rounds "
          f"x {args.seeds} seeds) ===")
    adverse = []
    for latency, loss in ((0.005, 0.0), (0.05, 0.0), (0.15, 0.0),
                          (0.05, 0.05), (0.05, 0.20)):
        # A shorter cap than the scale sweep on purpose: at n=10 the windows
        # are 0.2s doubling over four sub-rounds, so a height that is going to
        # commit has done so long before this.  Waiting 60s on each failure
        # makes the failure rate expensive to measure, not more accurate.
        runs = [await one(10, args.adversity_rounds, latency, latency / 2, loss,
                          args.seed + k, args.proposers,
                          timeout=args.adversity_timeout)
                for k in range(args.seeds)]
        agreed = sum(r["rounds_agreed"] for r in runs)
        total = sum(r["rounds"] for r in runs)
        record = {
            "latency_ms": runs[0]["latency_ms"],
            "loss": loss,
            "n": 10,
            "q": runs[0]["q"],
            "seeds": args.seeds,
            "rounds": total,
            "rounds_agreed": agreed,
            "round_failure_rate": round(1 - agreed / total, 3),
            "per_seed_agreed": [f"{r['rounds_agreed']}/{r['rounds']}" for r in runs],
            "median_round_s": round(statistics.median(
                r["median_round_s"] for r in runs), 3),
            "messages_per_round": int(statistics.median(
                r["messages_per_round"] for r in runs)),
        }
        adverse.append(record)
        print(f"  delay {record['latency_ms']:>5.0f} ms  loss {loss:>4.0%}  "
              f"committed {agreed}/{total}  "
              f"median {record['median_round_s']:.2f}s  "
              f"per seed {' '.join(record['per_seed_agreed'])}")

    print("\n" + table(
        ["one-way delay (ms)", "loss", "agreed", "failure rate", "median s",
         "messages", "per seed"],
        [[r["latency_ms"], f"{r['loss']:.0%}", f"{r['rounds_agreed']}/{r['rounds']}",
          r["round_failure_rate"], r["median_round_s"], r["messages_per_round"],
          " ".join(r["per_seed_agreed"])]
         for r in adverse],
    ))

    sub_rounds = []
    if args.sub_round_sweep:
        print(f"\n=== sub-round budget under loss (n=10, 50 ms delay, "
              f"{args.adversity_rounds} rounds x {args.seeds} seeds) ===")
        for loss in args.sub_round_losses:
            for budget in args.sub_round_budgets:
                runs = [await one(10, args.adversity_rounds, 0.05, 0.025, loss,
                                  args.seed + k, args.proposers,
                                  timeout=args.adversity_timeout,
                                  max_sub_rounds=budget)
                        for k in range(args.seeds)]
                agreed = sum(r["rounds_agreed"] for r in runs)
                total = sum(r["rounds"] for r in runs)
                sub_rounds.append({
                    "loss": loss, "max_sub_rounds": budget,
                    "rounds": total, "rounds_agreed": agreed,
                    "round_failure_rate": round(1 - agreed / total, 3),
                })
                print(f"  loss {loss:>4.0%}  max_sub_rounds {budget:>2}  "
                      f"committed {agreed}/{total}  "
                      f"failure rate {1 - agreed / total:.2f}")
        print("\n" + table(
            ["loss", "max_sub_rounds", "agreed", "failure rate"],
            [[f"{r['loss']:.0%}", r["max_sub_rounds"],
              f"{r['rounds_agreed']}/{r['rounds']}", r["round_failure_rate"]]
             for r in sub_rounds]))
        print("Read this column with care.  The budget is not monotone in these\n"
              "runs, and the seed-to-seed spread within one row is as large as\n"
              "the difference between rows, so it does not support a ranking.\n"
              "The likely coupling: attempt r waits collect_window * 2^r, so a\n"
              "larger budget also lengthens later attempts until one window\n"
              "outlives the caller's per-height deadline.")

    save("e3_consensus", {"scale": scale, "adversity": adverse,
                          "sub_round_budget": sub_rounds,
                          "proposers_per_round": args.proposers})
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
