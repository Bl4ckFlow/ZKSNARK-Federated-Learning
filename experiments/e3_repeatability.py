"""How reproducible is the loss-regime consensus number?

Two runs of ``e3_consensus`` with identical arguments returned 28/75 and 49/75
committed heights at 5% loss.  That is a factor of two on the headline figure of
Table 4, so before quoting either number we have to know what it is a sample of.

The simulator seeds the data split and the loss draws, but it runs on *real*
asyncio timers: collect windows and per-height deadlines are wall-clock.  Two
processes with the same seed therefore do not see the same interleaving, and at
5% loss the protocol sits near a tipping point where one lost vote decides
whether a height commits or the round collapses and drags its successors down.
So the seed does not pin the outcome, and a single run is one sample of a
distribution rather than a measurement of a constant.

This script separates the two sources of spread by running the same seed set
several times:

  * within a repeat, seed-to-seed spread -- different data and loss draws;
  * across repeats, run-to-run spread -- same seeds, different interleaving.

If the second is comparable to the first, no single number belongs in the paper
and the honest report is a range.

RUN THIS ALONE.  The first time we ran it, another experiment was running
concurrently, and the three repeats came out 0.71, 0.40, 0.33 in the order the
other job got heavier -- so most of that "run-to-run spread" was the host, not
the protocol.  Wall-clock timers make this script a load meter as much as a
protocol measurement, and a number taken on a busy machine is not reproducible
anywhere.  Nothing else should be running while it does.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics

from experiments.common import save, table
from experiments.e3_consensus import one


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--loss", type=float, default=0.05)
    ap.add_argument("--latency", type=float, default=0.05)
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    print(f"=== repeatability at {args.loss:.0%} loss, n=10, "
          f"{args.latency * 1000:.0f} ms one-way ===")
    print(f"{args.repeats} repeats x {args.seeds} seeds x {args.rounds} heights; "
          "same seeds every repeat, so any spread across repeats is timing\n")

    repeats = []
    for rep in range(args.repeats):
        per_seed = []
        for k in range(args.seeds):
            r = await one(10, args.rounds, args.latency, 0.025, args.loss,
                          k, 3, timeout=args.timeout)
            per_seed.append(r)
            print(f"  repeat {rep}  seed {k}  "
                  f"{r['rounds_agreed']}/{r['rounds']}  "
                  f"median {r['median_round_s']:.2f}s")
        agreed = sum(r["rounds_agreed"] for r in per_seed)
        total = sum(r["rounds"] for r in per_seed)
        repeats.append({
            "repeat": rep,
            "rounds": total,
            "rounds_agreed": agreed,
            "commit_rate": round(agreed / total, 3),
            "per_seed_agreed": [f"{r['rounds_agreed']}/{r['rounds']}"
                                for r in per_seed],
            "per_seed_rate": [round(r["rounds_agreed"] / r["rounds"], 3)
                              for r in per_seed],
        })
        print(f"  -> repeat {rep}: {agreed}/{total} "
              f"({agreed / total:.2f})\n")

    rates = [r["commit_rate"] for r in repeats]
    within = [rate for r in repeats for rate in r["per_seed_rate"]]

    print(table(
        ["repeat", "committed", "commit rate", "per seed"],
        [[r["repeat"], f"{r['rounds_agreed']}/{r['rounds']}", r["commit_rate"],
          " ".join(r["per_seed_agreed"])] for r in repeats],
    ))

    summary = {
        "loss": args.loss,
        "latency_ms": args.latency * 1000,
        "rounds_per_seed": args.rounds,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "repeat_rates": rates,
        "repeat_rate_min": min(rates),
        "repeat_rate_max": max(rates),
        "repeat_rate_spread": round(max(rates) - min(rates), 3),
        "per_seed_rates": within,
        "per_seed_rate_min": min(within),
        "per_seed_rate_max": max(within),
        "per_seed_rate_spread": round(max(within) - min(within), 3),
        "pooled_agreed": sum(r["rounds_agreed"] for r in repeats),
        "pooled_rounds": sum(r["rounds"] for r in repeats),
    }
    if len(rates) > 1:
        summary["repeat_rate_stdev"] = round(statistics.stdev(rates), 3)
    if len(within) > 1:
        summary["per_seed_rate_stdev"] = round(statistics.stdev(within), 3)

    print(f"\nrun-to-run  commit rate {min(rates):.2f}-{max(rates):.2f}"
          f"  (spread {summary['repeat_rate_spread']:.2f})")
    print(f"seed-to-seed commit rate {min(within):.2f}-{max(within):.2f}"
          f"  (spread {summary['per_seed_rate_spread']:.2f})")
    print(f"pooled      {summary['pooled_agreed']}/{summary['pooled_rounds']} "
          f"= {summary['pooled_agreed'] / summary['pooled_rounds']:.2f}")
    print("\nQuote the pooled rate and the range, not one run.")

    save("e3_repeatability", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
