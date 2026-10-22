"""Why the sub-round budget saturates: the window schedule outruns the deadline.

Attempt r waits ``collect_window * 2^min(r,6)`` -- and a sub-round opens two
such windows, one for prevote and one for precommit -- so the time a height has
consumed after r+1 attempts is a geometric sum.  At the settings used for the
adversity sweep (w = 0.2 s) that is 12.4 s through attempt 4 and 25.2 s through
attempt 5 when both phases time out, or 12.6 s and 25.4 s through attempts 5
and 6 when only the prevote window does.  Against a 15 s per-height deadline,
five or six attempts are reachable.  ``max_sub_rounds`` of 8 and 16 are
therefore not different configurations at all, which is the obvious candidate
explanation for the non-monotone sweep in e3_consensus.

This script tests that directly by varying the one quantity the reasoning says
matters -- the deadline -- while holding the budget fixed.  If the diagnosis is
right, a longer deadline buys commits and a larger budget alone does not.

Seed 0 is included deliberately: it is the seed that collapses reproducibly at
5% loss, committing 3 heights and then timing out on every subsequent one.
"""
from __future__ import annotations

import argparse
import asyncio

from experiments.common import save, table
from experiments.e3_consensus import one

WINDOW = 0.2          # ProtocolConfig.collect_window used by e3_consensus.one
CAP = 6               # the doubling is capped at 2^6


#: A sub-round opens *two* windows -- one for prevote, one for precommit -- and
#: `_open` sizes both with the same `_window_seconds`.  A sub-round in which
#: both time out therefore costs twice the window, so the honest accounting is a
#: range: PHASES=1 is the optimistic case where the precommit window closes
#: early because a polka landed, PHASES=2 the case where neither quorum arrives.
PHASES = (1, 2)


def schedule(n_sub_rounds: int, window: float = WINDOW,
             phases: int = 2) -> list[float]:
    """Cumulative seconds a height has spent after each attempt."""
    out, total = [], 0.0
    for r in range(n_sub_rounds):
        total += phases * window * (2 ** min(r, CAP))
        out.append(round(total, 2))
    return out


def reachable(deadline: float, window: float = WINDOW,
              phases: int = 2) -> int:
    """How many attempts fit inside the deadline."""
    return sum(1 for t in schedule(32, window, phases) if t <= deadline)


def reachable_range(deadline: float, window: float = WINDOW) -> tuple[int, int]:
    """(pessimistic, optimistic) attempt counts -- both phases vs one."""
    return reachable(deadline, window, 2), reachable(deadline, window, 1)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--loss", type=float, default=0.05)
    ap.add_argument("--deadlines", type=float, nargs="*",
                    default=[15.0, 60.0])
    ap.add_argument("--max-sub-rounds", type=int, default=8)
    args = ap.parse_args()

    cum1, cum2 = schedule(9, phases=1), schedule(9, phases=2)
    print("=== the window schedule, against the deadlines tested ===")
    print(f"collect_window {WINDOW}s, doubling capped at 2^{CAP}; each "
          f"sub-round opens a prevote and a precommit window\n")
    print(table(["attempt", "window s", "cumulative (1 phase)",
                 "cumulative (both)"],
                [[r, round(WINDOW * 2 ** min(r, CAP), 2), cum1[r], cum2[r]]
                 for r in range(9)]))
    for d in args.deadlines:
        lo, hi = reachable_range(d)
        print(f"  deadline {d:>5.0f}s -> {lo}-{hi} attempts reachable "
              f"(budget is {args.max_sub_rounds})")

    print(f"\n=== measured, {args.loss:.0%} loss, n=10, 50 ms one-way, "
          f"max_sub_rounds={args.max_sub_rounds} ===")
    rows = []
    for deadline in args.deadlines:
        runs = []
        for k in range(args.seeds):
            r = await one(10, args.rounds, 0.05, 0.025, args.loss, k, 3,
                          timeout=deadline,
                          max_sub_rounds=args.max_sub_rounds)
            runs.append(r)
            print(f"  deadline {deadline:>5.0f}s  seed {k}  "
                  f"{r['rounds_agreed']}/{r['rounds']}  "
                  f"median {r['median_round_s']:.2f}s")
        agreed = sum(r["rounds_agreed"] for r in runs)
        total = sum(r["rounds"] for r in runs)
        rows.append({
            "deadline_s": deadline,
            "attempts_reachable": list(reachable_range(deadline)),
            "max_sub_rounds": args.max_sub_rounds,
            "rounds": total,
            "rounds_agreed": agreed,
            "commit_rate": round(agreed / total, 3),
            "per_seed_agreed": [f"{r['rounds_agreed']}/{r['rounds']}"
                                for r in runs],
        })
        print(f"  -> {agreed}/{total} ({agreed / total:.2f})\n")

    print(table(
        ["deadline s", "attempts reachable", "committed", "commit rate",
         "per seed"],
        [[r["deadline_s"], "-".join(map(str, r["attempts_reachable"])),
          f"{r['rounds_agreed']}/{r['rounds']}", r["commit_rate"],
          " ".join(r["per_seed_agreed"])] for r in rows]))

    first, last = rows[0], rows[-1]

    def span(row):
        return "-".join(map(str, row["attempts_reachable"]))

    print(f"\ncommit rate {first['commit_rate']:.2f} at a "
          f"{first['deadline_s']:.0f}s deadline "
          f"({span(first)} attempts reachable) vs "
          f"{last['commit_rate']:.2f} at {last['deadline_s']:.0f}s "
          f"({span(last)} reachable), budget fixed at "
          f"{args.max_sub_rounds} throughout.")
    print("If that gap is large, the budget was never the binding constraint.")

    save("e3_deadline", {
        "collect_window_s": WINDOW,
        "doubling_cap_exponent": CAP,
        "cumulative_schedule_one_phase_s": cum1,
        "cumulative_schedule_both_phases_s": cum2,
        "loss": args.loss,
        "max_sub_rounds": args.max_sub_rounds,
        "rows": rows,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
