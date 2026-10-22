#!/usr/bin/env python3
"""E1 -- does the thing learn?

The paper reported two rounds and no accuracy at all, so "consistent
convergence" in the abstract had nothing behind it.  This runs the protocol to
convergence and puts two baselines beside it:

  * centralized SGD on the union of every node's data -- the ceiling;
  * plain FedAvg with no proofs and no consensus -- the cost of the machinery.

All three use the same integer fixed-point arithmetic and the same
piecewise-linear activation, so the comparison isolates the protocol rather
than the numerics.
"""
from __future__ import annotations

import argparse
import asyncio

from experiments.common import save, table
from zkfl.aggregation import aggregate
from zkfl.config import FixedPointConfig, ProtocolConfig
from zkfl.datasets import breast_cancer_binary, iris_binary, partition
from zkfl.fixedpoint import quantize
from zkfl.messages import Proposal
from zkfl.model import Model
from zkfl.simulation import Simulation
from zkfl.trainer import FixedPointTrainer, LocalDataset, accuracy, log_loss


def quantized_test(split, fp):
    xs = [[quantize(v, fp.scale) for v in row] for row in split.x_test]
    return xs, [int(v) for v in split.y_test]


def sklearn_reference(split):
    """Float64 logistic regression on the same split.

    Not a competitor -- a reference point.  The gap to it is the price of the
    piecewise-linear activation plus integer fixed point, which is a number the
    paper should state rather than leave to the reader's imagination.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    clf = LogisticRegression(max_iter=2000).fit(split.x_train, split.y_train)
    return float(clf.score(split.x_test, split.y_test))


def centralized(split, fp, protocol, rounds):
    """Everything in one place: the ceiling this protocol is paying to avoid."""
    dataset = LocalDataset(split.x_train.tolist(), split.y_train.tolist(), fp)
    trainer = FixedPointTrainer(fp, protocol.learning_rate)
    model = Model.zeros(fp)
    xs, ys = quantized_test(split, fp)
    curve = []
    for r in range(rounds):
        for b in range(dataset.n_batches):
            model, _ = trainer.step(model, dataset, b)
        curve.append({"round": r + 1, "accuracy": accuracy(model, xs, ys),
                      "loss": log_loss(model, xs, ys)})
    return curve


def plain_fedavg(split, fp, protocol, n_nodes, rounds, seed):
    """FedAvg with no proofs, no votes, no certificates."""
    parts = partition(split, n_nodes, fp, seed=seed)
    datasets = [LocalDataset(p.x.tolist(), p.y.tolist(), fp)
                for p in parts if len(p.x) >= fp.batch_size]
    trainer = FixedPointTrainer(fp, protocol.learning_rate)
    model = Model.zeros(fp)
    xs, ys = quantized_test(split, fp)
    curve = []
    for r in range(rounds):
        proposals = []
        for i, dataset in enumerate(datasets):
            after, _ = trainer.step(model, dataset, r % dataset.n_batches)
            proposals.append(Proposal(height=r + 1, prev_hash=model.digest(),
                                      model=after, proof=None, batch_index=0,
                                      proposer=f"n{i}"))
        model = aggregate(model, proposals, fp, rule="fedavg").model
        curve.append({"round": r + 1, "accuracy": accuracy(model, xs, ys),
                      "loss": log_loss(model, xs, ys)})
    return curve


async def protocol_run(split, fp, protocol, n_nodes, rounds, strict, seed, iid):
    sim = Simulation(split=split, fp=fp, protocol=protocol, n_nodes=n_nodes,
                     strict_zk=strict, latency=0.005, jitter=0.005, seed=seed,
                     iid=iid)
    result = await sim.run(rounds=rounds, proposers_per_round=n_nodes,
                           timeout=600 if strict else 30)
    return sim, result


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("iris", "breast-cancer"), default="iris")
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-seeds", type=int, default=2,
                    help="additional train/test splits to repeat the whole "
                         "comparison on.  The iris test set is 30 samples, so a "
                         "single run reports differences of one sample as if "
                         "they were results; the spread across seeds is what "
                         "says whether a gap means anything")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--strict-zk", action="store_true",
                    help="real Groth16 proofs (slow; use few rounds)")
    ap.add_argument("--non-iid", action="store_true",
                    help="Dirichlet label split instead of an even one")
    args = ap.parse_args()

    if args.dataset == "iris":
        split = iris_binary(seed=args.seed)
        fp = FixedPointConfig(n_features=4, batch_size=8, merkle_depth=4)
    else:
        split = breast_cancer_binary(seed=args.seed, n_features=8)
        fp = FixedPointConfig(n_features=8, batch_size=8, merkle_depth=5)
    fp.check()

    protocol = ProtocolConfig(
        collect_window=0.2, strict_zk=args.strict_zk, learning_rate=args.lr,
        aggregation_rule="trimmed_mean",
    ).with_nodes(args.nodes)

    print(split.describe())
    print(f"n={args.nodes} f={protocol.f_byzantine} q={protocol.effective_quorum()} "
          f"rounds={args.rounds} strict_zk={args.strict_zk} "
          f"split={'non-iid' if args.non_iid else 'iid'}\n")

    sim, result = await protocol_run(split, fp, protocol, args.nodes, args.rounds,
                                     args.strict_zk, args.seed, not args.non_iid)
    for note in sim.notes:
        print(f"  {note}")

    proto_curve = [
        {"round": r.height, "accuracy": r.accuracy, "loss": r.loss,
         "agreed": r.agreed, "proposals": r.proposals}
        for r in result.rounds
    ]
    central = centralized(split, fp, protocol, args.rounds)
    fedavg = plain_fedavg(split, fp, protocol, args.nodes, args.rounds, args.seed)
    reference = sklearn_reference(split)

    rows = []
    for i in range(args.rounds):
        p = proto_curve[i] if i < len(proto_curve) else {}
        rows.append([
            i + 1,
            f"{p.get('accuracy') or float('nan'):.3f}",
            f"{p.get('loss') or float('nan'):.4f}",
            f"{fedavg[i]['accuracy']:.3f}",
            f"{central[i]['accuracy']:.3f}",
            "yes" if p.get("agreed") else "no",
        ])
    print("\n" + table(
        ["round", "zkfl acc", "zkfl loss", "fedavg acc", "central acc", "agreed"],
        rows[:: max(1, args.rounds // 20)] + ([rows[-1]] if args.rounds > 20 else []),
    ))

    if reference is not None:
        print(f"\nfloat64 logistic regression on the same split: "
              f"accuracy {reference:.3f}  (the numerics ceiling, not a competitor)")

    print("\ninvariants after the run:")
    for name, ok, detail in sim.check_invariants():
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    # Repeat the whole comparison on further splits.  Reported as a range
    # beside the headline run, because on a 30-sample test set one sample is
    # 0.033 accuracy and a single number cannot distinguish that from a result.
    seed_runs = []
    for extra in range(1, args.extra_seeds + 1):
        seed = args.seed + extra
        alt = (iris_binary(seed=seed) if args.dataset == "iris"
               else breast_cancer_binary(seed=seed, n_features=8))
        _, alt_result = await protocol_run(alt, fp, protocol, args.nodes,
                                           args.rounds, args.strict_zk, seed,
                                           not args.non_iid)
        alt_fedavg = plain_fedavg(alt, fp, protocol, args.nodes, args.rounds, seed)
        alt_central = centralized(alt, fp, protocol, args.rounds)
        seed_runs.append({
            "seed": seed,
            "zkfl_accuracy": alt_result.rounds[-1].accuracy,
            "zkfl_loss": alt_result.rounds[-1].loss,
            "fedavg_accuracy": alt_fedavg[-1]["accuracy"],
            "central_accuracy": alt_central[-1]["accuracy"],
            "sklearn_reference": sklearn_reference(alt),
            "rounds_agreed": sum(1 for r in alt_result.rounds if r.agreed),
        })

    if seed_runs:
        every = [proto_curve[-1]["accuracy"]] + [r["zkfl_accuracy"] for r in seed_runs]
        fed = [fedavg[-1]["accuracy"]] + [r["fedavg_accuracy"] for r in seed_runs]
        print(f"\nacross {1 + len(seed_runs)} splits: "
              f"zkfl {min(every):.3f}-{max(every):.3f}, "
              f"fedavg {min(fed):.3f}-{max(fed):.3f}")
        print("On a 30-sample test set one sample is 0.033, so read the range,\n"
              "not the gap between two medians.")

    save(f"e1_learning_{args.dataset}_n{args.nodes}"
         + ("_strict" if args.strict_zk else "")
         + ("_noniid" if args.non_iid else ""), {
        "config": {"dataset": split.name, "nodes": args.nodes,
                   "rounds": args.rounds, "strict_zk": args.strict_zk,
                   "iid": not args.non_iid, "seed": args.seed,
                   "f": protocol.f_byzantine, "q": protocol.effective_quorum(),
                   "learning_rate": protocol.learning_rate,
                   "max_delta": fp.max_delta,
                   "rule": protocol.aggregation_rule},
        "notes": sim.notes,
        "sklearn_float64_reference": reference,
        "zkfl": proto_curve,
        "fedavg_no_zk": fedavg,
        "centralized": central,
        "summary": result.summary(),
        "other_seeds": seed_runs,
        "invariants": [{"name": n, "ok": o, "detail": d}
                       for n, o, d in sim.check_invariants()],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
