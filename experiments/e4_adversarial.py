#!/usr/bin/env python3
"""E4 -- the attacks, run against the live protocol.

Five scenarios, each with the specific defect it targets:

  a) POISON1 against v1 and v2 -- the circuit's own witness recipe used to
     prove an arbitrary target model.  Offline, because it never reaches a
     network in v2.
  b) equivocating voter -- two signed votes in one sub-round.
  c) equivocating proposer -- two proposals at one height, which vote-set
     cross-checking structurally could not see.
  d) poisoning with a *valid* proof on committed-but-adversarial data -- the
     attack the circuit cannot stop, so the aggregation rule has to.
  e) flooding -- a peer trying to evict the dedup window and amplify.

For each: was it detected, how fast, and what did it cost the model.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from experiments.common import save, table
from zkfl import legacy_v1
from zkfl.aggregation import aggregate, robustness_note
from zkfl.config import FixedPointConfig, ProtocolConfig
from zkfl.datasets import synthetic
from zkfl.fixedpoint import quantize, step_within_bound, training_step
from zkfl.messages import Proposal
from zkfl.model import Model
from zkfl.node import EquivocatingProposer, EquivocatingVoter, Flooder, Node
from zkfl.simulation import Simulation
from zkfl.trainer import (
    FixedPointTrainer,
    LocalDataset,
    TrainingWitness,
    accuracy,
)
from zkfl.zk import (
    CircuitArtifacts,
    Groth16Backend,
    ProofError,
    PublicInputs,
)


def scenario_poison1(fp) -> dict:
    """(a) Choose a target model; try to produce a proof for it."""
    out = {"name": "POISON1: prove an attacker-chosen model"}

    if legacy_v1.available():
        target = [9_999_999, -9_999_999, 4_242_424]
        forged = legacy_v1.forge_witness([0, 0, 0], target, 10_000)
        proof = legacy_v1.prove(forged)
        out["v1_accepted"] = bool(proof and legacy_v1.verify(proof))
        out["v1_effort"] = "one integer division per parameter; no data required"
    else:
        out["v1_accepted"] = None
        out["v1_effort"] = "v1 not built"

    artifacts = CircuitArtifacts.for_config(fp)
    if artifacts.missing():
        out["v2_accepted"] = None
        return out

    backend = Groth16Backend(fp)
    try:
        split = synthetic(n_samples=64, n_features=fp.n_features, seed=5)
        dataset = LocalDataset(split.x_train.tolist(), split.y_train.tolist(), fp)
        trainer = FixedPointTrainer(fp, 0.5)
        before = Model.zeros(fp)
        _, honest = trainer.step(before, dataset, 0)

        chosen = Model(w=(9_999_999, -9_999_999, 4_242_424, 1_000_000), b=0,
                       scale=fp.scale)
        forged_witness = TrainingWitness(
            model_before=before, model_after=chosen, eta=honest.eta,
            batch_index=0, xs=honest.xs, ys=honest.ys,
            merkle=honest.merkle, data_root=dataset.root,
        )
        try:
            backend.prove(forged_witness)
            out["v2_accepted"] = True
            out["v2_reason"] = "PROVED -- this should be impossible"
        except ProofError as exc:
            out["v2_accepted"] = False
            out["v2_reason"] = str(exc).split("\n")[0][:120]

        # And the same idea via invented data rather than an invented target.
        forged_x = tuple(tuple([15_000_000] * fp.n_features)
                         for _ in range(fp.batch_size))
        forged_y = tuple([0] * fp.batch_size)
        w_new, b_new = training_step(before.w, before.b,
                                     [list(r) for r in forged_x],
                                     list(forged_y), honest.eta, fp.scale)
        uncommitted = TrainingWitness(
            model_before=before,
            model_after=Model(w=tuple(w_new), b=b_new, scale=fp.scale),
            eta=honest.eta, batch_index=0, xs=forged_x, ys=forged_y,
            merkle=honest.merkle, data_root=dataset.root,
        )
        try:
            backend.prove(uncommitted)
            out["v2_uncommitted_data_accepted"] = True
        except ProofError as exc:
            out["v2_uncommitted_data_accepted"] = False
            out["v2_uncommitted_reason"] = str(exc).split("\n")[0][:120]
    finally:
        backend.close()
    return out


async def scenario_live(name, n_nodes, adversary_class, adversary_index, rounds):
    fp = FixedPointConfig()
    protocol = ProtocolConfig(collect_window=0.2, strict_zk=False,
                              aggregation_rule="trimmed_mean",
                              learning_rate=0.5).with_nodes(n_nodes)
    split = synthetic(n_samples=64 * n_nodes, n_features=fp.n_features, seed=21)
    sim = Simulation(split=split, fp=fp, protocol=protocol, n_nodes=n_nodes,
                     strict_zk=False, byzantine={adversary_index: adversary_class},
                     latency=0.005, jitter=0.005, seed=3)
    started = time.monotonic()
    result = await sim.run(rounds=rounds, proposers_per_round=min(4, n_nodes),
                           timeout=60)

    offender = sim.nodes[adversary_index].node_id
    honest = [n for n in sim.nodes if type(n) is Node]
    detected = [n for n in honest if n.engine.accountability.is_blacklisted(offender)]
    first_detection = None
    for node in honest:
        for record in node.engine.accountability.records:
            if record.offender == offender:
                t = record.detected_at - (started + 0)
                first_detection = t if first_detection is None else min(first_detection, t)

    blacklists = {frozenset(n.engine.accountability.blacklist) for n in honest}
    return {
        "name": name,
        "adversary": adversary_class.__name__,
        "n": n_nodes,
        "detected_by": f"{len(detected)}/{len(honest)}",
        "detected_everywhere": len(detected) == len(honest),
        "blacklists_converged": len(blacklists) == 1,
        "false_positives": sorted(
            b for b in set().union(*blacklists) if b != offender
        ) if blacklists else [],
        "rounds_agreed": sum(1 for r in result.rounds if r.agreed),
        "rounds": rounds,
        "final_heights": sorted({n.height for n in honest}),
        "rate_limited": sum(n.gate.stats.rate_limited for n in honest),
        "final_accuracy": result.rounds[-1].accuracy if result.rounds else None,
    }


#: Feature scalings the adversary is allowed to choose between.  It commits to
#: one dataset before training, so this is a design-time choice, not a per-round
#: one -- which is exactly why the calibration sweep belongs to the *attacker*.
MAGNITUDES = (0.25, 0.5, 1.0, 2.0, 4.0)


def _poison_shard(split, shard, fp, seed, magnitude=1.0):
    """Committed-but-adversarial data, pushing consistently in one direction.

    Two properties make this the attack worth measuring.  It is *directed*: all
    poisoned nodes drag the model the same way, so their contributions do not
    cancel and the damage accumulates over heights.  And it is *extremal*: that
    is precisely what an order statistic can discard, so this is the case that
    separates a mean from a trimmed mean.  Label flipping fails both tests --
    the flipped gradients point every which way, land in the middle of the
    order, and trimming has nothing to grip.

    ``magnitude`` scales the features.  Bigger is not automatically better for
    the adversary: past a point the resulting step exceeds the in-circuit
    clipping bound, becomes unprovable, and is simply lost.  The strongest
    adversary is therefore the one calibrated to sit just *under* that bound --
    see :func:`_calibrate_poison`.

    It remains an ordinary dataset: committed before training, Merkle-rooted,
    and the step taken on it is a correct gradient step, so the proof verifies.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    direction = np.ones(fp.n_features)
    direction[1::2] = -1.0                 # fixed, and shared by every attacker
    x = (np.abs(rng.normal(1.5, 0.3, size=(shard, fp.n_features)))
         * direction * magnitude)
    y = [0] * shard                        # always the class the features deny
    return x.tolist(), y


def _linf(a: Model, b: Model) -> int:
    return max([abs(x - y) for x, y in zip(a.w, b.w)] + [abs(a.b - b.b)])


def scenario_committed_poisoning(fp, seeds=(31, 32, 33)) -> dict:
    """Run the whole scenario once per data seed and report the spread.

    One run per row is thin: the test set is small, the trajectory is 40
    heights of integer arithmetic, and a single number gives a reader no way to
    tell a 0.11 accuracy loss from noise.  Every row below is therefore the
    median across independent seeds, with the range alongside it.
    """
    per_seed = [_committed_poisoning_once(fp, seed) for seed in seeds]

    merged = []
    for i in range(len(per_seed[0]["runs"])):
        rows = [p["runs"][i] for p in per_seed]
        losses = sorted(r["accuracy_lost"] for r in rows)
        accs = sorted(r["accuracy"] for r in rows)
        linfs = sorted(r["linf_from_clean"] for r in rows)
        merged.append({
            **{k: rows[0][k] for k in ("poisoned_of_10", "rule", "trim_holds")},
            "accuracy": statistics.median(accs),
            "accuracy_min": accs[0],
            "accuracy_max": accs[-1],
            "accuracy_lost": round(statistics.median(losses), 4),
            "accuracy_lost_min": losses[0],
            "accuracy_lost_max": losses[-1],
            "linf_from_clean": int(statistics.median(linfs)),
            "blocked_in_circuit": max(r["blocked_in_circuit"] for r in rows),
            "poison_magnitude": sorted(r["poison_magnitude"] for r in rows)[
                len(rows) // 2],
            "seeds": list(seeds),
        })

    clean = sorted(p["no_attack_accuracy"] for p in per_seed)
    return {
        **per_seed[0],
        "seeds": list(seeds),
        "no_attack_accuracy": statistics.median(clean),
        "no_attack_accuracy_min": clean[0],
        "no_attack_accuracy_max": clean[-1],
        "runs": merged,
        "per_seed": per_seed,
    }


def _committed_poisoning_once(fp, seed) -> dict:
    """(d) A proposer that trains honestly on data it committed -- but the data
    is adversarial.  The proof is valid; only the aggregation rule can help.

    Reported at two poisoned counts: 2 of 10, below the trimming threshold at
    250 permille, and 3 of 10 -- the configured f -- where trimmed mean and
    median hold and plain FedAvg does not.  Accuracy alone is a blunt
    instrument on an easy task, so the distance from the clean model is
    reported alongside it.
    """
    protocol = ProtocolConfig().with_nodes(10)
    # separation=0.4 rather than the default 1.5: on a well-separated task the
    # boundary can drift a long way before a single test point changes side,
    # and an attack measured there looks harmless whatever it does to the
    # model.  Overlapping classes are where poisoning is actually visible.
    split = synthetic(n_samples=1200, n_features=fp.n_features, seed=seed,
                      separation=0.4)
    xs = [[quantize(v, fp.scale) for v in row] for row in split.x_test]
    ys = [int(v) for v in split.y_test]

    trainer = FixedPointTrainer(fp, 0.5)
    # Size the shards from the split rather than assuming it: synthetic()
    # holds a test fraction back, so a hard-coded 8 x 64 asks for rows that
    # are not there and the last shard comes out empty.
    n_train = len(split.x_train)
    n_honest = 7
    shard = max(fp.batch_size,
                (n_train // (n_honest + 1)) // fp.batch_size * fp.batch_size)
    honest_sets = [
        LocalDataset(split.x_train[i * shard:(i + 1) * shard].tolist(),
                     split.y_train[i * shard:(i + 1) * shard].tolist(), fp)
        for i in range(n_honest)
    ]

    def train(datasets, rule, rounds=40):
        """Aggregate only proposals the circuit would actually accept.

        The in-circuit bound |w_new - w_old| <= maxDelta is not decoration: the
        most damaging poisoned step is simply unprovable, so it never reaches
        aggregation at all.  Modelling the adversary as able to submit it would
        overstate the attack; dropping it silently would overstate the defence.
        Both counts are reported.
        """
        model = Model.zeros(fp)
        blocked = 0
        for r in range(rounds):
            proposals = []
            for i, d in enumerate(datasets):
                after, _ = trainer.step(model, d, r % d.n_batches)
                if not step_within_bound(model.w, model.b, after.w, after.b,
                                         fp.max_delta):
                    blocked += 1          # the circuit refuses to prove this
                    continue
                proposals.append(Proposal(
                    height=r + 1, prev_hash=model.digest(), model=after,
                    proof=None, batch_index=0, proposer=f"n{i}"))
            if not proposals:
                continue
            model = aggregate(model, proposals, fp, rule=rule, permille=250).model
        return model, blocked

    def pad_to_ten(head):
        """Ten proposals, so the trim fraction is the same in every run."""
        datasets = list(head)
        while len(datasets) < 10:
            datasets.insert(0, honest_sets[len(datasets) % n_honest])
        return datasets

    # The reference: ten honest proposals, no adversary.  It has to be ten and
    # not n_honest, or `accuracy_lost` would fold the change in proposal count
    # -- and hence in the trim threshold -- into what we attribute to the
    # attack.
    clean, _ = train(pad_to_ten(honest_sets), "trimmed_mean")
    clean_accuracy = round(accuracy(clean, xs, ys), 4)

    def poison_sets(n_poison, magnitude):
        return [LocalDataset(*_poison_shard(split, shard, fp, seed * 100 + j,
                                            magnitude=magnitude), fp)
                for j in range(n_poison)]

    def calibrate(n_poison, rule):
        """The strongest *provable* adversary, not the loudest one.

        Turning the poison up does not monotonically increase damage: past some
        magnitude the resulting step exceeds the in-circuit clipping bound,
        becomes unprovable, and is discarded before it reaches aggregation --
        the attack loses its own rounds.  A real adversary would not do that;
        it would sit just under the bound.  The dataset is committed once and
        cannot be re-chosen per height, so the choice is a single magnitude,
        picked here by sweeping and keeping the one that does the most damage
        while staying provable throughout.
        """
        best = None
        for magnitude in MAGNITUDES:
            datasets = pad_to_ten(honest_sets[:10 - n_poison]
                                  + poison_sets(n_poison, magnitude))
            model, blocked = train(datasets, rule)
            damage = clean_accuracy - accuracy(model, xs, ys)
            record = (damage, magnitude, model, blocked)
            # Prefer an attack that never loses a step; among those, the most
            # damaging.  Fall back to the best overall if every setting is
            # partly blocked, so the row is never empty.
            if blocked == 0 and (best is None or damage > best[0]):
                best = record
        if best is None:
            best = max(
                ((clean_accuracy - accuracy(m, xs, ys), mag, m, b)
                 for mag, (m, b) in (
                     (mag, train(pad_to_ten(honest_sets[:10 - n_poison]
                                            + poison_sets(n_poison, mag)), rule))
                     for mag in MAGNITUDES)),
                key=lambda r: r[0])
        return best

    runs = []
    for n_poison in (2, 3):
        for rule in ("fedavg", "trimmed_mean", "median"):
            damage, magnitude, model, blocked = calibrate(n_poison, rule)
            runs.append({
                "poisoned_of_10": n_poison,
                "rule": rule,
                "poison_magnitude": magnitude,
                "accuracy": round(accuracy(model, xs, ys), 4),
                "accuracy_lost": round(damage, 4),
                "linf_from_clean": _linf(model, clean),
                "blocked_in_circuit": blocked,
                "trim_holds": "DOES NOT HOLD" not in robustness_note(
                    10, n_poison, 250, rule),
            })

    return {
        "name": "committed-but-adversarial data, calibrated to stay provable: "
                "directed features, wrong labels, magnitude swept so the step "
                "sits just under the in-circuit clipping bound",
        "magnitudes_tried": list(MAGNITUDES),
        "no_attack_accuracy": clean_accuracy,
        "runs": runs,
        "notes": [robustness_note(10, n, 250, "trimmed_mean") for n in (2, 3)],
        "why": ("every poisoned proposal carries a valid proof: the data was "
                "committed before training and the step is a correct gradient "
                "step on it. The circuit cannot and does not stop this -- only "
                "the aggregation rule can, and only while enough is trimmed "
                "per side."),
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    fp = FixedPointConfig()

    print("=== (a) POISON1 ===")
    poison1 = scenario_poison1(fp)
    for key in ("v1_accepted", "v1_effort", "v2_accepted", "v2_reason",
                "v2_uncommitted_data_accepted", "v2_uncommitted_reason"):
        if key in poison1:
            print(f"  {key}: {poison1[key]}")

    print("\n=== (b,c,e) live adversaries ===")
    live = []
    for name, cls in (("equivocating voter", EquivocatingVoter),
                      ("equivocating proposer", EquivocatingProposer),
                      ("flooder", Flooder)):
        record = await scenario_live(name, 10, cls, 0 if cls is not EquivocatingVoter
                                     else 9, args.rounds)
        live.append(record)
        print(f"  {name:<24} detected {record['detected_by']}  "
              f"converged={record['blacklists_converged']}  "
              f"false positives={len(record['false_positives'])}  "
              f"rounds agreed {record['rounds_agreed']}/{record['rounds']}")

    print("\n" + table(
        ["attack", "detected by", "converged", "false pos", "agreed",
         "rate-limited msgs"],
        [[r["name"], r["detected_by"], r["blacklists_converged"],
          len(r["false_positives"]), f"{r['rounds_agreed']}/{r['rounds']}",
          r["rate_limited"]] for r in live],
    ))

    print("\n=== (d) poisoning with a valid proof ===")
    poisoning = scenario_committed_poisoning(fp)
    print(f"  no attack (trimmed mean): {poisoning['no_attack_accuracy']:.3f}")
    print()
    print(table(["poisoned", "rule", "accuracy", "lost", "Linf from clean",
                 "blocked in-circuit", "bound holds"],
                [[f"{r['poisoned_of_10']}/10", r["rule"], f"{r['accuracy']:.3f}",
                  f"{r['accuracy_lost']:+.3f}", r["linf_from_clean"],
                  r["blocked_in_circuit"], "yes" if r["trim_holds"] else "no"]
                 for r in poisoning["runs"]]))
    for note in poisoning["notes"]:
        print(f"  {note}")
    print(f"  {poisoning['why']}")

    save("e4_adversarial", {"poison1": poison1, "live": live,
                            "committed_poisoning": poisoning})
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
