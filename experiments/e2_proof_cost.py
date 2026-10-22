#!/usr/bin/env python3
"""E2 -- what the proof costs, and what v1 cost instead.

Reports constraints, witness time, proving time, verification time and proof
size, for both circuits.  The v1 row is the ablation: three constraints, no
data, and a statement any target model satisfies -- so its cost is the price of
proving nothing.

Proving is measured two ways because the difference is the whole story of the
paper's original "1-2 s" figure: a cold call that spawns Node and loads the
zkey, and a warm call through the persistent worker.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from experiments.common import ROOT, constraint_count, save, table
from zkfl import legacy_v1
from zkfl.config import FixedPointConfig
from zkfl.datasets import synthetic
from zkfl.model import Model
from zkfl.trainer import FixedPointTrainer, LocalDataset
from zkfl.zk import CircuitArtifacts, Groth16Backend, ProofError, PublicInputs


def measure(cfg: FixedPointConfig, repeats: int) -> dict | None:
    artifacts = CircuitArtifacts.for_config(cfg)
    if artifacts.missing():
        return None

    backend = Groth16Backend(cfg)
    try:
        split = synthetic(n_samples=cfg.batch_size * 8, n_features=cfg.n_features,
                          seed=17)
        dataset = LocalDataset(split.x_train.tolist(), split.y_train.tolist(), cfg)
        trainer = FixedPointTrainer(cfg, 0.2)
        model = Model.zeros(cfg)

        prove_wall, verify_wall, sizes = [], [], []
        for i in range(repeats):
            after, witness = trainer.step(model, dataset, i % dataset.n_batches)
            t0 = time.perf_counter()
            proof = backend.prove(witness)
            prove_wall.append(time.perf_counter() - t0)

            expected = PublicInputs.from_models(model, after, witness.eta,
                                                dataset.root)
            t0 = time.perf_counter()
            ok = backend.verify(proof, expected)
            verify_wall.append(time.perf_counter() - t0)
            assert ok, "an honest proof failed to verify"

            import json
            sizes.append(len(json.dumps(proof["proof"]).encode()))
            model = after

        return {
            "circuit": artifacts.name,
            "constraints": constraint_count(artifacts.r1cs),
            "n_features": cfg.n_features,
            "batch_size": cfg.batch_size,
            "merkle_depth": cfg.merkle_depth,
            "prove_cold_s": round(prove_wall[0], 3),
            "prove_warm_median_s": round(statistics.median(prove_wall[1:] or prove_wall), 3),
            "prove_warm_min_s": round(min(prove_wall[1:] or prove_wall), 3),
            "verify_median_ms": round(statistics.median(verify_wall) * 1000, 2),
            "proof_bytes": sizes[0],
            "zkey_mb": round(artifacts.zkey.stat().st_size / 1e6, 1),
            "samples_per_proof": cfg.batch_size,
        }
    finally:
        backend.close()


def measure_v1(repeats: int) -> dict | None:
    if not legacy_v1.available():
        return None
    r1cs = ROOT / "artifacts" / "update_check_v1" / "update_check_v1.r1cs"
    prove_wall, verify_wall, sizes = [], [], []
    import json

    for i in range(repeats):
        forged = legacy_v1.forge_witness([0, 0, 0], [1000 * (i + 1)] * 3, 10_000)
        t0 = time.perf_counter()
        proof = legacy_v1.prove(forged)
        prove_wall.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ok = legacy_v1.verify(proof)
        verify_wall.append(time.perf_counter() - t0)
        assert ok
        sizes.append(len(json.dumps(proof["proof"]).encode()))

    return {
        "circuit": "update_check_v1 (ablation)",
        "constraints": constraint_count(r1cs),
        "n_features": 2,
        "batch_size": 0,
        "merkle_depth": 0,
        "prove_cold_s": round(prove_wall[0], 3),
        "prove_warm_median_s": round(statistics.median(prove_wall), 3),
        "prove_warm_min_s": round(min(prove_wall), 3),
        "verify_median_ms": round(statistics.median(verify_wall) * 1000, 2),
        "proof_bytes": sizes[0],
        "zkey_mb": round((ROOT / "artifacts" / "update_check_v1"
                          / "update_check_v1_final.zkey").stat().st_size / 1e6, 1),
        "samples_per_proof": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--configs", nargs="*", default=["4,8,4"],
                    help="features,batch,depth triples that are already built")
    args = ap.parse_args()

    rows, records = [], []
    for spec in args.configs:
        features, batch, depth = (int(v) for v in spec.split(","))
        cfg = FixedPointConfig(n_features=features, batch_size=batch,
                               merkle_depth=depth)
        record = measure(cfg, args.repeats)
        if record is None:
            print(f"skipping {spec}: not built "
                  f"(python scripts/setup_circuit.py --features {features} "
                  f"--batch {batch} --depth {depth} --local-ptau)")
            continue
        records.append(record)

    v1 = measure_v1(min(args.repeats, 3))
    if v1:
        records.append(v1)
    else:
        print("skipping the v1 ablation: not built (setup_circuit.py --v1)")

    for r in records:
        rows.append([r["circuit"], r["constraints"], r["samples_per_proof"],
                     r["prove_cold_s"], r["prove_warm_median_s"],
                     r["verify_median_ms"], r["proof_bytes"], r["zkey_mb"]])

    print("\n" + table(
        ["circuit", "constraints", "samples/proof", "prove cold (s)",
         "prove warm (s)", "verify (ms)", "proof (B)", "zkey (MB)"],
        rows,
    ))
    print("\nCold includes Node start-up and the zkey load; warm reuses the")
    print("persistent worker.  The gap is what the original per-proof figure")
    print("was mostly measuring.")

    save("e2_proof_cost", {"measurements": records})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
