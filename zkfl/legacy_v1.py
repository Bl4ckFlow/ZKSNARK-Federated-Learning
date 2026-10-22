"""Circuit v1, kept as an ablation.

This is the original design, unchanged, so the paper can show the attack rather
than describe it.  v1's statement is::

    exists g :  (w_old - w_new) * S == eta * g

which is satisfiable for *every* pair (w_old, w_new) by the single division
``g = (w_old - w_new) * S / eta``.  There is no data in the circuit at all --
the "gradient" is a free private input, unconstrained and unrelated to any
training.  The original prover computed exactly that witness itself, with the
comment "we force the gradient witness to match the difference exactly", so
producing a valid proof for an arbitrary target model was not an attack on the
system; it was the system's own code path.

Three constraints, and every one of them a multiplication that any target
satisfies.  Reported alongside v2's 19,402 in the evaluation.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
V1_DIR = ROOT / "artifacts" / "update_check_v1"
SCALE = 10 ** 6


def available() -> bool:
    return (V1_DIR / "update_check_v1_final.zkey").exists()


def forge_witness(
    old_params: Sequence[int], new_params: Sequence[int], eta: int
) -> dict[str, Any]:
    """Solve v1's constraint for an arbitrary target.

    This is the whole attack.  It is one line of arithmetic per parameter and
    needs no data, no training and no knowledge of anything private.
    """
    if SCALE % eta:
        raise ValueError(
            f"v1 only admits learning rates dividing {SCALE}; eta={eta} does not. "
            "(This is itself a defect: the original prover used integer "
            "division for the multiplier, so eta=0.03 could not produce a "
            "valid witness at all.)"
        )
    multiplier = SCALE // eta
    gradient = [(o - n) * multiplier for o, n in zip(old_params, new_params)]
    return {
        "old_params": [str(v) for v in old_params],
        "new_params": [str(v) for v in new_params],
        "learning_rate": str(eta),
        "gradient": [str(v) for v in gradient],
    }


def prove(inputs: dict[str, Any]) -> dict[str, Any] | None:
    wasm = V1_DIR / "update_check_v1_js" / "update_check_v1.wasm"
    gen = V1_DIR / "update_check_v1_js" / "generate_witness.js"
    zkey = V1_DIR / "update_check_v1_final.zkey"
    snarkjs = ROOT / "node_modules" / ".bin" / "snarkjs"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.json").write_text(json.dumps(inputs))
        try:
            subprocess.run(["node", str(gen), str(wasm), str(tmp / "in.json"),
                            str(tmp / "w.wtns")], check=True, capture_output=True)
            subprocess.run([str(snarkjs), "groth16", "prove", str(zkey),
                            str(tmp / "w.wtns"), str(tmp / "proof.json"),
                            str(tmp / "public.json")], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return None
        return {
            "proof": json.loads((tmp / "proof.json").read_text()),
            "public_signals": json.loads((tmp / "public.json").read_text()),
        }


def verify(proof_object: dict[str, Any] | None) -> bool:
    """v1's verification, exactly as the original: signals come from the prover."""
    if not proof_object:
        return False
    snarkjs = ROOT / "node_modules" / ".bin" / "snarkjs"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "proof.json").write_text(json.dumps(proof_object["proof"]))
        (tmp / "public.json").write_text(json.dumps(proof_object["public_signals"]))
        res = subprocess.run(
            [str(snarkjs), "groth16", "verify",
             str(V1_DIR / "verification_key.json"),
             str(tmp / "public.json"), str(tmp / "proof.json")],
            capture_output=True, text=True,
        )
        return "OK" in res.stdout
