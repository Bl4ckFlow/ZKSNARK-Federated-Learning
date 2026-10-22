#!/usr/bin/env python3
"""Generate Poseidon test vectors from a compiled circomlib circuit.

The Python implementation in :mod:`zkfl.poseidon` is checked against these, so
they must come from circom itself rather than from another Python run.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "data" / "poseidon_reference.json"
WIDTHS = (1, 2, 3, 5, 6, 9)


def sh(cmd: list[str], cwd: Path) -> None:
    env = {**os.environ,
           "PATH": f"{ROOT / 'node_modules' / '.bin'}:"
                   f"{Path.home() / '.local' / 'bin'}:{os.environ.get('PATH', '')}"}
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), env=env)
    if res.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)}\n{res.stdout}\n{res.stderr}")


def main() -> int:
    from zkfl.config import FIELD_PRIME

    rng = random.Random(20260908)
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        signals = "\n".join(
            f"    signal input in{w}[{w}];\n"
            f"    signal output out{w};\n"
            f"    component p{w} = Poseidon({w});\n"
            + "".join(f"    p{w}.inputs[{i}] <== in{w}[{i}];\n" for i in range(w))
            + f"    out{w} <== p{w}.out;"
            for w in WIDTHS
        )
        (tmp / "ref.circom").write_text(
            "pragma circom 2.1.6;\n"
            'include "circomlib/circuits/poseidon.circom";\n'
            "template Ref() {\n" + signals + "\n}\ncomponent main = Ref();\n"
        )
        sh(["circom", "ref.circom", "--r1cs", "--wasm", "-o", ".",
            "-l", str(ROOT / "node_modules")], tmp)

        inputs = {}
        for w in WIDTHS:
            values = [rng.randrange(FIELD_PRIME) for _ in range(w)]
            inputs[f"in{w}"] = [str(v) for v in values]
            cases.append({"inputs": values, "out": None})
        (tmp / "in.json").write_text(json.dumps(inputs))

        sh(["node", "ref_js/generate_witness.js", "ref_js/ref.wasm",
            "in.json", "w.wtns"], tmp)
        sh(["snarkjs", "wtns", "export", "json", "w.wtns", "w.json"], tmp)
        witness = json.loads((tmp / "w.json").read_text())

    # Outputs occupy witness slots 1..len(WIDTHS), in declaration order.
    for i, case in enumerate(cases):
        case["out"] = str(witness[1 + i])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=2))
    print(f"wrote {len(cases)} vectors to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
