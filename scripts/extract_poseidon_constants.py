#!/usr/bin/env python3
"""Lift the Poseidon round constants out of circomlib's own source.

The Python hash in :mod:`zkfl.poseidon` and the Poseidon inside the circuit
must agree on every one of these numbers, so we read them from the exact
``poseidon_constants.circom`` that circom compiles rather than from a second
copy that could drift.  ``tests/test_poseidon.py`` closes the loop by checking
the Python output against a witness generated from a real circuit.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "node_modules" / "circomlib" / "circuits" / "poseidon_constants.circom"
OUT = ROOT / "zkfl" / "poseidon_constants.json"

FUNCS = ("POSEIDON_C", "POSEIDON_M", "POSEIDON_P", "POSEIDON_S")
T_RANGE = range(2, 18)


def _function_body(text: str, name: str) -> str:
    start = text.index(f"function {name}(t) {{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    raise ValueError(f"unterminated function {name}")


def _parse_array(literal: str):
    # circom hex literals -> decimal, then it is plain JSON.
    literal = re.sub(r"//[^\n]*", "", literal)
    literal = re.sub(r"0x([0-9a-fA-F]+)", lambda m: str(int(m.group(1), 16)), literal)
    literal = re.sub(r",\s*\]", "]", literal)
    return json.loads(literal)


def extract(name: str, text: str) -> dict[str, object]:
    body = _function_body(text, name)
    out: dict[str, object] = {}
    branches = [(m.start(), int(m.group(1)))
                for m in re.finditer(r"t\s*==\s*(\d+)", body)]
    for idx, (pos, t) in enumerate(branches):
        end = branches[idx + 1][0] if idx + 1 < len(branches) else len(body)
        chunk = body[pos:end]
        ret = chunk.index("return")
        open_br = chunk.index("[", ret)
        depth = 0
        for i in range(open_br, len(chunk)):
            if chunk[i] == "[":
                depth += 1
            elif chunk[i] == "]":
                depth -= 1
                if depth == 0:
                    out[str(t)] = _parse_array(chunk[open_br:i + 1])
                    break
        else:
            raise ValueError(f"unterminated array for {name} t={t}")
    return out


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}; run `npm install` first", file=sys.stderr)
        return 1
    text = SRC.read_text()
    data = {name: extract(name, text) for name in FUNCS}

    # Sanity: the circuit sizes C as t*8 + nRoundsP and S as nRoundsP*(2t-1).
    n_rounds_p = [56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68]
    for t in T_RANGE:
        key = str(t)
        if key not in data["POSEIDON_C"]:
            continue
        p = n_rounds_p[t - 2]
        assert len(data["POSEIDON_C"][key]) == t * 8 + p, f"C length wrong for t={t}"
        assert len(data["POSEIDON_S"][key]) == p * (2 * t - 1), f"S length wrong for t={t}"
        assert len(data["POSEIDON_M"][key]) == t, f"M shape wrong for t={t}"
        assert len(data["POSEIDON_P"][key]) == t, f"P shape wrong for t={t}"

    OUT.write_text(json.dumps(data, separators=(",", ":")))
    sizes = ", ".join(f"t={t}" for t in sorted(map(int, data["POSEIDON_C"])))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB) covering {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
