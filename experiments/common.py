"""Shared plumbing for the evaluation scripts.

Every experiment writes a JSON file under ``results/`` and prints a table, so
the numbers in the paper can be regenerated with one command and diffed
against a previous run rather than retyped.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))


def environment() -> dict[str, Any]:
    """Recorded with every result -- a proving time means nothing without it."""

    def version(cmd: list[str]) -> str:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                                 cwd=str(ROOT))
            return (out.stdout or out.stderr).strip().splitlines()[0][:80]
        except Exception:
            return "unavailable"

    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu,
        "node": version(["node", "--version"]),
        "circom": version(["circom", "--version"]),
        "snarkjs": version([str(ROOT / "node_modules" / ".bin" / "snarkjs"), "--version"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def save(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps({"environment": environment(), **payload}, indent=2))
    print(f"\nwrote {path.relative_to(ROOT)}")
    return path


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    rule = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(c.ljust(w) for c, w in zip(row, widths))
                     for row in cells)
    return f"{line}\n{rule}\n{body}"


def constraint_count(r1cs: Path) -> int:
    import struct

    data = r1cs.read_bytes()
    n_sections = struct.unpack("<I", data[8:12])[0]
    off = 12
    for _ in range(n_sections):
        s_type = struct.unpack("<I", data[off:off + 4])[0]
        s_size = struct.unpack("<Q", data[off + 4:off + 12])[0]
        if s_type == 1:
            head = data[off + 12:off + 12 + s_size]
            fs = struct.unpack("<I", head[0:4])[0]
            o = 4 + fs
            return struct.unpack("<I", head[o + 24:o + 28])[0]
        off += 12 + s_size
    raise ValueError("no header in r1cs")
