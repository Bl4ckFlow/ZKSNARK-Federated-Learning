#!/usr/bin/env python3
"""Compile the circuit and run the phase-2 ceremony.

Groth16 needs a circuit-specific setup, and whoever holds its toxic waste can
forge a proof for any statement.  In a paper whose whole premise is "no trusted
party", that has to be said out loud and then dealt with -- so phase 2 here is
a real multi-contributor ceremony, one contribution per participating node,
sound as long as a single contributor discards their randomness.  That is a
natural fit for a P2P system: the participants are already there.

Phase 1 (the universal powers-of-tau) is a different matter.  We use a public
ceremony transcript when one is available and verify it; the ``--local-ptau``
fallback exists so the artifact still builds offline, and it prints a warning
because a phase 1 generated on the machine that runs the protocol is not a
trusted setup at all.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zkfl.config import DEFAULT_FP, FixedPointConfig  # noqa: E402
from zkfl.zk import CircuitArtifacts, assert_signal_order  # noqa: E402
from scripts.gen_circuit_main import write as write_main  # noqa: E402

PTAU_URLS = [
    "https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_{p}.ptau",
    "https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_{p}.ptau",
]


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{ROOT / 'node_modules' / '.bin'}:"
                                 f"{Path.home() / '.local' / 'bin'}:{os.environ.get('PATH', '')}"}
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT), **kw)
    if res.returncode != 0:
        raise SystemExit(
            f"command failed: {' '.join(cmd)}\n"
            f"stdout:\n{res.stdout[-2000:]}\nstderr:\n{res.stderr[-2000:]}"
        )
    return res


def entropy() -> str:
    return base64.b64encode(os.urandom(64)).decode()


def constraint_count(r1cs: Path) -> int:
    """Read the constraint count straight out of the .r1cs header."""
    import struct

    data = r1cs.read_bytes()
    assert data[:4] == b"r1cs", "not an r1cs file"
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
    raise ValueError("no header section in r1cs")


def power_for(constraints: int) -> int:
    p = 8
    while (1 << p) < constraints:
        p += 1
    return p


def ensure_ptau(power: int, allow_local: bool) -> Path:
    ptau_dir = ROOT / "artifacts" / "ptau"
    ptau_dir.mkdir(parents=True, exist_ok=True)
    public = ptau_dir / f"powersOfTau28_hez_final_{power:02d}.ptau"
    if public.exists():
        print(f"  phase 1: using public transcript {public.name}")
        sh(["snarkjs", "powersoftau", "verify", str(public)])
        return public

    import urllib.error
    import urllib.request

    for template in PTAU_URLS:
        url = template.format(p=f"{power:02d}")
        try:
            print(f"  phase 1: fetching {url}")
            with urllib.request.urlopen(url, timeout=120) as resp:
                if resp.status != 200:
                    continue
                public.write_bytes(resp.read())
            sh(["snarkjs", "powersoftau", "verify", str(public)])
            print(f"  phase 1: verified {public.name}")
            return public
        except (urllib.error.URLError, SystemExit, OSError) as exc:
            print(f"    unavailable ({type(exc).__name__})")
            public.unlink(missing_ok=True)

    local = ptau_dir / f"pot{power}_final.ptau"
    if local.exists():
        print(f"  phase 1: reusing local {local.name}")
        return local
    if not allow_local:
        raise SystemExit(
            "no public powers-of-tau transcript could be fetched.\n"
            "Re-run with --local-ptau to generate one locally for offline "
            "builds -- but note that a locally generated phase 1 is NOT a "
            "trusted setup and any result produced with it must say so."
        )

    print("\n  !! phase 1: generating LOCALLY. This is not a trusted setup:")
    print("  !! whoever runs this script could forge proofs. Offline builds only.\n")
    a = ptau_dir / f"pot{power}_0000.ptau"
    b = ptau_dir / f"pot{power}_0001.ptau"
    sh(["snarkjs", "powersoftau", "new", "bn128", str(power), str(a)])
    sh(["snarkjs", "powersoftau", "contribute", str(a), str(b),
        "--name=local-offline-build", f"-e={entropy()}"])
    sh(["snarkjs", "powersoftau", "prepare", "phase2", str(b), str(local)])
    a.unlink(missing_ok=True)
    b.unlink(missing_ok=True)
    return local


def phase2_ceremony(name: str, out: Path, r1cs: Path, ptau: Path,
                    contributors: list[str]) -> dict:
    """One zkey contribution per participant; sound if any one is honest."""
    stage = out / f"{name}_0000.zkey"
    sh(["snarkjs", "groth16", "setup", str(r1cs), str(ptau), str(stage)])

    transcript = []
    for i, who in enumerate(contributors, start=1):
        nxt = out / f"{name}_{i:04d}.zkey"
        t0 = time.perf_counter()
        sh(["snarkjs", "zkey", "contribute", str(stage), str(nxt),
            f"--name={who}", f"-e={entropy()}"])
        transcript.append({"index": i, "contributor": who,
                           "seconds": round(time.perf_counter() - t0, 3)})
        stage.unlink(missing_ok=True)
        stage = nxt

    final = out / f"{name}_final.zkey"
    # Beacon: makes the final key deterministic given the contributions and
    # removes the last contributor's ability to withhold.
    sh(["snarkjs", "zkey", "beacon", str(stage), str(final),
        "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20", "10",
        "-n=final beacon"])
    stage.unlink(missing_ok=True)

    verify = sh(["snarkjs", "zkey", "verify", str(r1cs), str(ptau), str(final)])
    ok = "ZKey Ok!" in verify.stdout or "ZKey Ok!" in verify.stderr
    if not ok:
        raise SystemExit(f"zkey verification failed:\n{verify.stdout}\n{verify.stderr}")

    sh(["snarkjs", "zkey", "export", "verificationkey", str(final),
        str(out / "verification_key.json")])
    return {"contributions": transcript, "beacon": True, "verified": True}


def build(cfg: FixedPointConfig, contributors: list[str], allow_local: bool,
          keep_existing: bool) -> CircuitArtifacts:
    cfg.check()
    name = cfg.circuit_name()
    art = CircuitArtifacts.for_config(cfg)

    if keep_existing and not art.missing():
        print(f"artifacts for {name} already present; nothing to do")
        return art

    main_path = write_main(cfg, ROOT / "circuits" / "main")
    art.directory.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] compiling {main_path.relative_to(ROOT)}")
    out = sh(["circom", str(main_path), "--r1cs", "--wasm", "--sym",
              "-o", str(art.directory), "-l", "node_modules"])
    print("      " + " | ".join(
        line.strip() for line in out.stdout.splitlines()
        if "constraints" in line or "inputs" in line
    ))

    n_constraints = constraint_count(art.r1cs)
    power = power_for(n_constraints)
    print(f"[2/4] {n_constraints} constraints -> 2^{power} powers of tau")

    ptau = ensure_ptau(power, allow_local)

    print(f"[3/4] phase-2 ceremony with {len(contributors)} contributors")
    ceremony = phase2_ceremony(name, art.directory, art.r1cs, ptau, contributors)

    print("[4/4] checking the public-input order matches zkfl/zk.py")
    assert_signal_order(art, cfg)

    (art.directory / "build_info.json").write_text(json.dumps({
        "circuit": name,
        "constraints": n_constraints,
        "ptau": ptau.name,
        "ptau_is_public_transcript": ptau.name.startswith("powersOfTau28_hez"),
        "config": {
            "scale": cfg.scale, "n_features": cfg.n_features,
            "batch_size": cfg.batch_size, "merkle_depth": cfg.merkle_depth,
            "x_bits": cfg.x_bits, "w_bits": cfg.w_bits, "z_bits": cfg.z_bits,
            "max_delta": cfg.max_delta,
        },
        "ceremony": ceremony,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))

    print(f"\nready: {art.directory.relative_to(ROOT)}")
    if not ptau.name.startswith("powersOfTau28_hez"):
        print("NOTE: built on a locally generated phase 1 -- not a trusted setup.")
    return art


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=int, default=DEFAULT_FP.n_features)
    ap.add_argument("--batch", type=int, default=DEFAULT_FP.batch_size)
    ap.add_argument("--depth", type=int, default=DEFAULT_FP.merkle_depth)
    ap.add_argument("--contributors", type=int, default=3,
                    help="phase-2 contributors to simulate (one per node in a "
                         "real deployment)")
    ap.add_argument("--local-ptau", action="store_true",
                    help="generate phase 1 locally if no public transcript is "
                         "reachable (offline builds; NOT a trusted setup)")
    ap.add_argument("--force", action="store_true", help="rebuild even if present")
    ap.add_argument("--v1", action="store_true",
                    help="also build the vacuous v1 circuit used as an ablation")
    args = ap.parse_args()

    for tool in ("circom", "snarkjs", "node"):
        if shutil.which(tool) is None and not (ROOT / "node_modules" / ".bin" / tool).exists():
            raise SystemExit(f"{tool} not found; see README.md for setup")

    cfg = FixedPointConfig(
        n_features=args.features, batch_size=args.batch, merkle_depth=args.depth
    )
    build(cfg, [f"node-{i}" for i in range(args.contributors)],
          args.local_ptau, keep_existing=not args.force)

    if args.v1:
        print("\n--- ablation: circuit v1 ---")
        v1_dir = ROOT / "artifacts" / "update_check_v1"
        v1_dir.mkdir(parents=True, exist_ok=True)
        sh(["circom", str(ROOT / "circuits" / "update_check_v1.circom"),
            "--r1cs", "--wasm", "--sym", "-o", str(v1_dir), "-l", "node_modules"])
        n = constraint_count(v1_dir / "update_check_v1.r1cs")
        ptau = ensure_ptau(power_for(max(n, 256)), args.local_ptau)
        phase2_ceremony("update_check_v1", v1_dir,
                        v1_dir / "update_check_v1.r1cs", ptau, ["ablation"])
        print(f"v1 built: {n} constraints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
