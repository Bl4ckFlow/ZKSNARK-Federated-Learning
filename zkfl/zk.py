"""Groth16 proving and verification, with the public inputs actually bound.

Two things changed relative to the previous implementation, and both are the
difference between a proof that means something and one that does not:

1. **Strict is the default and there is no fail-open mode.**  The old
   ``enable_strict_zk`` defaulted to False and a plain string counted as a
   proof.  Here, the only non-cryptographic path is
   :class:`NullProofBackend`, which refuses to run unless explicitly
   constructed for a test and which every protocol-level entry point declines
   to accept in production configuration.

2. **The verifier supplies the public inputs from its own state.**  Checking
   ``groth16.verify(vk, publicSignals, proof)`` alone proves only that the
   prover's *own* claimed inputs are consistent -- it says nothing about which
   model was trained.  :meth:`Groth16Backend.verify` therefore rebuilds the
   expected public signal vector from the verifier's committed model, the
   protocol learning rate and the proposer's registered data root, and rejects
   any mismatch before it even calls the pairing check.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from zkfl.config import FIELD_PRIME, FixedPointConfig
from zkfl.model import Model
from zkfl.trainer import TrainingWitness

ROOT = Path(__file__).resolve().parent.parent


class ProofError(RuntimeError):
    """Proving failed -- usually an unsatisfiable witness."""


@dataclass(frozen=True)
class PublicInputs:
    """The statement, in the circuit's declared signal order.

    Order matters and is not negotiable: it follows
    ``component main {public [wOld, bOld, wNew, bNew, eta, dataRoot]}``.
    :func:`assert_signal_order` re-derives it from the compiled ``.sym`` so a
    silent reordering after a circuit edit fails loudly instead of turning
    verification into a no-op.
    """

    w_old: tuple[int, ...]
    b_old: int
    w_new: tuple[int, ...]
    b_new: int
    eta: int
    data_root: int

    def to_signals(self) -> list[str]:
        values = [*self.w_old, self.b_old, *self.w_new, self.b_new, self.eta,
                  self.data_root]
        return [str(v % FIELD_PRIME) for v in values]

    @classmethod
    def from_models(
        cls,
        before: Model,
        after: Model,
        eta: int,
        data_root: int,
    ) -> "PublicInputs":
        return cls(
            w_old=tuple(before.w),
            b_old=before.b,
            w_new=tuple(after.w),
            b_new=after.b,
            eta=eta,
            data_root=data_root,
        )


def witness_input(witness: TrainingWitness) -> dict[str, Any]:
    """The circom input JSON for a training witness."""
    pub = PublicInputs.from_models(
        witness.model_before, witness.model_after, witness.eta, witness.data_root
    )
    f = lambda v: str(int(v) % FIELD_PRIME)  # noqa: E731
    return {
        "wOld": [f(v) for v in pub.w_old],
        "bOld": f(pub.b_old),
        "wNew": [f(v) for v in pub.w_new],
        "bNew": f(pub.b_new),
        "eta": f(pub.eta),
        "dataRoot": f(pub.data_root),
        "x": [[f(v) for v in row] for row in witness.xs],
        "y": [f(v) for v in witness.ys],
        "pathElements": [f(v) for v in witness.merkle.elements],
        "pathIndex": [f(v) for v in witness.merkle.indices],
    }


@dataclass
class CircuitArtifacts:
    """Where a compiled circuit lives."""

    name: str
    directory: Path

    @property
    def wasm(self) -> Path:
        return self.directory / f"{self.name}_js" / f"{self.name}.wasm"

    @property
    def zkey(self) -> Path:
        return self.directory / f"{self.name}_final.zkey"

    @property
    def vkey(self) -> Path:
        return self.directory / "verification_key.json"

    @property
    def r1cs(self) -> Path:
        return self.directory / f"{self.name}.r1cs"

    @property
    def sym(self) -> Path:
        return self.directory / f"{self.name}.sym"

    def missing(self) -> list[str]:
        return [
            str(p) for p in (self.wasm, self.zkey, self.vkey) if not p.exists()
        ]

    @classmethod
    def for_config(cls, cfg: FixedPointConfig, base: Path | None = None) -> "CircuitArtifacts":
        name = cfg.circuit_name()
        base = base or (ROOT / "artifacts")
        return cls(name=name, directory=base / name)


def assert_signal_order(artifacts: CircuitArtifacts, cfg: FixedPointConfig) -> None:
    """Confirm the compiled circuit's public inputs are in the order we assume.

    snarkjs emits ``public.json`` in witness-index order, so if a circuit edit
    ever reshuffles the ``main`` component's public list, the verifier would
    quietly start comparing weights against a data root.  Reading the ``.sym``
    makes that a startup failure instead.
    """
    if not artifacts.sym.exists():
        raise FileNotFoundError(f"missing {artifacts.sym}; recompile the circuit")

    expected = (
        [f"main.wOld[{i}]" for i in range(cfg.n_features)]
        + ["main.bOld"]
        + [f"main.wNew[{i}]" for i in range(cfg.n_features)]
        + ["main.bNew", "main.eta", "main.dataRoot"]
    )
    # .sym lines are  witnessIdx,nodeIdx,componentIdx,name
    by_index: dict[int, str] = {}
    for line in artifacts.sym.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        # keep the first (canonical) name for each witness slot
        by_index.setdefault(idx, parts[3])

    # witness slot 0 is the constant 1; public inputs follow immediately
    actual = [by_index.get(i) for i in range(1, len(expected) + 1)]
    if actual != expected:
        raise RuntimeError(
            "circuit public-input order changed.\n"
            f"  expected: {expected}\n"
            f"  compiled: {actual}\n"
            "Update PublicInputs.to_signals to match, or restore the order in "
            "circuits/main/."
        )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class ProofBackend:
    """Interface shared by the real prover and the test fixture."""

    strict = True

    def prove(self, witness: TrainingWitness) -> dict[str, Any]:
        raise NotImplementedError

    def verify(self, proof: Any, expected: PublicInputs) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass


class Groth16Backend(ProofBackend):
    """Real Groth16 over a compiled circuit, via a persistent Node worker."""

    strict = True

    def __init__(
        self,
        cfg: FixedPointConfig,
        artifacts: CircuitArtifacts | None = None,
        *,
        check_signal_order: bool = True,
    ):
        cfg.check()
        self.cfg = cfg
        self.artifacts = artifacts or CircuitArtifacts.for_config(cfg)
        missing = self.artifacts.missing()
        if missing:
            raise FileNotFoundError(
                "circuit artifacts missing: " + ", ".join(missing)
                + "\nRun: python scripts/setup_circuit.py"
            )
        if check_signal_order:
            assert_signal_order(self.artifacts, cfg)
        self._assert_build_matches(cfg)

        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 0
        self.prove_ms: list[float] = []
        self.verify_ms: list[float] = []

    def _assert_build_matches(self, cfg: FixedPointConfig) -> None:
        """Refuse a zkey compiled from different parameters.

        The artifact directory is named after (features, batch, depth), so a
        change to a bit width or the clipping bound leaves the name alone and
        would otherwise be discovered as an unexplained "proof rejected".
        """
        info_path = self.artifacts.directory / "build_info.json"
        if not info_path.exists():
            return
        built = json.loads(info_path.read_text()).get("config", {})
        current = {
            "scale": cfg.scale, "n_features": cfg.n_features,
            "batch_size": cfg.batch_size, "merkle_depth": cfg.merkle_depth,
            "x_bits": cfg.x_bits, "w_bits": cfg.w_bits, "z_bits": cfg.z_bits,
            "max_delta": cfg.max_delta,
        }
        drift = {k: (built.get(k), v) for k, v in current.items()
                 if k in built and built[k] != v}
        if drift:
            detail = ", ".join(f"{k}: built={a} now={b}" for k, (a, b) in drift.items())
            raise RuntimeError(
                f"{self.artifacts.name} was compiled with different parameters "
                f"({detail}). Re-run scripts/setup_circuit.py --force."
            )

    # -- worker lifecycle ---------------------------------------------------

    def _worker(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        node = shutil.which("node")
        if node is None:
            raise ProofError("node is not on PATH; the Groth16 backend needs it")
        self._proc = subprocess.Popen(
            [
                node,
                str(Path(__file__).with_name("prover_worker.js")),
                str(self.artifacts.wasm),
                str(self.artifacts.zkey),
                str(self.artifacts.vkey),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )
        return self._proc

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            proc = self._worker()
            self._next_id += 1
            payload = {"id": self._next_id, **payload}
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except (BrokenPipeError, ValueError) as exc:
                self._proc = None
                raise ProofError(f"prover worker died: {exc}") from exc
            if not line:
                err = ""
                if proc.stderr is not None:
                    try:
                        err = proc.stderr.read() or ""
                    except Exception:  # pragma: no cover - diagnostics only
                        pass
                self._proc = None
                raise ProofError(f"prover worker produced no output. {err[:400]}")
            return json.loads(line)

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    assert self._proc.stdin is not None
                    self._proc.stdin.close()
                    self._proc.wait(timeout=5)
                except Exception:  # pragma: no cover
                    self._proc.kill()
            self._proc = None

    # -- proving / verifying ------------------------------------------------

    def prove(self, witness: TrainingWitness) -> dict[str, Any]:
        resp = self._call({"op": "prove", "input": witness_input(witness)})
        if not resp.get("ok"):
            raise ProofError(resp.get("error", "unknown proving failure"))
        self.prove_ms.append(float(resp.get("ms", 0.0)))
        return {
            "scheme": "groth16",
            "circuit": self.artifacts.name,
            "proof": resp["proof"],
            "public_signals": resp["publicSignals"],
        }

    def verify(self, proof: Any, expected: PublicInputs) -> bool:
        """Verify against public inputs the *verifier* chose.

        A proof whose public signals do not equal ``expected`` is rejected here,
        before the pairing check: otherwise a proposer could prove a perfectly
        valid step from a model nobody agreed on.
        """
        if not isinstance(proof, dict):
            return False
        if proof.get("scheme") != "groth16":
            return False
        signals = proof.get("public_signals")
        if not isinstance(signals, list):
            return False
        if [str(s) for s in signals] != expected.to_signals():
            return False
        resp = self._call(
            {"op": "verify", "proof": proof["proof"], "publicSignals": signals}
        )
        if not resp.get("ok"):
            return False
        self.verify_ms.append(float(resp.get("ms", 0.0)))
        return bool(resp.get("valid"))


class NullProofBackend(ProofBackend):
    """A stand-in for tests that do not exercise the cryptography.

    It still binds the statement -- the "proof" is a hash of the public inputs,
    so a wrong model is rejected -- but it proves nothing about training.  It
    must be constructed deliberately, refuses to exist when
    ``ZKFL_FORBID_NULL_BACKEND`` is set, and every proposal it produces is
    tagged so an audit log can never be mistaken for a verified one.
    """

    strict = False

    def __init__(self, *, i_understand_this_is_not_a_proof: bool = False):
        if not i_understand_this_is_not_a_proof:
            raise RuntimeError(
                "NullProofBackend proves nothing; construct it with "
                "i_understand_this_is_not_a_proof=True and only in tests"
            )
        if os.environ.get("ZKFL_FORBID_NULL_BACKEND"):
            raise RuntimeError(
                "ZKFL_FORBID_NULL_BACKEND is set: refusing to run without real proofs"
            )

    @staticmethod
    def _tag(expected: PublicInputs) -> str:
        import hashlib

        return hashlib.sha256("|".join(expected.to_signals()).encode()).hexdigest()

    def prove(self, witness: TrainingWitness) -> dict[str, Any]:
        expected = PublicInputs.from_models(
            witness.model_before, witness.model_after, witness.eta, witness.data_root
        )
        return {
            "scheme": "null",
            "unverified": True,
            "public_signals": expected.to_signals(),
            "tag": self._tag(expected),
        }

    def verify(self, proof: Any, expected: PublicInputs) -> bool:
        if not isinstance(proof, dict) or proof.get("scheme") != "null":
            return False
        return proof.get("tag") == self._tag(expected)


def make_backend(
    cfg: FixedPointConfig, strict: bool, artifacts: CircuitArtifacts | None = None
) -> ProofBackend:
    """Pick a backend, defaulting to the real one and failing closed."""
    if strict:
        return Groth16Backend(cfg, artifacts)
    return NullProofBackend(i_understand_this_is_not_a_proof=True)
