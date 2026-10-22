"""The global model: an integer vector, not a float one.

Carrying the model in scaled integers is what makes cross-machine determinism
free.  Two nodes that apply the same committed update hold bitwise-identical
state because there is no floating-point reduction anywhere in the path -- the
old design asserted determinism and tested it on a single machine, which is a
different claim.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from zkfl.config import FixedPointConfig
from zkfl.fixedpoint import dequantize_vector


@dataclass(frozen=True)
class Model:
    """Scaled-integer logistic-regression parameters."""

    w: tuple[int, ...]
    b: int
    scale: int

    @classmethod
    def zeros(cls, cfg: FixedPointConfig) -> "Model":
        return cls(w=(0,) * cfg.n_features, b=0, scale=cfg.scale)

    @classmethod
    def from_floats(cls, w: Sequence[float], b: float, cfg: FixedPointConfig) -> "Model":
        from zkfl.fixedpoint import quantize, quantize_vector

        return cls(
            w=tuple(quantize_vector(w, cfg.scale)),
            b=quantize(b, cfg.scale),
            scale=cfg.scale,
        )

    # -- canonical encoding -------------------------------------------------

    def canonical_bytes(self) -> bytes:
        """The only serialisation used for hashing or signing.

        Integers with sorted keys: no float formatting, no rounding, no
        platform-dependent byte order.
        """
        return json.dumps(
            {"b": self.b, "scale": self.scale, "w": list(self.w)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    # -- interop ------------------------------------------------------------

    def to_floats(self) -> tuple[list[float], float]:
        return dequantize_vector(self.w, self.scale), self.b / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {"w": list(self.w), "b": self.b, "scale": self.scale}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Model":
        w = data["w"]
        if not isinstance(w, (list, tuple)) or not all(isinstance(v, int) for v in w):
            raise ValueError("model weights must be a list of integers")
        if not isinstance(data["b"], int) or not isinstance(data["scale"], int):
            raise ValueError("model bias and scale must be integers")
        return cls(w=tuple(w), b=data["b"], scale=data["scale"])

    def __len__(self) -> int:
        return len(self.w)
