"""Poseidon over BN254, mirroring circomlib's ``PoseidonEx`` exactly.

The circuit uses the optimised Hades layout -- full rounds with the MDS matrix
M, a pre-round with the sparse matrix P, then partial rounds each using one row
of S -- so this implementation follows the same shape rather than the textbook
one.  The two are not interchangeable: the constant vector C that circomlib
ships is sized ``t*8 + nRoundsP``, which only the optimised schedule consumes.

Constants come from ``poseidon_constants.json``, lifted straight out of the
circomlib source by ``scripts/extract_poseidon_constants.py``.
``tests/test_poseidon.py`` checks the output against a witness from a real
compiled circuit, so a drift here fails the suite rather than silently
producing roots no verifier will accept.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from zkfl.config import FIELD_PRIME as P

_CONSTANTS_PATH = Path(__file__).with_name("poseidon_constants.json")

N_ROUNDS_F = 8
N_ROUNDS_P = [56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68]


@lru_cache(maxsize=1)
def _all_constants() -> dict:
    if not _CONSTANTS_PATH.exists():
        raise FileNotFoundError(
            f"{_CONSTANTS_PATH} is missing; run "
            "scripts/extract_poseidon_constants.py after `npm install`"
        )
    return json.loads(_CONSTANTS_PATH.read_text())


@lru_cache(maxsize=32)
def _params(t: int):
    raw = _all_constants()
    key = str(t)
    if key not in raw["POSEIDON_C"]:
        raise ValueError(f"no Poseidon constants for t={t} (2..17 supported)")
    return (
        tuple(raw["POSEIDON_C"][key]),
        tuple(tuple(row) for row in raw["POSEIDON_M"][key]),
        tuple(tuple(row) for row in raw["POSEIDON_P"][key]),
        tuple(raw["POSEIDON_S"][key]),
    )


def _pow5(x: int) -> int:
    x2 = x * x % P
    return x2 * x2 % P * x % P


def _mix(state: Sequence[int], m: Sequence[Sequence[int]], t: int) -> list[int]:
    # circomlib's Mix computes out[i] = sum_j M[j][i] * in[j] -- the transpose
    # of the obvious reading.  Getting this backwards yields a plausible-looking
    # hash that no circuit agrees with.
    return [sum(m[j][i] * state[j] for j in range(t)) % P for i in range(t)]


def permute(inputs: Sequence[int], initial_state: int = 0) -> list[int]:
    """The full Poseidon permutation on ``[initial_state, *inputs]``."""
    t = len(inputs) + 1
    c, m, p, s = _params(t)
    half_f = N_ROUNDS_F // 2
    n_p = N_ROUNDS_P[t - 2]

    state = [initial_state % P] + [int(x) % P for x in inputs]
    state = [(state[i] + c[i]) % P for i in range(t)]

    # first half of the full rounds
    for r in range(half_f - 1):
        state = [_pow5(v) for v in state]
        state = [(state[j] + c[(r + 1) * t + j]) % P for j in range(t)]
        state = _mix(state, m, t)

    state = [_pow5(v) for v in state]
    state = [(state[j] + c[half_f * t + j]) % P for j in range(t)]
    state = _mix(state, p, t)

    # partial rounds, each mixing with one sparse row of S
    for r in range(n_p):
        state[0] = _pow5(state[0])
        state[0] = (state[0] + c[(half_f + 1) * t + r]) % P
        row = (2 * t - 1) * r
        acc = sum(s[row + i] * state[i] for i in range(t)) % P
        new = [acc] + [
            (state[i] + state[0] * s[row + t + i - 1]) % P for i in range(1, t)
        ]
        state = new

    # second half of the full rounds
    for r in range(half_f - 1):
        state = [_pow5(v) for v in state]
        base = (half_f + 1) * t + n_p + r * t
        state = [(state[j] + c[base + j]) % P for j in range(t)]
        state = _mix(state, m, t)

    state = [_pow5(v) for v in state]
    return _mix(state, m, t)


def poseidon(inputs: Iterable[int]) -> int:
    """circomlib ``Poseidon(nInputs)``: the first output word of the permutation."""
    values = list(inputs)
    if not 1 <= len(values) <= 16:
        raise ValueError(f"Poseidon takes 1..16 inputs, got {len(values)}")
    return permute(values)[0]


def poseidon2(left: int, right: int) -> int:
    """Two-input Poseidon, the Merkle compression function."""
    return poseidon((left, right))


def to_field(value: int) -> int:
    """Map a signed integer into the field the way circom does."""
    return int(value) % P
