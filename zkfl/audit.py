"""The audit log: a verifiable training history, and the basis for catch-up.

An entry is everything a third party needs to re-derive one model transition
offline, months later, without trusting the node that wrote it:

    (h, mu_{h-1}, mu_h, ids(P), the proposals with their proofs, the
     certificate, the aggregation rule, a timestamp)

From that you can re-check every proof against mu_{h-1}, re-check the q
signatures on the certificate, recompute Aggregate(P), and confirm it hashes to
mu_h.  The hash chain over (mu_0, ..., mu_H) then makes any edit to an
intermediate model break every later link.

It doubles as the sync protocol: a node at height h asks a peer for entry h+1,
verifies it end to end (:meth:`ConsensusEngine.handle_sync_entry`), applies it,
and repeats.  New nodes bootstrap from genesis the same way.  The old design
recorded missed heights and never acted on them, so a node whose window expired
one vote short was stuck forever.

Persistence is SQLite so a crashed node resumes at its committed height instead
of restarting from genesis.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from zkfl.crypto import canonical_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    height        INTEGER PRIMARY KEY,
    prev_hash     TEXT NOT NULL,
    model_hash    TEXT NOT NULL,
    entry         TEXT NOT NULL,
    chain_hash    TEXT NOT NULL,
    committed_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id   TEXT PRIMARY KEY,
    offender      TEXT NOT NULL,
    kind          TEXT NOT NULL,
    height        INTEGER NOT NULL,
    body          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

GENESIS_CHAIN_HASH = "0" * 64


def chain_hash(previous: str, entry: dict[str, Any]) -> str:
    """Link an entry to its predecessor.

    The digest covers the previous link *and* the entry's canonical bytes, so
    rewriting history requires reproducing every later link.
    """
    body = canonical_json({
        "height": entry["height"],
        "prev_model_hash": entry["prev_model_hash"],
        "model_hash": entry["model_hash"],
        "proposal_ids": sorted(entry.get("proposal_ids", [])),
        "certificate": entry["certificate"],
    })
    return hashlib.sha256(previous.encode() + b"|" + body).hexdigest()


class AuditLog:
    """Append-only committed history, optionally backed by SQLite."""

    def __init__(self, path: Optional[Path | str] = None):
        self.path = Path(path) if path else None
        self._entries: list[dict[str, Any]] = []
        self._chain: list[str] = []
        self._db: Optional[sqlite3.Connection] = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path))
            self._db.executescript(SCHEMA)
            self._db.commit()
            self._load()

    def _load(self) -> None:
        assert self._db is not None
        rows = self._db.execute(
            "SELECT entry, chain_hash FROM audit ORDER BY height"
        ).fetchall()
        for raw, link in rows:
            self._entries.append(json.loads(raw))
            self._chain.append(link)

    # -- writing ------------------------------------------------------------

    def append(self, entry: dict[str, Any]) -> str:
        expected = self.height + 1
        if entry["height"] != expected:
            raise ValueError(
                f"audit log is at height {self.height}; refusing entry "
                f"{entry['height']}"
            )
        if self._entries and entry["prev_model_hash"] != self._entries[-1]["model_hash"]:
            raise ValueError("entry does not extend the chain")

        link = chain_hash(self._chain[-1] if self._chain else GENESIS_CHAIN_HASH, entry)
        self._entries.append(entry)
        self._chain.append(link)

        if self._db is not None:
            self._db.execute(
                "INSERT OR REPLACE INTO audit "
                "(height, prev_hash, model_hash, entry, chain_hash, committed_at) "
                "VALUES (?,?,?,?,?,?)",
                (entry["height"], entry["prev_model_hash"], entry["model_hash"],
                 json.dumps(entry, separators=(",", ":")), link,
                 float(entry.get("committed_at_ms", 0)) / 1000.0),
            )
            self._db.commit()
        return link

    def record_evidence(self, evidence_id: str, offender: str, kind: str,
                        height: int, body: dict[str, Any]) -> None:
        if self._db is None:
            return
        self._db.execute(
            "INSERT OR IGNORE INTO evidence (evidence_id, offender, kind, height, body) "
            "VALUES (?,?,?,?,?)",
            (evidence_id, offender, kind, height,
             json.dumps(body, separators=(",", ":"))),
        )
        self._db.commit()

    # -- reading ------------------------------------------------------------

    @property
    def height(self) -> int:
        return self._entries[-1]["height"] if self._entries else 0

    @property
    def head(self) -> str:
        return self._chain[-1] if self._chain else GENESIS_CHAIN_HASH

    def entry(self, height: int) -> Optional[dict[str, Any]]:
        for candidate in self._entries:
            if candidate["height"] == height:
                return candidate
        return None

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._entries)

    # -- verification -------------------------------------------------------

    def verify_chain(self) -> tuple[bool, str]:
        """Re-derive every link.  Returns (ok, reason)."""
        previous = GENESIS_CHAIN_HASH
        for i, entry in enumerate(self._entries):
            if entry["height"] != i + 1:
                return False, f"height gap at index {i}"
            if i and entry["prev_model_hash"] != self._entries[i - 1]["model_hash"]:
                return False, f"broken parent link at height {entry['height']}"
            link = chain_hash(previous, entry)
            if link != self._chain[i]:
                return False, f"chain hash mismatch at height {entry['height']}"
            previous = link
        return True, "ok"

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None


def verify_offline(
    entries: Iterable[dict[str, Any]],
    genesis_model: dict[str, Any],
    quorum: int,
    members: frozenset[str],
    fp,
    protocol,
    backend,
) -> tuple[bool, str]:
    """Re-verify a whole audit log from scratch, as an outside auditor would.

    Deliberately shares no state with the running node: it re-parses the
    entries, re-checks the certificates and every proof against the model as of
    that height, and recomputes each aggregate.  ``experiments/e5_audit.py``
    runs this over a finished simulation.
    """
    from zkfl.aggregation import AggregationError, aggregate
    from zkfl.messages import Certificate, MalformedMessage, Proposal
    from zkfl.model import Model
    from zkfl.zk import PublicInputs
    from zkfl.fixedpoint import quantize

    eta = quantize(protocol.learning_rate, fp.scale)
    model = Model.from_dict(genesis_model)

    for entry in entries:
        try:
            certificate = Certificate.from_dict(entry["certificate"])
            proposals = [Proposal.from_dict(p) for p in entry["proposals"]]
        except (KeyError, TypeError, MalformedMessage) as exc:
            return False, f"height {entry.get('height')}: malformed ({exc})"

        if entry["prev_model_hash"] != model.digest():
            return False, f"height {entry['height']}: parent hash mismatch"

        # The entry carries the model twice: as a hash, which everything else
        # is checked against, and in full, which is what a reader actually
        # consumes.  Checking only the hash leaves the readable copy free to
        # say anything, so check that the two agree before trusting either.
        try:
            stated = Model.from_dict(entry["model"])
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"height {entry['height']}: unreadable model ({exc})"
        if stated.digest() != entry["model_hash"]:
            return False, f"height {entry['height']}: model does not match its hash"
        if not certificate.is_valid(quorum, members):
            return False, f"height {entry['height']}: certificate lacks a quorum"

        for proposal in proposals:
            if not proposal.verify_signature():
                return False, f"height {entry['height']}: bad proposal signature"
            expected = PublicInputs.from_models(
                before=model, after=proposal.model, eta=eta,
                data_root=int(entry["roots"][proposal.proposer])
                if "roots" in entry else 0,
            )
            if not backend.verify(proposal.proof, expected):
                return False, (f"height {entry['height']}: proof from "
                               f"{proposal.proposer[:8]} does not verify")

        try:
            result = aggregate(
                model, proposals, fp,
                rule=entry["aggregation"]["rule"],
                permille=entry["aggregation"]["trim_permille"],
            )
        except AggregationError as exc:
            return False, f"height {entry['height']}: aggregation failed ({exc})"
        if result.model.digest() != entry["model_hash"]:
            return False, f"height {entry['height']}: aggregate does not match"
        model = stated

    return True, "ok"
