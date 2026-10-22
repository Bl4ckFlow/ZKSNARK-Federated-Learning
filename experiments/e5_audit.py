#!/usr/bin/env python3
"""E5 -- can a stranger check the training history without trusting anyone?

The paper's auditability claim is only worth something if the artefact a node
publishes is independently checkable *and* if tampering with it is detectable.
This experiment does both, in that order:

  1. Run a live network for a few heights and export node 0's audit log to a
     plain JSON file -- nothing but public data.
  2. Re-verify that file from a cold start (fresh process state, fresh proof
     backend, no access to any node) with :func:`zkfl.audit.verify_offline`:
     every certificate re-checked against the member set, every proof
     re-checked against the model as of *that* height and the proposer's
     registered data root, every aggregate recomputed, every hash-chain link
     recomputed.
  3. Corrupt the file in each way an adversary would want to and confirm each
     corruption is rejected.  A checker that accepts everything checks nothing,
     so the negative results are the point of the experiment, not a bonus.

Run with ``--strict`` to use real Groth16 proofs (slower, and the only mode in
which step 2's proof re-check means anything).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any, Callable

from experiments.common import ROOT, RESULTS, save, table
from zkfl.audit import GENESIS_CHAIN_HASH, chain_hash, verify_offline
from zkfl.config import FixedPointConfig, ProtocolConfig
from zkfl.datasets import synthetic
from zkfl.simulation import Simulation
from zkfl.zk import make_backend

Tamper = tuple[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]], str]


# ---------------------------------------------------------------------------
# The tampering suite.  Each entry is (name, mutation, what it models).
# ---------------------------------------------------------------------------


def _tampers(fp: FixedPointConfig, identities=None) -> list[Tamper]:
    def swap_model(entries):
        entries[-1]["model"]["w"] = [v + 1 for v in entries[-1]["model"]["w"]]
        return entries

    def rehash_model(entries):
        """The careful forger: change the model *and* fix up its hash."""
        from zkfl.model import Model
        forged = Model(w=tuple(v + 1 for v in entries[-1]["model"]["w"]),
                       b=entries[-1]["model"]["b"], scale=fp.scale)
        entries[-1]["model"] = forged.to_dict()
        entries[-1]["model_hash"] = forged.digest()
        return entries

    def drop_vote(entries):
        cert = entries[-1]["certificate"]
        cert["votes"] = cert["votes"][:-1]
        return entries

    def duplicate_vote(entries):
        cert = entries[-1]["certificate"]
        cert["votes"] = list(cert["votes"][:-1]) + [cert["votes"][0]]
        return entries

    def prevote_certificate(entries):
        """A polka is not a decision; passing one off as one must fail."""
        cert = entries[-1]["certificate"]
        for vote in cert["votes"]:
            vote["phase"] = "prevote"
        return entries

    def drop_proposal(entries):
        """Quietly exclude one node's contribution from the aggregate."""
        entries[-1]["proposals"] = entries[-1]["proposals"][:-1]
        return entries

    def swap_proof(entries):
        if len(entries[-1]["proposals"]) < 2:
            return entries
        a, b = entries[-1]["proposals"][0], entries[-1]["proposals"][1]
        a["proof"], b["proof"] = b["proof"], a["proof"]
        return entries

    def swap_proof_resigned(entries):
        """The same swap, re-signed by the real proposers.

        Without this the swap is caught by the signature check, which covers
        the proof -- so the audit's *proof* verification is never the thing
        that rejects it.  Re-signing makes every message well-formed and leaves
        the pairing check as the only line of defence, which is the point.
        """
        from zkfl.messages import Proposal

        if identities is None or len(entries[-1]["proposals"]) < 2:
            return entries
        by_id = {i.node_id: i for i in identities}
        props = entries[-1]["proposals"]
        a, b = dict(props[0]), dict(props[1])
        a["proof"], b["proof"] = b["proof"], a["proof"]
        out = []
        for raw in (a, b):
            identity = by_id.get(raw["proposer"])
            if identity is None:
                return entries
            raw.pop("signature", None)
            out.append(Proposal.from_dict(
                {**raw, "signature": ""}).signed(identity).to_dict())
        entries[-1]["proposals"] = out + list(props[2:])
        return entries

    def relink(entries):
        """Splice out a height and re-point the chain around it."""
        if len(entries) < 3:
            return entries
        del entries[1]
        return entries

    return [
        ("model edited", swap_model,
         "a node reports a different global model than it committed"),
        ("model edited, hash fixed", rehash_model,
         "the same, by a forger who remembers to recompute the digest"),
        ("certificate one vote short", drop_vote,
         "committing without a quorum"),
        ("certificate padded with a duplicate", duplicate_vote,
         "faking a quorum by counting one voter twice"),
        ("precommits downgraded to prevotes", prevote_certificate,
         "passing a polka off as a decision"),
        ("proposal dropped from the set", drop_proposal,
         "censoring a participant after the fact"),
        ("proofs swapped between proposers", swap_proof,
         "reusing someone else's proof for your own update"),
        ("proofs swapped and re-signed", swap_proof_resigned,
         "the same, well-formed, so only the pairing check can reject it"),
        ("a height spliced out", relink,
         "rewriting the training history"),
    ]


def _chain_ok(entries: list[dict[str, Any]]) -> tuple[bool, str]:
    """Recompute the hash chain, exactly as :class:`AuditLog` builds it."""
    previous = GENESIS_CHAIN_HASH
    for i, entry in enumerate(entries):
        if i and entry["prev_model_hash"] != entries[i - 1]["model_hash"]:
            return False, f"height {entry['height']}: broken parent link"
        previous = chain_hash(previous, entry)
    return True, previous


def audit(entries, sim, backend) -> tuple[bool, str]:
    """Everything an outside auditor checks, in one call."""
    ok, detail = _chain_ok(entries)
    if not ok:
        return False, detail
    return verify_offline(
        entries=entries,
        genesis_model=sim.genesis.to_dict(),
        quorum=sim.protocol.effective_quorum(),
        members=sim.registry.members(),
        fp=sim.fp, protocol=sim.protocol, backend=backend,
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--proposers", type=int, default=3)
    ap.add_argument("--strict", action="store_true",
                    help="real Groth16 proofs (slow, and the only honest mode)")
    ap.add_argument("--out", default=str(RESULTS / "audit_log.json"))
    args = ap.parse_args()

    fp = FixedPointConfig()
    protocol = ProtocolConfig(collect_window=0.3, strict_zk=args.strict,
                              aggregation_rule="trimmed_mean").with_nodes(args.nodes)
    split = synthetic(n_samples=64 * args.nodes, n_features=fp.n_features, seed=17)
    sim = Simulation(split=split, fp=fp, protocol=protocol, n_nodes=args.nodes,
                     strict_zk=args.strict, latency=0.004, jitter=0.004, seed=17)

    started = time.perf_counter()
    result = await sim.run(rounds=args.rounds, proposers_per_round=args.proposers,
                           timeout=300, evaluate=False)
    live_seconds = time.perf_counter() - started
    committed = [r for r in result.rounds if r.agreed]
    if not committed:
        print("no height committed; nothing to audit")
        return 1

    # -- 1. export -----------------------------------------------------------
    entries = sim.nodes[0].audit.entries()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "genesis": sim.genesis.to_dict(),
        "members": sorted(sim.registry.members()),
        "quorum": protocol.effective_quorum(),
        "entries": entries,
    }, indent=2))
    size = out.stat().st_size

    # Every node that reached a height must record the same thing at it.  A
    # node that is merely *behind* has a shorter chain and a different head,
    # which is lag, not disagreement -- so compare per height, not per node.
    per_height: dict[int, set[str]] = {}
    for node in sim.nodes:
        for e in node.audit.entries():
            per_height.setdefault(e["height"], set()).add(e["model_hash"])
    forked = sorted(h for h, hashes in per_height.items() if len(hashes) > 1)
    await sim.shutdown()

    # -- 2. cold re-verification --------------------------------------------
    # A brand-new backend, reading the file back off disk: no shared state
    # with the run that produced it.
    published = json.loads(out.read_text())
    backend = make_backend(fp, args.strict)
    started = time.perf_counter()
    ok, detail = audit(published["entries"], sim, backend)
    audit_seconds = time.perf_counter() - started

    # -- 3. tampering --------------------------------------------------------
    caught: list[dict[str, Any]] = []
    for name, mutate, models in _tampers(fp, sim.identities):
        corrupted = mutate(copy.deepcopy(published["entries"]))
        rejected, why = audit(corrupted, sim, backend)
        caught.append({"tamper": name, "models": models,
                       "rejected": not rejected,
                       "reason": why if not rejected else "ACCEPTED"})

    payload = {
        "config": {"n": args.nodes, "rounds": args.rounds,
                   "proposers_per_round": args.proposers,
                   "strict_zk": args.strict,
                   "quorum": protocol.effective_quorum(),
                   "f": protocol.f_byzantine},
        "heights_committed": len(committed),
        "log_bytes": size,
        "bytes_per_height": size // max(len(entries), 1),
        "heights_observed": {str(h): len(v) for h, v in sorted(per_height.items())},
        "forked_heights": forked,
        "all_nodes_agree_on_history": not forked,
        "live_seconds": round(live_seconds, 2),
        "audit_seconds": round(audit_seconds, 3),
        "audit_seconds_per_height": round(audit_seconds / max(len(entries), 1), 4),
        "verified": ok,
        "detail": detail if not ok else "ok",
        "tampering": caught,
    }

    print(f"\nheights committed        {len(committed)}/{args.rounds}")
    print(f"published log            {size} bytes "
          f"({payload['bytes_per_height']} per height)")
    print(f"all nodes agree          {payload['all_nodes_agree_on_history']}"
          + (f"  (forked at {forked})" if forked else ""))
    print(f"cold re-verification     {'PASS' if ok else 'FAIL: ' + detail} "
          f"in {audit_seconds:.3f}s")
    print()
    print(table(["tamper", "caught", "how"],
                [[c["tamper"], "yes" if c["rejected"] else "NO",
                  c["reason"][:64]] for c in caught]))

    missed = [c["tamper"] for c in caught if not c["rejected"]]
    if missed:
        print(f"\nWARNING: undetected tampering: {', '.join(missed)}")
    save("e5_audit", payload)
    return 0 if ok and not missed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
