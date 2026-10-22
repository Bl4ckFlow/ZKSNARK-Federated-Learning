#!/usr/bin/env python3
"""Every number in the paper, checked against the JSON that produced it.

A paper is a claim about files on disk.  This script is what keeps that true:
it extracts the quantitative claims from ``paper/paper.tex`` and compares each
against ``results/*.json``.  A number that is edited in the prose and not in an
experiment -- or an experiment re-run whose output moved without the prose
following -- fails here rather than in review.

Run it after every experiment re-run:  python scripts/check_paper_numbers.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEX = (ROOT / "paper" / "paper.tex").read_text()


def load(name):
    path = ROOT / "results" / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


WORDS = {6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _flatten(text: str) -> str:
    r"""Strip the ways LaTeX writes a number so only the digits are compared.

    ``18{,}354``, ``18,354`` and ``18354`` are the same number; ``0.72\,s`` and
    ``0.72 s`` are the same measurement.  Comparing raw strings would report
    differences that are purely typographic and hide the ones that are not.
    """
    for old, new in (("\\,", ""), ("{,}", ""), (",", ""), ("~", " "),
                     ("\\textbf{", ""), ("\\emph{", "")):
        text = text.replace(old, new)
    return text


def tex_has(needle: str) -> bool:
    r"""Substring, except that a bare number must match as a whole token.

    Without this, ``72`` "matches" inside ``0.72\,s`` and the checker reports a
    stale message count as verified -- a false pass, which is worse than no
    check at all.  A needle that is purely digits (after flattening) therefore
    has to sit between non-digit, non-decimal-point characters.
    """
    flat_needle, flat_tex = _flatten(needle), _flatten(TEX)
    if re.fullmatch(r"\d+", flat_needle):
        return re.search(rf"(?<![\d.]){re.escape(flat_needle)}(?![\d.])",
                         flat_tex) is not None
    return flat_needle in flat_tex


CHECKS: list[tuple[str, str, object]] = []


def check(label, needle, ok=None):
    """Record a claim.  ``ok`` overrides the literal lookup when the paper
    writes the value some other way (spelled out, say) -- it is the authority,
    not an extra condition to satisfy on top of it."""
    CHECKS.append((label, needle, tex_has(needle) if ok is None else bool(ok)))


e2, e1, e3, e4, e5 = (load(n) for n in ("e2_proof_cost", "e1_learning_iris_n5",
                                        "e3_consensus", "e4_adversarial",
                                        "e5_audit"))
e3rep = load("e3_repeatability")
e3dl = load("e3_deadline")

if e2:
    main, abl = e2["measurements"][0], e2["measurements"][1]
    check("constraints", f"{main['constraints']:,}".replace(",", "{,}"))
    check("prove warm", f"{main['prove_warm_median_s']:.2f}")
    check("prove cold", f"{main['prove_cold_s']:.2f}")
    check("verify ms", f"{main['verify_median_ms']:.1f}")
    check("proof bytes", str(main["proof_bytes"]))
    check("zkey MB", f"{main['zkey_mb']:.1f}")
    # The prose spells small counts out; the table does not.  Accept either.
    check("ablation constraints",
          f"{abl['constraints']} R1CS constraints",
          ok=tex_has(f"{abl['constraints']} R1CS constraints")
             or tex_has(f"{WORDS.get(abl['constraints'], abl['constraints'])} "
                        f"R1CS constraints"))
    check("ablation verify ms", f"{abl['verify_median_ms']:.1f}")

if e1:
    for series in ("zkfl", "fedavg_no_zk", "centralized"):
        last = e1[series][-1]
        check(f"e1 {series} accuracy", f"{last['accuracy']:.3f}")
        check(f"e1 {series} loss", f"{last['loss']:.3f}")
    check("e1 all invariants held", "all held",
          ok=all(i["ok"] for i in e1["invariants"]))
    check("e1 every height agreed", "All 60 heights committed",
          ok=all(r["agreed"] for r in e1["zkfl"]))
    check("e1 sklearn reference", f"{e1['sklearn_float64_reference']:.3f}")
    cfg = e1.get("config", {})
    # The table caption states the configuration; if an experiment is re-run
    # with different settings the caption has to move with it.
    check("e1 learning rate", f"$\\eta = {cfg.get('learning_rate')}$")
    check("e1 nodes", f"{cfg.get('nodes')} nodes")
    check("e1 rounds", f"{cfg.get('rounds')} heights")
    others = e1.get("other_seeds") or []
    if others:
        # A 30-sample test set moves 0.033 per sample, so the single-split
        # numbers in the table mean nothing without the spread beside them.
        acc = [e1["zkfl"][-1]["accuracy"]] + [r["zkfl_accuracy"] for r in others]
        fed = [e1["fedavg_no_zk"][-1]["accuracy"]] + [r["fedavg_accuracy"]
                                                      for r in others]
        check("e1 zkfl across splits", f"{min(acc):.3f}--{max(acc):.3f}")
        check("e1 fedavg across splits", f"{min(fed):.3f}--{max(fed):.3f}")
        check("e1 split count", f"{1 + len(others)} splits")

if e3:
    for row in e3["scale"]:
        # Deliveries per height on a *mean* basis, which is what the per-kind
        # columns beside it are.  Older result files carry only the median
        # (``messages_per_round``); derive the mean from the raw totals so an
        # old file is checked on the same basis the paper states.
        total = row.get("deliveries_per_round")
        if total is None and "deliveries_total" in row:
            total = round(row["deliveries_total"] / row["rounds"], 1)
        if total is None:
            total = row["messages_per_round"]
        check(f"scale n={row['n']} deliveries",
              f"{round(total):,}".replace(",", "{,}"))
        kinds = row.get("deliveries_per_round_by_kind", {})
        if kinds:
            # The decomposition has to add up to the total printed next to it.
            # It stopped doing so once, because the total was a median and the
            # components were means -- a table that does not sum is exactly the
            # kind of thing a reader is right to distrust.
            CHECKS.append((f"scale n={row['n']} decomposition sums",
                           f"{round(sum(kinds.values()))} == {round(total)}",
                           abs(sum(kinds.values()) - total) <= 1.0))
        for kind in ("proposal", "vote", "polka", "certificate"):
            if kind in kinds:
                check(f"scale n={row['n']} {kind}",
                      f"{int(kinds[kind]):,}".replace(",", "{,}"))
        if "closed_form" in row:
            check(f"scale n={row['n']} closed form",
                  f"{row['closed_form']:,}".replace(",", "{,}"))
    for row in e3["adversity"]:
        check(f"adv {row['loss']:.0%} {row['latency_ms']:.0f}ms",
              f"{row['rounds_agreed']}/{row['rounds']}")
    for row in e3.get("sub_round_budget", []):
        if row["loss"] == 0.05:
            check(f"budget {row['max_sub_rounds']} failure",
                  f"{row['round_failure_rate']:.2f}")

if e3dl:
    # The claim is that moving the deadline moves the outcome while the budget
    # is held fixed, so both endpoints and the reachable-attempt spans have to
    # be in the paper -- the span is what makes "the budget was never binding"
    # an argument rather than an assertion.
    for row in e3dl["rows"]:
        check(f"deadline {row['deadline_s']:.0f}s committed",
              f"{row['rounds_agreed']}/{row['rounds']}")
        lo, hi = row["attempts_reachable"]
        check(f"deadline {row['deadline_s']:.0f}s attempts", f"{lo} to {hi}")
    CHECKS.append(("deadline sweep held the budget fixed",
                   f"max_sub_rounds={e3dl['max_sub_rounds']}",
                   len({r["max_sub_rounds"] for r in e3dl["rows"]}) == 1))

if e3rep:
    # The run-to-run spread at 5% loss is the number that says how much any
    # single loss figure is worth, so the paper has to carry it and this has to
    # check it.  Identical seeds every repeat: the spread is timing, not chance.
    check("repeat rate range",
          f"{e3rep['repeat_rate_min']:.2f}--{e3rep['repeat_rate_max']:.2f}")
    check("repeat pooled",
          f"{e3rep['pooled_agreed']}/{e3rep['pooled_rounds']}")

if e4:
    cp = e4["committed_poisoning"]
    check("clean accuracy", f"{cp['no_attack_accuracy']:.3f}")
    for r in cp["runs"]:
        tag = f"poison {r['poisoned_of_10']} {r['rule']}"
        check(f"{tag} acc", f"{r['accuracy']:.3f}")
        check(f"{tag} linf", f"{r['linf_from_clean']:,}".replace(",", "{,}"))
        if "accuracy_min" in r:
            check(f"{tag} range",
                  f"{r['accuracy_min']:.3f}--{r['accuracy_max']:.3f}")
        if "poison_magnitude" in r:
            # The magnitude column is the claim that the adversary is
            # calibrated rather than merely loud, so it is checked too.
            check(f"{tag} magnitude", f"{r['poison_magnitude']}")
    if "no_attack_accuracy_min" in cp:
        check("clean accuracy range",
              f"{cp['no_attack_accuracy_min']:.3f}--{cp['no_attack_accuracy_max']:.3f}")

if e5:
    # The paper says "in strict mode", and the whole point of the timing is that
    # it re-runs real pairing checks.  A run without --strict audits a log with
    # no proofs in it in three milliseconds, which would look like a much better
    # result and mean nothing.  So the mode is checked, not assumed.
    CHECKS.append(("audit ran in strict mode",
                   f"strict_zk={e5['config'].get('strict_zk')}",
                   bool(e5["config"].get("strict_zk"))))
    check("audit bytes", f"{e5['log_bytes'] / 1000:.1f}")
    check("audit per height", f"{e5['bytes_per_height'] / 1000:.1f}")
    check("audit seconds", f"{e5['audit_seconds']:.2f}")
    check("audit per height s", f"{e5['audit_seconds_per_height']:.2f}")
    n_tamper = len(e5["tampering"])
    rejected = sum(1 for t in e5["tampering"] if t["rejected"])
    check("tamperings all rejected", f"{rejected}", ok=(rejected == n_tamper))
    word = WORDS.get(n_tamper, n_tamper)
    check("tamper count in prose", f"{word} attacks, {word} rejections",
          ok=tex_has(f"{word.capitalize() if isinstance(word, str) else word} "
                     f"attacks, {word} rejections"))

# The test count is a claim too, and the easiest one to leave stale.
try:
    import subprocess
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    m = re.search(r"(\d+) tests collected", out.stdout)
    if m:
        check("test count", f"{m.group(1)} automated tests")
    m = re.search(r"(\d+)/\d+ tests collected", subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "circuit"],
        cwd=ROOT, capture_output=True, text=True, timeout=120).stdout)
    if m:
        check("circuit test count", f"{m.group(1)} require the compiled circuit")
except Exception as exc:                       # pytest absent is not a failure
    print(f"  (test count not checked: {exc})")

missing = [n for n, f in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4),
                          ("e5", e5)) if f is None]

bad = [(label, needle) for label, needle, ok in CHECKS if not ok]

width = max((len(c[0]) for c in CHECKS), default=10)
for label, needle, ok in CHECKS:
    print(f"  [{'ok  ' if ok else 'MISS'}] {label:<{width}}  {needle}")

print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} numbers verified against "
      f"results/*.json")
if missing:
    print(f"result files absent (not checked): {', '.join(missing)}")
for label, needle in bad:
    print(f"  FAIL {label}: paper does not state {needle!r}")

sys.exit(1 if bad else 0)
