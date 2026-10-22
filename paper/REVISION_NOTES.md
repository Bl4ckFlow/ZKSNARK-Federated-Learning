# Revision notes

What the review asked for, what changed in the artefact, and what still has to
change in the text. Every artefact claim below is backed by a file you can run;
`docs/CLAIMS.md` has the full claim → mechanism → evidence table.

Section numbers refer to the submitted draft.

---

## B1 — "The circuit is vacuously sound"

**The finding.** The committed circuit had 6 R1CS constraints and took the
gradient as a free witness signal. A prover picks any `w_new`, back-derives a
gradient that satisfies the single subtraction, and produces a valid proof.
Every possible update was provable, so the proof carried no information. §4 of
the paper describes a circuit that is not the one in the artefact.

**Artefact.** `circuits/lib/update_check.circom`, 19,402 constraints, enforcing:

- `Num2Bits` range checks on every input, so no signal can wrap the field;
- `y·(y−1) = 0`, and `0 ≤ η ≤ SCALE`;
- Merkle inclusion of every sample against a `dataRoot` fixed at registration;
- the forward pass, a 5-piece linear sigmoid (max error 0.0117), and the error;
- `w_new[j] = w_old[j] − ⌊η·g_j/(k·SCALE²)⌋` by exact integer division with a
  range-checked remainder — `DivConst` in `circuits/lib/fixedpoint.circom`,
  which offsets by `divisor·2^quotBits` first because circom's `\` sees field
  elements and a negative numerator would otherwise divide `p − |num|`;
- `b_new` constrained the same way (it was previously unconstrained);
- `|w_new − w_old| ≤ maxDelta`, coordinate-wise, by two `Num2Bits` rather than
  a power-of-two bound.

The verifier builds the public inputs `(w_old, b_old, η, dataRoot)` from **its
own** committed state and rejects any proof whose public signals differ, before
the pairing check (`zkfl/zk.py`). The trainer is integer fixed-point end to end
and matches the circuit bit-for-bit (`tests/test_circuit.py::test_witness_input_shape_matches_the_circuit` and `tests/test_fixedpoint.py::test_floor_division_convention_matches_the_circuit`).

`experiments/e4_adversarial.py` runs POISON1 against both circuits: accepted by
v1 (`zkfl/legacy_v1.py` keeps it for exactly this purpose), rejected by v2.

**Text to change.**

- §4: replace the circuit description with the one above; state the constraint
  count (19,402), the parameters it is for (`n=4, k=8, depth=4`), and that the
  count grows roughly linearly in `n·k`.
- §7 "Proof overhead": the "17-constraint circuit" figure must go. The measured
  numbers are 0.72 s warm median prove, 1.13 s cold, 13.9 ms verify, 740-byte
  proof, 9.1 MB proving key (`results/e2_proof_cost.json`).
- Add an explicit paragraph on what the proof does **not** establish: it binds
  the update to the committed data, not the data's honesty.

---

## B2 — "There is no aggregation, and voting is first-come"

**The finding.** Nodes voted for the first valid update they saw and exactly one
proposal was committed per height, so the global model was a chain of
single-node steps ordered by network latency. Nothing was federated.

**Artefact.** `zkfl/aggregation.py`. A height decides `Agg(μ_{h−1}, P)` over the
whole verified set: canonical ordering by `proposal_id`, coordinate-wise trimmed
mean / median / weighted mean, pure integer arithmetic. Because `Agg` is a
deterministic function of public inputs, two correct nodes holding the same set
necessarily vote identically — which is what makes a conflicting vote
misbehaviour rather than a timing artefact, and hence what makes equivocation a
meaningful accusation. Proposal equivocation is detected on the signed message
(`AccountabilityLog.observe_proposal`); `proposal_id` now hashes the whole
payload including height and proposer, so two nodes making the same step no
longer collide into one identity.

`robustness_note()` prints, at runtime, whether the trimmed-mean bound actually
holds for the run's `(|P|, f, permille)`, and prints `DOES NOT HOLD (bound is
vacuous)` when it does not.

**Text to change.**

- §5: rewrite as aggregation, not selection. The abstract's
  "distributed weighted-voting protocol" is wrong twice over — the votes are
  one-node-one-vote, and the thing voted on is an aggregate.
- §6.3: the "8 out of 10 votes ... the caa0563c update wins" narrative describes
  the selection protocol and must be replaced. The current runner
  (`scripts/run_simulation.py`) reports per-height aggregate, proposal count,
  messages and bytes.
- Cite the aggregation rules: Krum [blanchard2017krum], coordinate-wise trimmed
  mean [yin2018trimmedmean], FLTrust [cao2021fltrust]. The draft reported
  robustness without citing any of them.

---

## B3 — "Consensus safety and liveness are asserted, not argued"

**The finding.** No fault model, no quorum arithmetic, no safety argument.
The reported parameters `n = 10, q = 6` were described as tolerating 30%
Byzantine nodes; `2·6 > 10 + 3` is false, so they tolerate `f = 1`.

**Artefact.**

- `docs/THREAT_MODEL.md` states the fault model, the synchrony assumption, and —
  at equal length — what is *not* defended.
- `docs/PROTOCOL.md` §5 states and proves **Proposition 1 (Agreement)**: with
  `2q > n + f`, no two correct nodes commit different models at one height. §6
  states **Proposition 2 (Progress)** with a proof sketch under partial synchrony.
- `ProtocolConfig.check()` refuses any `(n, f, q)` with `2q ≤ n + f` or
  `n − f < q`, so a run cannot advertise a guarantee it lacks.
  `tests/test_consensus.py::test_the_papers_original_parameters_only_tolerate_one_fault`
  pins the arithmetic above.
- All votes are Ed25519-signed with domain separation; commit requires a
  **certificate** of `q` matching precommits, which is self-verifying and
  therefore usable for catch-up by a node whose own window expired.
- Equivocation evidence is the two conflicting signed messages, re-verified by
  each recipient — transferable, in the sense of the accountability literature
  [civit2021polygraph].
- Catch-up is implemented and retried (the previous code recorded missed heights
  in `_missed_heights` and never acted on them, so such a node stayed put
  forever).

**The substantive protocol change.** Voting is now two-phase — prevote → polka →
lock → precommit → certificate, following Tendermint [buchman2018tendermint].
This is not presentation. A single-phase protocol has to lock a node on its own
prevote, and correct nodes close their collect windows with different proposal
sets on any network with non-zero delay; each locks on its own aggregate,
nothing ever reaches `q`, and the height never decides. That failure was
observed here directly — two nodes, two locks, zero rejections, no certificate.
Locking on a polka means split prevotes produce no lock at all: everyone
precommits nil, the next sub-round opens with a doubled window, and by then the
proposal sets have converged.

**Text to change.**

- New subsection in §5 with the fault model and the quorum inequalities.
- State Proposition 1 and give the proof (or cite `docs/PROTOCOL.md` and give
  the sketch).
- Correct every occurrence of the `n = 10, q = 6` tolerance claim.
- §7 "Scalability": the `O(n²)` framing is right for the current design but the
  draft's parenthetical — "each update to the vote set triggers a re-broadcast"
  — describes an `O(n³)`-byte design. Either say that was the old cost and give
  the new one, or drop the parenthetical.
- Cite PBFT [castro1999pbft], Tendermint [buchman2018tendermint],
  HotStuff [yin2019hotstuff], FLP [fischer1985flp], and partial synchrony
  [dwork1988partialsynchrony].

---

## B4 — "There is no machine-learning result"

**The finding.** The paper reports consensus and proofs but never accuracy or
loss, so nothing establishes the system trains anything.

**Artefact.** Five experiments, each writing `results/*.json` with the CPU and
toolchain versions recorded alongside the numbers. `make evidence` regenerates
all of them.

| | measured |
|---|---|
| E1 learning | iris, 5 nodes, 60 heights: **0.867** accuracy vs **0.833** scikit-learn float64; baselines for centralised float64 and FedAvg-without-ZK included |
| E2 proof cost | 19,402 constraints; 0.72 s warm / 1.13 s cold prove; 13.9 ms verify; 740 B proof; 9.1 MB zkey |
| E3 consensus | `n` 5→50: messages/`n²` flat at 2.88→3.00. Under 150 ms one-way delay: 75/75 heights, 1.46 s median. Under 5% loss: 49/75 at the default `max_sub_rounds = 8` (per-seed 3/25 25/25 21/25), materially worse at 4. Under 20% loss: **almost nothing commits at any budget** |
| E4 adversarial | POISON1 accepted by v1, rejected by v2; equivocating voter and proposer detected by every honest node; flooding rate-limited without stalling |
| E5 audit | published log re-verified cold; **8/8** tamperings rejected |

The 20% loss result is a genuine degradation and is reported as one. A table
showing only the configurations that worked would be the same mistake the review
is objecting to.

The loss numbers also carry a methodological point worth making in the text.
Estimated from five rounds each, they moved between 5/5 and 3/5 across runs —
five samples is not a rate. The adversity sweep now runs 25 rounds across three
independent seeds and reports the per-seed split alongside the aggregate, and
at 5% loss the honest figure turned out to be far worse than the five-round
estimate suggested. Chasing that discrepancy produced a second, sharper
instance of the same lesson. A 10-height sweep suggested `max_sub_rounds = 8`
eliminated the 5%-loss failures (0.23 → 0.00), and we changed the default on
that basis; re-running at 25 heights gives 0.48, 0.29 and 0.53 at budgets 4, 8
and 16 — non-monotone, and within the seed-to-seed spread.

Pushing on *why* it was non-monotone found the actual answer, and it is not
statistical. Attempt `r` waits `collect_window · 2^min(r,6)`, and each sub-round
opens two such windows (prevote and precommit), so a height that has made five
attempts has already spent 12.6–25.2 s against the sweep's 15 s per-height
deadline. **Only five or six attempts are reachable at any budget**, so
`max_sub_rounds = 8` and `= 16` are the same configuration and the difference
between their rates was never a signal. Budget and window schedule are one
parameter; the sweep varied a knob that was not connected to anything.
`experiments/e3_deadline.py` tests this by varying the deadline at a fixed
budget.

A second measurement problem showed up alongside it. Re-running the same
configuration end to end gives commit rates of 0.71 and 0.40 with *identical*
seeds, because the simulator runs on wall-clock timers and the 5%-loss regime is
bimodal: a run either stays healthy (median 0.6 s) or collapses to the timeout
and commits almost nothing. Seed 0 collapses reproducibly at 3 heights; seeds 1
and 2 collapse sometimes. A single pooled rate is a mixture of two regimes and
describes neither, so `experiments/e3_repeatability.py` reports the range and
the paper quotes it that way.

Getting caught three times by the same class of error in one revision — an
under-sampled rate, a knob that was not connected, and a number quoted from one
run of a bimodal process — is itself worth reporting, since it is precisely the
reviewer's B4 objection applied to us.

**Text to change.** Rewrite §6 around these tables. State the dataset, the
number of samples, the split, and the baselines for every accuracy figure.

---

## B5 — Presentation defects

| Defect | Fix |
|---|---|
| Two `\thebibliography` blocks, the second with placeholder arXiv IDs (`2407.XXXXX`, `2406.XXXXX`, `2501.XXXXX`, `2502.XXXXX`, `2310.XXXXX`), inconsistent author initials for the same works, and at least one work ("TT-FL") that does not appear to exist | `paper/references.bib` — one list, no placeholders. Six entries are marked `VERIFY` where the identifier came from the draft and could not be independently confirmed; check those before submission |
| "44 automated tests" (abstract, §6) vs "Total 53" (Table 1) | The current suite is **151 tests**; `pytest tests/ -q` prints the count. Quote it from a run, not from memory |
| Abstract says "weighted-voting protocol" | It is one-node-one-vote on an aggregate. Both halves of the phrase are wrong |
| `Dataset_A` / `Dataset_B` never defined (§6.3) | Name the actual datasets, sizes and splits. E1 uses iris (versicolor vs virginica, 100 samples, 4 features, 70/30 split); E3–E5 use a seeded synthetic generator, `zkfl/datasets.py` |
| "17-constraint circuit" (§7) | 19,402 for the shipped parameters |
| §2.5 attributes `O(n²)` to classical BFT as a limitation, then §7 reports `O(n²)` for this system as acceptable | Pick one framing. The honest one: `O(n²)` is the cost of all-to-all agreement, this design pays it, and E3 shows the constant |

---

## §5 / §6 / §7 — related work the draft is missing

The nearest work to this paper's core claim was absent. It must be cited and
compared, and the comparison is not flattering, which is the reason to make it:

- **Proofs of training.** [garg2023zkpot] and Kaizen [abbaszadeh2024kaizen] prove
  that a model results from running a training procedure — the same statement,
  at far larger scale, at far greater cost. This paper's contribution is the
  *composition* with consensus, not the proof of training itself, and the text
  should say so.
- **Byzantine-robust aggregation.** [blanchard2017krum], [yin2018trimmedmean],
  [cao2021fltrust].
- **Accountable consensus.** [civit2021polygraph], alongside PBFT, Tendermint,
  HotStuff.
- **Gradient leakage.** [zhu2019deepleakage], [geiping2020inverting] — necessary
  for an honest privacy limitation, since the published update is exactly the
  quantity those attacks invert.

---

## Limitations the text must state

The draft's limitations paragraph covers the sigmoid approximation and the
single-sample circuit. These are missing and each is more serious:

1. **The proof does not protect privacy of the update.** `w_new` is public by
   construction — every node needs it to aggregate — and gradient inversion
   reconstructs training data from it. Publishing per-node updates weakens
   privacy no matter how good the proof is.
2. **The trusted setup.** Groth16 needs a per-circuit ceremony.
   `scripts/setup_circuit.py` runs a real multi-contributor phase 2 with a
   beacon, but `--local-ptau` generates phase 1 locally, and in that mode
   whoever ran it can forge proofs. The mode is recorded in `build_info.json`.
3. **Poisoning within the bound.** A node that commits to a poisoned dataset and
   trains correctly on it produces a valid proof. Only aggregation bounds this,
   and only when enough is trimmed per side.
4. **Security level.** BN254 is around 100 bits after the extended tower number
   field sieve [kim2016extnfs; barbulescu2019updating], not the 128 often quoted.
5. **Scale.** 4 features, batch of 8. Nothing here supports extrapolating to
   deep models.

---

# Second review round

A second reviewer read the revised draft. Two findings were correctness bugs in
the protocol, not presentation problems, and both are fixed here. Ordered as
they were raised.

## 1. Table 3 read as a formula, not a measurement

The message totals were exactly `3(n²−1)` at every size — 72, 297, 1197, 7497.
Four for four is not a coincidence a reader should have to accept.

It was a real measurement, and the reviewer's arithmetic (*"five nodes each
prevoting and precommitting is 40 deliveries per height minimum"*) is exactly
right and exactly what the counter contains. The problems were presentational:
the column was headed **Messages** when it counts *deliveries per height*, and a
single total cannot be told apart from a closed form.

Fixed by decomposing it. `MessageBus` now counts per envelope kind, and the
table reports proposals / votes / polkas / certificates alongside the total and
the closed form `(n−1)(p + 4n)`. At `n ≤ 20` the measurement lands on the
formula exactly — at zero loss nothing is retried, so there is nothing to differ
by — and at `n = 50` it does not: 10,094 against a predicted 9,947, the excess
being expired sub-rounds at a 26 s median height. The count also scales linearly
in heights (144 / 360 / 720 for 2 / 5 / 10 at `n=5`), which is the check that it
is per-height.

## 2. A real liveness deadlock: locked vs unlocked nodes

**The reviewer was right, and this was a genuine bug.** The liveness sketch said
correct nodes hold the same `P` after GST, so `Agg` is deterministic and they
prevote identically. But a locked node prevotes its *lock*, not `Agg(P)`. Under
loss only some nodes observe a given polka; the rest keep offering `Agg(P)`. If
neither camp reaches `q`, no polka forms — and only a polka moves a lock — so the
split reproduces itself forever. More sub-rounds cannot help: each one recreates
the same two camps.

Reproduced directly: at `n=10, q=7`, five nodes locked on `v` and five offering
`v'`, prevotes split 5/5, permanently stuck.

Fixed by making polkas **transferable**. `Certificate` now carries a `phase`; a
node that observes `q` matching prevotes gossips them, and a receiver verifies
the `q` signatures itself before adopting the lock. This is the leaderless
counterpart of Tendermint's proposer re-proposing with a proof-of-lock. Costs
`n(n−1)` deliveries per height, reported as its own column rather than folded
into the vote total. Pinned by
`test_a_gossiped_polka_breaks_a_locked_unlocked_split`.

Found alongside it, from the same probe: `_advance` returned `NeedSync`
*instead of* `OpenWindow` past `max_sub_rounds`, so a node that exhausted its
budget went silent. Sync is only answerable by a node that is *ahead*, and the
case that exhausts the budget is the one where no quorum formed — nobody is
ahead, nothing answers, and every node hits the budget together. At 5% loss a
run committed three heights and then nothing, reproducibly, for seed 0 on every
repeat. The implementation contradicted its own liveness proposition.

## 3. The induction hypothesis did not match what the proof used

Stated as *"if a correct node precommits `v` at `r`"*, but the step derives its
intersection from `C`, the whole quorum — one node's precommit does not
intersect a polka. Restated over the certificate, as the reviewer suggested:
*"if a commit certificate for `v` exists at `r`, then any polka at `r₁ ≥ r` is
for `v`."* The proof goes through unchanged.

## 4. Purging was outside the model the proof uses

Erasing a proven equivocator's votes retroactively changes who counts, and the
proof never mentioned it. Worse, the code rejected certificates containing a
blacklisted voter — making validity **view-dependent**, so two correct nodes
with different evidence disagreed about whether a height was decided.

Fixed by fixing the semantics and stating them as **(P)**: a certificate or
polka is valid *once formed*; validity is a function of its signatures and the
member set, never of the checker's evidence; evidence governs future tallies
only. This costs nothing in safety — `2q > n+f` exists precisely to budget for
`f` faulty nodes signing both sides. Pinned by
`test_a_certificate_stays_valid_after_one_of_its_voters_is_blacklisted`.

## 5. Votes were not durable

Crash-restart is in the fault model, so *"a correct node signs one vote per
slot"* — which both cases of the proof use — was simply false: a node that
crashed after broadcasting and restarted would sign a second vote in the slot
and be blacklisted on genuine evidence. Added `zkfl/votelog.py`: append, flush,
`fsync`, *then* broadcast; a restart re-sends the recorded vote byte for byte.
Tested by killing an engine mid-prevote and rebuilding it on the same log.

Writing the test found a second bug: `vote_log or InMemoryVoteLog()` silently
discarded an empty `DurableVoteLog`, because both define `__len__`. A node would
have believed it was crash-safe and was not.

## 6. DivConst numerators needed their own bound

`num === q·d + r` is a field equation, so the checks on `q` and `r` isolate a
unique integer pair only while `|num| ≤ d·2^quotBits`. Every numerator is an
assigned combination of range-checked signals rather than a free witness, so
each has a derivable bound — but the derivation lived nowhere.
`FixedPointConfig.check()` now re-derives all four and refuses a configuration
that violates the condition.

Writing that check found a real gap. The sigmoid numerator `m_sel·(a − t_sel)`
is *not* bounded by `|z|`; the bound comes from the table (flat outer piece ⇒
zero slope). That held for the honest witness because the selector is determined
by `a`, but it was derived from the comparator chain being sound rather than
enforced anywhere — and uniqueness must hold for *every* witness. The activation
output was likewise documented as lying in `[0, SCALE]` and not constrained,
though every downstream numerator bound depends on it. Both are now constrained:
**+1,048 constraints, 18,354 → 19,402**, about 5.7%.

## 7. Whether zero-knowledge earns its place

Answered in the limitations rather than deflected. The integrity requirement is
a succinct argument, not a zero-knowledge one. What hiding still buys: the
witness is the batch *and* its Merkle path, so which `k` of the committed
samples were used stays hidden, and with it the rest of the dataset. What
Groth16 buys: constant-size proofs and millisecond verification, in a protocol
where every node verifies every proposal at every height. Stated as an
engineering choice, not a cryptographic necessity.

## 8. The poisoning adversary was not the strongest one

It proposed steps that exceeded `δmax`, which were discarded as unprovable — the
attack threw away its own rounds. A real adversary commits to the dataset that
does the most damage *while staying provable*. Implemented by sweeping the
committed feature scale and keeping the best setting that never trips the bound.

The result is sharper than expected: against FedAvg the adversary picks the
largest scale it can, but against trimmed mean and median it picks a
**smaller** one (0.5, 0.25) — an extremal contribution is exactly what an order
statistic discards, so the better attack stays inside the kept range. Damage to
trimmed mean at 2 poisoned nodes roughly triples, 0.6% → 1.7%, and no step is
blocked in any row. Both tables now report medians over three seeds with ranges.

## Smaller items

- **740 B** is snarkjs' JSON encoding; the raw bn128 proof is 128 B. Both stated,
  with a note that the message-cost table uses the former.
- **Batch/height binding.** Nothing binds a batch to a height, so a node can
  replay its worst committed batch every round. Documented with the concrete
  fix (expose the Merkle leaf index as a public signal, constrain it to the path
  bits, have the verifier require a deterministic schedule) and *not*
  implemented — a gap in coverage, not in soundness. The poisoning numbers
  understate this adversary, and the text says so.
- **Empty proposal set.** Specified: the node prevotes nothing and opens the
  next sub-round; the height waits. A liveness dependency, not a safety one.
- **Comparison table** against zkFL / Federify / VeriBlock-FL / VerifBFL /
  ByzSFL / Byeon et al., on prover, ordering, decision and fault model.
- **Artifact link** is now an explicit placeholder for an anonymized URL rather
  than "are available".
