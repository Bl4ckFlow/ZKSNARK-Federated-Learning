# Claims, mechanisms, evidence

Every claim this work makes, the code that implements it, and the artefact that
demonstrates it. A claim with no evidence column is not made. Where a claim is
weaker than the one in the earlier draft of the paper, the difference is stated
explicitly — the point of this table is to make overclaiming visible.

Regenerate everything with `make evidence`.

---

## C1 — A valid proof implies the update really is the gradient step on committed data

**Mechanism.** `circuits/lib/update_check.circom`. The circuit constrains, in
this order: range of every input signal (`Num2Bits`), `y·(y−1) = 0`,
`0 ≤ η ≤ SCALE`, Merkle inclusion of every sample against `dataRoot`, the
forward pass `z = ⟨x,w⟩/SCALE + b`, a five-piece linear sigmoid, the error
`pred − y·SCALE`, and finally `w_new[j] = w_old[j] − ⌊η·g_j / (k·SCALE²)⌋` with a
range-checked remainder, plus `|w_new − w_old| ≤ maxDelta`.

**Evidence.**

| | |
|---|---|
| 19,402 R1CS constraints | `results/e2_proof_cost.json` |
| Honest step proves and verifies | `tests/test_circuit.py::test_honest_step_proves_and_verifies` |
| POISON1 accepted by v1, rejected by v2 | `test_poison1_succeeds_against_v1`, `test_poison1_fails_against_v2` |
| Uncommitted data rejected | `test_uncommitted_data_cannot_be_used` |
| Out-of-range features rejected | `test_out_of_range_features_are_rejected` |
| Non-binary labels rejected | `test_non_binary_labels_are_rejected` |
| Step beyond `maxDelta` rejected | `test_step_beyond_the_clipping_bound_is_unprovable` |
| Merkle commitment binds sample order | `test_batch_digest_binds_sample_order` |
| Python trainer and circuit agree bit-for-bit | `test_witness_input_shape_matches_the_circuit`, `tests/test_fixedpoint.py::test_floor_division_convention_matches_the_circuit`, `tests/test_poseidon.py::test_five_input_hash_matches_the_circuit` |

**What changed.** The circuit in the earlier artefact had **6 constraints** and
took the gradient as a free witness signal; the prover chose `w_new`, then
back-derived a gradient that satisfied the single subtraction. Every update was
provable — the soundness was vacuous. `experiments/e4_adversarial.py` runs the
POISON1 attack against both: it succeeds against v1 (a forged gradient witness
verifies) and fails against v2.

**What it does not claim.** The circuit proves the update *is* the gradient step
on the *committed* data. It does not and cannot prove the committed data is
honest. See §"Model poisoning within the bound" in `docs/THREAT_MODEL.md`.

---

## C2 — A verifier cannot be made to check a statement of the prover's choosing

**Mechanism.** `zkfl/zk.py`, `Groth16Backend.verify`. The verifier constructs
`PublicInputs` from its own committed model, its own `η`, and the proposer's
registered `dataRoot`, then rejects any proof whose public signals differ —
before the pairing check. `assert_signal_order` checks against the compiled
`.sym` that the ordering the Python side assumes is the ordering the circuit
actually has.

**Evidence.** `tests/test_circuit.py::test_public_inputs_are_bound_to_the_verifier_state`
(a proof is checked against a different base model, a different data root, a
different learning rate, and a chosen target — all four rejected),
`test_swapping_the_public_signals_inside_a_proof_is_rejected`,
`test_string_and_null_proofs_are_always_rejected`, and `tests/test_zk.py`,
which pins the public-signal ordering, the field encoding of negative weights,
and the guards that stop the stand-in backend from being mistaken for a real
one (`test_the_null_backend_will_not_be_constructed_by_accident`,
`test_the_null_backend_refuses_when_the_environment_forbids_it`).

**What changed.** The earlier design verified proofs against public inputs taken
from the proposal itself, which proves only that the prover was internally
consistent.

---

## C3 — Committed models are the aggregate of a verified set, not one node's step

**Mechanism.** `zkfl/aggregation.py`. Canonical ordering by `proposal_id`,
coordinate-wise trimmed mean / median / weighted mean, pure integer arithmetic.

**Evidence.** `tests/test_aggregation.py` (18 tests, including determinism under
permutation, integer-exactness, and the trimmed-mean bound);
`tests/test_integration.py::test_every_committed_model_is_the_aggregate_of_its_proposals`
recomputes every committed height from its own audit entry.

**What changed.** The earlier design committed exactly one proposal per height —
the first valid one to arrive — so the global model was a chain of single-node
steps ordered by network latency, and nothing was federated about it.

**Conditional, and the condition is not decorative.** Trimmed-mean robustness
holds only when the number trimmed per side is at least the number of Byzantine
proposals. `robustness_note()` prints `DOES NOT HOLD (bound is vacuous)` when it
is not, and `experiments/e4_adversarial.py` measures what happens on either
side of that line — directed poisoning by nodes that train honestly on
committed-but-adversarial data, 40 heights, accuracy against a clean run of
0.947 (0.928–0.958 across seeds):

| poisoned | rule | accuracy | lost | trim ≥ f |
|---|---|---|---|---|
| 2 of 10 | fedavg | 0.842 | −0.117 | — |
| 2 of 10 | trimmed mean | 0.931 | −0.017 | yes |
| 2 of 10 | median | 0.933 | −0.014 | yes |
| 3 of 10 | fedavg | 0.831 | −0.128 | — |
| 3 of 10 | trimmed mean | 0.844 | −0.114 | **no** |
| 3 of 10 | median | 0.906 | −0.033 | yes |

Median over three data seeds. The adversary is *calibrated*: it sweeps the
feature scale it commits to and keeps the most damaging setting that stays
inside the clipping bound, so no poisoned step is discarded as unprovable.
Against an order statistic it deliberately picks a smaller scale (0.5, 0.25)
than against FedAvg (1.0) — an extremal contribution is exactly what trimming
removes, so the better attack stays inside the kept range and drags it.

Trimmed mean fails exactly where its own bound says it will, and the runtime
warning fires on that row and no other.

---

## C4 — No two correct nodes commit different models at the same height

**Mechanism.** Two-phase voting with certificate-based commit and polka-based
locking; `2q > n + f` enforced by `ProtocolConfig.check()`.

**Evidence.** Proposition 1 with proof in `docs/PROTOCOL.md` §5.
`tests/test_consensus.py::test_commit_requires_a_full_quorum`,
`test_two_conflicting_certificates_need_more_than_f_equivocators`,
`test_a_polka_locks_the_node_and_the_lock_survives_new_proposals`,
`test_certificate_with_duplicate_or_foreign_voters_is_rejected`.
Every simulation run asserts bitwise-identical models across honest nodes
(`Simulation.check_invariants`).

**What changed.** The earlier paper reported `n = 10, q = 6` as tolerating 30%
Byzantine nodes. `2·6 > 10 + 3` is false: those parameters tolerate `f = 1`.
`test_the_papers_original_parameters_only_tolerate_one_fault` pins this.

---

## C5 — Every height eventually commits under partial synchrony

**Mechanism.** Windows double per sub-round; a polka from a sub-round ahead pulls
a node forward; a quorum of nil precommits starts the next sub-round at once;
certificates commit a node regardless of its own window; catch-up retries.

**Evidence.** Proposition 2 with proof sketch in `docs/PROTOCOL.md` §6.
`tests/test_consensus.py::test_no_polka_means_no_lock_and_the_next_sub_round_is_free`,
`test_a_quorum_of_nil_precommits_starts_the_next_sub_round`,
`tests/test_integration.py::test_a_node_that_misses_a_round_catches_up_by_sync`,
and `experiments/e3_consensus.py`'s adversity sweep — 25 heights across three
independent seeds per configuration, reporting the per-seed split as well as
the aggregate, because a rate estimated from five rounds moves by 0.2 whenever
one round goes the other way.

**Measured, including where it fails.** 150 ms one-way delay costs latency and
nothing else: 75/75 heights, 1.46 s median. Loss is a different problem — a
dropped vote costs a whole sub-round, and a two-phase sub-round needs two
quorums to land. We first concluded that the sub-round budget was the binding
constraint and that raising it from 4 to 8 was a large improvement; a 25-height
sweep does not support that. Failure rates come out 0.48, 0.29 and 0.53 at
budgets 4, 8 and 16 — not monotone, so not a ranking.

The reason is arithmetic rather than statistical. Attempt `r` waits
`collect_window · 2^min(r,6)`, and a sub-round opens two such windows (prevote
and precommit), so a height that has made five attempts has already spent
between 12.6 s and 25.2 s against a 15 s per-height deadline. **Only five or six
attempts are reachable**, so budgets of 8 and 16 are not distinct configurations
and the spread between them is noise.
Budget and window schedule are one parameter, not two, and we did not design
them that way. See `experiments/e3_deadline.py`.

At 5% loss the outcome is bimodal rather than merely variable: a run either
stays healthy (median 0.6 s, nearly every height) or collapses (median at the
timeout, almost nothing), and seed 0 collapses reproducibly. Quoting a single
pooled rate describes neither regime; `experiments/e3_repeatability.py` reports
the range. At 20% loss almost nothing commits at any budget, which is a limit of
the design rather than a parameter to tune, and is stated as one in
`docs/THREAT_MODEL.md`.

**What changed.** Single-phase locking on one's own vote deadlocked whenever two
correct nodes closed their windows with different proposal sets — that is, on
any network with non-zero delay. The strict-ZK live-round test never committed
until this was fixed.

---

## C6 — Misbehaviour produces transferable proof, not just local suspicion

**Mechanism.** `zkfl/accountability.py`. The two conflicting signed messages are
the evidence; recipients re-verify both signatures themselves.

**Evidence.** `tests/test_accountability.py` (13 tests, including that a fabricated
accusation against an honest node is rejected);
`tests/test_integration.py::test_an_equivocating_voter_is_caught_and_the_round_still_commits`
and `test_an_equivocating_proposer_is_caught` assert that **every** honest node
independently blacklists the offender.

**What changed.** Proposal equivocation was previously undetectable — vote-set
cross-checking only ever looked at voters.

**And detection alone is not enough.** An equivocating proposer's first message
is accepted before its second exposes it, and honest nodes see the two in
different orders, so they accept *different* firsts. Blacklisting the offender
without retracting what it already got accepted leaves the proposal sets
permanently split: prevotes divided 5/4 against `q = 7` at every sub-round and
the height never decided. `_purge_offender` erases a proven equivocator's votes
*and* proposals from every open round and from the future buffer, which is
deterministic given the (self-verifying, gossiped) evidence, so every honest
node performs the same purge.
`test_an_equivocating_proposer_cannot_stall_the_height` is the regression.

---

## C7 — The training history is checkable by a stranger, and tampering is detectable

**Mechanism.** `zkfl/audit.py`. Each entry carries everything needed to re-derive
one transition; entries are hash-chained; `verify_offline` re-checks every
certificate, every proof against the model as of that height, and every
aggregate, sharing no state with the node that wrote the log.

**Evidence.** `experiments/e5_audit.py` — publishes a log, re-verifies it cold,
then applies eight tamperings and confirms each is rejected:

| Tampering | Caught by |
|---|---|
| model edited | model does not match its hash |
| model edited, hash recomputed | aggregate does not match |
| certificate one vote short | certificate lacks a quorum |
| certificate padded with a duplicate voter | certificate lacks a quorum |
| precommits downgraded to prevotes | certificate lacks a quorum |
| a proposal dropped from the set | aggregate does not match |
| proofs swapped between proposers | bad proposal signature |
| a height spliced out | broken parent link |

Regression tests for the same eight live in `tests/test_audit.py`.

---

## C8 — Proving cost is what is reported

**Mechanism.** `experiments/e2_proof_cost.py` measures on a warm prover process,
reports cold and warm separately, and records the CPU and toolchain versions
alongside the numbers.

**Evidence.** `results/e2_proof_cost.json` is the source of truth for these
numbers; quote them from a run rather than from here. As measured on an
i5-11400H: 19,402 constraints, 0.84 s warm median prove, 1.09 s cold, 15.1 ms
verify, 742-byte proof, 9.5 MB proving key, for `n_features = 4`,
`batch_size = 8`, `merkle_depth = 4`. Timings move a few percent run to run;
the constraint count, proof size and key size do not.

**What changed.** The earlier artefact's 6-constraint circuit proved in 0.36 s.
Quoting that as the cost of proving a training step compares a proof of nothing
with a proof of something.

---

## C9 — The system learns

**Mechanism.** `experiments/e1_learning.py`. Integer fixed-point logistic
regression with the piecewise-linear sigmoid, against three baselines: centralised
float64 training, FedAvg without ZK, and scikit-learn.

**Evidence.** `results/e1_learning_iris_n5.json`. On iris (versicolor vs
virginica), 5 nodes, 60 heights: **0.867** final accuracy, against **0.833** for
centralised float64 training, **0.833** for FedAvg without ZK, and **0.833** for
scikit-learn on the same split. All 60 heights agreed; every invariant held.

**Caveats that must travel with the number.** The test set is 30 samples, so
0.867 and 0.833 differ by *one sample*. Two further splits (seeds 1 and 2) make
the point without argument: on both, this work and FedAvg score **identically**
(0.933 and 0.967), and the spread across the three splits — 0.867–0.967 here,
0.833–0.967 for FedAvg — is several times the gap being compared. That is not
evidence the integer pipeline beats float64 — it is evidence it does not lose to
it, which is the whole claim. This is a 4-feature binary logistic regression on 100 samples split
five ways: it shows the fixed-point / piecewise-linear pipeline does not destroy
learning, and it says nothing whatever about deep models. The piecewise sigmoid's
maximum error is 0.0117 (`tests/test_fixedpoint.py`).

---

## C10 — Message cost is quadratic in `n`, not cubic

**Mechanism.** Individual signed votes, sent once by their author; certificates
and evidence gossiped because both are self-verifying and terminal; proposals
relayed only after verification.

**Evidence.** `tests/test_integration.py::test_message_cost_is_quadratic_not_cubic`
(messages must grow less than 6× when `n` doubles) and `experiments/e3_consensus.py`'s
scale sweep at `n ∈ {5, 10, 20, 50}`.

**What changed.** The earlier design re-broadcast an `O(n)`-sized vote set with
every new vote: `O(n³)` bytes per height.

---

## Claims deliberately **not** made

- **Privacy of updates.** The proof hides the data, not `w_new`. Gradient
  inversion applies. See `docs/THREAT_MODEL.md` §4.
- **Robustness to poisoning by nodes that train honestly on poisoned data.**
  Out of scope for any circuit of this form.
- **Sybil resistance.** Membership is permissioned and out of scope.
- **A trustworthy setup by default.** `--local-ptau` is a development mode in
  which the operator can forge proofs, and it is labelled as such in
  `build_info.json`.
- **Deep-learning scale.** `n_features = 4`, `batch_size = 8`. Constraint count
  grows roughly linearly in `n_features · batch_size`; extrapolating from 18k
  constraints to a CNN is not supported by anything here.
