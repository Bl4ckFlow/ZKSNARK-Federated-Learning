# The protocol

A height is decided by a leaderless, certificate-based agreement on the
*aggregate* of a verified proposal set. This document specifies it and states
the safety and liveness propositions with their proofs.

## 1. Notation

| Symbol | Meaning |
|---|---|
| `n` | number of nodes |
| `f` | maximum number of Byzantine nodes |
| `q` | quorum size, `ProtocolConfig.effective_quorum()` |
| `h` | height (one global model update) |
| `r` | sub-round within a height |
| `μ_h` | the global model committed at height `h`; `μ_0` is genesis |
| `P_h` | the set of verified proposals a node holds for height `h` |
| `Agg(μ, P)` | the aggregation function, a deterministic function of `(μ, P)` |
| `H(·)` | SHA-256 over the canonical JSON encoding |

Throughout, `2q > n + f` and `n − f ≥ q`, which `ProtocolConfig.check()` enforces.

## 2. Messages

All four are signed with Ed25519 over `b"zkfl/v2|" + kind + b"|" + canonical_json(payload)`.
Domain separation by `kind` is what stops a signature over one message type from
being replayed as another. `canonical_json` rejects floats outright, because a
float's shortest repr is platform-dependent and a signature over a
platform-dependent encoding is not a signature.

**Proposal** `(h, H(μ_{h−1}), μ', π, batch, proposer)`.
`π` is a Groth16 proof that `μ'` is the fixed-point gradient step from `μ_{h−1}`
on a batch of the proposer's committed data. `proposal_id = H(payload)`, over the
*whole* payload including `h` and the proposer — an id derived from the update
alone collides whenever two nodes make the same step, and then two distinct
proposals share an identity.

**Vote** `(h, r, phase, model_hash, proposal_ids, voter)`, with
`phase ∈ {prevote, precommit}`. A voter's *slot* is `(voter, h, r, phase)`;
occupying one twice is equivocation. Prevoting and precommitting in the same
sub-round is normal, which is why the phase is part of the slot.

**Certificate** `(h, r, model_hash, votes)` — `q` precommits from distinct
registered members, all for the same `(h, r, model_hash)`, each signature valid,
`model_hash ≠ nil`. Self-verifying: anyone can check one without trusting its
bearer, which is what makes it usable for catch-up.

**Evidence** `(offender, kind, two conflicting signed messages)`. Also
self-verifying, hence transferable: a node that receives evidence re-checks the
two signatures itself rather than trusting the accuser.

## 3. A height

```
  h begins with every correct node holding μ_{h−1}
  ┌─────────────────────────────────────────────────────────────────┐
  │  collect        proposals arrive; each is verified (§4) before   │
  │                 it is stored or relayed                         │
  │                                                                 │
  │  PREVOTE        on timeout, broadcast                           │
  │                     prevote(h, r, H(Agg(μ_{h−1}, P_h)))         │
  │                 or, if locked, prevote the locked value         │
  │                                                                 │
  │  POLKA          q matching prevotes for v at sub-round r        │
  │                     ⇒ lock on (v, r), then                      │
  │                                                                 │
  │  PRECOMMIT      broadcast precommit(h, r, v)                    │
  │                 (no polka before the timer ⇒ precommit nil,     │
  │                  and start sub-round r+1 with a doubled window) │
  │                                                                 │
  │  COMMIT         q matching precommits = a certificate           │
  │                     ⇒ μ_h := Agg(μ_{h−1}, P), append to the     │
  │                       audit log, advance to h+1                 │
  └─────────────────────────────────────────────────────────────────┘
```

Three rules carry the weight.

**Aggregate, do not select.** The decision is `Agg(μ_{h−1}, P_h)` over the whole
verified set, not one chosen proposal. The earlier design voted for the first
valid update seen, so the global model was a chain of single-node steps ordered
by network latency — federated learning with `n = 1` per height, and a race for
which node's update happened to arrive first. Aggregation also makes the decision
a *deterministic function of public inputs*, so two correct nodes holding the same
set necessarily vote identically and a conflicting vote is misbehaviour rather
than a timing artefact. That is what makes equivocation a meaningful accusation.

**Commit on a certificate, never on a local tally.** The old rule — commit when
the leading candidate holds more than half of *the votes this node happened to
see* — has a per-node denominator and therefore says nothing about agreement.
A certificate is `q` signatures on one value, and its meaning does not depend on
who is holding it.

**Lock on a polka, never on your own vote.** This is the difference between a
protocol that runs and one that does not. Locking on one's own prevote deadlocks
the moment two correct nodes close their collect windows with different proposal
sets, which on any network with non-zero delay is immediately: each locks on its
own aggregate, neither ever reaches `q`, and the height never decides. The
observed failure was exactly this — two nodes with two proposals each, both
locked, zero rejections, no certificate, forever. Locking on a polka means split
prevotes produce *no* lock: everyone precommits nil, sub-round `r+1` opens with a
doubled window, and by then the proposal sets have converged. This is Tendermint's
rule (Buchman, Kwon and Milosevic, *The latest gossip on BFT consensus*, 2018),
and §5 is Tendermint's argument.

## 4. Proposal verification

A proposal is stored and relayed only after all of:

1. the proposer is a registered member and is not blacklisted;
2. its signature verifies;
3. it does not equivocate with one already seen at this height;
4. `prev_hash == H(μ_{h−1})` **for this node's own committed `μ_{h−1}`**;
5. `μ'` is within range, and `|μ' − μ_{h−1}| ≤ maxDelta` coordinate-wise;
6. `π` verifies against public inputs the **verifier** builds from its own state:
   `(μ_{h−1}, μ', η, dataRoot(proposer))`. The prover supplies only the proof.

Step 6 is the one the original design lacked. A proof whose public signals are
taken from the prover proves a statement the prover chose; `Groth16Backend.verify`
compares the proof's public signals to the verifier's expectation and rejects on
any mismatch *before* the pairing check.

Nothing unverified is ever relayed, so a node cannot be used to amplify garbage.

## 5. Safety

> **Proposition 1 (Agreement).** If `2q > n + f` and at most `f` nodes are
> Byzantine, then no two correct nodes commit different models at the same
> height.

*Proof.* Write `v ≠ v'` for the two committed hashes and `r ≤ r'` for their
sub-rounds. Two quorums intersect in at least `2q − n > f` members, so any pair
of them shares a **correct** node. Also `n ≥ 3f+1` and `2q > n+f` give `q ≥
2f+1`, so each certificate holds at least `q − f ≥ f+1` correct precommits.

The argument needs one property of the implementation, stated separately because
it is what closes the induction:

> **(I)** A correct node never prevotes at sub-round `s` after observing a polka
> at `s`. Observing one sets its step to `precommit` (`_on_polka` → `_precommit`),
> and the prevote action fires only in the `prevote` step. So a correct node's
> prevote at `s` reflects a lock formed *strictly before* `s`.
> Pinned by `test_a_node_that_sees_a_polka_does_not_then_prevote_in_that_sub_round`.

And one property of the *system*, because purging changes who counts:

> **(P)** A certificate or polka is valid **once formed**. Validity is a function
> of its `q` signatures and the registered member set — never of which
> equivocations the checker has heard about. Evidence governs future tallies
> only. Without (P), "a polka at `r₁`" is not a well-defined object: two correct
> nodes with different evidence would disagree about whether the same signatures
> constitute one. Pinned by
> `test_a_certificate_stays_valid_after_one_of_its_voters_is_blacklisted`.

*Case `r = r'`.* `C` and `C'` are quorums of precommits in the same sub-round, so
some correct node signed both — two votes in slot `(p, h, r, precommit)`. A
correct node signs one vote per slot (`RoundState.precommitted`, and
`zkfl/votelog.py` across restarts). Impossible.

*Case `r < r'`.* We prove the stronger claim, stated over the **certificate**
because that is what the induction step consumes — a single node's precommit
does not intersect a polka:

> **(C)** If a commit certificate for `v` exists at sub-round `r`, then for every
> `r₁ ≥ r`, any polka at `r₁` is for `v`.

`C'` exists, so some correct node precommitted `v'` at `r'`, which it does only
after locking on `v'` — so a polka for `v'` exists at `r' ≥ r`, and (C) applied
to `C` gives `v' = v`, contradicting `v ≠ v'`. It remains to prove (C), by strong
induction on `r₁`.

*Base `r₁ = r`.* `C` holds `≥ f+1` correct precommits for `v` at `r`, and a
correct node precommits non-nil only after observing a polka, so a polka for `v`
at `r` exists. Any polka for `u` at `r` is another quorum of prevotes in the same
sub-round, so some correct node signed both — two votes in slot
`(p, h, r, prevote)`. Hence `u = v`.

*Step.* Let `r₁ > r`, assuming (C) throughout `[r, r₁)`. Suppose a polka for `u`
at `r₁`. Its `q` prevotes and `C`'s `q` precommits — both quorums — share a
correct `p*`, which precommitted `v` at `r` and prevoted `u` at `r₁`.
Precommitting `v` at `r` means `p*` held lock `(v, r)`. A correct node prevotes
its locked value (`_prevote_value`), so if `u ≠ v` then `p*`'s lock moved off `v`
before it prevoted at `r₁`; locks move only on an observed polka, at a sub-round
no earlier than the locked round. So `p*` observed a polka for some `u₀ ≠ v` at
`r₀ ∈ [r, r₁]`, before prevoting at `r₁`. By **(I)**, `r₀ ≠ r₁`. So `r₀ ∈ [r, r₁)`
and the induction hypothesis makes that polka one for `v`, i.e. `u₀ = v` —
contradicting `u₀ ≠ v`. Therefore `u = v`. ∎

> **Corollary (Chain consistency).** Because entry `h` records `H(μ_{h−1})` and
> the audit chain hashes each entry together with its predecessor's link, two
> correct nodes agreeing at every height ≤ `h` have bit-identical logs.
> `Simulation.check_invariants` asserts this after every run.

Note what Proposition 1 does **not** say. It says nothing about the *quality* of
`μ_h` — `f` Byzantine nodes can submit valid proofs for genuinely trained but
adversarially-chosen updates, and only aggregation (§7) bounds their influence.
It also assumes exactly the fault model in `docs/THREAT_MODEL.md`; with `f + 1`
Byzantine nodes, two conflicting certificates are constructible and the argument
fails at the first step, as it must.

### A correct node must not forget that it voted

Both cases of Proposition 1 use *a correct node signs one vote per slot*, and the
fault model admits crash-restart. `RoundState.precommitted` lives in memory, so
a node that crashed after broadcasting a prevote and restarted without it would
sign a second, different vote in the same slot: not Byzantine, but
indistinguishable from it, blacklisted by its peers on evidence that is entirely
genuine, and a counterexample to the hypothesis the proof uses.

So `_sign_vote` appends the vote to a write-ahead log and `fsync`s it **before**
the vote is broadcast, and on restart a slot already in the log re-sends exactly
the vote it recorded, byte for byte. A crash between recording and sending loses
the broadcast, which is harmless — peers time out and the sub-round proceeds —
rather than losing the record, which is not.

`zkfl/votelog.py` holds both implementations. `InMemoryVoteLog` reports
`durable = False` and is what simulations use, since they never restart;
a deployment passes `Node(vote_log_path=...)` and gets `DurableVoteLog`, which
replays its log on construction and tolerates a torn final record. Pinned by
`test_a_restarted_node_resends_its_vote_instead_of_signing_a_new_one`,
`test_a_vote_is_on_disk_before_it_is_returned_for_broadcast` and
`test_the_log_survives_a_torn_final_record` in `tests/test_votelog.py`.

## 6. Liveness

> **Proposition 2 (Progress).** Under partial synchrony, with `n − f ≥ q`, at
> most `f` Byzantine nodes, and at least one correct node proposing, every
> height eventually commits at some correct node.

*Proof sketch.* After GST, delays are bounded by some `Δ`. The collect window is
`w·2^min(r,6)`, so there is an `r*` with `w·2^{r*} > 2Δ` — which presumes
sub-rounds continue without bound (see the fourth bullet below). From `r*` onward
every message a correct node sends at the start of a phase arrives before that
phase closes at every other correct node.

It does **not** follow that they prevote identically just because they hold the
same `P`. A locked node prevotes its *lock*, not `Agg(P)`. If some correct nodes
locked on `v` earlier and the rest never observed that polka, the locked ones
offer `v` and the unlocked ones offer `Agg(P) = v' ≠ v` once more proposals have
landed. If neither camp reaches `q`, no polka forms — and only a polka moves a
lock, so the split reproduces itself at every subsequent sub-round and the height
never finishes. More sub-rounds do not help; each recreates the same two camps.

**Transferable polkas** close this. A node observing `q` matching prevotes
gossips them as a `Certificate` with `phase = prevote`; a receiver verifies the
`q` signatures itself (`handle_polka`) and adopts the lock whenever the polka's
sub-round is `≥` its own locked round. After `r*` every such certificate reaches
every correct node, so all of them hold the lock of the highest sub-round any has
seen and prevote it. That is `n − f ≥ q` identical prevotes: a polka, hence `q`
matching precommits, hence a certificate. Byzantine nodes can withhold or send
garbage; neither prevents `q` correct nodes from agreeing, because `q ≤ n − f`. ∎

This is the leaderless counterpart of Tendermint's proposer re-proposing with a
proof-of-lock from the highest round. With no proposer to carry it, the evidence
travels by itself. It costs `n(n−1)` deliveries per height — a quarter of the
total in `results/e3_consensus.json`, reported as its own column there. Pinned by
`test_a_gossiped_polka_breaks_a_locked_unlocked_split`.

Four implementation details are load-bearing here and each was a bug at some
point in this repository's history:

- **The window must grow.** With a fixed window there is no `r*` and the argument
  fails outright. A run on loopback hides this because `Δ ≈ 0`.
- **A quorum of nil precommits starts the next sub-round immediately.** Since any
  two quorums overlap, `q` nils mean no value can be certified at that sub-round;
  waiting out the timer only adds latency.
- **A node that falls behind must keep asking.** Requesting a missing height once
  and never retrying leaves a node stuck forever when the answer is lost;
  `Node._sync_inflight` retries on an interval.
- **Exhausting `max_sub_rounds` must not stop the clock.** It used to return
  `NeedSync` *instead of* `OpenWindow`. But sync is only answerable by a node
  that is *ahead*, and the case that exhausts the budget is exactly the one where
  no quorum formed — nobody is ahead, nothing answers, and since every node hits
  the budget together the whole network went silent. At 5% loss a run committed
  three heights and then nothing, reproducibly. A node now keeps its clock
  running and asks for sync alongside. Pinned by
  `test_exhausting_the_sub_round_budget_asks_for_sync_without_stopping`.

### Delay and loss are not the same problem

Proposition 2 is stated for delay, and the doubling window is the answer to
delay. It is *not* an answer to loss: a dropped vote costs the whole sub-round
however long the window was, and a two-phase sub-round needs two full quorums to
land rather than one. What loss needs is more attempts — which `max_sub_rounds`
caps.

That reasoning is sound and the measurement does not cleanly support it. At
`n = 10` with 50 ms delay and 5% per-message loss, over 25 heights and three
seeds:

| `max_sub_rounds` | heights committed | failure rate |
|---|---|---|
| 4 | 39 / 75 | 0.48 |
| 8 | 53 / 75 | 0.29 |
| 16 | 35 / 75 | 0.53 |

Better at 8 than at 4, **worse at 16 than at either**. The relationship is not
monotone, and with three seeds — whose per-seed splits at this loss rate run
3/25, 25/25, 21/25 — the differences are not separable from noise. The default
is 8 because 8 measured best, and that is the entire strength of the claim.

An earlier 10-height version of this sweep reported 0.23 / 0.00 / 0.00 and we
briefly believed it. It was under-sampled. Quote `results/e3_consensus.json`
after a run you paid for, not a table you remember.

The explanation is arithmetic, and it is why the budget is not an independent
knob. Attempt `r` waits `w · 2^min(r,6)`, so the time a height has already spent
after `r+1` attempts is the running sum of that schedule. At the `w = 0.2 s`
used for this sweep:

A sub-round opens *two* windows, one for prevote and one for precommit, so a
sub-round in which neither quorum arrives costs twice the window. Both
accountings:

| attempt | window | cumulative (prevote only) | cumulative (both phases) |
|---|---|---|---|
| 0 | 0.20 s | 0.20 s | 0.40 s |
| 3 | 1.60 s | 3.00 s | 6.00 s |
| 4 | 3.20 s | 6.20 s | 12.40 s |
| 5 | 6.40 s | 12.60 s | **25.20 s** |
| 6 | 12.80 s | **25.40 s** | 50.80 s |

The caller's per-height deadline is 15 s, so **five or six attempts are
reachable at any budget**. `max_sub_rounds` of 8 and 16 are therefore not
different configurations — they are the same five or six attempts, and the gap
between their measured rates is noise. Budget and window schedule are
one parameter, not two, and they were not designed that way here. Reproduce with
`experiments/e3_deadline.py`, which varies the deadline while holding the budget
fixed.

The failures are also bimodal rather than merely variable. A run at 5% loss
either stays healthy — median 0.6 s, nearly every height committing — or
collapses to the timeout and commits almost nothing; seed 0 collapses
reproducibly, at 3 heights, on every repeat. A pooled commit rate is a mixture
of the two regimes and describes neither. `experiments/e3_repeatability.py`
reports the range instead.

At 20% loss nothing helps. This is a limit of the design, not a parameter to
tune, and `docs/THREAT_MODEL.md` states it as one.

## 7. Aggregation

`Agg(μ, P)` is a pure integer function (`zkfl/aggregation.py`):

1. order `P` canonically by `proposal_id` — floating-point order dependence
   would break determinism, and so would set iteration order;
2. per coordinate, apply the configured rule — `trimmed_mean` (drop
   `min(⌊|P|·permille/1000⌋, ⌊(|P|-1)/2⌋)` from each end -- the second
   term keeps at least one value, so the rule is total on any non-empty
   set), `median`, or weighted `mean`;
3. all arithmetic is integer floor division, never floating point.

`robustness_note()` states at runtime whether the trimmed-mean bound actually
holds for the run's `(|P|, f, permille)`; when it does not, the run says
`DOES NOT HOLD (bound is vacuous)` rather than staying quiet.

## 8. Accountability

Equivocation is detected on the signed message, before any expensive
verification. The two conflicting messages *are* the evidence; a recipient
re-verifies both signatures itself (`Evidence.is_valid`). An offender is
blacklisted, its votes purged from every open round, and its proposals refused.
Evidence is retained for `evidence_retention` heights so late-arriving proof of
misbehaviour at a committed height is still accepted and gossiped.

This is *accountability*, not prevention: `f` equivocators can still deny a
height and force sub-rounds. What they cannot do is deny it without leaving a
transferable proof that they did (cf. Civit et al., *Polygraph*, and the
accountability line of work).

## 9. Catch-up

A node at height `h` that sees a certificate for `h' > h + 1` requests audit
entries from `h + 1` upward. Each is verified end to end — certificate quorum,
every proof against the model as of that height, recomputed aggregate, hash-chain
link — before it is applied (`handle_sync_entry`). Requests retry on an interval.
A new node bootstraps from genesis by exactly the same path, and an outside
auditor runs the same checks offline (`zkfl/audit.py`, `verify_offline`).

## 10. Complexity

Per height, with `p` proposals: `O(n·p)` proposal messages, `O(n²)` vote
messages (two phases, each vote sent once by its author and not relayed), and
`O(n)` certificate messages. The earlier design re-broadcast an `O(n)`-sized
vote set with every new vote — `O(n³)` bytes per height.
`tests/test_integration.py::test_message_cost_is_quadratic_not_cubic` and
`experiments/e3_consensus.py` measure the growth rather than asserting it.
