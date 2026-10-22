# Verifiable federated learning with zk-SNARKs

Nodes train locally, prove in zero knowledge that the update they publish **is**
the gradient step on data they committed to beforehand, and agree on the
aggregate of the verified updates by a leaderless BFT protocol. The training
history is a hash-chained log that a stranger can re-verify offline.

The interesting part is what the proof rules out. A node cannot publish a model
it likes and call it training, cannot train on data it never committed to,
cannot reuse a proof against a different base model, and cannot move the model
further than the protocol allows — and none of that rests on trusting the node,
the proposer, or the network.

```
                proposal (μ', π)                      certificate
  ┌────────┐  ───────────────────►  ┌──────────┐  ───────────────►  ┌────────┐
  │ train  │                        │  verify  │                    │ commit │
  │ locally│  ◄───────────────────  │ + agree  │  ◄───────────────  │ + log  │
  └────────┘        μ_{h-1}         └──────────┘      audit sync    └────────┘
     data                            n nodes,                       hash chain
   committed                         q = quorum                      re-checkable
   to a Merkle root                                                  offline
```

---

## Quick start

```bash
make install          # python venv + node_modules
make setup            # compile the circuit, run the phase-2 ceremony  (~2 min)
make test             # 151 tests
make sim              # 10 nodes, 5 heights, live
```

Without `make setup`, everything still runs — the circuit tests skip and the
protocol uses a proof backend that refuses to pretend it is one (`NullProofBackend`
raises unless you pass `i_understand_this_is_not_a_proof=True`, and refuses
outright when `ZKFL_FORBID_NULL_BACKEND` is set).

Building offline uses `make setup-offline`, which generates powers-of-tau
locally. **That is not a trusted setup** — whoever ran it can forge proofs for
this circuit. The script says so, and records the mode in `build_info.json`.

## What is in the circuit

`circuits/lib/update_check.circom`, 19,402 R1CS constraints for
`n_features = 4`, `batch_size = 8`, `merkle_depth = 4`.

Public: `w_old`, `b_old`, `w_new`, `b_new`, `η`, `dataRoot`.
Private: the batch `(x, y)` and its Merkle paths.

```
range checks on every signal                    Num2Bits
y ∈ {0,1}                                       y·(y−1) = 0
0 ≤ η ≤ SCALE
each sample is in the committed dataset         Merkle inclusion vs dataRoot
z = ⟨x, w_old⟩/SCALE + b_old                    remainder-checked division
pred = σ(z)                                     5-piece linear, max error 0.0117
err = pred − y·SCALE
w_new[j] = w_old[j] − ⌊η·g_j / (k·SCALE²)⌋      exact integer floor division
|w_new − w_old| ≤ maxDelta                      two Num2Bits per coordinate
```

Fixed-point division is the subtle part: circom's `\` and `%` operate on field
elements, so a negative numerator divides `p − |num|` and the quotient is
garbage. `DivConst` offsets by `divisor · 2^quotBits` before dividing and
constrains `num = q·divisor + r` with `0 ≤ r < divisor` by two range checks —
so the circuit's division is Python's `//`, exactly, for negative numerators too.

## Repository

```
circuits/lib/          fixedpoint, activation, merkle, update_check
circuits/main/         generated top-level circuits (scripts/gen_circuit_main.py)
zkfl/config.py         one source of truth for every parameter; check() re-derives
zkfl/zk.py             Groth16 backend; the verifier builds its own public inputs
zkfl/poseidon.py       bit-identical to circomlib's PoseidonEx
zkfl/consensus.py      the state machine: effects in, effects out, no I/O
zkfl/aggregation.py    trimmed mean / median / weighted mean, pure integers
zkfl/accountability.py equivocation evidence, self-verifying and transferable
zkfl/audit.py          hash-chained log, sync, offline re-verification
zkfl/votelog.py        write-ahead log of our own votes, fsynced before broadcast
zkfl/node.py           the node, plus three adversaries
zkfl/simulation.py     n nodes on an in-memory bus with injected delay and loss
scripts/setup_circuit.py    compile + multi-contributor phase 2 with a beacon
scripts/run_simulation.py   live runner, in-memory or real WebSockets
experiments/e1..e5     the evaluation, each writing results/*.json
experiments/e3_repeatability.py  how far the loss-regime numbers move between runs
experiments/e3_deadline.py       why the sub-round budget saturates
scripts/check_paper_numbers.py   every number in the paper, against results/*.json
docs/PROTOCOL.md       the protocol, with safety and liveness propositions
docs/THREAT_MODEL.md   assumptions, defences, and what is *not* defended
docs/CLAIMS.md         claim → mechanism → evidence, for every claim made
```

## The protocol in one paragraph

Each height is prevote → polka → lock → precommit → certificate. A prevote says
what a node's verified proposal set aggregates to; `q` matching prevotes (a
*polka*) is proof a quorum could see that value, and only that licenses a lock;
`q` matching precommits is a certificate, and any node holding one commits
regardless of its own timers. **Polkas are gossiped, not just observed** — a
locked node offers its lock while an unlocked one offers `Agg(P)`, and if
neither camp reaches `q` the split is permanent, because only a polka moves a
lock. Sending the polka itself is what breaks the tie; it is the leaderless
stand-in for Tendermint's proposer re-proposing with a proof-of-lock. Safety needs `2q > n + f`, liveness needs
`n − f ≥ q`, and together `n ≥ 3f + 1`; `ProtocolConfig.check()` refuses anything
weaker rather than letting a run advertise a guarantee it lacks. Full statement
and proofs in [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

Locking on a *polka* rather than on one's own prevote is not a detail. Locking on
your own vote deadlocks the moment two correct nodes close their windows with
different proposal sets — which on any network with non-zero delay is
immediately.

## Results

Regenerate with `make evidence`; everything lands in `results/` with the CPU and
toolchain versions recorded alongside.

| | |
|---|---|
| Circuit | 19,402 constraints, 742-byte proof (snarkjs JSON; 128 B as raw bn128 group elements), 9.5 MB proving key |
| Prove | 0.84 s warm median, 1.09 s cold |
| Verify | 15.1 ms |
| Learning (iris, 5 nodes, 60 heights) | 0.867 accuracy; float64 centralised, FedAvg-without-ZK and scikit-learn all 0.833 on the same split. Across three splits this work is 0.867–0.967 and FedAvg 0.833–0.967, scoring *identically* on two of them — a 30-sample test set, so this is "does not lose to float64", not "beats it" |
| Consensus, `n` = 5 → 50 | messages/`n²` flat at 2.88 → 3.00 (quadratic, not cubic) |
| Consensus under 150 ms delay | 75/75 heights committed, 1.46 s median |
| Consensus under 5% loss | **bimodal**: a run either stays healthy (median 0.6 s) or collapses to the timeout. Identical seeds give 0.71 and 0.40 commit rates on separate runs, so quote the range, not one number (`make e3-repeat`) |
| Consensus under 20% loss | almost nothing commits at any sub-round budget — a real limit, reported, not hidden |
| Sub-round budget | not a usable knob: the window schedule (`w·2^min(r,6)`, two windows per sub-round) puts only 5–6 attempts inside a 15 s deadline, so `max_sub_rounds` 8 and 16 are the same configuration (`make e3-deadline`) |
| Offline audit | 9/9 tamperings rejected; 41.6 kB log re-verified cold in 0.37 s (0.12 s/height) |
| Directed poisoning, 2 of 10 | FedAvg −0.117 accuracy; trimmed mean −0.017; median −0.014 (median of 3 seeds) |
| Directed poisoning, 3 of 10 (`f` = 3, trim = 2) | trimmed mean −0.114 — it fails exactly where its own bound says it will |
| Poisoning adversary | calibrated, not just loud: it sweeps its committed feature scale and keeps the most damaging one that stays *inside* the clipping bound, so no poisoned step is wasted. Against order statistics it picks a **smaller** scale than against FedAvg |

## Running attacks

```bash
python -m experiments.e4_adversarial     # every attack, end to end
python scripts/run_simulation.py --nodes 10 --byzantine equivocating-voter:9
python scripts/run_simulation.py --nodes 10 --byzantine flooder:0
python scripts/run_simulation.py --nodes 10 --loss 0.2 --latency 0.05
```

`zkfl/legacy_v1.py` keeps the original 6-constraint circuit and its witness
recipe, so `e4_adversarial` can demonstrate the POISON1 attack succeeding
against it and failing against the current one. Its only purpose is to make the
difference measurable.

## What this does not do

Read [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) before citing anything here.
The short version:

- **The proof hides the data, not the update.** `w_new` is public by
  construction. Gradient-inversion attacks apply, so publishing per-node updates
  weakens privacy no matter how good the proof is.
- **It cannot tell honest data from poisoned data.** A node that commits to a
  poisoned dataset and trains correctly on it produces a valid proof. Only the
  aggregation rule bounds that, and only when enough is trimmed per side —
  `robustness_note()` says at runtime when the bound is vacuous.
- **Membership is permissioned.** No Sybil resistance is offered or implied.
- **Groth16 needs a per-circuit ceremony**, and `--local-ptau` builds are
  forgeable by whoever ran them.
- **4 features, batch of 8.** Constraints grow roughly linearly in
  `n_features · batch_size`. Nothing here supports extrapolating to deep models.

## Reproducing

```bash
git clone <this repo> && cd ZKSNARK-Federated-Learning
make install
make setup            # or setup-offline, with the caveat above
make test
make evidence
```

`make evidence` should be run on an otherwise idle machine. The consensus
medians and the prove/verify timings are wall-clock, and at 5% loss we have
measured the *same seeds* commit at 0.33 and at 0.71 depending only on what else
the host was doing — `make e3-repeat` exists to measure that spread rather than
let a single run stand in for it. Delivery counts and every accuracy figure are
deterministic and do not care.

Dependencies are pinned exactly — `requirements.txt` and `package.json` — because
a caret range on circomlib can silently change the Poseidon constants that
`zkfl/poseidon.py` mirrors, and then every proof in the repository stops
verifying.

## Licence

MIT.
