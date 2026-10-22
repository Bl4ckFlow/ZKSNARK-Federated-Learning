# Threat model

This document says what the system assumes, what it defends against, and — at
least as importantly — what it does not. Every claim elsewhere in the repository
is scoped by what is written here. Where a defence is partial, it is described
as partial.

## 1. Participants and setup

A **permissioned** set of `n` nodes. Membership is fixed for a run and known to
everyone: each node holds an Ed25519 key pair, and the mapping from node id
(the hex-encoded public key) to its registered data commitment is established
before height 1 and is immutable thereafter (`zkfl/crypto.py`,
`PeerRegistry.register`).

Nothing here provides **Sybil resistance**. An adversary who can enrol
identities at will controls the vote and the model, and no amount of proving
changes that. Admission is out of scope and must be solved by whatever operates
the deployment — a consortium agreement, a staking mechanism, a CA. A paper
that reports Byzantine tolerance for a permissionless deployment of this design
would be reporting a property it does not have.

## 2. Fault model

- At most `f` nodes are **Byzantine**: arbitrary behaviour, including equivocation,
  collusion between all `f`, and adaptive choice of when to deviate. They see
  every message on the network.
- The remaining `n − f` nodes are **correct**: they follow the protocol, though
  they may be slow, may crash and restart, and may be temporarily partitioned.
  Restart recovers committed state from the SQLite audit log and *in-flight
  votes* from an `fsync`ed write-ahead log (`zkfl/votelog.py`) — without the
  latter a restarted node would sign a second vote in a slot it had already
  broadcast, which is genuine equivocation evidence against a node that did
  nothing wrong.
- **Partial synchrony.** Messages between correct nodes are eventually delivered,
  and after some unknown global stabilisation time delays are bounded by an
  unknown constant. Before that, delays are arbitrary. The collect window doubles
  per sub-round (`ConsensusEngine._window_seconds`) precisely so that it
  eventually exceeds the real delay; a *fixed* window can starve forever, and
  a run on loopback where propagation is free will never reveal this.
- Byzantine nodes **cannot forge signatures** and cannot break the discrete-log
  assumptions on BN254.

The quorum size `q` must satisfy

```
2q > n + f        (safety:   any two quorums share an honest node)
n − f ≥ q         (liveness: the correct nodes alone can form one)
```

which together force `n ≥ 3f + 1`. `ProtocolConfig.check()` refuses any
configuration that does not, rather than letting a run advertise a guarantee it
does not hold. The parameters reported in the original paper — `n = 10`,
`q = 6` — satisfy this only for `f = 1`, not for the `f = 3` implied by "up to
30% Byzantine".

## 3. What is defended, and by what

| Attack | Mechanism | Where |
|---|---|---|
| Fabricated model update (no training done) | Groth16 proof that `w_new` is the gradient step from `w_old` on committed data | `circuits/lib/update_check.circom` |
| Training on data the node never committed to | Merkle inclusion of every sample against the registered `dataRoot` | `MerkleInclusion`, `PeerRegistry` |
| Proof replayed against a different base model | Verifier supplies `w_old`, `b_old`, `eta`, `dataRoot` from *its own* state; the proof's public signals must equal them before the pairing check | `zkfl/zk.py`, `Groth16Backend.verify` |
| Unbounded / poisoning update | `|w_new − w_old| ≤ maxDelta` and full range checks on every signal, all in-circuit | `AbsBound`, `SignedRangeCheck` |
| Out-of-range or non-binary labels | `y·(y−1) = 0`, `Num2Bits` bounds on features | `ModelUpdateCheck` |
| Equivocating voter (two votes in one slot) | Both signed votes are the evidence; any node holding them broadcasts a self-verifying accusation | `zkfl/accountability.py` |
| Equivocating proposer (two proposals at one height) | Same, keyed on `(proposer, height)` | `AccountabilityLog.observe_proposal` |
| A correct node equivocating against itself after a crash | Votes are `fsync`ed to a write-ahead log *before* broadcast; a restart re-sends the recorded vote byte for byte | `zkfl/votelog.py`, `ConsensusEngine._sign_vote` |
| A locked/unlocked split that no number of sub-rounds resolves | Polkas are gossiped as transferable certificates; a receiver verifies the `q` signatures itself and adopts the lock | `Certificate.is_polka`, `ConsensusEngine.handle_polka` |
| Commit without a quorum | Commit requires a certificate: `q` signed precommits for one `(height, sub_round, model_hash)` | `Certificate.is_valid` |
| A polka passed off as a decision | `Certificate.is_valid` requires `phase == precommit` | `zkfl/messages.py` |
| Message flooding | Per-peer rate limit and a bounded future-height buffer, both applied before any verification | `zkfl/network.py`, `MessageGate` |
| Replay of an old message | `(sender, height, seq, kind, content-hash)` dedup; stale heights rejected | `Envelope.dedup_key` |
| A minority of poisoned updates surviving aggregation | Coordinate-wise trimmed mean or median | `zkfl/aggregation.py` |
| A node stuck below the committed height | Certificate-driven sync with retry, verified end to end before it is applied | `ConsensusEngine.handle_sync_entry` |
| Tampering with the published history | Hash chain plus offline re-verification of every proof, certificate and aggregate | `zkfl/audit.py`, `experiments/e5_audit.py` |

## 4. What is **not** defended

These are limitations, not oversights, and none of them should be read past.

**Privacy of the model updates.** The proof hides the training data. It does not
hide `w_new`, which is public by construction — every node needs it to
aggregate. Gradient-inversion attacks (Zhu et al., *Deep Leakage from Gradients*,
NeurIPS 2019; Geiping et al., *Inverting Gradients*, NeurIPS 2020) reconstruct
training samples from exactly this quantity. **Publishing a per-node update
weakens privacy regardless of the ZK proof.** Composing this with differential
privacy or secure aggregation is future work; claiming privacy without it is
wrong.

**The trusted setup.** Groth16 needs a per-circuit ceremony. `scripts/setup_circuit.py`
performs a real multi-contributor phase-2 with a beacon over a public
powers-of-tau transcript, but with `--local-ptau` it generates phase 1 locally,
and in that mode **whoever ran it can forge proofs for this circuit**. The script
says so, loudly, and records the mode in `build_info.json`. Nothing verifies
that the phase-1 transcript you fetched is the one the ceremony produced beyond
its published hash.

**Model poisoning within the bound.** `maxDelta` and trimmed-mean bound how far
one node can move the model in one height. They do not stop a patient adversary
from moving it slowly over many heights, and no in-circuit check can: the circuit
proves the update *is* the gradient step on committed data, not that the
committed data is honest. A node that commits to a poisoned dataset and trains
correctly on it produces a valid proof. Defending that needs data-level
attestation, which is out of scope.

**Nothing binds a batch to a height.** The circuit proves the update is a
correct step on *some* batch whose digest is in the committed tree; it does not
constrain *which*. A node is free to train on its most damaging committed batch
at every height, and an adversary would. Honest nodes walk their batches
round-robin, and `experiments/e4_adversarial.py` does the same, so the poisoning
numbers **understate** this adversary: it commits one poisoned shard and replays
its worst batch indefinitely. The fix is small and is described rather than
claimed — expose the Merkle leaf index as a public signal, constrain it to the
path bits already in the circuit, and have the verifier require it to equal a
deterministic schedule such as `H(h ‖ proposer) mod 2^d`. Not implemented: a gap
in coverage, not in soundness.

**Trimmed-mean robustness is conditional.** Coordinate-wise trimmed mean bounds
the adversary's influence only when the number trimmed per side is at least the
number of Byzantine proposals. With few proposals per height, `trim_permille`
rounds down to zero and the bound is vacuous. The code says so at runtime
(`aggregation.robustness_note`) rather than staying quiet, and any run whose
notes contain `DOES NOT HOLD` has no robustness guarantee.

**Transport authentication.** `WebSocketTransport` binds a node id to an address
and drops unknown senders, but the channel itself is plain TCP. A network
adversary can delay, drop and reorder — which the fault model already grants
them — and can read everything, which matters given the privacy limitation
above. Deployments should terminate TLS.

**Availability.** A partition holding more than `f` nodes stops progress. This is
the FLP-mandated price of safety under partial synchrony and is not a bug, but a
deployment must expect it.

**Sustained packet loss.** Delay is handled — 150 ms one-way costs latency and
nothing else. Loss is not: a dropped vote costs a whole sub-round, and a
two-phase sub-round needs two quorums to land.

At `n = 10` and 5% loss the behaviour is **bimodal**, and this matters more than
any average of it. A run either stays healthy — median height 0.6 s, nearly
every height committing — or collapses to the per-height timeout and commits
almost nothing. Re-running the identical configuration with identical seeds
gives commit rates of 0.71 and 0.40, so a single figure is a sample of a mixture
rather than a measurement of a rate.

**No sub-round budget makes this reliable, and the budget is not the knob it
appears to be.** Attempt `r` waits `collect_window · 2^min(r,6)` and each
sub-round opens two such windows, so five attempts already consume 12.6–25.2 s
against a 15 s per-height deadline: only five or six attempts are reachable at
*any* budget, which is why `max_sub_rounds` 4, 8 and 16 measure 0.48, 0.29 and
0.53 — not even monotone. Budget and window schedule are one parameter.
**At 20% loss almost nothing commits at any budget.**

A lossy link needs a retransmitting transport underneath. This protocol does not
compensate for one, and no parameter in it will.

**Sub-round exhaustion is no longer a stopping condition, and used to be.**
`max_sub_rounds` once made a node return `NeedSync` *instead of* opening the next
window, so a node past the budget went quiet. Sync is only answerable by a node
that is *ahead*, and the case that exhausts the budget is exactly the one where
no quorum formed: nobody is ahead, nothing answers, and every node reaches the
budget at about the same time. A node now keeps its clock running and asks for
sync alongside, so a height recovers once the loss burst passes. What remains is
the ordinary partition limit above: if the correct nodes cannot reach `q`, the
height waits.

**Side channels.** Proving time depends on the witness. Nothing here is constant-time.

## 5. Assumptions inherited from the cryptography

- Groth16 soundness under the knowledge-of-exponent and `q`-power assumptions
  on BN254 (~100 bits of security after the Kim–Barbulescu improvements to the
  tower number field sieve, not the 128 sometimes quoted).
- Poseidon collision resistance at the circomlib parameters used here.
- Ed25519 EUF-CMA.
- SHA-256 collision resistance for the audit chain and proposal ids.

## 6. How to check these claims

Every entry in the "defended" table above has a test that fails if the defence
is removed:

```
pytest tests/test_circuit.py -m circuit    # the in-circuit constraints
pytest tests/test_consensus.py             # quorum arithmetic, locks, certificates
pytest tests/test_accountability.py        # equivocation evidence
pytest tests/test_integration.py           # live network, each adversary in turn
python -m experiments.e4_adversarial       # the attacks end to end
python -m experiments.e5_audit             # offline audit and eight tamperings
```

`docs/CLAIMS.md` maps each claim in the paper to the mechanism that implements it
and the artefact that demonstrates it.
