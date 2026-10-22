"""
Protocol constants.

Every number that has to agree between the Circom circuit, the Python trainer
and the on-the-wire encoding lives here.  ``scripts/setup_circuit.py`` renders
the circuit's ``main`` component from these values, and
``tests/test_fixedpoint.py`` fails if the two ever drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

#: The piecewise-linear sigmoid table, mirrored from
#: circuits/lib/activation.circom.  Only the two values the numerator bound
#: needs live here; changing the table without changing these makes check()
#: pass on a circuit it no longer describes.
PWL_MAX_SLOPE = 231059
PWL_LAST_BREAKPOINT = 5_000_000

# BN254 scalar field -- the field circom compiles into.
FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617


@dataclass(frozen=True)
class FixedPointConfig:
    """Fixed-point and circuit sizing.

    The bit widths are not decoration: every one of them is a range check the
    circuit actually enforces, and together they are what stops a prover from
    reading a huge field element as a small negative number.
    """

    scale: int = 10 ** 6
    n_features: int = 4
    batch_size: int = 8
    merkle_depth: int = 4

    #: |x| < 2^x_bits, i.e. features up to +-33.5 at scale 10^6
    x_bits: int = 25
    #: |w|, |b| < 2^w_bits, i.e. weights up to +-67
    w_bits: int = 26
    #: |z| < 2^z_bits for the pre-activation.  Leave at 0 to derive it from
    #: the other widths -- hand-setting it is how a wider feature vector
    #: silently produces a circuit whose witnesses overflow their own checks.
    z_bits: int = 0
    #: verifiable clipping bound on |w_new - w_old|_inf, scaled
    max_delta: int = 500_000

    def __post_init__(self) -> None:
        if self.z_bits == 0:
            object.__setattr__(self, "z_bits", self.derive_z_bits())

    def derive_z_bits(self) -> int:
        """Bits needed for z = (sum_j x_j w_j)/scale + b.

        The quotient needs ``quot_bits``; adding a bias below ``2^w_bits``
        cannot carry past one more bit, so ``max(quot, w) + 1`` is both
        sufficient and tight.  :meth:`check` re-derives it independently.
        """
        dot_bits = self.x_bits + self.w_bits + (self.n_features - 1).bit_length()
        quot_bits = dot_bits - (self.scale - 1).bit_length() + 1
        return max(quot_bits, self.w_bits) + 1

    @property
    def max_batches(self) -> int:
        """Committed batches a node can hold, i.e. the Merkle tree's width."""
        return 1 << self.merkle_depth

    @property
    def max_samples(self) -> int:
        return self.max_batches * self.batch_size

    @property
    def n_public_signals(self) -> int:
        # wOld[n], bOld, wNew[n], bNew, eta, dataRoot
        return 2 * self.n_features + 4

    def check(self) -> None:
        """Re-derive the range bounds and refuse a configuration that cannot
        be proved.  Cheap insurance against someone widening ``n_features`` or
        ``batch_size`` and silently producing a circuit whose witnesses
        overflow their own range checks."""
        s = self.scale
        assert s > 0 and (1 << 20) >= s, "scale must fit the 20-bit remainder checks"

        # dot product: sum_j x_j * w_j
        dot_bits = self.x_bits + self.w_bits + (self.n_features - 1).bit_length()
        quot_bits = dot_bits - (s - 1).bit_length() + 1
        assert quot_bits < self.z_bits, (
            f"z_bits={self.z_bits} too small: dot/scale needs {quot_bits} bits"
        )
        assert self.w_bits < self.z_bits, "z_bits must also cover the bias"
        assert self.z_bits + 1 < 253, "z_bits must stay well inside the field"

        # weight update: eta * gw / (k * scale^2) must fit the 32-bit quotient
        gw_bits = (self.batch_size * s * (1 << self.x_bits)).bit_length()
        num_bits = gw_bits + (s - 1).bit_length()
        div_bits = (self.batch_size * s * s).bit_length()
        assert num_bits - div_bits < 32, "weight-update quotient exceeds 32 bits"
        assert (1 << 48) >= self.batch_size * s * s, "48-bit remainder too small"
        assert (1 << 28) >= self.batch_size * s, "28-bit remainder too small"

        # DivConst's `num === q*d + r` is a *field* equation.  The checks on q
        # and r pin down a unique integer (q, r) only while the numerator cannot
        # wrap: we need |num| <= d * 2^quotBits, or a huge field element admits
        # an in-range (q, r) that means nothing.  Every numerator in the circuit
        # is an assigned combination of range-checked signals rather than a free
        # witness, so each has a derivable bound -- and here we derive it rather
        # than trust it.  The margin is stated because "it fits" is the kind of
        # claim that should not be re-established by hand after a config change.
        for label, num_b, quot_bits, divisor in (
            # dot product / scale, quotient sized by z_bits
            ("forward pass", dot_bits, self.z_bits, s),
            # eta * gw / (k * scale^2)
            ("weight update", num_bits, 32, self.batch_size * s * s),
            # eta * gb / (k * scale)
            ("bias update", (self.batch_size * s).bit_length()
             + (s - 1).bit_length(), 32, self.batch_size * s),
            # pwl slope * (a - anchor) / scale.  Bounded by the table -- the
            # largest slope times the last breakpoint -- and constrained to it
            # by an AbsBound in activation.circom, not by z_bits: on the flat
            # piece the slope is zero, so a large |z| cannot enlarge prod.
            ("sigmoid", (PWL_MAX_SLOPE * PWL_LAST_BREAKPOINT).bit_length(),
             24, s),
        ):
            headroom = (divisor.bit_length() + quot_bits) - num_b
            assert headroom > 0, (
                f"{label}: numerator may reach {num_b} bits but DivConst pins a "
                f"unique quotient only below {divisor.bit_length() + quot_bits}; "
                "a wrapped numerator would satisfy num == q*d + r with an "
                "in-range (q, r) that means nothing"
            )
            assert num_b + 2 < 253, f"{label}: numerator must stay inside the field"

        # clipping bound
        assert self.max_delta > 0
        assert self.max_delta + (1 << (self.w_bits + 1)) < (1 << (self.w_bits + 2)), \
            "w_bits + 2 too narrow for the AbsBound on the step"

    def circuit_name(self) -> str:
        return (f"update_check_n{self.n_features}"
                f"_k{self.batch_size}_d{self.merkle_depth}")


DEFAULT_FP = FixedPointConfig()
DEFAULT_FP.check()


@dataclass(frozen=True)
class ProtocolConfig:
    """Consensus, network and accountability parameters.

    ``n``, ``f`` and ``q`` are the fault model, not tuning knobs: safety needs
    ``q > (n + f) / 2`` and liveness needs ``n - f >= q``.  Together those force
    ``n >= 3f + 1``.  :meth:`check` refuses anything else rather than letting a
    run quietly claim a guarantee it does not have.
    """

    n_nodes: int = 10
    #: Byzantine nodes tolerated
    f_byzantine: int = 3
    #: quorum size; derived from (n, f) when left at None
    quorum: int | None = None

    #: collect window in seconds; adaptive_window scales it from measured delay
    collect_window: float = 3.0
    adaptive_window: bool = True
    #: Sub-rounds after which a node *also* starts asking peers for the height.
    #:
    #: Not a ceiling.  It used to be one, and that was a deadlock: `_advance`
    #: returned NeedSync instead of OpenWindow, so a node past the budget went
    #: quiet.  Sync is only answerable by a node that is ahead, and the case
    #: that exhausts the budget is exactly the one where no quorum formed --
    #: nobody is ahead, nothing answers, and since every node hits the budget at
    #: about the same time the whole network stopped.  At 5% loss a run
    #: committed three heights and then nothing, reproducibly.  A node now keeps
    #: its clock running and asks for sync alongside, so a height recovers once
    #: the loss burst passes.  See
    #: tests/test_consensus.py::test_exhausting_the_sub_round_budget_asks_for_sync_without_stopping
    #:
    #: Measured before that fix, at n=10, 50 ms delay and 5% loss over 25
    #: heights x 3 seeds, round-failure rates were 0.48, 0.29 and 0.53 at
    #: budgets 4, 8 and 16 -- better at 8, *worse* at 16, so not monotone, and
    #: not separable from the seed-to-seed spread.
    #:
    #: The reason the knob saturates is arithmetic, not statistical.  Attempt r
    #: waits collect_window * 2^min(r, 6), and a sub-round opens two such
    #: windows -- prevote and precommit -- so a height that has made five
    #: attempts has already spent between 12.6 s (only prevote windows expiring)
    #: and 25.2 s (both) against the 15 s per-height deadline the sweep used.
    #: Five or six attempts are reachable, so budgets of 8 and 16 are the same
    #: configuration and the gap between their rates is noise.  Budget and
    #: window schedule are one parameter, not two.  Reproduce with
    #: experiments/e3_deadline.py, which varies the deadline at a fixed budget;
    #: results/e3_deadline.json and results/e3_consensus.json are the numbers to
    #: quote.
    max_sub_rounds: int = 8

    learning_rate: float = 0.05
    #: fedavg | trimmed_mean | median
    aggregation_rule: str = "trimmed_mean"
    #: trim fraction per coordinate for trimmed_mean, in permille.  An integer
    #: because it ends up inside signed and hash-chained audit entries, where a
    #: float would make the encoding platform-dependent.
    trim_permille: int = 250

    #: strict ZK is the default; the test fixture must be asked for explicitly
    strict_zk: bool = True

    #: bounded look-ahead for proposals at heights above h+1
    future_window: int = 4
    #: per-peer inbound message budget per height
    peer_rate_limit: int = 64
    #: signed votes retained after commit, so late evidence stays checkable
    evidence_retention: int = 8

    #: Re-gossip messages received from peers.  Unnecessary on a full mesh --
    #: the originator already reached everyone -- and expensive: relaying every
    #: vote costs O(n^3) messages.  Turn on for partial-mesh topologies.
    relay_votes: bool = False
    relay_proposals: bool = False

    def effective_quorum(self) -> int:
        if self.quorum is not None:
            return self.quorum
        return (self.n_nodes + self.f_byzantine) // 2 + 1

    def check(self) -> None:
        n, f, q = self.n_nodes, self.f_byzantine, self.effective_quorum()
        assert n >= 1 and f >= 0
        assert 2 * q > n + f, (
            f"unsafe quorum: q={q} must exceed (n+f)/2 = {(n + f) / 2}; "
            "two quorums could otherwise share no honest node"
        )
        assert n - f >= q, (
            f"no liveness: {n - f} honest nodes cannot reach a quorum of {q}"
        )
        assert n >= 3 * f + 1, f"n={n} does not tolerate f={f}; need n >= {3 * f + 1}"
        assert self.aggregation_rule in ("fedavg", "trimmed_mean", "median")
        assert 0 <= self.trim_permille < 500, "trim fraction must be under 50%"
        assert self.max_sub_rounds >= 1

    def with_nodes(self, n: int, f: int | None = None) -> "ProtocolConfig":
        """A config for ``n`` nodes with the largest f that (n, f, q) supports."""
        f = (n - 1) // 3 if f is None else f
        cfg = replace(self, n_nodes=n, f_byzantine=f, quorum=None)
        cfg.check()
        return cfg


DEFAULT_PROTOCOL = ProtocolConfig()
DEFAULT_PROTOCOL.check()
