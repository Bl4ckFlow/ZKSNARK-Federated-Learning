pragma circom 2.1.6;

include "../../node_modules/circomlib/circuits/bitify.circom";

/*
 * Fixed-point primitives shared by the update circuit.
 *
 * Representation.  A real value v is carried as the integer  v~ = round(v * S)
 * with S = 10^6 (zkfl/config.py is the single source of truth for every
 * constant used here).  Negative values are the field elements p - |v~|, so
 * any signal that may be negative needs an explicit signed range check --
 * without one, a prover reads a huge field element as a small negative number
 * and the arithmetic constrains nothing.
 */

/*
 * SignedRangeCheck(bits) -- constrains  -2^bits <= in < 2^bits.
 * Costs bits + 2 constraints.
 */
template SignedRangeCheck(bits) {
    signal input in;

    assert(bits + 1 < 253);

    component n2b = Num2Bits(bits + 1);
    n2b.in <== in + (1 << bits);
}

/*
 * AbsBound(bits, bound) -- constrains  -bound <= in <= bound.
 *
 * Both shifted values are decomposed with Num2Bits, which admits only
 * non-negative numbers: if |in| exceeded bound one of them would be a field
 * element near p and could not be written in `bits` bits.  `bits` must cover
 * 2 * bound and stay far below the field size.
 */
template AbsBound(bits, bound) {
    signal input in;

    assert(bits < 253);

    component lo = Num2Bits(bits);
    lo.in <== bound + in;

    component hi = Num2Bits(bits);
    hi.in <== bound - in;
}

/*
 * BoundedAbs(bits) -- out = |in| with the sign bit exposed.
 * Requires -2^bits <= in < 2^bits; sign = 1 exactly when in < 0.
 */
template BoundedAbs(bits) {
    signal input in;
    signal output out;
    signal output sign;

    // in + 2^bits lands in [0, 2^(bits+1)); its top bit is 1 iff in >= 0.
    component n2b = Num2Bits(bits + 1);
    n2b.in <== in + (1 << bits);

    sign <== 1 - n2b.out[bits];

    signal sIn;
    sIn <== sign * in;
    out <== in - 2 * sIn;

    // |in| must itself fit in `bits` bits; this also rules out in = -2^bits.
    component absCheck = Num2Bits(bits);
    absCheck.in <== out;
}

/*
 * DivConst(divisor, remBits, quotBits)
 *
 * Euclidean division by a compile-time constant with a non-negative
 * remainder:
 *
 *     num = q * divisor + r,      0 <= r < divisor
 *
 * which is floor division for every sign of num -- the same convention as
 * Python's `//` and `%` on a positive divisor, so zkfl/fixedpoint.py
 * reproduces this bit for bit.
 *
 * Witness note: circom's `\` and `%` see field elements, so a negative
 * numerator is the astronomically large p - |num| and dividing it directly
 * yields nonsense.  We therefore divide num + divisor * 2^quotBits, which is a
 * genuine non-negative integer whenever the quotient fits in quotBits, and
 * subtract the shift back off.  The constraint below is what actually pins
 * (q, r) down; the assignment only has to find them.
 *
 * `num === q * divisor + r` is linear because divisor is a constant, so the
 * cost is just the two remainder decompositions and the signed check on q.
 */
template DivConst(divisor, remBits, quotBits) {
    signal input num;
    signal output q;
    signal output r;

    assert(divisor > 0);
    assert((1 << remBits) >= divisor);
    assert(quotBits + 1 < 253);

    var SHIFT = divisor * (1 << quotBits);

    q <-- (num + SHIFT) \ divisor - (1 << quotBits);
    r <-- (num + SHIFT) % divisor;

    num === q * divisor + r;

    // 0 <= r
    component rLow = Num2Bits(remBits);
    rLow.in <== r;

    // r <= divisor - 1.  Were r >= divisor, the difference would wrap to a
    // field element far larger than 2^remBits and the decomposition fails.
    component rHigh = Num2Bits(remBits);
    rHigh.in <== divisor - 1 - r;

    component qRange = SignedRangeCheck(quotBits);
    qRange.in <== q;
}
