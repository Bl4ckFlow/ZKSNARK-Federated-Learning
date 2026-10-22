pragma circom 2.1.6;

include "fixedpoint.circom";
include "activation.circom";
include "merkle.circom";

/*
 * ModelUpdateCheck -- proof of one mini-batch logistic-regression step.
 *
 * The statement proved is:
 *
 *   I know a batch of k samples whose digest is a leaf of the Merkle tree with
 *   root `dataRoot`, such that running exactly one full-batch gradient step of
 *   logistic regression with learning rate `eta` from (wOld, bOld) yields
 *   (wNew, bNew); and every intermediate quantity stays inside its declared
 *   range.
 *
 * What that buys, precisely:
 *
 *   - computation integrity: wNew really is the training output, not a chosen
 *     target with a solved-for witness;
 *   - input binding: the data was committed, signed and published before
 *     training, so it cannot be invented to reach a target;
 *   - a bounded step: |wNew - wOld|_inf <= maxDelta, verifiable, which the
 *     aggregation rule leans on.
 *
 * What it does not buy: resistance to a proposer who honestly trains on data
 * that is committed but adversarial.  That is the aggregation rule's job
 * (zkfl/aggregation.py), not the circuit's.
 *
 * Arithmetic is integer fixed point with scale SCALE throughout; every
 * division is Euclidean with a range-checked remainder, so zkfl/trainer.py
 * reproduces the witness exactly rather than solving backwards for it.
 *
 * Parameters
 *   n         number of features
 *   k         batch size
 *   depth     Merkle depth over the node's committed batches
 *   SCALE     fixed-point scale (10^6)
 *   xBits     |x| < 2^xBits          (features)
 *   wBits     |w| < 2^wBits          (weights and bias)
 *   zBits     |z| < 2^zBits          (pre-activation)
 *   maxDelta  |wNew - wOld|_inf <= maxDelta, scaled
 */
template ModelUpdateCheck(n, k, depth, SCALE, xBits, wBits, zBits, maxDelta) {
    // ---- public: supplied by the verifier from its own state --------------
    signal input wOld[n];
    signal input bOld;
    signal input wNew[n];
    signal input bNew;
    signal input eta;
    signal input dataRoot;

    // ---- private ----------------------------------------------------------
    signal input x[k][n];
    signal input y[k];
    signal input pathElements[depth];
    signal input pathIndex[depth];

    // =====================================================================
    // 1. Ranges.  Public inputs are checked too: it costs little and makes
    //    the circuit meaningful on its own terms rather than conditional on
    //    the caller behaving.
    // =====================================================================
    component wOldRange[n];
    component wNewRange[n];
    for (var j = 0; j < n; j++) {
        wOldRange[j] = SignedRangeCheck(wBits);
        wOldRange[j].in <== wOld[j];
        wNewRange[j] = SignedRangeCheck(wBits);
        wNewRange[j].in <== wNew[j];
    }
    component bOldRange = SignedRangeCheck(wBits);
    bOldRange.in <== bOld;
    component bNewRange = SignedRangeCheck(wBits);
    bNewRange.in <== bNew;

    // 0 <= eta <= SCALE.  A negative eta would turn descent into ascent.
    component etaRange = Num2Bits(21);
    etaRange.in <== eta;
    component etaCap = Num2Bits(21);
    etaCap.in <== SCALE - eta;

    component xRange[k][n];
    for (var i = 0; i < k; i++) {
        for (var j = 0; j < n; j++) {
            xRange[i][j] = SignedRangeCheck(xBits);
            xRange[i][j].in <== x[i][j];
        }
        // labels are binary
        y[i] * (y[i] - 1) === 0;
    }

    // =====================================================================
    // 2. The batch is one of the proposer's committed batches.
    // =====================================================================
    component batch = BatchDigest(n, k);
    for (var i = 0; i < k; i++) {
        for (var j = 0; j < n; j++) {
            batch.x[i][j] <== x[i][j];
        }
        batch.y[i] <== y[i];
    }

    component incl = MerkleInclusion(depth);
    incl.leaf <== batch.out;
    for (var l = 0; l < depth; l++) {
        incl.pathElements[l] <== pathElements[l];
        incl.pathIndex[l] <== pathIndex[l];
    }
    incl.root === dataRoot;

    // =====================================================================
    // 3. Forward pass, per sample.
    //
    //    z_i    = (sum_j x_ij * wOld_j) / SCALE + bOld       [scale SCALE]
    //    pred_i = sigma_pwl(z_i)                             [scale SCALE]
    //    err_i  = pred_i - y_i * SCALE                       [scale SCALE]
    // =====================================================================
    signal prods[k][n];
    signal dot[k];
    component dotDiv[k];
    component act[k];
    signal z[k];
    signal err[k];

    for (var i = 0; i < k; i++) {
        var acc = 0;
        for (var j = 0; j < n; j++) {
            prods[i][j] <== x[i][j] * wOld[j];
            acc += prods[i][j];
        }
        dot[i] <== acc;

        // |dot| < 2^(xBits+wBits+ceil(log2 n)); the quotient loses log2(SCALE)
        // of that.  zBits is sized for it in zkfl/config.py.
        dotDiv[i] = DivConst(SCALE, 20, zBits);
        dotDiv[i].num <== dot[i];

        z[i] <== dotDiv[i].q + bOld;

        act[i] = PWLSigmoid(zBits, SCALE);
        act[i].z <== z[i];

        err[i] <== act[i].out - y[i] * SCALE;
    }

    // =====================================================================
    // 4. Batch gradient and the update rule.
    //
    //    gw_j = sum_i err_i * x_ij                    [scale SCALE^2]
    //    gb   = sum_i err_i                           [scale SCALE]
    //
    //    wNew_j = wOld_j - (eta * gw_j) / (k * SCALE^2)
    //    bNew   = bOld   - (eta * gb)   / (k * SCALE)
    //
    //    Both divisions are Euclidean (floor), matching zkfl/fixedpoint.py.
    // =====================================================================
    signal gterm[k][n];
    signal gw[n];
    component wDiv[n];
    component wDelta[n];

    for (var j = 0; j < n; j++) {
        var g = 0;
        for (var i = 0; i < k; i++) {
            gterm[i][j] <== err[i] * x[i][j];
            g += gterm[i][j];
        }
        gw[j] <== g;

        wDiv[j] = DivConst(k * SCALE * SCALE, 48, 32);
        wDiv[j].num <== eta * gw[j];

        wNew[j] === wOld[j] - wDiv[j].q;

        // Verifiable clipping: the aggregation rule is allowed to assume every
        // accepted proposal moved each coordinate by at most maxDelta.  This
        // has to be a two-sided non-negative bound -- a signed range check
        // would leave a whole extra power of two of slack on the wrong side.
        wDelta[j] = AbsBound(wBits + 2, maxDelta);
        wDelta[j].in <== wNew[j] - wOld[j];
    }

    var gbAcc = 0;
    for (var i = 0; i < k; i++) {
        gbAcc += err[i];
    }
    signal gb;
    gb <== gbAcc;

    component bDiv = DivConst(k * SCALE, 28, 32);
    bDiv.num <== eta * gb;

    bNew === bOld - bDiv.q;

    component bDelta = AbsBound(wBits + 2, maxDelta);
    bDelta.in <== bNew - bOld;
}
