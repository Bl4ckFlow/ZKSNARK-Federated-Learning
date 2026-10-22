pragma circom 2.1.6;

include "../../node_modules/circomlib/circuits/poseidon.circom";

/*
 * Data binding.
 *
 * Proving "w_new is a correct gradient step on *some* (x, y)" is vacuous: a
 * prover picks the target first and solves for the data.  What makes the proof
 * mean anything is that the (x, y) it used were fixed before training, in
 * public, by the proposer itself.
 *
 * Each node quantizes its dataset once, cuts it into batches of k, and
 * publishes the Poseidon-Merkle root over the batch digests when it joins.
 * That root is registered in the genesis state and signed.  A proof then
 * carries a membership path showing the batch it trained on is one of the
 * committed ones, and the verifier supplies the root from its own copy of the
 * registry -- never from the proposal.
 *
 * Committing per batch rather than per sample keeps the cost at one Merkle
 * path per proof instead of k of them.
 */

/*
 * SampleDigest(nFeatures) -- Poseidon over one (x, y) pair.
 */
template SampleDigest(nFeatures) {
    signal input x[nFeatures];
    signal input y;
    signal output out;

    component h = Poseidon(nFeatures + 1);
    for (var j = 0; j < nFeatures; j++) {
        h.inputs[j] <== x[j];
    }
    h.inputs[nFeatures] <== y;
    out <== h.out;
}

/*
 * BatchDigest(nFeatures, batchSize) -- a Poseidon hash chain over the batch.
 *
 *     h_0     = 0
 *     h_{i+1} = Poseidon(h_i, SampleDigest(x_i, y_i))
 *
 * The chain binds sample order as well as content, which is what makes a batch
 * a single committed object.
 */
template BatchDigest(nFeatures, batchSize) {
    signal input x[batchSize][nFeatures];
    signal input y[batchSize];
    signal output out;

    component leaf[batchSize];
    component chain[batchSize];
    signal acc[batchSize + 1];
    acc[0] <== 0;

    for (var i = 0; i < batchSize; i++) {
        leaf[i] = SampleDigest(nFeatures);
        for (var j = 0; j < nFeatures; j++) {
            leaf[i].x[j] <== x[i][j];
        }
        leaf[i].y <== y[i];

        chain[i] = Poseidon(2);
        chain[i].inputs[0] <== acc[i];
        chain[i].inputs[1] <== leaf[i].out;
        acc[i + 1] <== chain[i].out;
    }

    out <== acc[batchSize];
}

/*
 * MerkleInclusion(depth) -- binary Poseidon Merkle path.
 *
 * pathIndex[l] = 0 means the running node is the left child at level l.
 * Trees are padded to a power of two with the zero leaf, matching
 * zkfl/commitment.py.
 */
template MerkleInclusion(depth) {
    signal input leaf;
    signal input pathElements[depth];
    signal input pathIndex[depth];
    signal output root;

    component h[depth];
    signal cur[depth + 1];
    signal left[depth];
    signal right[depth];
    cur[0] <== leaf;

    for (var l = 0; l < depth; l++) {
        // pathIndex must be a bit, otherwise the mux below is meaningless.
        pathIndex[l] * (pathIndex[l] - 1) === 0;

        left[l]  <== cur[l] + pathIndex[l] * (pathElements[l] - cur[l]);
        right[l] <== pathElements[l] + pathIndex[l] * (cur[l] - pathElements[l]);

        h[l] = Poseidon(2);
        h[l].inputs[0] <== left[l];
        h[l].inputs[1] <== right[l];
        cur[l + 1] <== h[l].out;
    }

    root <== cur[depth];
}
