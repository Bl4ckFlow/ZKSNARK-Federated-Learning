pragma circom 2.0.0;

// Template to verify Federated Logistic Regression weight update
template FederatedUpdate(nFeatures, SCALE) {
    
    // -------------------------------
    // 1. PUBLIC INPUTS
    // -------------------------------
    signal input w_old[nFeatures];  // Global model weights
    signal input b_old;             // Global model bias
    signal input eta;               // Learning rate (scaled)
    signal input w_new[nFeatures];  // Updated weights
    signal input b_new;             // Updated bias

    // -------------------------------
    // 2. PRIVATE INPUTS
    // -------------------------------
    signal input x[nFeatures];      // Client local features
    signal input y_true;            // Client label (scaled)

    // -------------------------------
    // 3. INTERMEDIATE SIGNALS
    // -------------------------------
    signal dot_product;
    signal z;
    signal pred;
    signal error;
    signal scaled_grad[nFeatures];
    signal update_neg[nFeatures];
    signal bias_update_neg;

    // -------------------------------
    // FORWARD PASS: dot_product = sum(w_i * x_i) / SCALE
    // -------------------------------
    var sum = 0;
    for (var i = 0; i < nFeatures; i++) {
        sum += w_old[i] * x[i];
    }
    // Division by constant using witness assignment
    dot_product <-- sum / SCALE;

    // Add bias
    z <== dot_product + b_old;

    // Prediction approximation
    pred <== z + (SCALE / 2);

    // Error
    error <== pred - y_true;

    // -------------------------------
    // BACKWARD PASS: compute weight updates
    // -------------------------------
    for (var i = 0; i < nFeatures; i++) {
        // Gradient = error * x_i
        scaled_grad[i] <-- (error * x[i]) / SCALE;

        // Update = eta * gradient / SCALE
        update_neg[i] <-- -(eta * scaled_grad[i] / SCALE);

        // Quadratic constraint: w_new = w_old + (-update)
        w_new[i] * 1 === w_old[i] + update_neg[i];
    }

    // Bias update
    bias_update_neg <-- -(eta * error / SCALE);
    b_new * 1 === b_old + bias_update_neg;
}

// Instantiate the circuit: 2 features, SCALE = 10000
component main {public [w_old, b_old, eta, w_new, b_new]} = FederatedUpdate(2, 10000);
