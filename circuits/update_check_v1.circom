pragma circom 2.0.0;

// Template to prove: new_weight = old_weight - (learning_rate * gradient)
template ModelUpdateCheck(n) {
    // PUBLIC INPUTS
    signal input old_params[n];      // Scaled integers
    signal input new_params[n];      // Scaled integers
    signal input learning_rate;      // Scaled integer

    // PRIVATE INPUTS (The Secret Witness)
    signal input gradient[n];        // Scaled integers

    // Internal signals
    signal delta[n];

    // Scaling factor (10^6) to handle fixed-point arithmetic
    var SCALE = 1000000;

    for (var i = 0; i < n; i++) {
        // 1. Calculate the change: delta = learning_rate * gradient
        // The result of (int * int) has scale 10^12
        delta[i] <== learning_rate * gradient[i];

        // 2. Enforce update rule: (old - new) * SCALE === delta
        // We scale (old - new) by 10^6 to match the 10^12 scale of delta
        (old_params[i] - new_params[i]) * SCALE === delta[i];
    }
}

// Instantiate for 3 parameters (2 weights + 1 bias)
component main {public [old_params, new_params, learning_rate]} = ModelUpdateCheck(3);