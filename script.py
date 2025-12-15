import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
SCALE = 10000  # Scale factor for fixed-point representation
LEARNING_RATE = 0.1

# ============================================================================
# STEP 1: SERVER SIDE - INITIAL TRAINING
# ============================================================================
print("="*80)
print("STEP 1: SERVER INITIALIZATION")
print("="*80)

# Load Data
iris = datasets.load_iris()
X = iris.data[:, 2:]  # Only 2 features to keep circuit small
y = (iris.target == 2).astype(int)  # Binary target: Virginica = 1

# Train the global model
server_model = LogisticRegression(random_state=42)
server_model.fit(X, y)

# Extract weights
w_server = server_model.coef_[0]
b_server = server_model.intercept_[0]

print(f"Global Model Weights: {w_server}")
print(f"Global Model Bias:    {b_server}")

# ============================================================================
# STEP 2: CLIENT SIDE - SELECT PRIVATE DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CLIENT DATA SELECTION (PRIVATE)")
print("="*80)

client_idx = 100
x_private = X[client_idx]
y_private = y[client_idx]

print(f"Client Private X: {x_private}")
print(f"Client Private y: {y_private}")

# ============================================================================
# STEP 3: ARITHMETIZATION (PREPARING FOR SNARK)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: ARITHMETIZATION & CIRCUIT SIMULATION")
print("="*80)

# Helper to scale floats to integers
def encode(val):
    return int(val * SCALE)

# Encode all values
w_old_int = np.array([encode(w) for w in w_server])
b_old_int = encode(b_server)
x_int = np.array([encode(x) for x in x_private])
y_true_int = encode(y_private)
eta_int = encode(LEARNING_RATE)

# Forward pass (dot product)
dot_sum = sum([w * xi for w, xi in zip(w_old_int, x_int)])
dot_product = dot_sum // SCALE
z = dot_product + b_old_int

# Linear approximation of prediction
pred = z + (SCALE // 2)

# Compute error
error = pred - y_true_int

# Backward pass (weight updates)
w_new_int = []
for i in range(len(w_old_int)):
    grad = (error * x_int[i]) // SCALE
    update = (eta_int * grad) // SCALE
    w_new_int.append(w_old_int[i] - update)

# Bias update
b_new_int = b_old_int - (eta_int * error // SCALE)

# Print results
w_new_float = [val / SCALE for val in w_new_int]
print(f"\nCalculated w_new (int): {w_new_int}")
print(f"Calculated w_new (float): {w_new_float}")
print(f"Calculated b_new (int): {b_new_int}")
print(f"Calculated b_new (float): {b_new_int / SCALE}")

# ============================================================================
# STEP 4: GENERATE JSON INPUT
# ============================================================================
print("\n" + "="*80)
print("STEP 4: SAVING 'input_fl.json'")
print("="*80)

snark_input = {
    "w_old": [str(x) for x in w_old_int],
    "b_old": str(b_old_int),
    "eta": str(eta_int),
    "w_new": [str(x) for x in w_new_int],
    "b_new": str(b_new_int),
    "x": [str(x) for x in x_int],
    "y_true": str(y_true_int)
}

with open("input_fl.json", "w") as f:
    json.dump(snark_input, f, indent=2)

print(f"File 'input_fl.json' generated successfully.")
print("You can now generate the proof using:")
print(">> snarkjs groth16 prove ...")
