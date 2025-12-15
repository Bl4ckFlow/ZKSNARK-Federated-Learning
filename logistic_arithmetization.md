# Arithmetization Example: Logistic Regression for zk-SNARKs

## Setup

Let's work through a concrete example with **small numbers** to see how logistic regression gets arithmetized.

### Problem Parameters
- **Features**: d = 3 (three-dimensional input)
- **Training sample**: Just 1 data point for simplicity
- **Field**: Fp where p = 101 (small prime for illustration)
- **Fixed-point scaling**: SCALE = 10 (so 0.5 becomes 5, 1.2 becomes 12, etc.)

### Initial Data
- **Global weights**: W = [0.2, -0.3, 0.5] 
- **Client's private data point**: x = [1.0, 2.0, -1.0], y = 1 (positive class)
- **Learning rate**: η = 0.1

---

## Step 1: Fixed-Point Encoding

Convert everything to integers in F₁₀₁:

```
W_encoded = [2, -3, 5]  → [2, 98, 5] (mod 101)
x_encoded = [10, 20, -10] → [10, 20, 91] (mod 101)
η_encoded = 1
y = 1 (already integer)
```

---

## Step 2: Arithmetize Forward Pass

### 2.1 Compute Dot Product z = w·x

We need constraints for:
- z₁ = w₁ · x₁ = 2 · 10 = 20
- z₂ = w₂ · x₂ = 98 · 20 = 1960 ≡ 34 (mod 101)
- z₃ = w₃ · x₃ = 5 · 91 = 455 ≡ 51 (mod 101)

**R1CS Constraints for multiplication:**

```
Constraint 1: (w₁) · (x₁) = (z₁)
  Variables: [1, w₁, w₂, w₃, x₁, x₂, x₃, z₁, z₂, z₃, ...]
  A = [0, 1, 0, 0, 0, 0, 0, ...]  (selects w₁)
  B = [0, 0, 0, 0, 1, 0, 0, ...]  (selects x₁)
  C = [0, 0, 0, 0, 0, 0, 0, 1, ...] (selects z₁)

Constraint 2: (w₂) · (x₂) = (z₂)
Constraint 3: (w₃) · (x₃) = (z₃)
```

### 2.2 Sum the products

Now sum: z = z₁ + z₂ + z₃ = 20 + 34 + 51 = 105 ≡ 4 (mod 101)

**R1CS Constraint for addition:**
```
Constraint 4: (z₁ + z₂ + z₃) · (1) = (z)
  A = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, ...]  (coefficient vector)
  B = [1, 0, 0, ...]  (selects constant 1)
  C = [0, ..., 0, 1, ...] (selects z)
```

So **z = 4** (in scaled representation, this is ≈ 0.4 in real numbers)

---

## Step 3: Arithmetize Sigmoid Approximation

Real sigmoid: σ(z) = 1/(1 + e^(-z))

We use polynomial approximation: **σ(z) ≈ 0.5 + 0.25z** (for small z)

In fixed-point (scaled by 10):
```
σ̃(z) = 5 + (25 · z) / 100
```

With z = 4:
```
σ̃(4) = 5 + (25 · 4) / 100 = 5 + 100/100 = 5 + 1 = 6
```

**R1CS Constraints:**
```
Constraint 5: (25) · (z) = (temp)
  → temp = 100

Constraint 6: (temp) · (inv_100) = (temp2)
  → temp2 = 1  (using modular inverse of 100 in F₁₀₁)

Constraint 7: (5 + temp2) · (1) = (σ̃)
  → σ̃ = 6
```

So **ŷ = σ̃ = 6** (prediction ≈ 0.6 in real numbers)

---

## Step 4: Arithmetize Gradient

For logistic regression: **g = (ŷ - y) · x**

Compute error: e = ŷ - y = 6 - 1 = 5

**Gradient for each weight:**
```
g₁ = e · x₁ = 5 · 10 = 50
g₂ = e · x₂ = 5 · 20 = 100
g₃ = e · x₃ = 5 · 91 = 455 ≡ 51 (mod 101)
```

**R1CS Constraints:**
```
Constraint 8: (ŷ - y) · (1) = (e)
  → e = 5

Constraint 9: (e) · (x₁) = (g₁)
Constraint 10: (e) · (x₂) = (g₂)
Constraint 11: (e) · (x₃) = (g₃)
```

---

## Step 5: Arithmetize Update Rule

**w'ⱼ = wⱼ - η · gⱼ**

For each weight:
```
w'₁ = 2 - 1 · 50 = 2 - 50 = -48 ≡ 53 (mod 101)
w'₂ = 98 - 1 · 100 = 98 - 100 = -2 ≡ 99 (mod 101)
w'₃ = 5 - 1 · 51 = 5 - 51 = -46 ≡ 55 (mod 101)
```

**R1CS Constraints:**
```
Constraint 12: (η) · (g₁) = (δ₁)
  → δ₁ = 50

Constraint 13: (w₁ - δ₁) · (1) = (w'₁)
  → w'₁ = 53

Constraint 14-17: Similar for w'₂ and w'₃
```

---

## Step 6: Public vs Private

### Public Inputs (known to verifier):
- W = [2, 98, 5]
- W' = [53, 99, 55]
- η = 1

### Private Witness (known only to prover):
- x = [10, 20, 91]
- y = 1
- All intermediate values: z₁, z₂, z₃, z, σ̃, e, g₁, g₂, g₃, δ₁, δ₂, δ₃

---

## Step 7: What the SNARK Proves

The proof π demonstrates:

> "I know private data (x, y) and intermediate computations such that:
> - Computing w·x gives z
> - Applying sigmoid approximation to z gives ŷ
> - Computing gradient (ŷ - y)·x gives g
> - Updating W using gradient descent with rate η produces exactly W'
> 
> **All without revealing x or y!**"

The server can verify the proof in milliseconds without learning the private data.

---

## Summary of Constraints

Total R1CS constraints for this example: **≈ 17 constraints**

- 3 for dot product multiplications
- 1 for summation
- 3 for sigmoid approximation
- 1 for error computation
- 3 for gradient multiplications
- 6 for weight updates (3 multiplications + 3 subtractions)

Each constraint is one row in the R1CS matrix system, which then gets converted to QAP polynomials for the actual SNARK proof generation.