#!/usr/bin/env python3
"""
Setup script for zk-SNARK circuit
Run this once to compile the circuit and generate keys
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.zk_proof import setup_circuit, ZKSNARK
import json


def test_proof_generation():
    """Test that proof generation works"""
    print("\n🧪 Testing proof generation...")
    
    # Sample model parameters (2 weights + 1 bias)
    old_model = {
        "weights": [0.5, 0.3],
        "bias": 0.1
    }
    
    # Simulate training: new = old - 0.01 * gradient
    learning_rate = 0.01
    gradient = {
        "weights": [0.2, 0.15],  # Arbitrary gradient
        "bias": 0.05
    }
    
    new_model = {
        "weights": [
            old_model["weights"][0] - learning_rate * gradient["weights"][0],
            old_model["weights"][1] - learning_rate * gradient["weights"][1]
        ],
        "bias": old_model["bias"] - learning_rate * gradient["bias"]
    }
    
    print(f"  Old model: {old_model}")
    print(f"  New model: {new_model}")
    print(f"  Learning rate: {learning_rate}")
    
    # Generate proof
    zksnark = ZKSNARK(circuit_dir="./circuits")
    proof = zksnark.generate_proof(old_model, new_model, learning_rate)
    
    if proof is None:
        print("❌ Proof generation failed")
        return False
    
    print("✅ Proof generated successfully")
    
    # Verify proof
    print("\n🔍 Testing proof verification...")
    is_valid = zksnark.verify_proof(proof)
    
    if is_valid:
        print("✅ Proof verified successfully!")
        return True
    else:
        print("❌ Proof verification failed")
        return False


def main():
    print("=" * 60)
    print("ZK-SNARK Circuit Setup for Federated Learning")
    print("=" * 60)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    
    import subprocess
    
    try:
        result = subprocess.run(["circom", "--version"], capture_output=True, text=True)
        print(f"✓ Circom: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Circom not found. Install from: https://docs.circom.io/getting-started/installation/")
        print("   Quick install: curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh")
        print("                  cargo install circom")
        return
    
    try:
        result = subprocess.run(["snarkjs", "--version"], capture_output=True, text=True)
        print(f"✓ SnarkJS: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ SnarkJS not found. Install with: npm install -g snarkjs")
        return
    
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        print(f"✓ Node.js: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Node.js not found. Install from: https://nodejs.org/")
        return
    
    # Setup circuit
    print("\n🔧 Compiling circuit and generating keys...")
    print("(This may take 1-2 minutes for the first time)")
    
    if not os.path.exists("./circuits/update_check.circom"):
        print("❌ Circuit file not found: ./circuits/update_check.circom")
        print("Please create the circuit file first.")
        return
    
    success = setup_circuit(
        circuit_path="./circuits/update_check.circom",
        output_dir="./circuits"
    )
    
    if not success:
        print("\n❌ Circuit setup failed")
        return
    
    # Test proof generation
    if test_proof_generation():
        print("\n" + "=" * 60)
        print("✅ Setup complete! Your zk-SNARK system is ready.")
        print("=" * 60)
        print("\nYou can now run your federated learning nodes with:")
        print("  enable_strict_zk=True")
    else:
        print("\n❌ Setup succeeded but proof testing failed")
        print("Check the error messages above for details")


if __name__ == "__main__":
    main()