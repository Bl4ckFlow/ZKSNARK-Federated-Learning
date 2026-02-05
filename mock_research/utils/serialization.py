# utils/serialization.py
import numpy as np
import json

def serialize_model(params: dict) -> bytes:
    """
    Serializes model parameters deterministically.
    Handles both numpy arrays (local training) and lists (network JSON).
    """
    if isinstance(params["weights"], list):
        weights = np.array(params["weights"], dtype=np.float64)
    else:
        weights = params["weights"].astype(np.float64)
        
    # 2. Normalize bias
    bias = float(params["bias"])
    
    # 3. Use JSON for consistency with ZK proofs (easier debugging than raw bytes)
    # We round to 8 decimals to avoid floating point inconsistencies across nodes
    data = {
        "weights": np.round(weights, 8).tolist(),
        "bias": round(bias, 8)
    }
    
    # sort_keys=True is CRITICAL for deterministic hashing
    return json.dumps(data, sort_keys=True).encode('utf-8')