# nodes/fl_types.py
import numpy as np
from dataclasses import dataclass, asdict, field, fields
from typing import Any, Dict, Optional
import time


@dataclass
class UpdateLogEntry:
    """
    Audit log entry for applied updates.
    """
    source_node: str
    previous_model_hash: str
    params_hash: str
    proof: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class UpdateType:
    """
    Model update message with ZK proof and signature.
    """
    prev_model_hash: str
    params_hash: str
    new_params: Dict[str, Any]
    model_height: int
    zk_proof: Any
    timestamp: float = field(default_factory=time.time)

    sender_pubkey: Optional[str] = None
    signature: Optional[str] = None
    update_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpdateType":
        # FIX 8: Discard unknown keys so that extra network metadata never
        # causes a TypeError when the dataclass is instantiated.
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        if "new_params" in filtered:
            params = filtered["new_params"]
            if "weights" in params:
                w = params["weights"]
                # If it's a string like '[-0.39...]', clean it up
                if isinstance(w, str):
                    w = w.replace('[', '').replace(']', '').split()
                params["weights"] = np.array(w, dtype=float)
            if "bias" in params:
                params["bias"] = float(params["bias"])

        return cls(**filtered)


@dataclass
class VoteMessage:
    """
    Vote message for distributed consensus.
    Each node votes for the first update it sees for a given height.
    """
    voter_node_id: str
    target_height: int
    voted_update_id: str  # Which update this node votes for
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoteMessage":
        # FIX 8: Same guard as UpdateType — strip unknown fields.
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)