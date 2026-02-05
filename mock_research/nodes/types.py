# nodes/types.py
from dataclasses import dataclass, asdict, field
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
        return cls(**data)


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
        return cls(**data)