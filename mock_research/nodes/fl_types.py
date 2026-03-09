# nodes/fl_types.py
import hashlib
import numpy as np
from dataclasses import dataclass, asdict, field, fields
from typing import Any, Dict, List, Optional
import time


@dataclass
class UpdateLogEntry:
    """Audit log entry for applied updates."""
    source_node: str
    previous_model_hash: str
    params_hash: str
    proof: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class UpdateType:
    """Model update message with ZK proof and signature."""
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
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        if "new_params" in filtered:
            params = filtered["new_params"]
            if "weights" in params:
                w = params["weights"]
                if isinstance(w, str):
                    w = w.replace('[', '').replace(']', '').split()
                params["weights"] = np.array(w, dtype=float)
            if "bias" in params:
                params["bias"] = float(params["bias"])
        return cls(**filtered)


@dataclass
class VoteMessage:
    """A single vote cast by one node for one update at one height."""
    voter_node_id: str
    target_height: int
    voted_update_id: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoteMessage":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class VoteSetBroadcast:
    """
    Byzantine-resilient vote gossip message.

    Instead of only sharing its own vote, a node periodically broadcasts
    the FULL set of votes it has accumulated for a given height.
    Receiving nodes cross-check: if they already know that voter X chose
    update A, but this message claims X chose update B, X is an equivocator
    and gets permanently flagged — their vote is excluded from consensus.

    This makes it impossible for a Byzantine node to secretly vote for A
    with half the network and B with the other half without being caught.
    """
    sender_node_id: str
    target_height: int
    votes: List[VoteMessage]
    timestamp: float = field(default_factory=time.time)

    def fingerprint(self) -> str:
        """Stable hash of the vote content — used for dedup in NetworkLayer."""
        key = "|".join(sorted(
            f"{v.voter_node_id}:{v.voted_update_id}" for v in self.votes
        ))
        return hashlib.sha256(
            f"{self.sender_node_id}:{self.target_height}:{key}".encode()
        ).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_node_id": self.sender_node_id,
            "target_height": self.target_height,
            "votes": [v.to_dict() for v in self.votes],
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoteSetBroadcast":
        votes = [VoteMessage.from_dict(v) for v in data.get("votes", [])]
        return cls(
            sender_node_id=data["sender_node_id"],
            target_height=data["target_height"],
            votes=votes,
            timestamp=data.get("timestamp", time.time()),
        )