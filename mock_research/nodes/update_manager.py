# nodes/update_manager.py
import hashlib
import time
from typing import Dict, Any, Optional

from utils.crypto import sign_update, verify_signature
from utils.zk_proof import generate_zk_proof, verify_zk_proof
from nodes.types import UpdateType
from utils.serialization import serialize_model

class UpdateManager:
    """
    Builds, verifies, and manages model updates.
    Includes buffering support for out-of-order updates.
    NOW WITH REAL ZK-SNARK SUPPORT!
    """

    def __init__(self, private_key: Optional[str], enable_strict_zk: bool = False, learning_rate: float = 0.01):
        """
        Args:
            private_key: Ed25519 private key for signing (None for PoC)
            enable_strict_zk: If True, use zk-SNARK proofs
            learning_rate: Learning rate for zk-SNARK gradient verification
        """
        self.private_key = private_key
        self.enable_strict_zk = enable_strict_zk
        self.learning_rate = learning_rate
        self.candidate_pool: list[UpdateType] = []

    def build_update(
        self, 
        prev_model_hash: str, 
        new_params: Dict[str, Any], 
        model_height: int,
        prev_model: Optional[Dict[str, Any]] = None,
        sender_pubkey: Optional[str] = None
    ) -> UpdateType:
        """
        Builds a new update with ZK proof and signature.
        
        Args:
            prev_model_hash: Hash of previous model
            new_params: New model parameters
            model_height: Height of new model
            prev_model: Previous model (REQUIRED for zk-SNARK)
            sender_pubkey: Public key of sender
        """
        if model_height is None:
            raise ValueError("model_height must be provided")

        serialized = serialize_model(new_params)
        params_hash = hashlib.sha256(serialized).hexdigest()
        
        # Generate ZK proof (zk-SNARK if enabled, simple otherwise)
        zk_proof = generate_zk_proof(
            prev_model_hash=prev_model_hash,
            serialized_new_params=serialized,
            prev_model=prev_model,
            new_model=new_params,
            learning_rate=self.learning_rate,
            enable_strict_verification=self.enable_strict_zk
        )

        update = UpdateType(
            prev_model_hash=prev_model_hash,
            new_params=new_params,
            params_hash=params_hash,
            model_height=model_height,
            sender_pubkey=sender_pubkey,
            zk_proof=zk_proof,
            timestamp=time.time()
        )

        payload = self.serialize_for_signing(update.to_dict())
        sig_bytes = sign_update(self.private_key, payload) if self.private_key else None
        update.signature = sig_bytes.hex() if sig_bytes is not None else None
        update.update_id = hashlib.sha256(payload).hexdigest()

        return update

    def verify_update(self, update: UpdateType, prev_model: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verifies the ZK-proof, cryptographic signature, and parameter hash 
        of an UpdateType object.
        
        Args:
            update: UpdateType to verify
            prev_model: Previous model (optional, for zk-SNARK verification)
        """
        serialized = self.serialize_params(update.new_params)

        # Verify ZK proof (zk-SNARK if enabled)
        if not verify_zk_proof(
            prev_model_hash=update.prev_model_hash,
            serialized_new_params=serialized,
            proof=update.zk_proof,
            prev_model=prev_model,
            enable_strict_verification=self.enable_strict_zk
        ):
            print(f"[UpdateManager] ZK proof verification failed")
            return False

        # Verify cryptographic signature (if present)
        if update.sender_pubkey and update.signature:
            payload = self.serialize_for_signing(update.to_dict())
            try:
                sig_bytes = bytes.fromhex(update.signature)
                if not verify_signature(update.sender_pubkey, payload, sig_bytes):
                    print(f"[UpdateManager] Signature verification failed")
                    return False
            except ValueError:
                print(f"[UpdateManager] Invalid signature format")
                return False

        # Verify parameter hash
        computed_hash = hashlib.sha256(serialized).hexdigest()
        if computed_hash != update.params_hash:
            print(f"[UpdateManager] Hash mismatch: expected {update.params_hash[:8]}, got {computed_hash[:8]}")
            return False

        return True

    def add_candidate(self, update: UpdateType) -> None:
        """
        Add update to candidate pool with bounded size.
        Supports buffering of future updates.
        """
        # Check if this update is already in the pool
        if any(u.update_id == update.update_id for u in self.candidate_pool):
            return
        
        self.candidate_pool.append(update)

        # Maintain bounded pool size (FIFO eviction)
        if len(self.candidate_pool) > 100:
            self.candidate_pool.pop(0)

    def get_update_by_id(self, update_id: str) -> Optional[UpdateType]:
        """
        Retrieve update from candidate pool by its ID.
        """
        for update in self.candidate_pool:
            if update.update_id == update_id:
                return update
        return None

    def get_candidates_for_height(self, height: int) -> list[UpdateType]:
        """
        Returns all candidate updates for a specific height.
        Used for checking buffered future updates.
        """
        return [u for u in self.candidate_pool if u.model_height == height]

    def clear_candidates_for_height(self, height: int) -> None:
        """
        Remove all candidates for a specific height.
        Call this after consensus is reached to free memory.
        """
        self.candidate_pool = [u for u in self.candidate_pool if u.model_height != height]

    def get_buffered_heights(self) -> list[int]:
        """
        Returns list of all heights that have buffered updates.
        Useful for debugging and recovery.
        """
        heights = set(u.model_height for u in self.candidate_pool)
        return sorted(heights)

    # ===========================
    # ====== SERIALIZATION ======
    # ===========================

    @staticmethod
    def serialize_for_signing(update: Dict[str, Any]) -> bytes:
        """
        Serialize update fields for signing.
        """
        fields = [
            update["prev_model_hash"],
            update["params_hash"],
            str(update["model_height"]),
            str(update["timestamp"]),
        ]
        return "|".join(fields).encode()

    @staticmethod
    def serialize_params(params: Dict[str, Any]) -> bytes:
        """
        Serialize model parameters.
        """
        return serialize_model(params)