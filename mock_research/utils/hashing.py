import json
import hashlib
from typing import Any, Optional, List

def hash_object(obj: Any) -> str:
    """
    Deterministic hash for models / params / metadata
    """
    data = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()