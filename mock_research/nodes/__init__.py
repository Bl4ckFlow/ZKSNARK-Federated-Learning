# nodes/__init__.py

# Optional: expose main node classes for easy imports
from .node import Node
from .training_engine import TrainingEngine
from .update_manager import UpdateManager
from .network_layer import NetworkLayer

# __all__ is optional but recommended for clarity
__all__ = ["Node", "TrainingEngine", "UpdateManager", "NetworkLayer"]
