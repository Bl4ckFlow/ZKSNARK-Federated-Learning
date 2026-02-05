# nodes/training_engine.py
import numpy as np
import hashlib
from typing import Optional, Dict, Any


class TrainingEngine:
    """
    Stateless Logistic Regression Training Engine (PoC)
    """

    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
        return self.sigmoid(X @ weights + bias)

    def compute_loss(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float) -> float:
        y_pred = self.predict_proba(X, weights, bias)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    def train_one_epoch(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float) -> tuple[np.ndarray, float]:
        """Returns updated (weights, bias)"""
        y_pred = self.predict_proba(X, weights, bias)
        error = y_pred - y

        dw = (1.0 / X.shape[0]) * (X.T @ error)
        db = float(np.mean(error))

        new_weights = weights - self.lr * dw
        new_bias = bias - self.lr * db
        
        return new_weights, new_bias

    def train_until_converged(
        self, 
        data: Dict[str, Any], 
        initial_model: Optional[Dict[str, Any]] = None,
        max_epochs: int = 1000, 
        tol: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Train the logistic regression model until convergence or max_epochs.
        Takes initial_model as input, returns updated model.
        """
        X = np.asarray(data["X"], dtype=float)
        y = np.asarray(data["y"], dtype=float).reshape(-1)

        # Initialize from provided model or zeros
        if initial_model is None:
            weights = np.zeros(X.shape[1], dtype=float)
            bias = 0.0
        else:
            weights = np.array(initial_model["weights"], dtype=float)
            bias = float(initial_model["bias"])

        prev_loss = self.compute_loss(X, y, weights, bias)
        loss = prev_loss

        for _ in range(max_epochs):
            weights, bias = self.train_one_epoch(X, y, weights, bias)
            loss = self.compute_loss(X, y, weights, bias)
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

        return {
            "weights": weights.copy(),
            "bias": bias,
            "loss": loss
        }

    @staticmethod
    def get_model_hash(model: Dict[str, Any]) -> str:
        """Compute hash of model parameters"""
        weights = np.array(model["weights"], dtype=float)
        bias = float(model["bias"])
        
        return hashlib.sha256(
            weights.tobytes() + np.array([bias]).tobytes()
        ).hexdigest()
