"""
client_app.py
-------------
Flower ClientApp for federated SOH estimation.

Each client:
1. Receives the current global model (or an empty seed) from the server.
2. Loads its local partition of discharge data.
3. Trains a local RandomForestRegressor.
4. Sends the fitted model back together with benchmark metrics.
5. On evaluate rounds, runs local evaluation and returns metrics.
"""

from __future__ import annotations

import os
import numpy as np

import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from sohfed.task import (
    load_data,
    train,
    evaluate,
    bytes_to_ndarray,
    ndarray_to_bytes,
    bytes_to_model,
    model_to_bytes,
)


# ---------------------------------------------------------------------------
# NumPyClient implementation
# ---------------------------------------------------------------------------

class SOHClient(NumPyClient):
    """
    Flower NumPyClient wrapping a scikit-learn RandomForestRegressor.

    Because sklearn models are not differentiable, we serialise the entire
    fitted model as a byte-string transported inside a 1-D uint8 NumPy array.
    The server aggregates by combining all client forests into a FederatedForest.
    """

    def __init__(self, context: Context) -> None:
        self.context = context

        # Read configuration from pyproject.toml / run-config
        cfg = context.run_config
        node_cfg = context.node_config

        self.partition_id: int = int(node_cfg.get("partition-id", 0))
        self.num_partitions: int = int(node_cfg.get("num-partitions", 5))
        self.base_path: str = cfg.get("data-base-path", "data")
        self.metadata_path: str = cfg.get("metadata-path", "metadata.csv")
        self.n_estimators: int = int(cfg.get("n-estimators", 100))
        self.test_size: float = float(cfg.get("test-size", 0.2))
        self.partition_strategy: str = cfg.get("partition-strategy", "by_battery")
        self.dirichlet_alpha: float = float(cfg.get("dirichlet-alpha", 0.5))

        # Load data once at construction
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = load_data(
            base_path=self.base_path,
            metadata_path=self.metadata_path,
            partition_id=self.partition_id,
            num_partitions=self.num_partitions,
            test_size=self.test_size,
            partition_strategy=self.partition_strategy,
            dirichlet_alpha=self.dirichlet_alpha,
        )

        self.local_model = None  # will be set after first fit

    # ------------------------------------------------------------------
    # Flower API: get / set parameters
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        """Return local model as a serialised byte array."""
        if self.local_model is None:
            # Before first training: return empty placeholder
            return [np.array([], dtype=np.uint8)]
        return [bytes_to_ndarray(model_to_bytes(self.local_model))]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Deserialise the global model received from the server (ignored for RF)."""
        # For Random Forests we always retrain from scratch on local data,
        # so we intentionally do not initialise from the server model.
        # The aggregation happens on the SERVER side (prediction averaging).
        pass

    # ------------------------------------------------------------------
    # Flower API: fit
    # ------------------------------------------------------------------

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        """Train a local RandomForestRegressor and return it with metrics."""
        self.set_parameters(parameters)

        # Train
        self.local_model, bench = train(
            self.X_train,
            self.y_train,
            n_estimators=self.n_estimators,
        )

        # Evaluate on local test set so we can report train-side loss
        eval_metrics = evaluate(self.local_model, self.X_test, self.y_test)

        # Serialise model for transport; measure payload size
        model_bytes = model_to_bytes(self.local_model)
        model_size = len(model_bytes)

        metrics = {
            # Benchmark: timing & memory
            "train_time_s": bench["train_time_s"],
            "peak_memory_kb": bench["peak_memory_kb"],
            "model_size_bytes": float(model_size),
            "n_train_samples": float(bench["n_train_samples"]),
            # Local accuracy
            "local_mae": eval_metrics["mae"],
            "local_rmse": eval_metrics["rmse"],
            "local_accuracy_1pct": eval_metrics["accuracy_1pct"],
            # Required by FedAvg weighting
            "num_examples": float(len(self.X_train)),
        }

        print(
            f"[Client {self.partition_id}] fit() "
            f"train_time={bench['train_time_s']:.3f}s  "
            f"local_mae={eval_metrics['mae']:.4f}  "
            f"payload={model_size/1024:.1f} KB"
        )

        return [bytes_to_ndarray(model_bytes)], len(self.X_train), metrics

    # ------------------------------------------------------------------
    # Flower API: evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[float, int, dict]:
        """Evaluate the local model on the local test set."""
        # Use our locally trained model (already fitted during fit())
        if self.local_model is None:
            return 0.0, len(self.X_test), {"mae": 0.0}

        metrics = evaluate(self.local_model, self.X_test, self.y_test)

        print(
            f"[Client {self.partition_id}] evaluate() "
            f"mae={metrics['mae']:.4f}  "
            f"rmse={metrics['rmse']:.4f}  "
            f"acc_1pct={metrics['accuracy_1pct']:.3f}"
        )

        return float(metrics["mae"]), len(self.X_test), {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "accuracy_1pct": metrics["accuracy_1pct"],
        }


# ---------------------------------------------------------------------------
# ClientApp entry point
# ---------------------------------------------------------------------------

def client_fn(context: Context) -> fl.client.Client:
    return SOHClient(context).to_client()


app = ClientApp(client_fn=client_fn)
