"""
task.py
-------
Model definition, data loading, training and evaluation for SOH estimation
using a Random Forest Regressor federated with Flower.

Because sklearn's RandomForestRegressor is not a neural network, we serialise
its parameters (the individual tree structures) as a flat NumPy array so they
can be exchanged via Flower's ArrayRecord.

Partitioning strategies
-----------------------
Three strategies are supported, chosen via the `partition_strategy` argument
of `load_data()`:

  "iid"       — Random equal-size slices across all discharge cycles (default
                in many FL tutorials, but unrealistic for battery data).

  "by_battery" — Each client owns a disjoint subset of battery IDs
                (NaturalIdPartitioner equivalent).  This is the most realistic
                scenario: organisation A owns batteries B0047/B0048, organisation
                B owns B0049/B0050, etc.  Clients see very different degradation
                trajectories → high non-IID-ness.

  "dirichlet"  — Dirichlet(α) sampling over battery IDs.  Lower α → more
                heterogeneous assignment (some clients get lots of one battery,
                very little of another).  α=100 approaches IID.  Good for
                studying the effect of data heterogeneity on federated accuracy.

Benchmarking hooks
------------------
Every public function that touches training / evaluation also emits timing
and memory information into a shared BENCHMARK_LOG list so the ServerApp can
aggregate and report them.
"""

from __future__ import annotations

import io
import os
import pickle
import time
import tracemalloc
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Global benchmark log – appended to by clients, read by server
# ---------------------------------------------------------------------------
BENCHMARK_LOG: list[dict] = []


# ---------------------------------------------------------------------------
# Coulomb-counting helper
# ---------------------------------------------------------------------------

def coulomb_capacity(time_s: np.ndarray, current_a: np.ndarray) -> float:
    """Compute capacity in Ah via Coulomb counting (trapezoidal integration)."""
    time_h = time_s / 3600.0
    return float(np.trapezoid(np.abs(current_a), time_h))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_voltage_features(df: pd.DataFrame) -> list[float]:
    """Return [V_mean, V_min, V_std, V_area] from a discharge DataFrame."""
    v = df["Voltage_measured"].values
    t = df["Time"].values
    return [
        float(v.mean()),
        float(v.min()),
        float(v.std()),
        float(np.trapz(v, t)),
    ]


# ---------------------------------------------------------------------------
# Internal: build full feature matrix from all discharge files
# ---------------------------------------------------------------------------

def _build_global_dataset(
    base_path: str,
    metadata_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Read all discharge files and return:
        X_all        : (N, 4) voltage features
        SOH_all      : (N,)   Coulomb-counting SOH (normalised per battery)
        battery_ids  : (N,)   battery_id string for each row
        filenames    : list of N filenames (in row order)

    SOH normalisation: each battery's capacity sequence is divided by *its own*
    first discharge capacity, not the global first.  This matches the
    per-battery SoH definition in your Kaggle notebook.
    """
    metadata = pd.read_csv(metadata_path)
    discharge_meta = metadata[metadata["type"].str.lower() == "discharge"].copy()
    discharge_meta["Capacity"] = pd.to_numeric(
        discharge_meta["Capacity"], errors="coerce"
    )
    discharge_meta = discharge_meta.dropna(subset=["Capacity"]).reset_index(drop=True)

    X_rows: list[list[float]] = []
    soh_rows: list[float] = []
    bat_rows: list[str] = []
    fname_rows: list[str] = []

    # Process per battery so SOH is normalised correctly
    for bat_id, grp in discharge_meta.groupby("battery_id", sort=True):
        grp = grp.sort_values("start_time").reset_index(drop=True)
        cap0 = grp["Capacity"].iloc[0]
        if cap0 == 0:
            continue
        for _, row in grp.iterrows():
            fname = str(row["filename"])
            fp = os.path.join(base_path, fname)
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            X_rows.append(extract_voltage_features(df))
            soh_rows.append(float(row["Capacity"]) / cap0)
            bat_rows.append(str(bat_id))
            fname_rows.append(fname)

    return (
        np.array(X_rows),
        np.array(soh_rows),
        np.array(bat_rows),
        fname_rows,
    )


# ---------------------------------------------------------------------------
# Partitioning strategies
# ---------------------------------------------------------------------------

def _partition_iid(
    X: np.ndarray,
    SOH: np.ndarray,
    partition_id: int,
    num_partitions: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Random shuffle then equal split — IID baseline."""
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X))
    chunks = np.array_split(idx, num_partitions)
    part_idx = chunks[partition_id % num_partitions]
    return X[part_idx], SOH[part_idx]


def _partition_by_battery(
    X: np.ndarray,
    SOH: np.ndarray,
    battery_ids: np.ndarray,
    partition_id: int,
    num_partitions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    NaturalIdPartitioner equivalent: assign whole batteries to clients.

    Batteries are sorted alphabetically and split into num_partitions groups.
    Each client receives ALL cycles from its assigned batteries → the most
    realistic non-IID scenario for this dataset.

    Example with 4 batteries and 2 clients:
        client 0 → B0047, B0048
        client 1 → B0049, B0050
    """
    unique_batteries = sorted(np.unique(battery_ids))
    if num_partitions > len(unique_batteries):
        raise ValueError(
            f"num_partitions ({num_partitions}) > number of batteries "
            f"({len(unique_batteries)}).  Reduce --num-clients."
        )
    battery_chunks = np.array_split(unique_batteries, num_partitions)
    assigned = battery_chunks[partition_id % num_partitions]
    mask = np.isin(battery_ids, assigned)
    print(
        f"  [Partition {partition_id}] by_battery → "
        f"batteries {list(assigned)}  ({mask.sum()} cycles)"
    )
    return X[mask], SOH[mask]


def _partition_dirichlet(
    X: np.ndarray,
    SOH: np.ndarray,
    battery_ids: np.ndarray,
    partition_id: int,
    num_partitions: int,
    alpha: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    DirichletPartitioner equivalent: probabilistic, heterogeneous assignment.

    For each battery, sample a Dirichlet(α) distribution over num_partitions
    clients to decide what fraction of that battery's cycles go to each client.
    Lower α → more skewed (non-IID); higher α → more uniform (approaches IID).

    Typical values:
        α = 0.1  → very non-IID (one client dominates each battery)
        α = 1.0  → moderately non-IID
        α = 100  → nearly IID
    """
    rng = np.random.default_rng(random_state)
    unique_batteries = sorted(np.unique(battery_ids))
    selected_indices: list[int] = []

    for bat in unique_batteries:
        bat_idx = np.where(battery_ids == bat)[0]
        # Sample proportion for each client from Dirichlet
        proportions = rng.dirichlet(np.full(num_partitions, alpha))
        # Convert to integer counts (at least 0 samples per client)
        counts = (proportions * len(bat_idx)).astype(int)
        # Fix rounding so counts sum to exactly len(bat_idx)
        counts[-1] = len(bat_idx) - counts[:-1].sum()
        counts = np.clip(counts, 0, None)

        # Shuffle indices then take this client's share
        shuffled = rng.permutation(bat_idx)
        boundaries = np.concatenate([[0], np.cumsum(counts)])
        start = int(boundaries[partition_id])
        end   = int(boundaries[partition_id + 1])
        selected_indices.extend(shuffled[start:end].tolist())

    idx = np.array(selected_indices)
    print(
        f"  [Partition {partition_id}] dirichlet(α={alpha}) → "
        f"{len(idx)} cycles"
    )
    return X[idx], SOH[idx]


# ---------------------------------------------------------------------------
# Public load_data entry point
# ---------------------------------------------------------------------------

def load_data(
    base_path: str,
    metadata_path: str,
    partition_id: int,
    num_partitions: int,
    test_size: float = 0.2,
    random_state: int = 42,
    partition_strategy: Literal["iid", "by_battery", "dirichlet"] = "by_battery",
    dirichlet_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and partition discharge data, returning (X_train, X_test, y_train, y_test).

    Parameters
    ----------
    base_path           : directory containing the discharge CSV files
    metadata_path       : path to metadata.csv
    partition_id        : index of this client's partition (0-based)
    num_partitions      : total number of clients / partitions
    test_size           : fraction of local data held out for testing
    random_state        : random seed for reproducibility
    partition_strategy  : one of "iid", "by_battery", "dirichlet"
    dirichlet_alpha     : α parameter for Dirichlet partitioning (ignored otherwise)

    Returns
    -------
    X_train, X_test, y_train, y_test  (NumPy arrays)
    """
    from sklearn.model_selection import train_test_split

    X_all, SOH_all, battery_ids, _ = _build_global_dataset(base_path, metadata_path)

    if partition_strategy == "iid":
        X_part, y_part = _partition_iid(
            X_all, SOH_all, partition_id, num_partitions, random_state
        )
    elif partition_strategy == "by_battery":
        X_part, y_part = _partition_by_battery(
            X_all, SOH_all, battery_ids, partition_id, num_partitions
        )
    elif partition_strategy == "dirichlet":
        X_part, y_part = _partition_dirichlet(
            X_all, SOH_all, battery_ids,
            partition_id, num_partitions,
            dirichlet_alpha, random_state,
        )
    else:
        raise ValueError(
            f"Unknown partition_strategy '{partition_strategy}'. "
            "Choose from: 'iid', 'by_battery', 'dirichlet'."
        )

    if len(X_part) < 4:
        raise ValueError(
            f"Partition {partition_id} has only {len(X_part)} samples with "
            f"strategy='{partition_strategy}'.  Reduce num_partitions or "
            f"increase dirichlet_alpha."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_part, y_part, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# RF serialisation helpers
# ---------------------------------------------------------------------------

def model_to_bytes(rf: RandomForestRegressor) -> bytes:
    """Pickle the fitted RandomForestRegressor to bytes."""
    buf = io.BytesIO()
    pickle.dump(rf, buf)
    return buf.getvalue()


def bytes_to_model(data: bytes) -> RandomForestRegressor:
    """Unpickle a RandomForestRegressor from bytes."""
    return pickle.loads(data)  # noqa: S301  (controlled internal usage)


def bytes_to_ndarray(data: bytes) -> np.ndarray:
    """Convert raw bytes to a uint8 NumPy array (for ArrayRecord transport)."""
    return np.frombuffer(data, dtype=np.uint8)


def ndarray_to_bytes(arr: np.ndarray) -> bytes:
    """Convert a uint8 NumPy array back to raw bytes."""
    return arr.tobytes()


def get_model_size_bytes(rf: RandomForestRegressor) -> int:
    """Return the pickled size of the model in bytes."""
    return len(model_to_bytes(rf))


# ---------------------------------------------------------------------------
# Aggregation (FedAvg on trees: average predictions over all client forests)
# ---------------------------------------------------------------------------

class FederatedForest:
    """
    A lightweight container that holds one RandomForestRegressor per client
    and produces predictions by averaging across all forests.  This is the
    server-side aggregated model.
    """

    def __init__(self) -> None:
        self.forests: list[RandomForestRegressor] = []

    def add(self, rf: RandomForestRegressor) -> None:
        self.forests.append(rf)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.forests:
            raise RuntimeError("No forests added yet.")
        preds = np.stack([f.predict(X) for f in self.forests], axis=0)
        return preds.mean(axis=0)

    def clear(self) -> None:
        self.forests.clear()

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> "FederatedForest":
        return pickle.loads(data)  # noqa: S301


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, dict[str, Any]]:
    """
    Fit a RandomForestRegressor and return (model, benchmark_metrics).
    benchmark_metrics contains: train_time_s, peak_memory_kb, model_size_bytes.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    train_time = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = {
        "train_time_s": train_time,
        "peak_memory_kb": peak_mem / 1024,
        "model_size_bytes": get_model_size_bytes(rf),
        "n_train_samples": len(X_train),
    }
    return rf, metrics


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    model: RandomForestRegressor | FederatedForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate model; returns dict with mae, mse, rmse, accuracy_1pct.
    accuracy_1pct = fraction of predictions within 1 % absolute error.
    """
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    acc_1pct = float(np.mean(np.abs(y_pred - y_test) < 0.01))
    return {"mae": mae, "mse": mse, "rmse": rmse, "accuracy_1pct": acc_1pct}
