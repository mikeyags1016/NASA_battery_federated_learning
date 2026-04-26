"""
benchmarks.py
-------------
Utilities for tracking and reporting federated learning benchmarks:
  - Data transmission size (bytes per round per client)
  - Communication rounds
  - Training time (per client, per round)
  - Hardware usage (CPU / peak RAM via tracemalloc)
  - Accuracy (MAE, RMSE) and loss per round
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RoundMetrics:
    """Metrics collected for a single federated round."""
    round_num: int = 0

    # Communication
    bytes_sent_to_clients: int = 0       # server → clients (model payload)
    bytes_received_from_clients: int = 0 # clients → server (updated models)
    num_clients_trained: int = 0
    num_clients_evaluated: int = 0

    # Timing (seconds)
    round_wall_time_s: float = 0.0
    avg_client_train_time_s: float = 0.0
    avg_client_cpu_time_s: float = 0.0

    # Hardware (KB)
    avg_client_peak_memory_kb: float = 0.0

    # Accuracy / loss (aggregated across clients)
    avg_train_loss: float = 0.0          # MAE on train set
    avg_eval_mae: float = 0.0
    avg_eval_rmse: float = 0.0
    avg_eval_accuracy_1pct: float = 0.0

    # Server-side global evaluation (if available)
    global_mae: float = 0.0
    global_rmse: float = 0.0
    global_r2: float = 0.0
    global_accuracy_1pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Aggregates RoundMetrics across all rounds."""
    mode: str = "federated"
    rounds: list[RoundMetrics] = field(default_factory=list)
    total_wall_time_s: float = 0.0
    total_bytes_uploaded: int = 0
    total_bytes_downloaded: int = 0

    def add_round(self, rm: RoundMetrics) -> None:
        self.rounds.append(rm)
        self.total_bytes_uploaded += rm.bytes_received_from_clients
        self.total_bytes_downloaded += rm.bytes_sent_to_clients

    def summary(self) -> dict[str, Any]:
        if not self.rounds:
            return {}
        final = self.rounds[-1]
        total_bytes_transmitted = self.total_bytes_uploaded + self.total_bytes_downloaded
        return {
            "mode": self.mode,
            "num_rounds": len(self.rounds),
            "total_wall_time_s": round(self.total_wall_time_s, 3),
            "total_bytes_uploaded_MB": round(self.total_bytes_uploaded / 1_048_576, 4),
            "total_bytes_downloaded_MB": round(self.total_bytes_downloaded / 1_048_576, 4),
            "total_bytes_transmitted_MB": round(
                total_bytes_transmitted / 1_048_576, 4
            ),
            "final_global_mae": round(final.global_mae, 6),
            "final_global_rmse": round(final.global_rmse, 6),
            "final_global_r2": round(final.global_r2, 6),
            "final_global_accuracy_1pct": round(final.global_accuracy_1pct, 4),
            "avg_round_time_s": round(
                sum(r.round_wall_time_s for r in self.rounds) / len(self.rounds), 3
            ),
            "avg_client_cpu_time_s": round(
                sum(r.avg_client_cpu_time_s for r in self.rounds) / len(self.rounds), 3
            ),
            "avg_client_peak_memory_kb": round(
                sum(r.avg_client_peak_memory_kb for r in self.rounds) / len(self.rounds), 2
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        data = {
            "summary": self.summary(),
            "rounds": [r.to_dict() for r in self.rounds],
        }
        return json.dumps(data, indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())
        print(f"[Benchmarks] Report saved -> {path}")


class Timer:
    """Simple context-manager wall-clock timer."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
