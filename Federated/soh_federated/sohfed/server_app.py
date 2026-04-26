"""
server_app.py
-------------
Flower ServerApp for federated SOH estimation.

Strategy: custom FedAvg variant that:
  - Aggregates client RandomForestRegressors into a FederatedForest
    (prediction-averaging ensemble).
  - Tracks all benchmark metrics per round:
      * Data transmission (bytes sent / received)
      * Communication round count
      * Per-client training time & peak memory
      * Global accuracy (MAE, RMSE) evaluated on a held-out test set

A BenchmarkReport JSON file is written to disk after training completes.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

import flwr as fl
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Context

from sohfed.task import (
    FederatedForest,
    evaluate,
    load_global_splits,
    bytes_to_model,
    ndarray_to_bytes,
    bytes_to_ndarray,
)
from sohfed.benchmarks import BenchmarkReport, RoundMetrics, Timer


# ---------------------------------------------------------------------------
# Custom strategy
# ---------------------------------------------------------------------------

class FederatedForestStrategy(FedAvg):
    """
    Extends FedAvg to:
      1. Aggregate RandomForestRegressors by building a FederatedForest.
      2. Evaluate the aggregated FederatedForest on a server-side test set.
      3. Collect and report detailed benchmark metrics.
    """

    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        num_partitions: int,
        global_test_size: float = 0.2,
        benchmark_output: str = "benchmark_report.json",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.num_partitions = num_partitions
        self.benchmark_output = benchmark_output
        self.report = BenchmarkReport(mode="federated")
        self._total_timer = Timer()
        self._round_start: float = 0.0
        self._current_round_metrics = RoundMetrics()
        self.global_forest = FederatedForest()

        (
            _X_train_pool,
            self.X_test_global,
            _y_train_pool,
            self.y_test_global,
            _train_battery_ids,
            _test_battery_ids,
            _train_filenames,
            _test_filenames,
        ) = load_global_splits(
            base_path=self.base_path,
            metadata_path=self.metadata_path,
            global_test_size=global_test_size,
        )
        print(
            f"[Server] Global eval set: {len(self.X_test_global)} samples"
        )

    # ------------------------------------------------------------------
    # Aggregate fit results → FederatedForest
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}

        rm = self._current_round_metrics
        rm.round_num = server_round
        rm.num_clients_trained = len(results)

        # ---- Reconstruct FederatedForest from client payloads ----
        round_forest = FederatedForest()
        total_bytes_received = 0

        train_times: list[float] = []
        cpu_times: list[float] = []
        peak_mems: list[float] = []
        local_maes: list[float] = []

        for _proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            if len(ndarrays) == 0 or ndarrays[0].size == 0:
                continue

            model_bytes = ndarray_to_bytes(ndarrays[0])
            total_bytes_received += len(model_bytes)

            rf = bytes_to_model(model_bytes)

            m = fit_res.metrics or {}
            train_weight = float(m.get("n_train_samples", fit_res.num_examples))
            round_forest.add(rf, weight=train_weight)
            train_times.append(float(m.get("train_time_s", 0)))
            cpu_times.append(float(m.get("cpu_time_s", 0)))
            peak_mems.append(float(m.get("peak_memory_kb", 0)))
            local_maes.append(float(m.get("local_mae", 0)))

        self.global_forest.extend(round_forest)

        rm.bytes_received_from_clients = total_bytes_received
        rm.avg_client_train_time_s = float(np.mean(train_times)) if train_times else 0.0
        rm.avg_client_cpu_time_s = float(np.mean(cpu_times)) if cpu_times else 0.0
        rm.avg_client_peak_memory_kb = float(np.mean(peak_mems)) if peak_mems else 0.0
        rm.avg_train_loss = float(np.mean(local_maes)) if local_maes else 0.0

        # ---- Server-side global evaluation ----
        if self.global_forest.forests:
            global_metrics = evaluate(self.global_forest, self.X_test_global, self.y_test_global)
            rm.global_mae = global_metrics["mae"]
            rm.global_rmse = global_metrics["rmse"]
            rm.global_r2 = global_metrics["r2"]
            rm.global_accuracy_1pct = global_metrics["accuracy_1pct"]

        # ---- Serialise aggregated forest as parameters ----
        agg_bytes = self.global_forest.to_bytes()
        agg_array = bytes_to_ndarray(agg_bytes)
        rm.bytes_sent_to_clients = len(agg_bytes) * len(results)

        # Wall time for round
        rm.round_wall_time_s = time.perf_counter() - self._round_start

        print(
            f"\n[Server] Round {server_round} complete  "
            f"clients={len(results)}  "
            f"global_mae={rm.global_mae:.4f}  "
            f"global_rmse={rm.global_rmse:.4f}  "
            f"acc_1pct={rm.global_accuracy_1pct:.3f}  "
            f"rx={total_bytes_received/1024:.1f} KB  "
            f"round_time={rm.round_wall_time_s:.2f}s"
        )

        self.report.add_round(rm)
        self._current_round_metrics = RoundMetrics()  # reset for next round
        self._round_start = time.perf_counter()        # start next round timer
        self.report.total_wall_time_s = time.perf_counter() - self._total_start
        self.report.save(self.benchmark_output)

        agg_metrics: dict[str, Scalar] = {
            "global_mae": rm.global_mae,
            "global_rmse": rm.global_rmse,
            "global_r2": rm.global_r2,
            "accuracy_1pct": rm.global_accuracy_1pct,
            "round_wall_time_s": rm.round_wall_time_s,
        }
        return ndarrays_to_parameters([agg_array]), agg_metrics

    # ------------------------------------------------------------------
    # Aggregate evaluate results
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Any],
    ) -> tuple[float | None, dict[str, Scalar]]:
        if not results:
            return None, {}

        # Weighted average of client MAEs
        total_examples = sum(r.num_examples for _, r in results)
        weighted_mae = sum(
            r.loss * r.num_examples for _, r in results
        ) / max(total_examples, 1)

        maes = [r.metrics.get("mae", 0.0) for _, r in results]
        rmses = [r.metrics.get("rmse", 0.0) for _, r in results]
        r2s = [r.metrics.get("r2", 0.0) for _, r in results]
        accs = [r.metrics.get("accuracy_1pct", 0.0) for _, r in results]

        # Patch the current (already-stored) round's eval metrics
        if self.report.rounds:
            self.report.rounds[-1].avg_eval_mae = float(np.mean(maes))
            self.report.rounds[-1].avg_eval_rmse = float(np.mean(rmses))
            self.report.rounds[-1].avg_eval_accuracy_1pct = float(np.mean(accs))
            self.report.rounds[-1].num_clients_evaluated = len(results)
            self.report.total_wall_time_s = time.perf_counter() - self._total_start
            self.report.save(self.benchmark_output)

        return weighted_mae, {
            "avg_mae": float(np.mean(maes)),
            "avg_rmse": float(np.mean(rmses)),
            "avg_r2": float(np.mean(r2s)),
            "avg_accuracy_1pct": float(np.mean(accs)),
        }

    def initialize_parameters(self, client_manager: Any) -> Parameters | None:
        """Provide empty initial parameters (clients train from scratch)."""
        self._total_start = time.perf_counter()
        self._round_start = time.perf_counter()
        return ndarrays_to_parameters([np.array([], dtype=np.uint8)])


# ---------------------------------------------------------------------------
# ServerApp factory
# ---------------------------------------------------------------------------

def server_fn(context: Context) -> fl.server.Server:
    cfg = context.run_config

    strategy = FederatedForestStrategy(
        base_path=cfg.get("data-base-path", "data"),
        metadata_path=cfg.get("metadata-path", "metadata.csv"),
        num_partitions=int(cfg.get("num-supernodes", 5)),
        benchmark_output=cfg.get("benchmark-output", "benchmark_report.json"),
        global_test_size=float(cfg.get("global-test-size", 0.2)),
        min_fit_clients=int(cfg.get("min-fit-clients", 2)),
        min_evaluate_clients=int(cfg.get("min-evaluate-clients", 2)),
        min_available_clients=int(cfg.get("min-available-clients", 2)),
        fraction_fit=float(cfg.get("fraction-fit", 1.0)),
        fraction_evaluate=float(cfg.get("fraction-evaluate", 1.0)),
    )

    num_rounds = int(cfg.get("num-server-rounds", 3))
    server_config = ServerConfig(num_rounds=num_rounds)

    return fl.server.Server(
        client_manager=fl.server.SimpleClientManager(),
        strategy=strategy,
    )


def on_fit_config_fn(server_round: int) -> dict[str, Scalar]:
    return {"server_round": server_round}


app = ServerApp(server_fn=server_fn)
