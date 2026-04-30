from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise SystemExit(
        "xgboost is required for this benchmark. Install dependencies with "
        "`pip install -r requirements.txt` or `pip install xgboost>=2.0`."
    ) from exc

import flwr as fl
from flwr.client import Client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Scalar,
    Status,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedXgbCyclic

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sohfed.benchmarks import BenchmarkReport, RoundMetrics
from sohfed.task import evaluate, load_data, load_global_splits


PartitionStrategy = Literal["iid", "by_battery", "dirichlet"]


def _ok() -> Status:
    return Status(code=Code.OK, message="OK")


def _model_to_parameters(model_bytes: bytes | None) -> Parameters:
    return Parameters(tensor_type="", tensors=[] if not model_bytes else [model_bytes])


def _parameters_to_model_bytes(parameters: Parameters) -> bytes | None:
    if not parameters.tensors:
        return None
    return bytes(parameters.tensors[0])


def _booster_from_bytes(model_bytes: bytes, params: dict[str, Any]) -> xgb.Booster:
    booster = xgb.Booster(params=params)
    booster.load_model(bytearray(model_bytes))
    return booster


def _predict(model_bytes: bytes, params: dict[str, Any], X: np.ndarray) -> np.ndarray:
    booster = _booster_from_bytes(model_bytes, params)
    return booster.predict(xgb.DMatrix(X.astype(np.float32)))


def _eval_model(
    model_bytes: bytes,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    class _XgbModel:
        def predict(self, X_eval: np.ndarray) -> np.ndarray:
            return _predict(model_bytes, params, X_eval)

    return evaluate(_XgbModel(), X, y)


class XgbSOHClient(Client):
    def __init__(
        self,
        cid: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        params: dict[str, Any],
        local_rounds: int,
    ) -> None:
        self.cid = cid
        self.X_train = X_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_train = y_train.astype(np.float32)
        self.y_test = y_test.astype(np.float32)
        self.params = params
        self.local_rounds = local_rounds

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(status=_ok(), parameters=_model_to_parameters(None))

    def fit(self, ins: FitIns) -> FitRes:
        server_round = int(ins.config.get("server_round", 1))
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        incoming = _parameters_to_model_bytes(ins.parameters)

        tracemalloc.start()
        t0 = time.perf_counter()
        cpu_t0 = time.process_time()
        if incoming is None:
            booster = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.local_rounds,
            )
        else:
            booster = _booster_from_bytes(incoming, self.params)
            booster = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.local_rounds,
                xgb_model=booster,
            )
        train_time = time.perf_counter() - t0
        cpu_time = time.process_time() - cpu_t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        model_bytes = bytes(booster.save_raw("json"))
        local_metrics = _eval_model(model_bytes, self.params, self.X_test, self.y_test)
        print(
            f"[XGBoost client {self.cid}] round={server_round} "
            f"trees={booster.num_boosted_rounds()} local_mae={local_metrics['mae']:.5f} "
            f"payload={len(model_bytes) / 1024:.1f} KB"
        )
        return FitRes(
            status=_ok(),
            parameters=_model_to_parameters(model_bytes),
            num_examples=len(self.X_train),
            metrics={
                "train_time_s": train_time,
                "cpu_time_s": cpu_time,
                "peak_memory_kb": peak_mem / 1024,
                "local_mae": local_metrics["mae"],
                "local_rmse": local_metrics["rmse"],
                "local_r2": local_metrics["r2"],
                "local_accuracy_1pct": local_metrics["accuracy_1pct"],
                "server_round": float(server_round),
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        model_bytes = _parameters_to_model_bytes(ins.parameters)
        if model_bytes is None:
            return EvaluateRes(status=_ok(), loss=0.0, num_examples=len(self.X_test), metrics={})

        metrics = _eval_model(model_bytes, self.params, self.X_test, self.y_test)
        return EvaluateRes(
            status=_ok(),
            loss=metrics["mae"],
            num_examples=len(self.X_test),
            metrics=metrics,
        )


class BenchmarkFedXgbCyclic(FedXgbCyclic):
    def __init__(
        self,
        params: dict[str, Any],
        X_test_global: np.ndarray,
        y_test_global: np.ndarray,
        benchmark_output: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.params = params
        self.X_test_global = X_test_global.astype(np.float32)
        self.y_test_global = y_test_global.astype(np.float32)
        self.benchmark_output = benchmark_output
        self.report = BenchmarkReport(mode="federated_xgboost_cyclic")
        self._total_start = time.perf_counter()
        self._round_start = time.perf_counter()
        self._previous_model_bytes = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        previous_model_bytes = self._previous_model_bytes
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is None or not parameters.tensors:
            return parameters, metrics

        model_bytes = bytes(parameters.tensors[0])
        self._previous_model_bytes = len(model_bytes)
        global_metrics = _eval_model(
            model_bytes,
            self.params,
            self.X_test_global,
            self.y_test_global,
        )

        train_times = [float(fit_res.metrics.get("train_time_s", 0.0)) for _, fit_res in results]
        cpu_times = [float(fit_res.metrics.get("cpu_time_s", 0.0)) for _, fit_res in results]
        peak_mems = [float(fit_res.metrics.get("peak_memory_kb", 0.0)) for _, fit_res in results]
        local_maes = [float(fit_res.metrics.get("local_mae", 0.0)) for _, fit_res in results]

        rm = RoundMetrics(
            round_num=server_round,
            bytes_sent_to_clients=previous_model_bytes * len(results),
            bytes_received_from_clients=sum(len(fit_res.parameters.tensors[0]) for _, fit_res in results),
            num_clients_trained=len(results),
            round_wall_time_s=time.perf_counter() - self._round_start,
            avg_client_train_time_s=float(np.mean(train_times)) if train_times else 0.0,
            avg_client_cpu_time_s=float(np.mean(cpu_times)) if cpu_times else 0.0,
            avg_client_peak_memory_kb=float(np.mean(peak_mems)) if peak_mems else 0.0,
            avg_train_loss=float(np.mean(local_maes)) if local_maes else 0.0,
            global_mae=global_metrics["mae"],
            global_rmse=global_metrics["rmse"],
            global_r2=global_metrics["r2"],
            global_accuracy_1pct=global_metrics["accuracy_1pct"],
        )
        self.report.add_round(rm)
        self.report.total_wall_time_s = time.perf_counter() - self._total_start
        self.report.save(self.benchmark_output)
        self._round_start = time.perf_counter()

        print(
            f"[XGBoost server] round={server_round} global_mae={rm.global_mae:.5f} "
            f"global_r2={rm.global_r2:.5f} model={len(model_bytes) / 1024:.1f} KB"
        )

        metrics.update(
            {
                "global_mae": rm.global_mae,
                "global_rmse": rm.global_rmse,
                "global_r2": rm.global_r2,
                "accuracy_1pct": rm.global_accuracy_1pct,
            }
        )
        return parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Any],
    ) -> tuple[float | None, dict[str, Scalar]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if self.report.rounds and results:
            maes = [float(res.metrics.get("mae", 0.0)) for _, res in results]
            rmses = [float(res.metrics.get("rmse", 0.0)) for _, res in results]
            accs = [float(res.metrics.get("accuracy_1pct", 0.0)) for _, res in results]
            self.report.rounds[-1].avg_eval_mae = float(np.mean(maes))
            self.report.rounds[-1].avg_eval_rmse = float(np.mean(rmses))
            self.report.rounds[-1].avg_eval_accuracy_1pct = float(np.mean(accs))
            self.report.rounds[-1].num_clients_evaluated = len(results)
            self.report.total_wall_time_s = time.perf_counter() - self._total_start
            self.report.save(self.benchmark_output)
        return loss, metrics


def on_fit_config_fn(server_round: int) -> dict[str, Scalar]:
    return {"server_round": server_round}


def _client_fn_builder(
    data_path: str,
    metadata_path: str,
    num_clients: int,
    params: dict[str, Any],
    local_rounds: int,
    global_test_size: float,
    partition_strategy: PartitionStrategy,
    dirichlet_alpha: float,
    random_state: int,
):
    def client_fn(context) -> Client:
        cid = str(int(context.node_config.get("partition-id", 0)))
        X_train, X_test, y_train, y_test = load_data(
            base_path=data_path,
            metadata_path=metadata_path,
            partition_id=int(cid),
            num_partitions=num_clients,
            global_test_size=global_test_size,
            partition_strategy=partition_strategy,
            dirichlet_alpha=dirichlet_alpha,
            random_state=random_state,
        )
        return XgbSOHClient(
            cid=cid,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=params,
            local_rounds=local_rounds,
        )

    return client_fn


def plot_xgboost_rounds(report: BenchmarkReport, output_path: str) -> None:
    if not report.rounds:
        return
    rounds = [r.round_num for r in report.rounds]
    mae = [r.global_mae for r in report.rounds]
    rmse = [r.global_rmse for r in report.rounds]
    r2 = [r.global_r2 for r in report.rounds]
    total_mb = [
        (r.bytes_sent_to_clients + r.bytes_received_from_clients) / 1_048_576
        for r in report.rounds
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, vals, title in [
        (axes[0, 0], mae, "Global MAE"),
        (axes[0, 1], rmse, "Global RMSE"),
        (axes[1, 0], r2, "Global R2"),
        (axes[1, 1], total_mb, "Communication / Round (MB)"),
    ]:
        ax.plot(rounds, vals, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xticks(rounds)
        ax.grid(True, linestyle=":", alpha=0.35)
    fig.suptitle("Federated XGBoost Cyclic SOH Benchmark", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_xgboost_benchmark(
    data_path: str,
    metadata_path: str,
    num_clients: int,
    num_rounds: int,
    local_rounds: int,
    output_dir: str,
    eta: float = 0.05,
    max_depth: int = 3,
    subsample: float = 0.9,
    global_test_size: float = 0.2,
    partition_strategy: PartitionStrategy = "by_battery",
    dirichlet_alpha: float = 0.5,
    random_state: int = 42,
) -> BenchmarkReport:
    os.makedirs(output_dir, exist_ok=True)
    params: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "seed": random_state,
        "nthread": 1,
    }

    (
        _X_train_pool,
        X_test_global,
        _y_train_pool,
        y_test_global,
        *_,
    ) = load_global_splits(
        base_path=data_path,
        metadata_path=metadata_path,
        global_test_size=global_test_size,
        random_state=random_state,
    )

    benchmark_path = os.path.join(output_dir, "benchmark_report.json")
    strategy = BenchmarkFedXgbCyclic(
        params=params,
        X_test_global=X_test_global,
        y_test_global=y_test_global,
        benchmark_output=benchmark_path,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=num_clients,
        on_fit_config_fn=on_fit_config_fn,
    )

    fl.simulation.start_simulation(
        client_fn=_client_fn_builder(
            data_path=data_path,
            metadata_path=metadata_path,
            num_clients=num_clients,
            params=params,
            local_rounds=local_rounds,
            global_test_size=global_test_size,
            partition_strategy=partition_strategy,
            dirichlet_alpha=dirichlet_alpha,
            random_state=random_state,
        ),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    strategy.report.total_wall_time_s = time.perf_counter() - strategy._total_start
    strategy.report.save(benchmark_path)
    plot_xgboost_rounds(strategy.report, os.path.join(output_dir, "xgboost_rounds.png"))
    return strategy.report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run federated XGBoost SOH benchmark.")
    parser.add_argument("--data-path", default="../../cleaned_dataset/data")
    parser.add_argument("--metadata-path", default="../../cleaned_dataset/metadata.csv")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=25)
    parser.add_argument("--local-rounds", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--global-test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--partition-strategy", default="by_battery", choices=["iid", "by_battery", "dirichlet"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results_xgboost")
    args = parser.parse_args()

    run_xgboost_benchmark(
        data_path=args.data_path,
        metadata_path=args.metadata_path,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_rounds=args.local_rounds,
        output_dir=args.output_dir,
        eta=args.eta,
        max_depth=args.max_depth,
        subsample=args.subsample,
        global_test_size=args.global_test_size,
        partition_strategy=args.partition_strategy,
        dirichlet_alpha=args.dirichlet_alpha,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
