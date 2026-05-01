from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sohfed.benchmarks import BenchmarkReport, RoundMetrics
from sohfed.task import evaluate, load_data, load_global_splits


PartitionStrategy = Literal["iid", "by_battery", "dirichlet"]


def model_to_bytes(model: GradientBoostingRegressor | None) -> bytes:
    if model is None:
        return b""
    return pickle.dumps(model)


def _fit_boosting_step(
    model: GradientBoostingRegressor | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    local_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    subsample: float,
    random_state: int,
) -> tuple[GradientBoostingRegressor, dict[str, float]]:
    tracemalloc.start()
    t0 = time.perf_counter()
    cpu_t0 = time.process_time()

    if model is None:
        model = GradientBoostingRegressor(
            n_estimators=local_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
            warm_start=True,
        )
    else:
        model.set_params(
            n_estimators=model.n_estimators + local_estimators,
            warm_start=True,
        )

    model.fit(X_train, y_train)

    train_time = time.perf_counter() - t0
    cpu_time = time.process_time() - cpu_t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return model, {
        "train_time_s": train_time,
        "cpu_time_s": cpu_time,
        "peak_memory_kb": peak_mem / 1024,
    }


def plot_boosted_rounds(report: BenchmarkReport, output_path: str) -> None:
    if not report.rounds:
        return

    rounds = [r.round_num for r in report.rounds]
    mae = [r.global_mae for r in report.rounds]
    rmse = [r.global_rmse for r in report.rounds]
    r2 = [r.global_r2 for r in report.rounds]
    acc = [r.global_accuracy_1pct for r in report.rounds]
    mb = [
        (r.bytes_received_from_clients + r.bytes_sent_to_clients) / 1_048_576
        for r in report.rounds
    ]
    wall = [r.round_wall_time_s for r in report.rounds]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("Cyclic Federated Boosted Trees", fontsize=16, fontweight="bold")
    plots = [
        (axes[0, 0], mae, "Global MAE"),
        (axes[0, 1], rmse, "Global RMSE"),
        (axes[0, 2], r2, "Global R2"),
        (axes[1, 0], acc, "Accuracy (<1% abs err)"),
        (axes[1, 1], mb, "Communication / Round (MB)"),
        (axes[1, 2], wall, "Round Wall Time (s)"),
    ]
    for ax, vals, title in plots:
        ax.plot(rounds, vals, marker="o", linewidth=2, color="#0f6cbd")
        ax.set_title(title)
        ax.set_xticks(rounds)
        ax.grid(True, linestyle=":", alpha=0.35)

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_boosted_benchmark(
    data_path: str,
    metadata_path: str,
    num_clients: int,
    num_rounds: int,
    local_estimators: int,
    output_dir: str,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    min_samples_leaf: int = 2,
    subsample: float = 0.9,
    global_test_size: float = 0.2,
    partition_strategy: PartitionStrategy = "by_battery",
    dirichlet_alpha: float = 0.5,
    random_state: int = 42,
) -> BenchmarkReport:
    os.makedirs(output_dir, exist_ok=True)

    local_splits = [
        load_data(
            base_path=data_path,
            metadata_path=metadata_path,
            partition_id=partition_id,
            num_partitions=num_clients,
            global_test_size=global_test_size,
            partition_strategy=partition_strategy,
            dirichlet_alpha=dirichlet_alpha,
            random_state=random_state,
        )
        for partition_id in range(num_clients)
    ]
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

    report = BenchmarkReport(mode="federated_boosted_cyclic")
    model: GradientBoostingRegressor | None = None
    total_t0 = time.perf_counter()

    for server_round in range(1, num_rounds + 1):
        round_t0 = time.perf_counter()
        client_id = (server_round - 1) % num_clients
        X_train, X_test, y_train, y_test = local_splits[client_id]
        download_bytes = len(model_to_bytes(model))

        model, bench = _fit_boosting_step(
            model=model,
            X_train=X_train,
            y_train=y_train,
            local_estimators=local_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state + server_round,
        )
        upload_bytes = len(model_to_bytes(model))
        global_metrics = evaluate(model, X_test_global, y_test_global)
        local_metrics = evaluate(model, X_test, y_test)

        rm = RoundMetrics(
            round_num=server_round,
            bytes_sent_to_clients=download_bytes,
            bytes_received_from_clients=upload_bytes,
            num_clients_trained=1,
            num_clients_evaluated=1,
            round_wall_time_s=time.perf_counter() - round_t0,
            avg_client_train_time_s=bench["train_time_s"],
            avg_client_cpu_time_s=bench["cpu_time_s"],
            avg_client_peak_memory_kb=bench["peak_memory_kb"],
            avg_train_loss=local_metrics["mae"],
            avg_eval_mae=local_metrics["mae"],
            avg_eval_rmse=local_metrics["rmse"],
            avg_eval_accuracy_1pct=local_metrics["accuracy_1pct"],
            global_mae=global_metrics["mae"],
            global_rmse=global_metrics["rmse"],
            global_r2=global_metrics["r2"],
            global_accuracy_1pct=global_metrics["accuracy_1pct"],
        )
        report.add_round(rm)

        print(
            f"[Boosted] round={server_round} client={client_id} "
            f"trees={model.n_estimators} global_mae={rm.global_mae:.5f} "
            f"global_r2={rm.global_r2:.5f} tx={(download_bytes + upload_bytes) / 1024:.1f} KB"
        )

    report.total_wall_time_s = time.perf_counter() - total_t0
    report_path = os.path.join(output_dir, "benchmark_report.json")
    report.save(report_path)
    plot_boosted_rounds(report, os.path.join(output_dir, "boosted_rounds.png"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cyclic federated boosted-tree SOH benchmark."
    )
    parser.add_argument("--data-path", default="../../cleaned_dataset/data")
    parser.add_argument("--metadata-path", default="../../cleaned_dataset/metadata.csv")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=25)
    parser.add_argument("--local-estimators", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--global-test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--partition-strategy",
        default="by_battery",
        choices=["iid", "by_battery", "dirichlet"],
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results_boosted")
    args = parser.parse_args()

    run_boosted_benchmark(
        data_path=args.data_path,
        metadata_path=args.metadata_path,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_estimators=args.local_estimators,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        global_test_size=args.global_test_size,
        partition_strategy=args.partition_strategy,
        dirichlet_alpha=args.dirichlet_alpha,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
