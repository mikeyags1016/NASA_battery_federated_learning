"""
simulate.py
-----------
Standalone simulation runner for the SOH federated learning system.

Usage:
    python simulate.py \
        --data-path     /path/to/cleaned_dataset/data \
        --metadata-path /path/to/cleaned_dataset/metadata.csv \
        --num-clients   5 \
        --num-rounds    3 \
        --n-estimators  100 \
        --output-dir    ./results

This script does NOT require a running Flower SuperLink.  It uses
flwr.simulation.start_simulation() to run everything in a single process,
which is ideal for research and benchmarking.

All benchmark metrics are printed to stdout and saved as JSON + a matplotlib
figure.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import flwr as fl
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from sohfed.task import (
    FederatedForest,
    evaluate,
    load_data,
    train,
    bytes_to_model,
    bytes_to_ndarray,
    ndarray_to_bytes,
    model_to_bytes,
)
from sohfed.benchmarks import BenchmarkReport, RoundMetrics


# ---------------------------------------------------------------------------
# Inline client function (used by flwr.simulation)
# ---------------------------------------------------------------------------

def build_client_fn(
    base_path: str,
    metadata_path: str,
    num_partitions: int,
    n_estimators: int,
    partition_strategy: str = "by_battery",
    dirichlet_alpha: float = 0.5,
):
    """Return a closure compatible with flwr.simulation.start_simulation."""

    def client_fn(cid: str) -> fl.client.NumPyClient:
        partition_id = int(cid)

        (X_train, X_test, y_train, y_test) = load_data(
            base_path=base_path,
            metadata_path=metadata_path,
            partition_id=partition_id,
            num_partitions=num_partitions,
            partition_strategy=partition_strategy,
            dirichlet_alpha=dirichlet_alpha,
        )

        class _Client(fl.client.NumPyClient):
            def get_parameters(self, config):
                return [np.array([], dtype=np.uint8)]

            def fit(self, parameters, config):
                rf, bench = train(X_train, y_train, n_estimators=n_estimators)
                eval_m = evaluate(rf, X_test, y_test)
                model_bytes = model_to_bytes(rf)
                metrics = {
                    "train_time_s":       bench["train_time_s"],
                    "peak_memory_kb":     bench["peak_memory_kb"],
                    "model_size_bytes":   float(len(model_bytes)),
                    "local_mae":          eval_m["mae"],
                    "local_rmse":         eval_m["rmse"],
                    "local_accuracy_1pct": eval_m["accuracy_1pct"],
                    "num_examples":       float(len(X_train)),
                }
                print(
                    f"  [client {cid}] train_time={bench['train_time_s']:.3f}s  "
                    f"local_mae={eval_m['mae']:.4f}  "
                    f"payload={len(model_bytes)/1024:.1f} KB"
                )
                return [bytes_to_ndarray(model_bytes)], len(X_train), metrics

            def evaluate(self, parameters, config):
                # Reconstruct the model sent from server (FederatedForest)
                ndarrays = parameters
                if ndarrays[0].size == 0:
                    return 0.0, len(X_test), {"mae": 0.0}
                raw = ndarray_to_bytes(ndarrays[0])
                try:
                    fed_forest = FederatedForest.from_bytes(raw)
                    eval_m = evaluate(fed_forest, X_test, y_test)
                except Exception:
                    # Fallback: treat as single RF
                    try:
                        rf = bytes_to_model(raw)
                        eval_m = evaluate(rf, X_test, y_test)
                    except Exception:
                        return 0.0, len(X_test), {"mae": 0.0}
                print(
                    f"  [client {cid}] eval  mae={eval_m['mae']:.4f}  "
                    f"rmse={eval_m['rmse']:.4f}"
                )
                return float(eval_m["mae"]), len(X_test), {
                    "mae": eval_m["mae"],
                    "rmse": eval_m["rmse"],
                    "accuracy_1pct": eval_m["accuracy_1pct"],
                }

        return _Client()

    return client_fn


# ---------------------------------------------------------------------------
# Custom strategy with benchmark collection
# ---------------------------------------------------------------------------

class BenchmarkStrategy(FedAvg):
    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        num_partitions: int,
        benchmark_output: str = None,  # ← ADD THIS LINE
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.report = BenchmarkReport()
        self.benchmark_output = benchmark_output  # ← ADD THIS LINE (optional)
        self._round_start = time.perf_counter()
        self._total_start = time.perf_counter()

        # Server-side global test set
        _, self.X_test_g, _, self.y_test_g = load_data(
            base_path=base_path,
            metadata_path=metadata_path,
            partition_id=0,
            num_partitions=num_partitions,
            test_size=0.3,
        )
        print(f"[Server] Global eval set: {len(self.X_test_g)} samples")

    def initialize_parameters(self, client_manager):
        self._round_start = time.perf_counter()
        self._total_start = time.perf_counter()
        return ndarrays_to_parameters([np.array([], dtype=np.uint8)])

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        rm = RoundMetrics(round_num=server_round)
        rm.num_clients_trained = len(results)

        fed_forest = FederatedForest()
        total_rx = 0
        train_times, peak_mems, local_maes = [], [], []

        for _proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            if not ndarrays or ndarrays[0].size == 0:
                continue
            raw = ndarray_to_bytes(ndarrays[0])
            total_rx += len(raw)
            rf = bytes_to_model(raw)
            fed_forest.add(rf)
            m = fit_res.metrics or {}
            train_times.append(float(m.get("train_time_s", 0)))
            peak_mems.append(float(m.get("peak_memory_kb", 0)))
            local_maes.append(float(m.get("local_mae", 0)))

        rm.bytes_received_from_clients = total_rx
        rm.avg_client_train_time_s = float(np.mean(train_times)) if train_times else 0
        rm.avg_client_peak_memory_kb = float(np.mean(peak_mems)) if peak_mems else 0
        rm.avg_train_loss = float(np.mean(local_maes)) if local_maes else 0

        # Global eval
        if fed_forest.forests:
            gm = evaluate(fed_forest, self.X_test_g, self.y_test_g)
            rm.global_mae  = gm["mae"]
            rm.global_rmse = gm["rmse"]
            rm.global_accuracy_1pct = gm["accuracy_1pct"]

        # Serialise aggregated forest
        agg_bytes = fed_forest.to_bytes()
        agg_arr   = bytes_to_ndarray(agg_bytes)
        rm.bytes_sent_to_clients = len(agg_bytes)

        rm.round_wall_time_s = time.perf_counter() - self._round_start
        self._round_start    = time.perf_counter()

        self.report.add_round(rm)

        total_tx = rm.bytes_sent_to_clients + rm.bytes_received_from_clients
        print(
            f"\n[Server] ─── Round {server_round} ───\n"
            f"  clients trained   : {rm.num_clients_trained}\n"
            f"  global MAE        : {rm.global_mae:.5f}\n"
            f"  global RMSE       : {rm.global_rmse:.5f}\n"
            f"  accuracy (<1% err): {rm.global_accuracy_1pct:.3f}\n"
            f"  data transmitted  : {total_tx/1024:.1f} KB\n"
            f"  avg train time    : {rm.avg_client_train_time_s:.3f}s\n"
            f"  avg peak memory   : {rm.avg_client_peak_memory_kb:.1f} KB\n"
            f"  round wall time   : {rm.round_wall_time_s:.2f}s\n"
        )

        return ndarrays_to_parameters([agg_arr]), {
            "global_mae":  rm.global_mae,
            "global_rmse": rm.global_rmse,
        }

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        total_ex = sum(r.num_examples for _, r in results)
        w_mae = sum(r.loss * r.num_examples for _, r in results) / max(total_ex, 1)
        maes  = [r.metrics.get("mae",  0.0) for _, r in results]
        rmses = [r.metrics.get("rmse", 0.0) for _, r in results]
        if self.report.rounds:
            self.report.rounds[-1].avg_eval_mae  = float(np.mean(maes))
            self.report.rounds[-1].avg_eval_rmse = float(np.mean(rmses))
            self.report.rounds[-1].num_clients_evaluated = len(results)
        return w_mae, {"avg_mae": float(np.mean(maes))}

    def finalize(self) -> None:
        self.report.total_wall_time_s = time.perf_counter() - self._total_start


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_benchmarks(report: BenchmarkReport, output_dir: str) -> None:
    rounds = [r.round_num for r in report.rounds]
    mae    = [r.global_mae   for r in report.rounds]
    rmse   = [r.global_rmse  for r in report.rounds]
    acc    = [r.global_accuracy_1pct for r in report.rounds]
    tx_kb  = [
        (r.bytes_sent_to_clients + r.bytes_received_from_clients) / 1024
        for r in report.rounds
    ]
    train_t = [r.avg_client_train_time_s   for r in report.rounds]
    mem_kb  = [r.avg_client_peak_memory_kb for r in report.rounds]
    wall_t  = [r.round_wall_time_s         for r in report.rounds]

    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    fig.suptitle(
        "Federated SOH Estimation — Benchmark Dashboard",
        fontsize=16, color="white", fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT = "#00e5ff"
    RED    = "#ff4d6d"
    GREEN  = "#00ff9d"
    YELLOW = "#ffd166"
    AX_BG  = "#161b22"

    def style_ax(ax, title):
        ax.set_facecolor(AX_BG)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.title.set_color("white")

    # 1. MAE
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rounds, mae, marker="o", color=ACCENT, linewidth=2)
    ax1.fill_between(rounds, mae, alpha=0.15, color=ACCENT)
    style_ax(ax1, "Global MAE per Round")
    ax1.set_ylabel("MAE", color="gray", fontsize=8)

    # 2. RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(rounds, rmse, marker="s", color=RED, linewidth=2)
    ax2.fill_between(rounds, rmse, alpha=0.15, color=RED)
    style_ax(ax2, "Global RMSE per Round")
    ax2.set_ylabel("RMSE", color="gray", fontsize=8)

    # 3. Accuracy (<1 % error)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(rounds, acc, marker="^", color=GREEN, linewidth=2)
    ax3.fill_between(rounds, acc, alpha=0.15, color=GREEN)
    style_ax(ax3, "Accuracy (<1% abs error)")
    ax3.set_ylabel("Fraction", color="gray", fontsize=8)

    # 4. Data transmitted per round
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.bar(rounds, tx_kb, color=YELLOW, alpha=0.8, width=0.6)
    style_ax(ax4, "Data Transmitted / Round (KB)")
    ax4.set_ylabel("KB", color="gray", fontsize=8)

    # 5. Avg client training time
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.bar(rounds, train_t, color="#a78bfa", alpha=0.85, width=0.6)
    style_ax(ax5, "Avg Client Train Time (s)")
    ax5.set_ylabel("seconds", color="gray", fontsize=8)

    # 6. Avg peak memory
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.bar(rounds, mem_kb, color="#fb923c", alpha=0.85, width=0.6)
    style_ax(ax6, "Avg Client Peak Memory (KB)")
    ax6.set_ylabel("KB", color="gray", fontsize=8)

    # 7. Round wall time
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(rounds, wall_t, marker="D", color="#f472b6", linewidth=2)
    style_ax(ax7, "Round Wall Time (s)")
    ax7.set_ylabel("seconds", color="gray", fontsize=8)

    # 8. Summary text box
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")
    ax8.set_facecolor(AX_BG)
    summ = report.summary()
    lines = [
        f"Rounds:           {summ.get('num_rounds', '-')}",
        f"Total time:       {summ.get('total_wall_time_s', '-'):.2f}s",
        f"Total TX:         {summ.get('total_bytes_transmitted_MB', '-'):.3f} MB",
        f"Final MAE:        {summ.get('final_global_mae', '-'):.5f}",
        f"Final RMSE:       {summ.get('final_global_rmse', '-'):.5f}",
        f"Final acc <1%:    {summ.get('final_global_accuracy_1pct', '-'):.3f}",
        f"Avg round time:   {summ.get('avg_round_time_s', '-'):.2f}s",
        f"Avg peak mem:     {summ.get('avg_client_peak_memory_kb', '-'):.0f} KB",
    ]
    ax8.text(
        0.05, 0.95,
        "\n".join(lines),
        transform=ax8.transAxes,
        color="white",
        fontsize=8.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d", edgecolor="#30363d"),
    )
    ax8.set_title("Summary", color="white", fontsize=10, pad=6)

    plt.savefig(
        os.path.join(output_dir, "benchmark_dashboard.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    print(f"[Plot] Saved → {output_dir}/benchmark_dashboard.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run federated SOH estimation simulation with Flower."
    )
    parser.add_argument("--data-path",     default="data",          help="Path to discharge CSV files")
    parser.add_argument("--metadata-path", default="metadata.csv",  help="Path to metadata.csv")
    parser.add_argument("--num-clients",   type=int, default=5,     help="Number of federated clients")
    parser.add_argument("--num-rounds",    type=int, default=3,     help="Number of FL rounds")
    parser.add_argument("--n-estimators",  type=int, default=100,   help="Trees per RandomForest")
    parser.add_argument("--output-dir",    default="results",       help="Directory for outputs")
    parser.add_argument(
        "--partition-strategy",
        default="by_battery",
        choices=["iid", "by_battery", "dirichlet"],
        help=(
            "How to split data across clients:\n"
            "  iid         — random equal slices (FL baseline, unrealistic)\n"
            "  by_battery  — each client owns whole batteries (most realistic)\n"
            "  dirichlet   — probabilistic skew via Dirichlet(alpha)"
        ),
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.5,
        help="α for Dirichlet partitioning. Lower → more non-IID. (default: 0.5)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Federated SOH Estimation — Flower Simulation")
    print("=" * 60)
    print(f"  data path          : {args.data_path}")
    print(f"  metadata           : {args.metadata_path}")
    print(f"  clients            : {args.num_clients}")
    print(f"  rounds             : {args.num_rounds}")
    print(f"  n_estimators       : {args.n_estimators}")
    print(f"  partition strategy : {args.partition_strategy}", end="")
    if args.partition_strategy == "dirichlet":
        print(f"  (α={args.dirichlet_alpha})", end="")
    print(f"\n  output dir         : {args.output_dir}")
    print("=" * 60 + "\n")

    strategy = BenchmarkStrategy(
        base_path=args.data_path,
        metadata_path=args.metadata_path,
        num_partitions=args.num_clients,
        benchmark_output=os.path.join(args.output_dir, "benchmark_report.json"),
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    client_fn = build_client_fn(
        base_path=args.data_path,
        metadata_path=args.metadata_path,
        num_partitions=args.num_clients,
        n_estimators=args.n_estimators,
        partition_strategy=args.partition_strategy,
        dirichlet_alpha=args.dirichlet_alpha,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    strategy.finalize()
    report = strategy.report

    # Save JSON report
    json_path = os.path.join(args.output_dir, "benchmark_report.json")
    report.save(json_path)

    # Print summary
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    for k, v in report.summary().items():
        print(f"  {k:<35}: {v}")
    print("=" * 60)

    # Plot
    plot_benchmarks(report, args.output_dir)


if __name__ == "__main__":
    main()
