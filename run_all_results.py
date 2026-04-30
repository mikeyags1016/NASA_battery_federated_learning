from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
TRADITIONAL_DIR = ROOT / "Traditional"
FEDERATED_DIR = ROOT / "Federated" / "soh_federated"

for extra_path in (TRADITIONAL_DIR, FEDERATED_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
os.environ["PYTHONPATH"] = (
    str(FEDERATED_DIR)
    if not os.environ.get("PYTHONPATH")
    else str(FEDERATED_DIR) + os.pathsep + os.environ["PYTHONPATH"]
)

from benchmark_traditional import run_benchmark as run_traditional_benchmark
from boosted_simulate import run_boosted_benchmark
from run_benchmark_suite import plot_federated_rounds
from simulate import run_simulation_benchmark

try:
    from xgboost_simulate import run_xgboost_benchmark
except SystemExit:
    run_xgboost_benchmark = None


def _summary(report: dict) -> dict:
    return report["summary"] if "summary" in report else report.summary()


def _plot_model_comparison(results: dict[str, dict], output_path: Path) -> None:
    labels = list(results.keys())
    summaries = [results[label] for label in labels]
    metrics = [
        ("final_global_mae", "MAE", True),
        ("final_global_rmse", "RMSE", True),
        ("final_global_r2", "R2", False),
        ("final_global_accuracy_1pct", "Acc <1%", False),
        ("total_bytes_transmitted_MB", "Total MB", True),
        ("total_wall_time_s", "Wall Time", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("SOH Model Comparison", fontsize=16, fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, (key, title, lower_is_better) in zip(axes.flat, metrics):
        values = [float(summary[key]) for summary in summaries]
        ax.bar(labels, values, color=colors[: len(labels)])
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        best_idx = int(np.argmin(values) if lower_is_better else np.argmax(values))
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.4g}", ha="center", va="bottom", fontsize=9)
        ax.get_xticklabels()[best_idx].set_fontweight("bold")

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_summary(results: dict[str, dict], output_path: Path) -> None:
    lines = [
        "# SOH Results Summary",
        "",
        "| Model | MAE | RMSE | R2 | Acc <1% | Total MB | Wall s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in results.items():
        lines.append(
            "| "
            f"{label} | "
            f"{summary['final_global_mae']:.6f} | "
            f"{summary['final_global_rmse']:.6f} | "
            f"{summary['final_global_r2']:.6f} | "
            f"{summary['final_global_accuracy_1pct']:.4f} | "
            f"{summary['total_bytes_transmitted_MB']:.4f} | "
            f"{summary['total_wall_time_s']:.3f} |"
        )

    best_mae = min(results.items(), key=lambda item: item[1]["final_global_mae"])
    best_comm = min(results.items(), key=lambda item: item[1]["total_bytes_transmitted_MB"])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Best predictive MAE: {best_mae[0]} ({best_mae[1]['final_global_mae']:.6f}).",
            f"- Lowest communication: {best_comm[0]} ({best_comm[1]['total_bytes_transmitted_MB']:.4f} MB).",
            "- Use the boosted cyclic curve to judge whether more communication rounds are buying real accuracy.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all SOH benchmarks needed for the current result set."
    )
    parser.add_argument("--data-path", default=str((ROOT.parent / "cleaned_dataset" / "data").resolve()))
    parser.add_argument("--metadata-path", default=str((ROOT.parent / "cleaned_dataset" / "metadata.csv").resolve()))
    parser.add_argument("--output-dir", default=str((ROOT / "benchmark_outputs_latest").resolve()))
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--partition-strategy", default="by_battery", choices=["iid", "by_battery", "dirichlet"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--global-test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--traditional-estimators", type=int, default=200)
    parser.add_argument("--rf-rounds", type=int, default=1)
    parser.add_argument("--rf-estimators", type=int, default=100)
    parser.add_argument("--rf-max-depth", type=int, default=10)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--boosted-rounds", type=int, default=25)
    parser.add_argument("--boosted-local-estimators", type=int, default=5)
    parser.add_argument("--boosted-learning-rate", type=float, default=0.05)
    parser.add_argument("--boosted-max-depth", type=int, default=3)
    parser.add_argument("--boosted-min-samples-leaf", type=int, default=2)
    parser.add_argument("--boosted-subsample", type=float, default=0.9)
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--xgboost-rounds", type=int, default=25)
    parser.add_argument("--xgboost-local-rounds", type=int, default=5)
    parser.add_argument("--xgboost-eta", type=float, default=0.05)
    parser.add_argument("--xgboost-max-depth", type=int, default=3)
    parser.add_argument("--xgboost-subsample", type=float, default=0.9)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    traditional_dir = output_dir / "traditional"
    federated_rf_dir = output_dir / "federated_rf"
    boosted_dir = output_dir / "federated_boosted"
    xgboost_dir = output_dir / "federated_xgboost"
    for path in (traditional_dir, federated_rf_dir, boosted_dir, xgboost_dir):
        path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("1/3 Traditional centralized Random Forest")
    print("=" * 70)
    traditional_report = run_traditional_benchmark(
        data_base_path=args.data_path,
        metadata_path=args.metadata_path,
        n_estimators=args.traditional_estimators,
        global_test_size=args.global_test_size,
        random_state=args.random_state,
        output_path=str(traditional_dir / "benchmark_report.json"),
        num_satellites=args.num_clients,
    )

    print("=" * 70)
    print("2/3 Federated Random Forest one-shot ensemble")
    print("=" * 70)
    federated_rf_report_obj = run_simulation_benchmark(
        data_path=args.data_path,
        metadata_path=args.metadata_path,
        num_clients=args.num_clients,
        num_rounds=args.rf_rounds,
        n_estimators=args.rf_estimators,
        output_dir=str(federated_rf_dir),
        global_test_size=args.global_test_size,
        partition_strategy=args.partition_strategy,
        dirichlet_alpha=args.dirichlet_alpha,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        max_features="sqrt",
    )
    federated_rf_report = {
        "summary": federated_rf_report_obj.summary(),
        "rounds": [round_info.to_dict() for round_info in federated_rf_report_obj.rounds],
    }

    print("=" * 70)
    print("3/3 Federated cyclic Gradient Boosted Trees")
    print("=" * 70)
    boosted_report_obj = run_boosted_benchmark(
        data_path=args.data_path,
        metadata_path=args.metadata_path,
        num_clients=args.num_clients,
        num_rounds=args.boosted_rounds,
        local_estimators=args.boosted_local_estimators,
        output_dir=str(boosted_dir),
        learning_rate=args.boosted_learning_rate,
        max_depth=args.boosted_max_depth,
        min_samples_leaf=args.boosted_min_samples_leaf,
        subsample=args.boosted_subsample,
        global_test_size=args.global_test_size,
        partition_strategy=args.partition_strategy,
        dirichlet_alpha=args.dirichlet_alpha,
        random_state=args.random_state,
    )
    boosted_report = {
        "summary": boosted_report_obj.summary(),
        "rounds": [round_info.to_dict() for round_info in boosted_report_obj.rounds],
    }

    results = {
        "Traditional RF": traditional_report["summary"],
        "Federated RF": federated_rf_report["summary"],
        "Federated Boosted": boosted_report["summary"],
    }
    reports = {
        "traditional": traditional_report,
        "federated_rf": federated_rf_report,
        "federated_boosted": boosted_report,
    }

    if args.include_xgboost:
        if run_xgboost_benchmark is None:
            raise RuntimeError(
                "XGBoost benchmark requested, but xgboost is not installed. "
                "Run `pip install -r requirements.txt` first."
            )
        print("=" * 70)
        print("4/4 Federated XGBoost cyclic")
        print("=" * 70)
        xgboost_report_obj = run_xgboost_benchmark(
            data_path=args.data_path,
            metadata_path=args.metadata_path,
            num_clients=args.num_clients,
            num_rounds=args.xgboost_rounds,
            local_rounds=args.xgboost_local_rounds,
            output_dir=str(xgboost_dir),
            eta=args.xgboost_eta,
            max_depth=args.xgboost_max_depth,
            subsample=args.xgboost_subsample,
            global_test_size=args.global_test_size,
            partition_strategy=args.partition_strategy,
            dirichlet_alpha=args.dirichlet_alpha,
            random_state=args.random_state,
        )
        xgboost_report = {
            "summary": xgboost_report_obj.summary(),
            "rounds": [round_info.to_dict() for round_info in xgboost_report_obj.rounds],
        }
        results["Federated XGBoost"] = xgboost_report["summary"]
        reports["federated_xgboost"] = xgboost_report

    combined = {
        "config": vars(args),
        "results": results,
        "reports": reports,
    }

    (output_dir / "all_results.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    _plot_model_comparison(results, output_dir / "model_comparison.png")
    plot_federated_rounds(federated_rf_report, str(output_dir / "federated_rf_rounds.png"))
    plot_federated_rounds(boosted_report, str(output_dir / "federated_boosted_rounds.png"))
    if "federated_xgboost" in reports:
        plot_federated_rounds(reports["federated_xgboost"], str(output_dir / "federated_xgboost_rounds.png"))
    _write_summary(results, output_dir / "summary.md")

    print("=" * 70)
    print("All results complete")
    print(f"Output directory: {output_dir}")
    print(f"Summary:          {output_dir / 'summary.md'}")
    print(f"Combined JSON:    {output_dir / 'all_results.json'}")
    print(f"Comparison chart: {output_dir / 'model_comparison.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
