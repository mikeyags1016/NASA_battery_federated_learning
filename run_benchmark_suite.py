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
from simulate import run_simulation_benchmark


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _winner(traditional: float, federated: float, lower_is_better: bool) -> str:
    if traditional == federated:
        return "tie"
    if lower_is_better:
        return "traditional" if traditional < federated else "federated"
    return "traditional" if traditional > federated else "federated"


def _build_comparison(
    traditional_report: dict,
    federated_report: dict,
    num_satellites: int,
    fed_rounds: int,
    traditional_estimators: int,
    federated_estimators: int,
    partition_strategy: str,
    fed_max_depth: int | None,
    fed_min_samples_leaf: int,
    fed_max_features: str,
) -> dict:
    traditional = traditional_report["summary"]
    federated = federated_report["summary"]

    comparison = {
        "config": {
            "num_satellites": num_satellites,
            "fed_rounds": fed_rounds,
            "traditional_estimators": traditional_estimators,
            "federated_estimators_per_client": federated_estimators,
            "partition_strategy": partition_strategy,
            "federated_max_depth": fed_max_depth,
            "federated_min_samples_leaf": fed_min_samples_leaf,
            "federated_max_features": fed_max_features,
        },
        "traditional": traditional,
        "federated": federated,
        "derived_metrics": {
            "traditional_total_mb_per_satellite": _safe_div(
                traditional["total_bytes_transmitted_MB"], num_satellites
            ),
            "federated_total_mb_per_satellite": _safe_div(
                federated["total_bytes_transmitted_MB"], num_satellites
            ),
            "traditional_accuracy_1pct_per_second": _safe_div(
                traditional["final_global_accuracy_1pct"], traditional["total_wall_time_s"]
            ),
            "federated_accuracy_1pct_per_second": _safe_div(
                federated["final_global_accuracy_1pct"], federated["total_wall_time_s"]
            ),
            "traditional_r2_per_second": _safe_div(
                traditional["final_global_r2"], traditional["total_wall_time_s"]
            ),
            "federated_r2_per_second": _safe_div(
                federated["final_global_r2"], federated["total_wall_time_s"]
            ),
        },
        "winner_by_metric": {
            "mae": _winner(
                traditional["final_global_mae"], federated["final_global_mae"], lower_is_better=True
            ),
            "rmse": _winner(
                traditional["final_global_rmse"], federated["final_global_rmse"], lower_is_better=True
            ),
            "r2": _winner(
                traditional["final_global_r2"], federated["final_global_r2"], lower_is_better=False
            ),
            "accuracy_1pct": _winner(
                traditional["final_global_accuracy_1pct"],
                federated["final_global_accuracy_1pct"],
                lower_is_better=False,
            ),
            "upload_mb": _winner(
                traditional["total_bytes_uploaded_MB"],
                federated["total_bytes_uploaded_MB"],
                lower_is_better=True,
            ),
            "download_mb": _winner(
                traditional["total_bytes_downloaded_MB"],
                federated["total_bytes_downloaded_MB"],
                lower_is_better=True,
            ),
            "total_mb": _winner(
                traditional["total_bytes_transmitted_MB"],
                federated["total_bytes_transmitted_MB"],
                lower_is_better=True,
            ),
            "wall_time_s": _winner(
                traditional["total_wall_time_s"], federated["total_wall_time_s"], lower_is_better=True
            ),
            "cpu_time_s": _winner(
                traditional["avg_client_cpu_time_s"], federated["avg_client_cpu_time_s"], lower_is_better=True
            ),
            "peak_memory_kb": _winner(
                traditional["avg_client_peak_memory_kb"],
                federated["avg_client_peak_memory_kb"],
                lower_is_better=True,
            ),
        },
    }
    return comparison


def _plot_grouped_bar(ax, title: str, values: dict[str, float], color_a: str, color_b: str) -> None:
    labels = list(values.keys())
    vals = np.array([[v[0], v[1]] for v in values.values()], dtype=float)
    x = np.arange(len(labels))
    width = 0.34

    ax.bar(x - width / 2, vals[:, 0], width=width, label="Traditional", color=color_a)
    ax.bar(x + width / 2, vals[:, 1], width=width, label="Federated", color=color_b)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlim(-0.65, len(labels) - 0.35)
    ax.set_ylabel("Value")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for bars in ax.containers:
        labels = []
        for bar in bars:
            height = bar.get_height()
            labels.append(f"{height:.3f}" if abs(height) < 100 else f"{height:.1f}")
        ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ymin, ymax = ax.get_ylim()
    if ymax > ymin:
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.12)


def write_summary_markdown(comparison: dict, output_path: str) -> None:
    traditional = comparison["traditional"]
    federated = comparison["federated"]
    winners = comparison["winner_by_metric"]
    derived = comparison["derived_metrics"]

    lines = [
        "# SOH Benchmark Summary",
        "",
        "## Headline",
        "",
        f"- Traditional wins on predictive quality: MAE {traditional['final_global_mae']:.3f} vs {federated['final_global_mae']:.3f}, RMSE {traditional['final_global_rmse']:.3f} vs {federated['final_global_rmse']:.3f}, R2 {traditional['final_global_r2']:.3f} vs {federated['final_global_r2']:.3f}.",
        f"- Federated wins on communication and reported memory in the current 5-satellite setup: total MB {federated['total_bytes_transmitted_MB']:.1f} vs {traditional['total_bytes_transmitted_MB']:.1f}, peak memory KB {federated['avg_client_peak_memory_kb']:.1f} vs {traditional['avg_client_peak_memory_kb']:.1f}.",
        f"- Traditional is faster end-to-end: wall time {traditional['total_wall_time_s']:.2f}s vs {federated['total_wall_time_s']:.2f}s.",
        "",
        "## Winner By Metric",
        "",
    ]

    for metric, winner in winners.items():
        lines.append(f"- `{metric}`: {winner}")

    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            f"- Traditional accuracy_1pct per second: {derived['traditional_accuracy_1pct_per_second']:.4f}",
            f"- Federated accuracy_1pct per second: {derived['federated_accuracy_1pct_per_second']:.4f}",
            f"- Traditional R2 per second: {derived['traditional_r2_per_second']:.4f}",
            f"- Federated R2 per second: {derived['federated_r2_per_second']:.4f}",
            "",
            "## Next Step",
            "",
            "- Treat the Random Forest federated path as a one-shot ensemble baseline; extra rounds add independently seeded forests rather than optimizing a shared model.",
            "- For a multi-round federated curve that genuinely improves, move the SOH task to an iterative method such as gradient-based FedAvg or boosted trees.",
        ]
    )

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def plot_comparison_dashboard(comparison: dict, output_path: str) -> None:
    traditional = comparison["traditional"]
    federated = comparison["federated"]
    derived = comparison["derived_metrics"]
    winners = comparison["winner_by_metric"]
    cfg = comparison["config"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    fig.suptitle("SOH Benchmark Comparison", fontsize=18, fontweight="bold")

    color_trad = "#1f77b4"
    color_fed = "#ff7f0e"

    _plot_grouped_bar(
        axes[0, 0],
        "Accuracy (Error)",
        {
            "MAE": (traditional["final_global_mae"], federated["final_global_mae"]),
            "RMSE": (traditional["final_global_rmse"], federated["final_global_rmse"]),
        },
        color_trad,
        color_fed,
    )
    _plot_grouped_bar(
        axes[0, 1],
        "Accuracy (Higher Better)",
        {
            "R2": (traditional["final_global_r2"], federated["final_global_r2"]),
            "Acc<1%": (
                traditional["final_global_accuracy_1pct"],
                federated["final_global_accuracy_1pct"],
            ),
        },
        color_trad,
        color_fed,
    )
    _plot_grouped_bar(
        axes[0, 2],
        "Communication (MB)",
        {
            "Upload": (
                traditional["total_bytes_uploaded_MB"],
                federated["total_bytes_uploaded_MB"],
            ),
            "Download": (
                traditional["total_bytes_downloaded_MB"],
                federated["total_bytes_downloaded_MB"],
            ),
            "Total": (
                traditional["total_bytes_transmitted_MB"],
                federated["total_bytes_transmitted_MB"],
            ),
        },
        color_trad,
        color_fed,
    )
    _plot_grouped_bar(
        axes[0, 3],
        "Communication Per Satellite (MB)",
        {
            "Total/sat": (
                derived["traditional_total_mb_per_satellite"],
                derived["federated_total_mb_per_satellite"],
            ),
        },
        color_trad,
        color_fed,
    )
    _plot_grouped_bar(
        axes[1, 0],
        "Time (s)",
        {
            "Wall": (traditional["total_wall_time_s"], federated["total_wall_time_s"]),
            "CPU": (traditional["avg_client_cpu_time_s"], federated["avg_client_cpu_time_s"]),
        },
        color_trad,
        color_fed,
    )
    _plot_grouped_bar(
        axes[1, 1],
        "Hardware",
        {
            "Peak Mem KB": (
                traditional["avg_client_peak_memory_kb"],
                federated["avg_client_peak_memory_kb"],
            ),
        },
        color_trad,
        color_fed,
    )
    _plot_grouped_bar(
        axes[1, 2],
        "Efficiency",
        {
            "Acc/s": (
                derived["traditional_accuracy_1pct_per_second"],
                derived["federated_accuracy_1pct_per_second"],
            ),
            "R2/s": (
                derived["traditional_r2_per_second"],
                derived["federated_r2_per_second"],
            ),
        },
        color_trad,
        color_fed,
    )

    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.02,
        0.98,
        "\n".join(
            [
                "Run Config",
                f"Satellites/clients: {cfg['num_satellites']}",
                f"Federated rounds: {cfg['fed_rounds']}",
                f"Traditional trees: {cfg['traditional_estimators']}",
                f"Federated trees/client: {cfg['federated_estimators_per_client']}",
                f"Fed max depth: {cfg['federated_max_depth']}",
                f"Fed min leaf: {cfg['federated_min_samples_leaf']}",
                f"Fed max features: {cfg['federated_max_features']}",
                f"Partition: {cfg['partition_strategy']}",
                "",
                "Current Story",
                f"Best MAE: {winners['mae']}",
                f"Best RMSE: {winners['rmse']}",
                f"Best R2: {winners['r2']}",
                f"Best communication: {winners['total_mb']}",
                f"Best wall time: {winners['wall_time_s']}",
                f"Best memory: {winners['peak_memory_kb']}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f7fb", edgecolor="#b7c4d3"),
    )

    metric_names = list(winners.keys())
    winner_values = [0 if winners[m] == "traditional" else 1 if winners[m] == "federated" else 0.5 for m in metric_names]
    axes[2, 0].barh(metric_names, winner_values, color=["#1f77b4" if v == 0 else "#ff7f0e" if v == 1 else "#888888" for v in winner_values])
    axes[2, 0].set_title("Winner By Metric")
    axes[2, 0].set_xlim(0, 1)
    axes[2, 0].set_xticks([0, 1])
    axes[2, 0].set_xticklabels(["Traditional", "Federated"])
    axes[2, 0].set_xlabel("Winning method")

    trad_vals = [
        traditional["final_global_mae"],
        traditional["final_global_rmse"],
        traditional["final_global_r2"],
        traditional["final_global_accuracy_1pct"],
    ]
    fed_vals = [
        federated["final_global_mae"],
        federated["final_global_rmse"],
        federated["final_global_r2"],
        federated["final_global_accuracy_1pct"],
    ]
    labels = ["MAE", "RMSE", "R2", "Acc<1%"]
    x = np.arange(len(labels))
    axes[2, 1].plot(x, trad_vals, marker="o", linewidth=2, label="Traditional", color=color_trad)
    axes[2, 1].plot(x, fed_vals, marker="o", linewidth=2, label="Federated", color=color_fed)
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(labels)
    axes[2, 1].set_title("Final Metric Profile")
    axes[2, 1].set_ylabel("Value")
    axes[2, 1].set_xlim(-0.25, len(labels) - 0.75)
    axes[2, 1].grid(axis="y", linestyle=":", alpha=0.35)
    axes[2, 1].legend()

    axes[2, 2].bar(
        ["Traditional", "Federated"],
        [traditional["total_bytes_transmitted_MB"], federated["total_bytes_transmitted_MB"]],
        color=[color_trad, color_fed],
    )
    axes[2, 2].set_title("Total Communication (MB)")
    axes[2, 2].set_ylabel("MB")
    axes[2, 2].grid(axis="y", linestyle=":", alpha=0.35)
    axes[2, 2].margins(x=0.25, y=0.18)
    for i, value in enumerate([traditional["total_bytes_transmitted_MB"], federated["total_bytes_transmitted_MB"]]):
        axes[2, 2].text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    axes[2, 3].bar(
        ["Traditional", "Federated"],
        [traditional["total_wall_time_s"], federated["total_wall_time_s"]],
        color=[color_trad, color_fed],
    )
    axes[2, 3].set_title("Total Wall Time (s)")
    axes[2, 3].set_ylabel("Seconds")
    axes[2, 3].grid(axis="y", linestyle=":", alpha=0.35)
    axes[2, 3].margins(x=0.25, y=0.18)
    for i, value in enumerate([traditional["total_wall_time_s"], federated["total_wall_time_s"]]):
        axes[2, 3].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    for ax in axes.flat:
        if ax.has_data():
            for spine in ax.spines.values():
                spine.set_alpha(0.3)

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_federated_rounds(federated_report: dict, output_path: str) -> None:
    rounds = federated_report["rounds"]
    if not rounds:
        return

    round_ids = [r["round_num"] for r in rounds]
    mae = [r["global_mae"] for r in rounds]
    rmse = [r["global_rmse"] for r in rounds]
    acc = [r["global_accuracy_1pct"] for r in rounds]
    total_mb = [
        (r["bytes_received_from_clients"] + r["bytes_sent_to_clients"]) / 1_048_576
        for r in rounds
    ]
    wall = [r["round_wall_time_s"] for r in rounds]
    train_time = [r["avg_client_train_time_s"] for r in rounds]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("Federated Round Metrics", fontsize=16, fontweight="bold")

    plots = [
        (axes[0, 0], mae, "Global MAE", "MAE"),
        (axes[0, 1], rmse, "Global RMSE", "RMSE"),
        (axes[0, 2], acc, "Accuracy (<1% abs err)", "Fraction"),
        (axes[1, 0], total_mb, "Communication / Round (MB)", "MB"),
        (axes[1, 1], wall, "Round Wall Time (s)", "Seconds"),
        (axes[1, 2], train_time, "Avg Client Train Time (s)", "Seconds"),
    ]
    max_ticks = 12
    tick_step = max(1, int(np.ceil(len(round_ids) / max_ticks)))
    tick_ids = round_ids[::tick_step]
    if round_ids[-1] not in tick_ids:
        tick_ids.append(round_ids[-1])

    for ax, vals, title, ylabel in plots:
        ax.plot(round_ids, vals, marker="o", linewidth=2, color="#0f6cbd")
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(tick_ids)
        ax.set_xlim(min(round_ids) - 0.5, max(round_ids) + 0.5)
        ax.margins(y=0.15)
        ax.grid(True, linestyle=":", alpha=0.35)

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run traditional and federated SOH benchmarks, then generate a comparison dashboard."
    )
    parser.add_argument("--data-path", default=str((ROOT.parent / "cleaned_dataset" / "data").resolve()))
    parser.add_argument("--metadata-path", default=str((ROOT.parent / "cleaned_dataset" / "metadata.csv").resolve()))
    parser.add_argument("--output-dir", default=str((ROOT / "benchmark_outputs").resolve()))
    parser.add_argument("--num-satellites", type=int, default=5)
    parser.add_argument("--fed-rounds", type=int, default=1)
    parser.add_argument("--traditional-estimators", type=int, default=200)
    parser.add_argument("--federated-estimators", type=int, default=100)
    parser.add_argument("--fed-max-depth", type=int, default=None)
    parser.add_argument("--fed-min-samples-leaf", type=int, default=1)
    parser.add_argument("--fed-max-features", default="sqrt")
    parser.add_argument("--global-test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--partition-strategy",
        default="by_battery",
        choices=["iid", "by_battery", "dirichlet"],
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    traditional_dir = output_dir / "traditional"
    federated_dir = output_dir / "federated"
    traditional_dir.mkdir(parents=True, exist_ok=True)
    federated_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Running traditional benchmark...")
    print("=" * 70)
    traditional_report = run_traditional_benchmark(
        data_base_path=args.data_path,
        metadata_path=args.metadata_path,
        n_estimators=args.traditional_estimators,
        global_test_size=args.global_test_size,
        random_state=args.random_state,
        output_path=str(traditional_dir / "traditional_benchmark_report.json"),
        num_satellites=args.num_satellites,
    )

    print("=" * 70)
    print("Running federated benchmark...")
    print("=" * 70)
    federated_benchmark = run_simulation_benchmark(
        data_path=args.data_path,
        metadata_path=args.metadata_path,
        num_clients=args.num_satellites,
        num_rounds=args.fed_rounds,
        n_estimators=args.federated_estimators,
        output_dir=str(federated_dir),
        global_test_size=args.global_test_size,
        partition_strategy=args.partition_strategy,
        dirichlet_alpha=args.dirichlet_alpha,
        max_depth=args.fed_max_depth,
        min_samples_leaf=args.fed_min_samples_leaf,
        max_features=args.fed_max_features,
    )
    federated_report = {
        "summary": federated_benchmark.summary(),
        "rounds": [round_info.to_dict() for round_info in federated_benchmark.rounds],
    }

    comparison = _build_comparison(
        traditional_report=traditional_report,
        federated_report=federated_report,
        num_satellites=args.num_satellites,
        fed_rounds=args.fed_rounds,
        traditional_estimators=args.traditional_estimators,
        federated_estimators=args.federated_estimators,
        partition_strategy=args.partition_strategy,
        fed_max_depth=args.fed_max_depth,
        fed_min_samples_leaf=args.fed_min_samples_leaf,
        fed_max_features=args.fed_max_features,
    )

    comparison_path = output_dir / "comparison_report.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))

    plot_comparison_dashboard(comparison, str(output_dir / "comparison_dashboard.png"))
    plot_federated_rounds(federated_report, str(output_dir / "federated_rounds.png"))
    write_summary_markdown(comparison, str(output_dir / "summary.md"))

    print("=" * 70)
    print("Benchmark suite complete")
    print(f"Traditional report: {traditional_dir / 'traditional_benchmark_report.json'}")
    print(f"Federated report:   {federated_dir / 'benchmark_report.json'}")
    print(f"Comparison report:  {comparison_path}")
    print(f"Comparison chart:   {output_dir / 'comparison_dashboard.png'}")
    print(f"Round chart:        {output_dir / 'federated_rounds.png'}")
    print(f"Summary:            {output_dir / 'summary.md'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
