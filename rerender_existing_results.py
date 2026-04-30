from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_all_results import _plot_model_comparison, _write_summary
from run_benchmark_suite import plot_federated_rounds


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerender existing benchmark graphs from JSON reports without training."
    )
    parser.add_argument("--output-dir", default="benchmark_outputs_latest")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    all_results = _read_json(output_dir / "all_results.json")

    if all_results and "results" in all_results:
        _plot_model_comparison(all_results["results"], output_dir / "model_comparison.png")
        _write_summary(all_results["results"], output_dir / "summary.md")

    round_reports = {
        "federated_rf": output_dir / "federated_rf" / "benchmark_report.json",
        "federated_boosted": output_dir / "federated_boosted" / "benchmark_report.json",
        "federated_boosted_25": output_dir / "federated_boosted_25" / "benchmark_report.json",
        "federated_xgboost": output_dir / "federated_xgboost" / "benchmark_report.json",
        "federated_xgboost_smoke": output_dir / "federated_xgboost_smoke" / "benchmark_report.json",
        "federated_xgboost_smoke2": output_dir / "federated_xgboost_smoke2" / "benchmark_report.json",
    }
    for name, report_path in round_reports.items():
        report = _read_json(report_path)
        if not report:
            continue
        plot_federated_rounds(report, str(output_dir / f"{name}_rounds.png"))

    print(f"Rerendered graphs in {output_dir}")


if __name__ == "__main__":
    main()
