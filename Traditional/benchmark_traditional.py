from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SOHFED_ROOT = THIS_DIR.parent / "Federated" / "soh_federated"
if str(SOHFED_ROOT) not in sys.path:
    sys.path.insert(0, str(SOHFED_ROOT))

from sohfed.benchmarks import BenchmarkReport, RoundMetrics
from sohfed.task import (
    build_global_dataset,
    evaluate,
    get_model_size_bytes,
    get_raw_bytes_for_filenames,
    load_global_splits,
    train,
)


def run_benchmark(
    data_base_path: str,
    metadata_path: str,
    n_estimators: int,
    global_test_size: float,
    random_state: int,
    output_path: str,
    num_satellites: int | None,
) -> dict:
    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_battery_ids,
        _test_battery_ids,
        train_filenames,
        _test_filenames,
    ) = load_global_splits(
        base_path=data_base_path,
        metadata_path=metadata_path,
        global_test_size=global_test_size,
        random_state=random_state,
    )

    model, bench = train(
        X_train,
        y_train,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    metrics = evaluate(model, X_test, y_test)

    total_batteries = len(set(train_battery_ids.tolist()))
    satellite_count = num_satellites if num_satellites is not None else total_batteries
    raw_upload_bytes = get_raw_bytes_for_filenames(data_base_path, train_filenames)
    model_download_bytes = get_model_size_bytes(model) * satellite_count

    round_metrics = RoundMetrics(
        round_num=1,
        bytes_sent_to_clients=model_download_bytes,
        bytes_received_from_clients=raw_upload_bytes,
        num_clients_trained=satellite_count,
        num_clients_evaluated=1,
        round_wall_time_s=bench["train_time_s"],
        avg_client_train_time_s=bench["train_time_s"],
        avg_client_cpu_time_s=bench["cpu_time_s"],
        avg_client_peak_memory_kb=bench["peak_memory_kb"],
        avg_train_loss=metrics["mae"],
        avg_eval_mae=metrics["mae"],
        avg_eval_rmse=metrics["rmse"],
        avg_eval_accuracy_1pct=metrics["accuracy_1pct"],
        global_mae=metrics["mae"],
        global_rmse=metrics["rmse"],
        global_r2=metrics["r2"],
        global_accuracy_1pct=metrics["accuracy_1pct"],
    )

    report = BenchmarkReport(mode="traditional")
    report.add_round(round_metrics)
    report.total_wall_time_s = bench["train_time_s"]
    report.save(output_path)

    summary = report.summary()
    summary.update(
        {
            "num_train_samples": len(X_train),
            "num_test_samples": len(X_test),
            "num_satellites": satellite_count,
            "raw_upload_bytes": raw_upload_bytes,
            "model_download_bytes": model_download_bytes,
            "n_estimators": n_estimators,
        }
    )
    return {"summary": summary, "rounds": [round_metrics.to_dict()]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the centralized SOH benchmark with the shared benchmark schema."
    )
    parser.add_argument("--data-path", default="../../cleaned_dataset/data")
    parser.add_argument("--metadata-path", default="../../cleaned_dataset/metadata.csv")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--global-test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-satellites", type=int, default=None)
    parser.add_argument(
        "--output-path",
        default="traditional_benchmark_report.json",
        help="Path to the output JSON report.",
    )
    args = parser.parse_args()

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = run_benchmark(
        data_base_path=args.data_path,
        metadata_path=args.metadata_path,
        n_estimators=args.n_estimators,
        global_test_size=args.global_test_size,
        random_state=args.random_state,
        output_path=output_path,
        num_satellites=args.num_satellites,
    )

    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
