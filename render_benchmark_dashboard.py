from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_benchmark_suite import (
    plot_comparison_dashboard,
    plot_federated_rounds,
    write_summary_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render comparison dashboard and summaries from existing benchmark JSON reports."
    )
    parser.add_argument("--comparison-report", required=True)
    parser.add_argument("--federated-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = json.loads(Path(args.comparison_report).read_text(encoding="utf-8"))
    federated = json.loads(Path(args.federated_report).read_text(encoding="utf-8"))

    plot_comparison_dashboard(comparison, str(output_dir / "comparison_dashboard.png"))
    plot_federated_rounds(federated, str(output_dir / "federated_rounds.png"))
    write_summary_markdown(comparison, str(output_dir / "summary.md"))

    print(f"Comparison chart: {output_dir / 'comparison_dashboard.png'}")
    print(f"Round chart:      {output_dir / 'federated_rounds.png'}")
    print(f"Summary:          {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
