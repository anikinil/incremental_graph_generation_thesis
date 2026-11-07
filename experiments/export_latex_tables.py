import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.export_quality_table import export_summary as export_quality
from experiments.export_timing_table import export_summary as export_timing


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "artifacts/results"


def latest_results(batch_name):
    candidates = sorted(
        path.parent
        for path in (RESULTS_ROOT / batch_name).glob("*/quality_results.csv")
        if (path.parent / "timing_results.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No quality and timing results found in {RESULTS_ROOT / batch_name}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Export both quality and timing tables for both update regimes.")
    parser.add_argument("--percent-results-dir", type=Path)
    parser.add_argument("--single-results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    sources = (
        ("5_percent_change", args.percent_results_dir or latest_results("evaluation_matrix_5_percent_change")),
        ("one_obstacle_change", args.single_results_dir or latest_results("evaluation_matrix_one_obstacle_change")),
    )
    timestamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or RESULTS_ROOT / "latex_tables" / timestamp
    output_dir.mkdir(parents=True, exist_ok=args.output_dir is not None)

    for regime, results_dir in sources:
        export_quality(results_dir, output_dir / f"quality_summary_{regime}_task_metrics_only.tex")
        export_timing(results_dir, output_dir / f"timing_summary_{regime}.tex")


if __name__ == "__main__":
    main()
