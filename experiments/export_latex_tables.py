from pathlib import Path
import sys

from experiments.export_quality_table import export_summary as export_quality
from experiments.export_timing_table import export_summary as export_timing


def main():
    results_dir = Path(sys.argv[1])
    export_quality(results_dir)
    export_timing(results_dir)


if __name__ == "__main__":
    main()
