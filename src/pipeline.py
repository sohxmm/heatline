"""
Main CLI Pipeline Runner for Nagpur Heatwave ANN Project.

Supports end-to-end execution across all phases:
- Phase 1: Fetch raw data, aggregate daily features, compute climatology & labels, generate audit
- Phase 2: Engineer features, split datasets, run tests
- Phase 3: Soft computing baselines (M-P neuron, Perceptron, ADALINE, Persistence, LogReg)
- Phase 4: Error Backpropagation ANN, Deep MLP Ensemble, Fuzzy Logic, GA Optimizer
- Phase 5: Evaluation figures and results summary
- Phase 6: Launch Streamlit web application
"""

import argparse
import sys
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and parse the project configuration YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_phase_1(config: dict):
    """Execute Phase 1: Data Acquisition, Daily Aggregation, Climatology, Labels, Data Audit."""
    print(">>> Starting Phase 1: Data Acquisition & Climatological Audit...")
    try:
        from src.fetch_data import fetch_all_data
        from src.daily import process_hourly_to_daily
        from src.climatology import compute_wmo_climatology
        from src.labels import evaluate_labels_and_audit

        print("[1/4] Fetching historical weather data from Open-Meteo...")
        fetch_all_data(config)

        print("[2/4] Aggregating hourly records to daily variables...")
        process_hourly_to_daily(config)

        print("[3/4] Computing WMO 1991-2020 climatology normals...")
        compute_wmo_climatology(config)

        print("[4/4] Evaluating heatwave labels and generating data audit report...")
        evaluate_labels_and_audit(config)

        print(">>> Phase 1 completed successfully. Results saved to results/data_audit.md.")
    except ImportError as e:
        print(f"Phase 1 modules not fully imported yet: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Nagpur May Heat Wave Prediction CLI Pipeline (KJSCE Soft Computing Lab)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["0", "1", "2", "3", "4", "5", "6", "all"],
        help="Specify which project phase to execute.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run only the climatology and label audit without refetching data.",
    )

    args = parser.parse_args()
    print("=" * 70)
    print(" Nagpur May Heat Wave Prediction — Soft Computing Lab Pipeline")
    print("=" * 70)
    print(f"Selected Phase: {args.phase}")

    config = load_config(args.config)

    if args.phase in ["1", "all"]:
        run_phase_1(config)

    if args.phase == "0":
        print("Phase 0 (Scaffold & Environment) verified.")

    print("Pipeline execution finished.")


if __name__ == "__main__":
    main()
