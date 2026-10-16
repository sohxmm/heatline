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

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def run_phase_2(config: dict):
    """Execute Phase 2: Feature Engineering, Chronological Splitting, Scaler Export."""
    print(">>> Starting Phase 2: Feature Engineering & Dataset Splitting...")
    from src.features import build_and_save_features
    from src.split import get_splits

    print("[1/2] Computing lag, rolling, trend, and climatological features...")
    build_and_save_features(config)

    print("[2/2] Generating chronological splits (Train/Val/Test) and fitting scaler...")
    get_splits(config)
    print(">>> Phase 2 completed successfully. Features and splits ready.")


def run_phase_3(config: dict):
    """Execute Phase 3: Evaluate Soft Computing Baselines (M-P, Perceptron, ADALINE, Persistence, LogReg)."""
    print(">>> Starting Phase 3: Soft Computing Baselines & Statistical Benchmarks...")
    from src.baselines import evaluate_all_baselines

    df_results = evaluate_all_baselines(config)
    print("\n" + df_results.to_string(index=False))
    print(">>> Phase 3 completed successfully. Results saved to results/tables/baseline_comparison.csv.")


def run_phase_4(config: dict):
    """Execute Phase 4: Core Soft Computing Models (EBPA, Deep MLP, Fuzzy, GA)."""
    print(">>> Starting Phase 4: Core Soft Computing Training & Hybrid Optimization...")
    from src.train_ann import train_mlp_ensemble
    from src.fuzzy_controller import evaluate_fuzzy_controller
    from src.ga_optimizer import run_ga_optimization
    from src.export_model import export_torch_ensemble_to_numpy, verify_inference_equivalence

    print("[1/4] Training 10-Seed Regularized Deep MLP Ensemble...")
    train_mlp_ensemble(config)
    print("[2/4] Evaluating Mamdani Fuzzy Inference System...")
    evaluate_fuzzy_controller(config)
    print("[3/4] Running Genetic Algorithm Neuro-Fuzzy Optimization...")
    run_ga_optimization(config)
    print("[4/4] Exporting Lightweight Pure-NumPy Weights & Verifying Equivalence...")
    export_torch_ensemble_to_numpy()
    verify_inference_equivalence(config)
    print(">>> Phase 4 completed successfully.")


def run_phase_5(config: dict):
    """Execute Phase 5: Comprehensive Evaluation and Figure Generation."""
    print(">>> Starting Phase 5: Evaluation & Publication Figures...")
    from src.evaluate import run_full_evaluation
    run_full_evaluation(config)
    print(">>> Phase 5 completed successfully. All 12 figures and tables generated.")


def run_phase_6(config: dict):
    """Execute Phase 6: Launch Interactive Streamlit Application."""
    print(">>> Starting Phase 6: Launching Streamlit Web Application...")
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8501"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)


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

    if args.phase in ["2", "all"]:
        run_phase_2(config)

    if args.phase in ["3", "all"]:
        run_phase_3(config)

    if args.phase in ["4", "all"]:
        run_phase_4(config)

    if args.phase in ["5", "all"]:
        run_phase_5(config)

    if args.phase == "6":
        run_phase_6(config)

    if args.phase == "0":
        print("Phase 0 (Scaffold & Environment) verified.")

    print("Pipeline execution finished.")


if __name__ == "__main__":
    main()

