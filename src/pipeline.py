"""Heatline pipeline entry point. Legacy phases are preserved in the archive."""
import argparse
from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def main():
    parser=argparse.ArgumentParser(description="Reproduce the assessed Nagpur May ANN experiment")
    parser.add_argument("--prepare-only",action="store_true",help="Rebuild daily data, frozen reference normals, labels and features")
    parser.add_argument("--fetch",action="store_true",help="Fetch missing historical ERA5 files before processing")
    parser.add_argument("--max-epochs",type=int,default=400)
    args=parser.parse_args()
    config=load_config()
    if args.fetch:
        from src.fetch_data import fetch_all_data
        fetch_all_data(config)
    if args.prepare_only:
        from src.daily import process_hourly_to_daily
        from src.climatology import compute_wmo_climatology
        from src.labels import evaluate_labels_and_audit
        from src.features import build_and_save_features
        process_hourly_to_daily(config)
        compute_wmo_climatology(config)
        evaluate_labels_and_audit(config)
        build_and_save_features(config)
    else:
        from src.experiment import run
        run(args.max_epochs)


if __name__=="__main__":
    main()
