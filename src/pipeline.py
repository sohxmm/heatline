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
    args=parser.parse_args()
    if args.fetch:
        from src.seasonal_data import fetch_all
        fetch_all('era5')
        fetch_all('nasa')
    if args.prepare_only:
        from src.seasonal_data import prepare_era5
        prepare_era5()
        from src.provider_benchmark import prepare_nasa
        daily, _, _ = prepare_nasa()
        daily.to_csv(Path(__file__).resolve().parents[1] / 'data/processed/nasa_daily.csv', index=False)
    else:
        from src.full_dataset_study import run
        run()


if __name__=="__main__":
    main()
