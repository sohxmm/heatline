"""
Heat Wave Label Definitions and Climatological Evaluation Module.

Evaluates three candidate heatwave label definitions for Nagpur Interior Plains:
1. `imd_reanalysis_plains` (Candidate A):
   tmax >= 43.5°C OR (tmax >= 40.0°C AND tmax - normal_tmax >= 3.5°C)
2. `climatology_p85_42c` (Candidate B):
   tmax >= max(42.0°C, P85_tmax_1991_2020)
3. `imd_strict_raw` (Reference):
   tmax >= 45.0°C OR (tmax >= 40.0°C AND tmax - normal_tmax >= 4.5°C)

Applies the Phase 1 selection rule to choose the primary binary target for training.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.climatology import load_climatology_lookup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_candidate_labels(df_daily: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute all 3 candidate heatwave labels for all records in df_daily."""
    lookup = load_climatology_lookup(config)

    normal_tmax = []
    p85_tmax = []
    p90_tmax = []

    for _, row in df_daily.iterrows():
        key = (int(row["month"]), int(row["day"]))
        norm = lookup.get(key)
        if norm:
            normal_tmax.append(norm["normal_tmax"])
            p85_tmax.append(norm["p85_tmax"])
            p90_tmax.append(norm["p90_tmax"])
        else:
            normal_tmax.append(np.nan)
            p85_tmax.append(np.nan)
            p90_tmax.append(np.nan)

    df = df_daily.copy()
    df["normal_tmax"] = normal_tmax
    df["p85_tmax"] = p85_tmax
    df["p90_tmax"] = p90_tmax
    df["tmax_anomaly"] = round(df["tmax"] - df["normal_tmax"], 2)

    # 1. Candidate A: imd_reanalysis_plains
    cand_a_cfg = config["labels"]["candidates"]["imd_reanalysis_plains"]
    abs_thresh_a = cand_a_cfg["absolute_threshold"]
    dep_base_a = cand_a_cfg["departure_base"]
    dep_thresh_a = cand_a_cfg["departure_threshold"]

    df["imd_reanalysis_plains"] = (
        (df["tmax"] >= abs_thresh_a)
        | ((df["tmax"] >= dep_base_a) & (df["tmax_anomaly"] >= dep_thresh_a))
    ).astype(int)

    # 2. Candidate B: climatology_p85_42c
    cand_b_cfg = config["labels"]["candidates"]["climatology_p85_42c"]
    floor_b = cand_b_cfg["floor_threshold"]
    p85_effective = np.maximum(floor_b, df["p85_tmax"])
    df["climatology_p85_42c"] = (df["tmax"] >= p85_effective).astype(int)

    # 3. Reference: imd_strict_raw
    cand_ref_cfg = config["labels"]["candidates"]["imd_strict_raw"]
    abs_thresh_ref = cand_ref_cfg["absolute_threshold"]
    dep_base_ref = cand_ref_cfg["departure_base"]
    dep_thresh_ref = cand_ref_cfg["departure_threshold"]

    df["imd_strict_raw"] = (
        (df["tmax"] >= abs_thresh_ref)
        | ((df["tmax"] >= dep_base_ref) & (df["tmax_anomaly"] >= dep_thresh_ref))
    ).astype(int)

    return df


def evaluate_split_metrics(df_may: pd.DataFrame, config: dict) -> dict:
    """Evaluate candidate label distributions across Train, Val, and Test splits."""
    splits = {
        "train": (
            config["splits"]["train"]["start_year"],
            config["splits"]["train"]["end_year"],
        ),
        "val": (
            config["splits"]["val"]["start_year"],
            config["splits"]["val"]["end_year"],
        ),
        "test": (
            config["splits"]["test"]["start_year"],
            config["splits"]["test"]["end_year"],
        ),
        "overall": (
            config["data"]["years"]["start"],
            config["data"]["years"]["end"],
        ),
    }

    candidates = ["imd_reanalysis_plains", "climatology_p85_42c", "imd_strict_raw"]
    metrics = {}

    for split_name, (y_start, y_end) in splits.items():
        sub_df = df_may[(df_may["year"] >= y_start) & (df_may["year"] <= y_end)]
        n_days = len(sub_df)
        split_res = {"total_days": n_days, "years": f"{y_start}-{y_end}"}

        for cand in candidates:
            pos_cnt = int(sub_df[cand].sum())
            pos_pct = round((pos_cnt / n_days) * 100, 2) if n_days > 0 else 0.0
            split_res[cand] = {"count": pos_cnt, "pct": pos_pct}

        metrics[split_name] = split_res

    return metrics


def select_primary_label(metrics: dict) -> tuple[str, str]:
    """
    Select primary label based on Training split (1981-2010) positive rate:
    Must be between 15% and 35%, choosing the candidate closest to 22%.
    """
    train_metrics = metrics["train"]
    rate_a = train_metrics["imd_reanalysis_plains"]["pct"]
    rate_b = train_metrics["climatology_p85_42c"]["pct"]

    target_rate = 22.0
    diff_a = abs(rate_a - target_rate)
    diff_b = abs(rate_b - target_rate)

    in_range_a = 15.0 <= rate_a <= 35.0
    in_range_b = 15.0 <= rate_b <= 35.0

    if in_range_a and not in_range_b:
        selected = "imd_reanalysis_plains"
        reason = f"Candidate A rate ({rate_a}%) is inside [15%, 35%] target range while Candidate B ({rate_b}%) is outside."
    elif in_range_b and not in_range_a:
        selected = "climatology_p85_42c"
        reason = f"Candidate B rate ({rate_b}%) is inside [15%, 35%] target range while Candidate A ({rate_a}%) is outside."
    else:
        # Both inside or both outside: choose closest to 22%
        if diff_a <= diff_b:
            selected = "imd_reanalysis_plains"
            reason = f"Candidate A ({rate_a}%, diff {diff_a:.2f}%) is closer to optimal 22% target than Candidate B ({rate_b}%, diff {diff_b:.2f}%)."
        else:
            selected = "climatology_p85_42c"
            reason = f"Candidate B ({rate_b}%, diff {diff_b:.2f}%) is closer to optimal 22% target than Candidate A ({rate_a}%, diff {diff_a:.2f}%)."

    return selected, reason


def evaluate_labels_and_audit(config: dict) -> dict:
    """Run full label evaluation, update daily dataset with labels, and return audit stats."""
    daily_csv = Path(config["data"]["daily_file"])
    df_daily = pd.read_csv(daily_csv)

    df_labeled = compute_candidate_labels(df_daily, config)
    df_may = df_labeled[df_labeled["month"] == 5].copy()

    metrics = evaluate_split_metrics(df_may, config)
    selected_label, selection_reason = select_primary_label(metrics)

    logger.info(f"Selected Primary Target Label: '{selected_label}'")
    logger.info(f"Selection Reason: {selection_reason}")

    # Set primary label column
    df_labeled["heatwave_day"] = df_labeled[selected_label]
    df_labeled.to_csv(daily_csv, index=False)
    logger.info(f"Updated {daily_csv} with candidate labels and primary 'heatwave_day' target.")

    # Update config.yaml with selected primary label
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["labels"]["primary"] = selected_label
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Updated config.yaml with primary label: {selected_label}")

    return {
        "metrics": metrics,
        "selected_label": selected_label,
        "selection_reason": selection_reason,
        "df_labeled": df_labeled,
    }


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as cfg_f:
        config_data = yaml.safe_load(cfg_f)
    res = evaluate_labels_and_audit(config_data)
    print("\n--- EVALUATION SUMMARY ---")
    print(f"Selected Label: {res['selected_label']}")
    print(f"Reason: {res['selection_reason']}")
    print("\nSplit Metrics:")
    for split, data in res["metrics"].items():
        print(f"  {split.upper()} ({data['years']}): {data['total_days']} days")
        for k in ["imd_reanalysis_plains", "climatology_p85_42c", "imd_strict_raw"]:
            print(f"    {k}: {data[k]['count']} days ({data[k]['pct']}%)")
