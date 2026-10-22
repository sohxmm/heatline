# LEGACY/OPTIONAL: original experiment only. Use src.experiment and frontend/ for Heatline.
# Old weights, label definitions and reported metrics are not current assessed evidence.
"""
Comprehensive Model Evaluation, Sensitivity Analysis & Publication Figures.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).

Generates:
- All CSV metric tables in `results/tables/`:
  - `final_model_comparison_val.csv`
  - `final_model_comparison_test.csv`
  - `sensitivity_analysis.csv`
  - `test_confusion_matrices.json`
- All 12 publication figures at 200 DPI in `results/figures/`:
  - fig01_study_area.png
  - fig02_tmax_climatology.png
  - fig03_positives_per_year.png
  - fig04_feature_correlation.png
  - fig05_training_curves.png
  - fig06_roc_pr_curves.png
  - fig07_confusion_matrix.png
  - fig08_calibration.png
  - fig09_permutation_importance.png
  - fig10_error_analysis.png
  - fig11_fuzzy_membership.png
  - fig12_ga_convergence.png
"""

import json
import logging
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    average_precision_score,
)
import yaml

from src.split import get_splits
from src.sc_baselines import McCullochPittsNeuron, PerceptronRuleClassifier, DeltaRuleADALINE
from src.baselines import PersistenceBaseline, fit_logistic_regression
from src.ann_numpy import EBPANetwork
from src.export_model import NumpyMLPEnsemble
from src.fuzzy_controller import MamdaniFuzzyController, trimf, trapmf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
})


def load_all_predictions(config: dict) -> Dict[str, Any]:
    """Load splits, fit/load all models, and collect prediction vectors."""
    splits = get_splits(config)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    feature_names = splits["feature_names"]

    # Raw features dataframe for M-P, Persistence, and Fuzzy
    df_features = pd.read_csv(config["data"]["features_file"])
    val_raw = df_features[(df_features["year"] >= 2011) & (df_features["year"] <= 2018)].reset_index(drop=True)
    test_raw = df_features[(df_features["year"] >= 2019) & (df_features["year"] <= 2026)].reset_index(drop=True)

    models_dict = {}

    # 1. Persistence Baseline
    persist = PersistenceBaseline()
    persist.fit(df_features[df_features["year"] <= 2010])
    p_persist_val = persist.predict_proba(val_raw)[:, 1]
    p_persist_test = persist.predict_proba(test_raw)[:, 1]
    models_dict["Persistence Baseline"] = {"val": p_persist_val, "test": p_persist_test, "is_prob": False}

    # 2. McCulloch-Pitts Neuron (Exp 1)
    mp = McCullochPittsNeuron(tmax_threshold=42.0, anomaly_threshold=3.5, rh_min_threshold=25.0, wind_threshold=15.0, theta=2.0)
    p_mp_val = mp.predict_proba(val_raw)[:, 1]
    p_mp_test = mp.predict_proba(test_raw)[:, 1]
    models_dict["McCulloch-Pitts (Exp 1)"] = {"val": p_mp_val, "test": p_mp_test, "is_prob": False}

    # 3. Perceptron Rule (Exp 4)
    perceptron = PerceptronRuleClassifier(eta=0.01, max_epochs=100, random_state=42)
    perceptron.fit(X_train, y_train)
    p_perc_val = perceptron.predict_proba(X_val)[:, 1]
    p_perc_test = perceptron.predict_proba(X_test)[:, 1]
    models_dict["Perceptron Rule (Exp 4)"] = {"val": p_perc_val, "test": p_perc_test, "is_prob": False}

    # 4. ADALINE Delta Rule (Exp 5)
    adaline = DeltaRuleADALINE(eta=0.001, max_epochs=200, random_state=42)
    adaline.fit(X_train, y_train)
    p_ada_val = adaline.predict_proba(X_val)[:, 1]
    p_ada_test = adaline.predict_proba(X_test)[:, 1]
    models_dict["Delta Rule ADALINE (Exp 5)"] = {"val": p_ada_val, "test": p_ada_test, "is_prob": True}

    # 5. Logistic Regression
    lr = fit_logistic_regression(X_train, y_train)
    p_lr_val = lr.predict_proba(X_val)[:, 1]
    p_lr_test = lr.predict_proba(X_test)[:, 1]
    models_dict["Logistic Regression"] = {"val": p_lr_val, "test": p_lr_test, "is_prob": True}

    # 6. NumPy EBPA Network (Exp 2, 6, 7)
    ebpa = EBPANetwork(n_input=X_train.shape[1], n_hidden=32, activation="bipolar_sigmoid", seed=42)
    ebpa_path = Path("models/ebpa_weights.npz")
    if ebpa_path.exists():
        ebpa.load_weights(str(ebpa_path))
    else:
        ebpa.train(X_train, y_train, X_val, y_val, max_epochs=200, patience=30)
    p_ebpa_val = ebpa.predict_proba(X_val)
    p_ebpa_test = ebpa.predict_proba(X_test)
    models_dict["NumPy EBPA (Exp 7)"] = {"val": p_ebpa_val, "test": p_ebpa_test, "is_prob": True}

    # 7. 10-Seed Deep MLP Ensemble (via pure-NumPy export)
    mlp_ensemble = NumpyMLPEnsemble("models/ensemble_weights.npz")
    p_mlp_val = mlp_ensemble.predict_proba(X_val)
    p_mlp_test = mlp_ensemble.predict_proba(X_test)
    models_dict["10-Seed Deep MLP Ensemble"] = {"val": p_mlp_val, "test": p_mlp_test, "is_prob": True}

    # 8. Mamdani Fuzzy Controller (CO4)
    fuzzy = MamdaniFuzzyController()
    p_fuz_val = fuzzy.predict_proba(val_raw)
    p_fuz_test = fuzzy.predict_proba(test_raw)
    models_dict["Mamdani Fuzzy (CO4)"] = {"val": p_fuz_val, "test": p_fuz_test, "is_prob": True}

    # 9. GA Neuro-Fuzzy Hybrid (CO5)
    with open("models/threshold.json", "r") as f:
        ga_params = json.load(f)
    w_ann = ga_params["w_ann"]
    w_ebpa = ga_params["w_ebpa"]
    w_fuzzy = ga_params["w_fuzzy"]
    tau_star = ga_params["threshold"]

    p_hybrid_val = w_ann * p_mlp_val + w_ebpa * p_ebpa_val + w_fuzzy * p_fuz_val
    p_hybrid_test = w_ann * p_mlp_test + w_ebpa * p_ebpa_test + w_fuzzy * p_fuz_test
    models_dict["GA Neuro-Fuzzy Hybrid (CO5)"] = {
        "val": p_hybrid_val,
        "test": p_hybrid_test,
        "is_prob": True,
        "tau_star": tau_star,
    }

    return {
        "splits": splits,
        "models_dict": models_dict,
        "val_raw": val_raw,
        "test_raw": test_raw,
        "df_features": df_features,
        "ga_params": ga_params,
        "mlp_ensemble": mlp_ensemble,
        "ebpa": ebpa,
        "fuzzy": fuzzy,
    }


def compute_metrics_for_predictions(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5, is_prob: bool = True) -> Dict[str, float]:
    """Compute comprehensive performance metrics."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    if is_prob and len(np.unique(probs)) > 2:
        roc = roc_auc_score(y_true, probs)
        pr = average_precision_score(y_true, probs)
        brier = brier_score_loss(y_true, probs)
    else:
        # Step / binary models
        roc = roc_auc_score(y_true, preds)
        pr = average_precision_score(y_true, preds)
        brier = float(np.mean((preds - y_true) ** 2))

    return {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1 Score": round(float(f1), 4),
        "ROC-AUC": round(float(roc), 4),
        "PR-AUC": round(float(pr), 4),
        "Brier Score": round(float(brier), 4),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def build_evaluation_tables(data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate final model comparison tables on Validation and locked Test splits."""
    models_dict = data["models_dict"]
    y_val = data["splits"]["y_val"]
    y_test = data["splits"]["y_test"]
    tau_star = data["ga_params"]["threshold"]

    val_records = []
    test_records = []

    for name, m_info in models_dict.items():
        is_p = m_info["is_prob"]
        t = tau_star if "tau_star" in m_info else 0.5

        # Validation
        val_m = compute_metrics_for_predictions(y_val, m_info["val"], threshold=t, is_prob=is_p)
        val_records.append({
            "Model": name,
            "Threshold": t,
            **val_m
        })

        # Test
        test_m = compute_metrics_for_predictions(y_test, m_info["test"], threshold=t, is_prob=is_p)
        test_records.append({
            "Model": name,
            "Threshold": t,
            **test_m
        })

    df_val = pd.DataFrame(val_records)
    df_test = pd.DataFrame(test_records)

    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_val.to_csv(tables_dir / "final_model_comparison_val.csv", index=False)
    df_test.to_csv(tables_dir / "final_model_comparison_test.csv", index=False)
    logger.info("Saved final validation and test comparison tables.")

    return df_val, df_test


def evaluate_sensitivity(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Evaluate model sensitivity against alternative heatwave definitions on the Test set:
    - Primary: imd_reanalysis_plains
    - Candidate B: climatology_p85_42c
    - Reference: imd_strict_raw
    """
    test_raw = data["test_raw"]
    models_dict = data["models_dict"]

    targets = {
        "Primary (imd_reanalysis_plains)": test_raw["heatwave_day"].values,
        "Candidate B (climatology_p85_42c)": test_raw["climatology_p85_42c"].values,
        "Reference (imd_strict_raw)": test_raw["imd_strict_raw"].values,
    }

    eval_models = [
        "Persistence Baseline",
        "Logistic Regression",
        "10-Seed Deep MLP Ensemble",
        "Mamdani Fuzzy (CO4)",
        "GA Neuro-Fuzzy Hybrid (CO5)",
    ]

    records = []
    for tgt_name, y_true in targets.items():
        n_pos = int(np.sum(y_true))
        pct_pos = round(float(n_pos / len(y_true) * 100), 2)

        for m_name in eval_models:
            m_info = models_dict[m_name]
            t = m_info.get("tau_star", 0.5)
            metrics = compute_metrics_for_predictions(y_true, m_info["test"], threshold=t, is_prob=m_info["is_prob"])
            records.append({
                "Target Definition": tgt_name,
                "Positives (Test)": f"{n_pos}/248 ({pct_pos}%)",
                "Model": m_name,
                "Threshold": t,
                "F1 Score": metrics["F1 Score"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "ROC-AUC": metrics["ROC-AUC"],
                "PR-AUC": metrics["PR-AUC"],
            })

    df_sens = pd.DataFrame(records)
    df_sens.to_csv(Path("results/tables/sensitivity_analysis.csv"), index=False)
    logger.info("Saved sensitivity analysis table to results/tables/sensitivity_analysis.csv")
    return df_sens


# ---------------------------------------------------------------------------
# Figure Generators (200 DPI)
# ---------------------------------------------------------------------------

def plot_all_figures(data: Dict[str, Any]) -> None:
    """Generate all 12 figures saved at 200 DPI in results/figures/."""
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating all 12 publication figures at 200 DPI...")

    df_features = data["df_features"]
    models_dict = data["models_dict"]
    y_test = data["splits"]["y_test"]
    y_val = data["splits"]["y_val"]
    test_raw = data["test_raw"]

    # -----------------------------------------------------------------------
    # Fig 1: Study Area & Meteorological Context
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    # Comparative climate profile: Nagpur (Inland CHZ) vs Mumbai (Coastal)
    regions = ["Nagpur (Vidarbha, Inland CHZ)", "Mumbai (Konkan Coast)"]
    may_tmax = [41.59, 33.40]
    may_rh = [33.2, 72.5]
    peak_tmax = [46.90, 36.20]

    x = np.arange(len(regions))
    width = 0.25

    rects1 = ax.bar(x - width, may_tmax, width, label="May Mean Tmax (°C)", color="#d95f02")
    rects2 = ax.bar(x, peak_tmax, width, label="Historical Peak Tmax (°C)", color="#e41a1c")
    rects3 = ax.bar(x + width, may_rh, width, label="May Mean Relative Humidity (%)", color="#377eb8")

    ax.set_ylabel("Value (°C / %)")
    ax.set_title("Study Area Meteorological Profile: Nagpur Urban Agglomeration vs Coastal Maharashtra")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontweight="bold")
    ax.legend(frameon=True)
    ax.set_ylim(0, 85)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(figures_dir / "fig01_study_area.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 2: Nagpur May Tmax Seasonal Curve & Climatology
    # -----------------------------------------------------------------------
    climatology_df = pd.read_csv("data/processed/climatology_normals.csv")
    may_clim = climatology_df[climatology_df["month"] == 5].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    days = may_clim["day"].values
    normal = may_clim["normal_tmax"].values
    p85 = may_clim["p85_tmax"].values
    p90 = may_clim["p90_tmax"].values

    ax.plot(days, normal, "b-", linewidth=2.2, label="WMO 1991–2020 Normal Tmax")
    ax.plot(days, p85, "orange", linestyle="--", linewidth=2.0, label="85th Percentile (P85)")
    ax.plot(days, p90, "r-.", linewidth=2.0, label="90th Percentile (P90)")
    ax.fill_between(days, normal, p90, color="orange", alpha=0.15, label="Severe Departure Zone (≥ +3.5°C)")
    ax.axhline(43.5, color="darkred", linestyle=":", linewidth=2, label="Absolute Heatwave Threshold (43.5°C)")

    ax.set_xlabel("Day of May")
    ax.set_ylabel("Maximum Temperature (°C)")
    ax.set_title("Nagpur May Climatological Normals & Extreme Thresholds (WMO 1991–2020)")
    ax.set_xticks(np.arange(1, 32, 2))
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(figures_dir / "fig02_tmax_climatology.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 3: Heat Wave Days per May (1981–2026) & Chronological Splits
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
    yearly_hw = df_features.groupby("year")["heatwave_day"].sum()
    years = yearly_hw.index.values
    counts = yearly_hw.values

    # Colors by split
    colors = []
    for y in years:
        if y <= 2010:
            colors.append("#2b83ba")  # Train: blue
        elif y <= 2018:
            colors.append("#fdae61")  # Val: orange
        else:
            colors.append("#d7191c")  # Test: red

    ax.bar(years, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(2010.5, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(2018.5, color="black", linestyle="--", linewidth=1.5)

    ax.text(1995, 23, "Training Split\n(1981–2010: 24.3% HW)", ha="center", fontsize=11, fontweight="bold", color="#2b83ba")
    ax.text(2014.5, 23, "Validation\n(2011–2018: 35.5%)", ha="center", fontsize=10, fontweight="bold", color="#d95f02")
    ax.text(2022.5, 23, "Locked Test\n(2019–2026: 14.9%)", ha="center", fontsize=10, fontweight="bold", color="#d7191c")

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of May Heat Wave Days")
    ax.set_title("Nagpur May Heat Wave Days per Year (1981–2026) across Chronological Splits")
    ax.set_xlim(1980, 2027)
    ax.set_ylim(0, 27)
    plt.tight_layout()
    fig.savefig(figures_dir / "fig03_positives_per_year.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 4: Feature Correlation Heatmap with Target
    # -----------------------------------------------------------------------
    feature_cols = data["splits"]["feature_names"]
    corr_df = df_features[feature_cols + ["heatwave_day"]].corr()
    target_corr = corr_df[["heatwave_day"]].drop("heatwave_day").sort_values(by="heatwave_day", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 9), dpi=200)
    sns.heatmap(target_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.6, vmax=0.8, cbar_kws={"label": "Pearson Correlation with Day T Heatwave"}, ax=ax)
    ax.set_title("Feature Correlations with Target (heatwave_day)")
    plt.tight_layout()
    fig.savefig(figures_dir / "fig04_feature_correlation.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 5: Training & Validation Loss Curves across 10 Seeds + Exp 2 Study
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

    # Load 10-seed histories
    with open("models/ensemble_train_history.json", "r") as f:
        hist_data = json.load(f)
    
    # Pad to equal length
    all_train_losses = [h["train_loss"] for h in hist_data["histories"]]
    all_val_losses = [h["val_loss"] for h in hist_data["histories"]]
    max_len = max(len(l) for l in all_train_losses)

    padded_train = np.full((len(all_train_losses), max_len), np.nan)
    padded_val = np.full((len(all_val_losses), max_len), np.nan)
    for i in range(len(all_train_losses)):
        padded_train[i, :len(all_train_losses[i])] = all_train_losses[i]
        padded_val[i, :len(all_val_losses[i])] = all_val_losses[i]

    mean_tr = np.nanmean(padded_train, axis=0)
    std_tr = np.nanstd(padded_train, axis=0)
    mean_val = np.nanmean(padded_val, axis=0)
    std_val = np.nanstd(padded_val, axis=0)

    epochs = np.arange(1, max_len + 1)
    ax1.plot(epochs, mean_tr, "b-", label="Train Loss (Mean 10 Seeds)", linewidth=2)
    ax1.fill_between(epochs, mean_tr - std_tr, mean_tr + std_tr, color="blue", alpha=0.15)
    ax1.plot(epochs, mean_val, "r-", label="Val Loss (Mean 10 Seeds)", linewidth=2)
    ax1.fill_between(epochs, mean_val - std_val, mean_val + std_val, color="red", alpha=0.15)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Class-Weighted BCE Loss")
    ax1.set_title("10-Seed Deep MLP Convergence (Mean ± 1 SD)")
    ax1.set_ylim(0.2, 1.2)
    ax1.legend(loc="upper right", frameon=True)

    # Exp 2 Activation Comparison
    act_df = pd.read_csv("results/tables/exp2_activation_comparison.csv")
    x_act = np.arange(len(act_df))
    ax2.bar(x_act - 0.2, act_df["val_pr_auc"], 0.35, label="Val PR-AUC", color="#1b9e77")
    ax2.bar(x_act + 0.2, act_df["val_f1"], 0.35, label="Val F1 Score", color="#d95f02")
    ax2.set_xticks(x_act)
    ax2.set_xticklabels(act_df["activation"], fontweight="bold")
    ax2.set_ylabel("Validation Score")
    ax2.set_title("KJSCE Exp 2: Hidden Activation Function Study")
    ax2.set_ylim(0.70, 0.95)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(figures_dir / "fig05_training_curves.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 6: ROC and PR Curves on Locked Test Set (2019–2026)
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    curve_models = [
        ("Delta Rule (ADALINE)", models_dict["Delta Rule ADALINE (Exp 5)"]["test"], "green", "--"),
        ("Logistic Regression", models_dict["Logistic Regression"]["test"], "gray", ":"),
        ("NumPy EBPA (Exp 7)", models_dict["NumPy EBPA (Exp 7)"]["test"], "purple", "-."),
        ("10-Seed Deep MLP", models_dict["10-Seed Deep MLP Ensemble"]["test"], "blue", "-"),
        ("Mamdani Fuzzy (CO4)", models_dict["Mamdani Fuzzy (CO4)"]["test"], "orange", "--"),
        ("GA Neuro-Fuzzy (CO5)", models_dict["GA Neuro-Fuzzy Hybrid (CO5)"]["test"], "red", "-"),
    ]

    for label, probs, col, ls in curve_models:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=col, linestyle=ls, linewidth=2, label=f"{label} (AUC={roc_val:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_val = auc(rec, prec)
        ax2.plot(rec, prec, color=col, linestyle=ls, linewidth=2, label=f"{label} (AUC={pr_val:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Test Set ROC Curves (2019–2026)")
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    # Baseline positive rate on Test
    no_skill = float(np.sum(y_test) / len(y_test))
    ax2.axhline(no_skill, color="k", linestyle="--", alpha=0.5, label=f"No Skill Baseline ({no_skill:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Test Set Precision-Recall Curves (2019–2026)")
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    fig.savefig(figures_dir / "fig06_roc_pr_curves.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 7: Test Set Confusion Matrices (Key Models)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), dpi=200)
    cm_display_models = [
        ("McCulloch-Pitts", models_dict["McCulloch-Pitts (Exp 1)"]["test"], 0.5),
        ("ADALINE (Delta Rule)", models_dict["Delta Rule ADALINE (Exp 5)"]["test"], 0.5),
        ("10-Seed Deep MLP", models_dict["10-Seed Deep MLP Ensemble"]["test"], 0.5),
        ("GA Neuro-Fuzzy", models_dict["GA Neuro-Fuzzy Hybrid (CO5)"]["test"], data["ga_params"]["threshold"]),
    ]

    for ax, (m_label, probs, t) in zip(axes, cm_display_models):
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No HW", "HW"], yticklabels=["No HW", "HW"])
        ax.set_title(f"{m_label}\n(Threshold={t:.2f})", fontsize=10.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    fig.savefig(figures_dir / "fig07_confusion_matrix.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 8: Reliability / Calibration Diagram
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    for label, probs, col in [
        ("10-Seed Deep MLP Ensemble", models_dict["10-Seed Deep MLP Ensemble"]["test"], "blue"),
        ("Mamdani Fuzzy Controller", models_dict["Mamdani Fuzzy (CO4)"]["test"], "orange"),
        ("GA Neuro-Fuzzy Hybrid", models_dict["GA Neuro-Fuzzy Hybrid (CO5)"]["test"], "red"),
        ("Logistic Regression", models_dict["Logistic Regression"]["test"], "gray"),
    ]:
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=8, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color=col, label=label)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Empirical Frequency")
    ax.set_title("Probability Calibration Diagram (Locked Test Set 2019–2026)")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(figures_dir / "fig08_calibration.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 9: Permutation Feature Importance (Validation Set)
    # -----------------------------------------------------------------------
    mlp = data["mlp_ensemble"]
    X_val = data["splits"]["X_val"]
    feature_names = data["splits"]["feature_names"]
    
    baseline_pr_auc = average_precision_score(y_val, mlp.predict_proba(X_val))
    importances = []

    np.random.seed(42)
    for j in range(X_val.shape[1]):
        X_perm = X_val.copy()
        X_perm[:, j] = np.random.permutation(X_perm[:, j])
        drop = baseline_pr_auc - average_precision_score(y_val, mlp.predict_proba(X_perm))
        importances.append(drop)

    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    sns.barplot(data=imp_df, y="Feature", x="Importance", hue="Feature", palette="mako", legend=False, ax=ax)
    ax.set_xlabel("PR-AUC Drop upon Permutation (Validation Set)")
    ax.set_title("Top 12 Most Critical Meteorological Features (Deep MLP Ensemble)")
    plt.tight_layout()
    fig.savefig(figures_dir / "fig09_permutation_importance.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 10: Test-Set Timeline Error Analysis (2019–2026)
    # -----------------------------------------------------------------------
    daily_df = pd.read_csv("data/processed/daily.csv")
    test_merged = test_raw.merge(daily_df[["date", "tmax"]], on="date", how="left")
    
    fig, ax = plt.subplots(figsize=(13, 5), dpi=200)
    test_dates = pd.to_datetime(test_merged["date"])
    actual_tmax = test_merged["tmax"].values
    y_test_arr = y_test
    hybrid_preds = (models_dict["GA Neuro-Fuzzy Hybrid (CO5)"]["test"] >= data["ga_params"]["threshold"]).astype(int)

    tp_mask = (y_test_arr == 1) & (hybrid_preds == 1)
    fp_mask = (y_test_arr == 0) & (hybrid_preds == 1)
    fn_mask = (y_test_arr == 1) & (hybrid_preds == 0)

    ax.plot(test_dates, actual_tmax, color="gray", alpha=0.6, linewidth=1, label="Actual Tmax (°C)")
    ax.scatter(test_dates[tp_mask], actual_tmax[tp_mask], color="green", s=40, zorder=4, label=f"True Positive ({np.sum(tp_mask)})")
    ax.scatter(test_dates[fp_mask], actual_tmax[fp_mask], color="orange", marker="^", s=45, zorder=4, label=f"False Alarm ({np.sum(fp_mask)})")
    ax.scatter(test_dates[fn_mask], actual_tmax[fn_mask], color="red", marker="x", s=50, zorder=4, label=f"Missed Heatwave ({np.sum(fn_mask)})")
    ax.axhline(43.5, color="darkred", linestyle=":", alpha=0.7, label="43.5°C Plains Threshold")

    ax.set_ylabel("Maximum Temperature (°C)")
    ax.set_title("Locked Test-Set Error Timeline (May 2019–2026): GA Neuro-Fuzzy Forecasts")
    ax.legend(loc="upper left", frameon=True, ncol=5)
    plt.tight_layout()
    fig.savefig(figures_dir / "fig10_error_analysis.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 11: Mamdani Fuzzy Membership Functions & Control Surface
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200)
    
    # 1. Tmax Lag 1
    x_tmax = np.linspace(35, 48, 200)
    axes[0, 0].plot(x_tmax, trapmf(x_tmax, (-100, -100, 38, 40.5)), "b-", label="Normal (<39°C)")
    axes[0, 0].plot(x_tmax, trimf(x_tmax, (39, 41.5, 43.5)), "orange", label="Hot (39–43.5°C)")
    axes[0, 0].plot(x_tmax, trapmf(x_tmax, (42, 44, 100, 100)), "r-", label="Extreme (>42.5°C)")
    axes[0, 0].set_title("Antecedent 1: Tmax Lag 1 (°C)")
    axes[0, 0].set_ylabel("Degree of Membership")
    axes[0, 0].legend(loc="upper left", fontsize=8.5)

    # 2. Tmax Anomaly Lag 1
    x_anom = np.linspace(-2, 7, 200)
    axes[0, 1].plot(x_anom, trapmf(x_anom, (-100, -100, 1, 2)), "b-", label="Normal (<1.5°C)")
    axes[0, 1].plot(x_anom, trimf(x_anom, (1, 2.5, 4)), "orange", label="Elevated (1–4°C)")
    axes[0, 1].plot(x_anom, trapmf(x_anom, (3, 4.5, 100, 100)), "r-", label="Severe (>3.5°C)")
    axes[0, 1].set_title("Antecedent 2: Tmax Anomaly Lag 1 (°C)")
    axes[0, 1].legend(loc="upper left", fontsize=8.5)

    # 3. Minimum Relative Humidity Lag 1
    x_rh = np.linspace(5, 60, 200)
    axes[1, 0].plot(x_rh, trapmf(x_rh, (-100, -100, 12, 20)), "r-", label="Arid / Dry (<18%)")
    axes[1, 0].plot(x_rh, trimf(x_rh, (15, 25, 38)), "orange", label="Moderate (15–35%)")
    axes[1, 0].plot(x_rh, trapmf(x_rh, (30, 45, 100, 100)), "b-", label="Humid (>30%)")
    axes[1, 0].set_title("Antecedent 3: Min Relative Humidity Lag 1 (%)")
    axes[1, 0].set_ylabel("Degree of Membership")
    axes[1, 0].legend(loc="upper right", fontsize=8.5)

    # 4. Consequent Heatwave Risk
    x_risk = np.linspace(0, 100, 200)
    axes[1, 1].plot(x_risk, trapmf(x_risk, (-10, -10, 15, 35)), "g-", label="Low Risk")
    axes[1, 1].plot(x_risk, trimf(x_risk, (25, 50, 75)), "orange", label="Watch")
    axes[1, 1].plot(x_risk, trapmf(x_risk, (65, 85, 110, 110)), "r-", label="Heatwave Warning")
    axes[1, 1].set_title("Consequent: Defuzzified Risk Index [0, 100]")
    axes[1, 1].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(figures_dir / "fig11_fuzzy_membership.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 12: Genetic Algorithm Fitness Convergence across 50 Generations
    # -----------------------------------------------------------------------
    with open("models/ga_convergence_history.json", "r") as f:
        ga_hist = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    gens = ga_hist["generation"]
    best_f = ga_hist["best_fitness"]
    mean_f = ga_hist["mean_fitness"]

    ax.plot(gens, best_f, "r-", linewidth=2.2, label="Champion Population Fitness (Validation F1)")
    ax.plot(gens, mean_f, "b--", linewidth=1.8, label="Mean Population Fitness")

    ax.scatter([gens[0], gens[-1]], [best_f[0], best_f[-1]], color="red", zorder=4)
    ax.annotate(f"Initial: {best_f[0]:.4f}", (gens[0], best_f[0]), textcoords="offset points", xytext=(10, -10), fontsize=9)
    ax.annotate(f"Optimal: {best_f[-1]:.4f}", (gens[-1], best_f[-1]), textcoords="offset points", xytext=(-60, 5), fontsize=9, fontweight="bold")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Validation F1 Score)")
    ax.set_title("Genetic Algorithm Optimization of Neuro-Fuzzy Ensembles (CO5)")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(figures_dir / "fig12_ga_convergence.png")
    plt.close(fig)

    logger.info("All 12 publication figures successfully generated at 200 DPI in results/figures/.")


def run_full_evaluation(config: dict) -> None:
    """Run full evaluation suite, generate all tables, and produce all 12 figures."""
    data = load_all_predictions(config)
    build_evaluation_tables(data)
    evaluate_sensitivity(data)
    plot_all_figures(data)
    logger.info("Evaluation and figure generation fully complete.")


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_full_evaluation(cfg)
