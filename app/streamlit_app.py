# LEGACY/OPTIONAL: original experiment only. Use src.experiment and frontend/ for Heatline.
# Old weights, label definitions and reported metrics are not current assessed evidence.
"""
Interactive Streamlit Web Application for Nagpur May Heat Wave Prediction using ANNs.
Course: Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP), K. J. Somaiya School of Engineering.

Implements the demo application (Replay, Leakage-Safe Day T-1 What-If Simulator,
Model & Method Evaluation, and KJSCE Syllabus & Viva Guide) using 100% pure-NumPy inference.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.ann_numpy import EBPANetwork
from src.export_model import NumpyMLPEnsemble
from src.features import FEATURE_COLS
from src.fuzzy_controller import MamdaniFuzzyController
from src.sc_baselines import McCullochPittsNeuron

st.set_page_config(
    page_title="Nagpur May Heatwave ANN Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    """Load configuration, datasets, scaler parameters, and pure-NumPy ANN models."""
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df_daily = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "daily.csv")
    df_features = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features.csv")
    df_normals = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "climatology_normals.csv")

    with open(PROJECT_ROOT / "models" / "scaler.json", "r", encoding="utf-8") as f:
        scaler_data = json.load(f)

    with open(PROJECT_ROOT / "models" / "threshold.json", "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    mlp_ensemble = NumpyMLPEnsemble(str(PROJECT_ROOT / "models" / "ensemble_weights.npz"))

    ebpa_net = EBPANetwork(n_input=len(FEATURE_COLS), n_hidden=32, activation="bipolar_sigmoid")
    ebpa_net.load_weights(str(PROJECT_ROOT / "models" / "ebpa_weights.npz"))

    fuzzy_ctrl = MamdaniFuzzyController()
    mp_neuron = McCullochPittsNeuron()

    return {
        "config": config,
        "daily": df_daily,
        "features": df_features,
        "normals": df_normals,
        "scaler_mean": np.array(scaler_data["mean"], dtype=np.float64),
        "scaler_scale": np.array(scaler_data["scale"], dtype=np.float64),
        "threshold_data": threshold_data,
        "mlp": mlp_ensemble,
        "ebpa": ebpa_net,
        "fuzzy": fuzzy_ctrl,
        "mp": mp_neuron,
    }


def scale_feature_vector(x_raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Standardize a raw 26-feature vector using Training split (1981-2010) statistics."""
    return (np.asarray(x_raw, dtype=np.float64) - mean) / scale


def run_all_models_on_row(row_series: pd.Series, bundle: dict) -> dict:
    """Compute predictions across the primary ANN Ensemble, NumPy EBPA, Fuzzy, and M-P Neuron."""
    x_raw = row_series[FEATURE_COLS].to_numpy(dtype=np.float64)
    x_scaled = scale_feature_vector(x_raw, bundle["scaler_mean"], bundle["scaler_scale"])

    p_mlp = float(bundle["mlp"].predict_proba(x_scaled)[0])
    p_ebpa = float(bundle["ebpa"].predict_proba(x_scaled)[0])
    fuzzy_score, p_fuzzy = bundle["fuzzy"].evaluate_sample(
        float(row_series["tmax_lag1"]),
        float(row_series["tmax_anomaly_lag1"]),
        float(row_series["rh_min_lag1"]),
    )

    th = bundle["threshold_data"]
    p_hybrid = th["w_ann"] * p_mlp + th["w_ebpa"] * p_ebpa + th["w_fuzzy"] * p_fuzzy
    tau_star = float(th["threshold"])

    # McCulloch-Pitts synaptic breakdown
    e1 = int(row_series["tmax_lag1"] >= 42.0)
    e2 = int(row_series["tmax_anomaly_lag1"] >= 3.5)
    e3 = int(row_series["rh_min_lag1"] <= 25.0)
    inh = int(row_series["wind_mean_lag1"] >= 15.0)
    mp_net = (e1 + e2 + e3) - 2 * inh
    mp_fire = int(mp_net >= 2)

    return {
        "p_mlp": p_mlp,
        "p_ebpa": p_ebpa,
        "p_fuzzy": p_fuzzy,
        "fuzzy_score": fuzzy_score,
        "p_hybrid": p_hybrid,
        "tau_star": tau_star,
        "mp_synapses": {
            "e1_tmax": e1,
            "e2_anom": e2,
            "e3_dry": e3,
            "inh_wind": inh,
            "net": mp_net,
            "fires": mp_fire,
        },
    }


def main():
    bundle = load_artifacts()
    df_features = bundle["features"]
    df_daily = bundle["daily"]
    df_normals = bundle["normals"]
    th_info = bundle["threshold_data"]

    # Sidebar
    with st.sidebar:
        st.markdown("### ☀️ Nagpur Heatwave ANN")
        st.caption(
            "**Problem Statement:** Design and develop an artificial neural network to predict "
            "heat wave in May month considering climate conditions of a specific demographic area."
        )
        st.divider()
        st.markdown("**Demographic Study Area**")
        st.markdown("- **City:** Nagpur (Vidarbha, MH)\n- **Coords:** `21.1458°N, 79.0882°E`\n- **Elevation:** `311 m` (Inland CHZ)")
        st.markdown("**Chronological Dataset (1981–2026)**")
        st.markdown(
            "- **Train (1981–2010):** 930 May days\n"
            "- **Val (2011–2018):** 248 May days\n"
            "- **Test (2019–2026):** 248 May days"
        )
        st.divider()
        decision_mode = st.radio(
            "Primary Decision Engine",
            ["10-Seed Deep MLP ANN (Default)", "Pure-NumPy EBPA ANN (Exp 7)", "Neuro-Fuzzy Hybrid (ANN + Fuzzy)"],
            index=0,
        )
        st.caption("Inference runs 100% in pure NumPy using exported weights (`models/ensemble_weights.npz`).")

    st.title("☀️ May Heat Wave Prediction in Nagpur Using an Artificial Neural Network")
    st.markdown(
        "**K. J. Somaiya School of Engineering** · Soft Computing Lab Mini Project · "
        "*Leakage-Safe Next-Day Forecasting from Day $T-1$ Atmospheric Conditions*"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 1. Historical May Replay (1981–2026)",
        "🎛️ 2. Interactive What-If Simulator (Day T-1)",
        "📊 3. Model, Method & Benchmark Results",
        "📘 4. Syllabus Alignment & Viva Guide",
    ])

    # =========================================================================
    # TAB 1: HISTORICAL MAY REPLAY
    # =========================================================================
    with tab1:
        st.subheader("Historical May Day Replay (1981–2026)")
        st.markdown(
            "Select any historical year and May calendar day $T$. The Artificial Neural Network uses **only** "
            "atmospheric conditions observed up to the end of day $T-1$ (plus deterministic WMO climatology for day $T$) "
            "to predict whether day $T$ will experience a heat wave."
        )

        col_sel1, col_sel2, col_sel3 = st.columns([1.2, 1.2, 1.6])
        with col_sel1:
            sel_year = st.selectbox("Select Year (1981–2026)", list(range(2026, 1980, -1)), index=2)
        with col_sel2:
            sel_day = st.slider("Select Day of May (T)", min_value=1, max_value=31, value=18)
        with col_sel3:
            split_label = (
                "Training Split (1981–2010)" if sel_year <= 2010
                else ("Validation Split (2011–2018)" if sel_year <= 2018 else "Locked Test Split (2019–2026)")
            )
            st.info(f"**Selected Date:** `{sel_year}-05-{sel_day:02d}`  \n**Partition:** {split_label}")

        row_match = df_features[(df_features["year"] == sel_year) & (df_features["day"] == sel_day)].iloc[0]
        daily_match = df_daily[(df_daily["year"] == sel_year) & (df_daily["month"] == 5) & (df_daily["day"] == sel_day)].iloc[0]
        preds = run_all_models_on_row(row_match, bundle)

        if "10-Seed" in decision_mode:
            active_prob = preds["p_mlp"]
            active_thresh = 0.50
            active_name = "10-Seed Deep MLP ANN"
        elif "EBPA" in decision_mode:
            active_prob = preds["p_ebpa"]
            active_thresh = 0.50
            active_name = "Pure-NumPy EBPA ANN"
        else:
            active_prob = preds["p_hybrid"]
            active_thresh = preds["tau_star"]
            active_name = "Neuro-Fuzzy Hybrid"

        pred_label = int(active_prob >= active_thresh)
        actual_label = int(row_match["heatwave_day"])
        actual_tmax = float(daily_match["tmax"])
        normal_tmax = float(row_match["normal_tmax_T"])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            f"{active_name} Prob.",
            f"{active_prob:.1%}",
            f"Threshold: {active_thresh:.2f}",
        )
        m2.metric(
            "Model Decision (Day T)",
            "🔥 HEAT WAVE" if pred_label == 1 else "✅ NORMAL",
            "Correct Forecast" if pred_label == actual_label else "Mismatch vs Actual",
            delta_color="normal" if pred_label == actual_label else "inverse",
        )
        m3.metric(
            "Actual Outcome (Day T)",
            "🔥 HEAT WAVE" if actual_label == 1 else "✅ NORMAL",
            f"Actual Tmax: {actual_tmax:.1f}°C (Norm: {normal_tmax:.1f}°C)",
        )
        m4.metric(
            "NumPy EBPA ANN (Exp 7)",
            f"{preds['p_ebpa']:.1%}",
            "Heat Wave" if preds["p_ebpa"] >= 0.5 else "Normal",
        )
        m5.metric(
            "Mamdani Fuzzy Risk (CO4)",
            f"{preds['fuzzy_score']:.1f} / 100",
            f"Prob equiv: {preds['p_fuzzy']:.1%}",
        )

        # 10-day window trajectory chart around selected May day
        st.markdown("#### 📈 10-Day Temperature Trajectory & Climatological Thresholds")
        year_may_df = df_daily[(df_daily["year"] == sel_year) & (df_daily["month"] == 5)].sort_values("day")
        may_normals = df_normals[df_normals["month"] == 5].sort_values("day")

        start_d = max(1, sel_day - 7)
        end_d = min(31, sel_day + 3)
        sub_daily = year_may_df[(year_may_df["day"] >= start_d) & (year_may_df["day"] <= end_d)]
        sub_norm = may_normals[(may_normals["day"] >= start_d) & (may_normals["day"] <= end_d)]

        fig_replay, ax_r = plt.subplots(figsize=(10, 3.6), dpi=150)
        ax_r.plot(sub_daily["day"], sub_daily["tmax"], "o-", color="#d7191c", linewidth=2.2, label=f"Observed Tmax ({sel_year})")
        ax_r.plot(sub_norm["day"], sub_norm["normal_tmax"], "--", color="#2b83ba", linewidth=1.8, label="WMO 1991–2020 Normal")
        ax_r.plot(sub_norm["day"], sub_norm["p90_tmax"], ":", color="#fdae61", linewidth=1.8, label="90th Percentile (P90)")
        ax_r.axhline(43.5, color="darkred", linestyle="-.", alpha=0.7, label="43.5°C Plains Heatwave Threshold")
        ax_r.axvline(sel_day, color="black", linestyle="--", alpha=0.6, label=f"Target Day T (May {sel_day})")
        ax_r.set_xlabel("Day of May")
        ax_r.set_ylabel("Max Temperature (°C)")
        ax_r.set_title(f"Nagpur May {sel_year} Temperature Progression (Window: May {start_d} to May {end_d})")
        ax_r.legend(loc="best", frameon=True, fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_replay)
        plt.close(fig_replay)

        # Feature inspection table
        with st.expander("🔍 Inspect All 26 Leakage-Safe Input Features (Known at End of Day T-1)", expanded=True):
            feat_disp = pd.DataFrame({
                "Feature Name": FEATURE_COLS,
                "Raw Value (at T-1 / Clim T)": [round(float(row_match[c]), 4) for c in FEATURE_COLS],
                "Standardized Z-Score (ANN Input)": np.round(
                    scale_feature_vector(row_match[FEATURE_COLS].to_numpy(), bundle["scaler_mean"], bundle["scaler_scale"]), 3
                ),
            })
            st.dataframe(feat_disp, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: INTERACTIVE WHAT-IF SIMULATOR (LEAKAGE-SAFE DAY T-1)
    # =========================================================================
    with tab2:
        st.subheader("Interactive What-If Simulator (Day T-1 Atmospheric Controls)")
        st.markdown(
            "Start from the currently selected May day in Tab 1 (`"
            f"{sel_year}-05-{sel_day:02d}"
            "`) and perturb yesterday's ($T-1$) temperature, humidity, wind speed, and hot-streak length. "
            "**All dependent engineered features (`tmax_anomaly_lag1`, `tmax_trend3`, `tmax_roll3`, `tmax_roll7`, `dtr_lag1`) "
            "are dynamically recomputed on the fly.**"
        )

        sim_col1, sim_col2 = st.columns([1.3, 1.7])
        with sim_col1:
            st.markdown("##### 🎚️ Adjust Day T-1 Weather Observations")
            s_tmax_lag1 = st.slider(
                "Yesterday Max Temp: `tmax_lag1` (°C)",
                min_value=32.0, max_value=48.0, value=float(row_match["tmax_lag1"]), step=0.2
            )
            s_tmin_lag1 = st.slider(
                "Yesterday Min Temp: `tmin_lag1` (°C)",
                min_value=20.0, max_value=36.0, value=float(row_match["tmin_lag1"]), step=0.2
            )
            s_rh_mean_lag1 = st.slider(
                "Yesterday Mean Humidity: `rh_mean_lag1` (%)",
                min_value=10.0, max_value=80.0, value=float(row_match["rh_mean_lag1"]), step=1.0
            )
            s_rh_min_lag1 = st.slider(
                "Yesterday Min Humidity: `rh_min_lag1` (%)",
                min_value=4.0, max_value=60.0, value=float(row_match["rh_min_lag1"]), step=1.0
            )
            s_wind_lag1 = st.slider(
                "Yesterday Mean Wind Speed: `wind_mean_lag1` (km/h)",
                min_value=4.0, max_value=30.0, value=float(row_match["wind_mean_lag1"]), step=0.5
            )
            s_streak = st.slider(
                "Prior Consecutive Heatwave Days: `hot_streak_len`",
                min_value=0, max_value=12, value=int(row_match["hot_streak_len"]), step=1
            )

        # Dynamically recompute dependent features (strictly leakage-safe)
        sim_row = row_match.copy()
        delta_tmax = s_tmax_lag1 - float(row_match["tmax_lag1"])
        sim_row["tmax_lag1"] = s_tmax_lag1
        sim_row["tmin_lag1"] = min(s_tmin_lag1, s_tmax_lag1 - 2.0)
        sim_row["tmean_lag1"] = round(0.55 * s_tmax_lag1 + 0.45 * sim_row["tmin_lag1"], 2)
        sim_row["rh_mean_lag1"] = s_rh_mean_lag1
        sim_row["rh_min_lag1"] = min(s_rh_min_lag1, s_rh_mean_lag1)
        sim_row["wind_mean_lag1"] = s_wind_lag1
        sim_row["hot_streak_len"] = s_streak

        # Recompute dependent engineered features
        sim_row["dtr_lag1"] = round(s_tmax_lag1 - sim_row["tmin_lag1"], 2)
        sim_row["tmax_trend3"] = round(s_tmax_lag1 - float(row_match["tmax_lag3"]), 2)
        sim_row["tmax_roll3"] = round((s_tmax_lag1 + float(row_match["tmax_lag2"]) + float(row_match["tmax_lag3"])) / 3.0, 4)
        sim_row["tmax_roll7"] = round(float(row_match["tmax_roll7"]) + delta_tmax / 7.0, 4)
        normal_lag1 = float(row_match["tmax_lag1"]) - float(row_match["tmax_anomaly_lag1"])
        sim_row["tmax_anomaly_lag1"] = round(s_tmax_lag1 - normal_lag1, 2)
        sim_row["apparent_tmax_lag1"] = round(float(row_match["apparent_tmax_lag1"]) + delta_tmax + 0.05 * (s_rh_mean_lag1 - float(row_match["rh_mean_lag1"])), 2)

        sim_preds = run_all_models_on_row(sim_row, bundle)

        with sim_col2:
            st.markdown("##### 🧠 Live Neural Network & Soft Computing Response")
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "10-Seed Deep MLP ANN",
                f"{sim_preds['p_mlp']:.1%}",
                f"{sim_preds['p_mlp'] - preds['p_mlp']:+.1%} vs baseline",
            )
            c2.metric(
                "Pure-NumPy EBPA ANN",
                f"{sim_preds['p_ebpa']:.1%}",
                f"{sim_preds['p_ebpa'] - preds['p_ebpa']:+.1%} vs baseline",
            )
            c3.metric(
                "Mamdani Fuzzy Risk",
                f"{sim_preds['fuzzy_score']:.1f} / 100",
                f"{sim_preds['p_fuzzy'] - preds['p_fuzzy']:+.1%} vs baseline",
            )

            st.markdown("###### Recomputed Engineered Features (Dynamic Cascade)")
            recomp_df = pd.DataFrame([
                {"Engineered Feature": "tmax_anomaly_lag1 (°C)", "Original": row_match["tmax_anomaly_lag1"], "Simulated": sim_row["tmax_anomaly_lag1"]},
                {"Engineered Feature": "tmax_trend3 (°C)", "Original": row_match["tmax_trend3"], "Simulated": sim_row["tmax_trend3"]},
                {"Engineered Feature": "tmax_roll3 (°C)", "Original": row_match["tmax_roll3"], "Simulated": sim_row["tmax_roll3"]},
                {"Engineered Feature": "tmax_roll7 (°C)", "Original": row_match["tmax_roll7"], "Simulated": sim_row["tmax_roll7"]},
                {"Engineered Feature": "dtr_lag1 (°C)", "Original": row_match["dtr_lag1"], "Simulated": sim_row["dtr_lag1"]},
            ])
            st.dataframe(recomp_df, use_container_width=True, hide_index=True)

            mp_s = sim_preds["mp_synapses"]
            st.markdown("###### Exp 1: McCulloch–Pitts Biological Synapse Status")
            st.markdown(
                f"- **Excitatory 1 (`tmax_lag1 >= 42°C`):** `{'ACTIVE (+1)' if mp_s['e1_tmax'] else 'Inactive (0)'}`  \n"
                f"- **Excitatory 2 (`anomaly_lag1 >= +3.5°C`):** `{'ACTIVE (+1)' if mp_s['e2_anom'] else 'Inactive (0)'}`  \n"
                f"- **Excitatory 3 (`rh_min_lag1 <= 25%`):** `{'ACTIVE (+1)' if mp_s['e3_dry'] else 'Inactive (0)'}`  \n"
                f"- **Inhibitory Veto (`wind_mean_lag1 >= 15 km/h`):** `{'INHIBITING (-2)' if mp_s['inh_wind'] else 'Inactive (0)'}`  \n"
                f"- **Net Soma Potential:** `{mp_s['net']} (Threshold θ = 2)` $\\rightarrow$ "
                f"**{'🔥 NEURON FIRES (1)' if mp_s['fires'] else '✅ QUIESCENT (0)'}**"
            )

    # =========================================================================
    # TAB 3: MODEL, METHOD & BENCHMARK RESULTS
    # =========================================================================
    with tab3:
        st.subheader("Model Architecture, Evaluation Metrics & Diagnostic Figures")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 🏆 Locked Test Set Benchmark (2019–2026, 248 May Days)")
            test_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "final_model_comparison_test.csv")
            st.dataframe(test_df, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("##### 🧪 Validation Set Benchmark (2011–2018, 248 May Days)")
            val_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "final_model_comparison_val.csv")
            st.dataframe(val_df, use_container_width=True, hide_index=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("##### ⚡ KJSCE Exp 2: Hidden Activation Function Study (`src/ann_numpy.py`)")
            act_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "exp2_activation_comparison.csv")
            st.dataframe(act_df, use_container_width=True, hide_index=True)
        with col_d:
            st.markdown("##### 🎯 Label Sensitivity Analysis (Alternative Heatwave Definitions)")
            sens_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "sensitivity_analysis.csv")
            st.dataframe(sens_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("##### 🖼️ Diagnostic & Publication Figures Explorer (200 DPI)")
        fig_options = {
            "Fig 06 — Test ROC & Precision-Recall Curves (ANN vs Baselines)": "fig06_roc_pr_curves.png",
            "Fig 07 — Test Confusion Matrices": "fig07_confusion_matrix.png",
            "Fig 08 — Probability Calibration / Reliability Diagram": "fig08_calibration.png",
            "Fig 05 — 10-Seed ANN Training Loss Curves & Exp 2 Activation Study": "fig05_training_curves.png",
            "Fig 09 — Permutation Feature Importance (Top 12 Predictors)": "fig09_permutation_importance.png",
            "Fig 10 — Test Set Timeline Error Analysis (2019–2026)": "fig10_error_analysis.png",
            "Fig 01 — Study Area Profile: Nagpur vs Coastal Mumbai": "fig01_study_area.png",
            "Fig 02 — WMO 1991–2020 May Climatological Normals & Thresholds": "fig02_tmax_climatology.png",
            "Fig 03 — Heat Wave Days per May (1981–2026) & Split Boundaries": "fig03_positives_per_year.png",
            "Fig 04 — Feature Correlation Heatmap with Heatwave Target": "fig04_feature_correlation.png",
            "Fig 11 — Mamdani Fuzzy Membership Functions (Module 4 / CO4)": "fig11_fuzzy_membership.png",
            "Fig 12 — Evolutionary Ensemble Optimization Curve (Module 5 / CO5)": "fig12_ga_convergence.png",
        }
        selected_fig_title = st.selectbox("Choose Figure to Inspect", list(fig_options.keys()))
        fig_path = PROJECT_ROOT / "results" / "figures" / fig_options[selected_fig_title]
        if fig_path.exists():
            st.image(str(fig_path), caption=selected_fig_title, use_container_width=True)

    # =========================================================================
    # TAB 4: SYLLABUS ALIGNMENT & VIVA GUIDE
    # =========================================================================
    with tab4:
        guide_path = PROJECT_ROOT / "PROJECT_GUIDE.md"
        if guide_path.exists():
            st.markdown(guide_path.read_text(encoding="utf-8"))
        else:
            st.info("PROJECT_GUIDE.md not found.")


if __name__ == "__main__":
    main()
