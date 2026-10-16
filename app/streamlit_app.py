"""
Nagpur May Heat Wave Prediction — Interactive Soft Computing System.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).
Location: Nagpur Urban Agglomeration (Vidarbha, Central India Core Heatwave Zone).

Pure lightweight implementation: Zero PyTorch imports at runtime.
Uses:
- NumpyMLPEnsemble (src.export_model)
- EBPANetwork (src.ann_numpy)
- MamdaniFuzzyController (src.fuzzy_controller)
- McCullochPittsNeuron (src.sc_baselines)
"""

import json
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.export_model import NumpyMLPEnsemble
from src.ann_numpy import EBPANetwork
from src.fuzzy_controller import MamdaniFuzzyController
from src.sc_baselines import McCullochPittsNeuron

# Page Configuration
st.set_page_config(
    page_title="Nagpur Heatwave Soft Computing System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #495057;
        margin-top: 4px;
    }
    .badge-alert {
        display: inline-block;
        background-color: #dc3545;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-normal {
        display: inline-block;
        background-color: #28a745;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-synapse-active {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-synapse-inactive {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_datasets():
    """Load preprocessed features, daily meteorological records, and normal climatology."""
    features_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features.csv")
    daily_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "daily.csv")
    with open(PROJECT_ROOT / "models" / "scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)
    with open(PROJECT_ROOT / "models" / "threshold.json", "r", encoding="utf-8") as f:
        threshold_cfg = json.load(f)
    return features_df, daily_df, scaler, threshold_cfg


@st.cache_resource
def load_models():
    """Load lightweight pure-NumPy inference engines (zero PyTorch imports)."""
    mlp_ensemble = NumpyMLPEnsemble(str(PROJECT_ROOT / "models" / "ensemble_weights.npz"))
    
    ebpa_net = EBPANetwork(n_input=26)
    ebpa_net.load_weights(str(PROJECT_ROOT / "models" / "ebpa_weights.npz"))
    
    fuzzy_ctrl = MamdaniFuzzyController()
    mp_neuron = McCullochPittsNeuron()
    return mlp_ensemble, ebpa_net, fuzzy_ctrl, mp_neuron


features_df, daily_df, scaler, threshold_cfg = load_datasets()
mlp_ensemble, ebpa_net, fuzzy_ctrl, mp_neuron = load_models()

FEATURE_COLS = scaler["feature_names"]
SCALER_MEAN = np.array(scaler["mean"], dtype=np.float64)
SCALER_SCALE = np.array(scaler["scale"], dtype=np.float64)

W_ANN = threshold_cfg["w_ann"]
W_EBPA = threshold_cfg["w_ebpa"]
W_FUZZY = threshold_cfg["w_fuzzy"]
TAU_STAR = threshold_cfg["threshold"]


def standardize_vector(feat_dict: Dict[str, float]) -> np.ndarray:
    """Standardize a single dictionary of features according to training scaler."""
    vec = np.array([feat_dict[col] for col in FEATURE_COLS], dtype=np.float64)
    return (vec - SCALER_MEAN) / SCALER_SCALE


# Sidebar Header
st.sidebar.image("https://img.shields.io/badge/KJSCE-Soft_Computing_Lab-blue.svg")
st.sidebar.title("Nagpur Heatwave ANN")
st.sidebar.markdown(
    """
    **Autonomous Soft Computing System for Extreme Weather Forecasting**  
    *Location:* Nagpur Plateau, Central India (21.1458°N, 79.0882°E)  
    *Primary Climatology:* WMO 1991–2020 Normals  
    *Lead Time:* Next-Day Operational Warning (T+1)
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **GA Neuro-Fuzzy Ensembling Parameters:**
    - $w_{{\\text{{ANN}}}}$: `{W_ANN:.4f}`
    - $w_{{\\text{{EBPA}}}}$: `{W_EBPA:.4f}`
    - $w_{{\\text{{Fuzzy}}}}$: `{W_FUZZY:.4f}`
    - Decision Threshold $\\tau^*$: `{TAU_STAR:.4f}`
    """
)
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 Built with 100% pure NumPy inference. No PyTorch / CUDA runtime overhead required."
)

# App Header
st.title("🔥 Nagpur May Heat Wave Prediction System")
st.markdown(
    "**Course Project: Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech Computer Engineering)**  \n"
    "Integrating McCulloch-Pitts Logic, Error Backpropagation (EBPA), Deep MLP Ensembles, Mamdani Fuzzy Inference, and Genetic Algorithms."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📅 Historical May Replay (1981–2026)",
        "🧪 Interactive What-If Simulator",
        "📊 KJSCE Soft Computing Lab Explorer",
        "📐 Mathematical Foundations & Viva Guide",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1: Historical May Replay
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Historical May Replay: Observation vs Soft Computing Forecasts")
    st.markdown(
        "Select any year and day between **1981 and 2026** to inspect atmospheric conditions, "
        "verify ground-truth thermal outcomes, and observe model alert consensus."
    )

    c_year, c_day, c_space = st.columns([2, 2, 4])
    with c_year:
        year_sel = st.selectbox(
            "Select Historical Year",
            options=sorted(features_df["year"].unique(), reverse=True),
            index=2,  # Default to 2024
        )
    with c_day:
        day_sel = st.slider("Select Day of May", min_value=1, max_value=31, value=24)

    target_date = f"{year_sel}-05-{day_sel:02d}"
    row_feat = features_df[features_df["date"] == target_date]
    row_daily = daily_df[daily_df["date"] == target_date]

    if row_feat.empty or row_daily.empty:
        st.warning(f"No meteorological records found for date {target_date}.")
    else:
        feat_data = row_feat.iloc[0].to_dict()
        daily_data = row_daily.iloc[0].to_dict()

        # Compute model inference
        X_scaled = standardize_vector(feat_data).reshape(1, -1)
        p_mlp = float(mlp_ensemble.predict_proba(X_scaled)[0])
        p_ebpa = float(ebpa_net.predict_proba(X_scaled)[0])
        fuzzy_risk, p_fuzzy = fuzzy_ctrl.evaluate_sample(
            tmax_lag1=float(feat_data["tmax_lag1"]),
            tmax_anomaly_lag1=float(feat_data["tmax_anomaly_lag1"]),
            rh_min_lag1=float(feat_data["rh_min_lag1"]),
        )
        p_hybrid = W_ANN * p_mlp + W_EBPA * p_ebpa + W_FUZZY * p_fuzzy
        is_hybrid_alert = p_hybrid >= TAU_STAR

        actual_tmax = float(daily_data["tmax"])
        actual_hw = bool(feat_data["imd_reanalysis_plains"])
        wmo_norm = float(feat_data["normal_tmax_T"])

        # Display Metrics Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            hw_badge = (
                '<span class="badge-alert">🔥 HEATWAVE DAY</span>'
                if actual_hw
                else '<span class="badge-normal">✅ NORMAL DAY</span>'
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Actual Day Tmax (Day T)</div>
                    <div class="metric-value">{actual_tmax:.1f} °C</div>
                    <div class="metric-sub">{hw_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Deep MLP Ensemble</div>
                    <div class="metric-value">{p_mlp * 100:.1f}%</div>
                    <div class="metric-sub">10-Seed Average Prob</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Pure-NumPy EBPA</div>
                    <div class="metric-value">{p_ebpa * 100:.1f}%</div>
                    <div class="metric-sub">Bipolar Sigmoid (Exp 2)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Mamdani Fuzzy Risk</div>
                    <div class="metric-value">{fuzzy_risk:.1f}/100</div>
                    <div class="metric-sub">Centroid Prob: {p_fuzzy * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m5:
            alert_badge = (
                '<span class="badge-alert">🚨 ALERT TRIGGERED</span>'
                if is_hybrid_alert
                else '<span class="badge-normal">🛡️ SAFE (NO ALERT)</span>'
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">GA Neuro-Fuzzy Hybrid</div>
                    <div class="metric-value">{p_hybrid * 100:.1f}%</div>
                    <div class="metric-sub">{alert_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Historical Trajectory Visualization
        st.subheader(f"May {year_sel} Temperature Trajectory (Nagpur 21.15°N, 79.09°E)")

        month_daily = daily_df[
            (daily_df["year"] == year_sel) & (daily_df["month"] == 5)
        ].sort_values("day")

        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)
        ax.plot(
            month_daily["day"],
            month_daily["tmax"],
            marker="o",
            color="#d9534f",
            linewidth=2,
            label="Actual Daily Tmax (°C)",
        )
        
        # Merge normal_tmax_T from features
        month_feat = features_df[
            (features_df["year"] == year_sel) & (features_df["month"] == 5)
        ].sort_values("day")
        if not month_feat.empty:
            ax.plot(
                month_feat["day"],
                month_feat["normal_tmax_T"],
                linestyle="--",
                color="#0275d8",
                linewidth=1.8,
                label="WMO 1991–2020 Normal",
            )

        ax.axhline(
            43.5,
            color="#d9534f",
            linestyle=":",
            alpha=0.8,
            linewidth=1.5,
            label="IMD Severe Heatwave Benchmark (43.5°C)",
        )

        # Highlight selected day
        ax.scatter(
            [day_sel],
            [actual_tmax],
            color="#292b2c" if not actual_hw else "#b02a37",
            s=180,
            zorder=5,
            edgecolors="black",
            linewidth=2,
            label=f"Selected Day ({target_date})",
        )

        ax.set_title(
            f"May {year_sel} Thermal Regime & Selected Forecast Verification Day",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Day of May", fontsize=10)
        ax.set_ylabel("Maximum Temperature (°C)", fontsize=10)
        ax.set_xticks(range(1, 32, 2))
        ax.set_ylim(32, 50)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Lagged Atmospheric Features Table
        with st.expander(f"📋 Prior-Day Meteorological Inputs (T-1 Conditions) for {target_date}"):
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                st.write(f"**Tmax (T-1):** `{feat_data['tmax_lag1']:.1f} °C`")
                st.write(f"**Tmin (T-1):** `{feat_data['tmin_lag1']:.1f} °C`")
                st.write(f"**Diurnal Range (DTR):** `{feat_data['dtr_lag1']:.1f} °C`")
                st.write(f"**Climatological Anomaly:** `{feat_data['tmax_anomaly_lag1']:+.2f} °C`")
            with c_f2:
                st.write(f"**Minimum Relative Humidity:** `{feat_data['rh_min_lag1']:.1f} %`")
                st.write(f"**Mean Relative Humidity:** `{feat_data['rh_mean_lag1']:.1f} %`")
                st.write(f"**Mean Wind Speed:** `{feat_data['wind_mean_lag1']:.1f} km/h`")
                st.write(f"**Mean Surface Pressure:** `{feat_data['pressure_mean_lag1']:.1f} hPa`")
            with c_f3:
                st.write(f"**Accumulated Solar Radiation:** `{feat_data['solar_sum_lag1']:.0f} kJ/m²`")
                st.write(f"**Soil Moisture (0-7 cm):** `{feat_data['soil_moisture_lag1']:.3f} m³/m³`")
                st.write(f"**Consecutive Hot Streak:** `{int(feat_data['hot_streak_len'])} days`")
                st.write(f"**WMO Baseline Normal:** `{feat_data['normal_tmax_T']:.1f} °C`")


# ---------------------------------------------------------------------------
# TAB 2: Interactive What-If Simulator
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Interactive Atmospheric Simulator: What-If Stress Testing")
    st.markdown(
        "Manipulate critical land-atmosphere boundary layer parameters to simulate advection, "
        "turbulent ventilation, or prolonged desiccation over the Vidarbha plateau."
    )

    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        sim_tmax = st.slider("Prior-Day Tmax (tmax_lag1) [°C]", 35.0, 48.0, 43.5, 0.1)
        sim_tmin = st.slider("Prior-Day Tmin (tmin_lag1) [°C]", 20.0, 35.0, 28.5, 0.1)
    with c_s2:
        sim_rh = st.slider("Prior-Day Min Humidity (rh_min_lag1) [%]", 4.0, 60.0, 16.0, 0.5)
        sim_wind = st.slider("Prior-Day Mean Wind (wind_mean_lag1) [km/h]", 5.0, 30.0, 11.0, 0.5)
    with c_s3:
        sim_streak = st.slider("Hot Streak Length [Days >= 42°C]", 0, 10, 3, 1)
        sim_normal = st.slider("WMO Baseline Normal Tmax [°C]", 40.0, 44.5, 42.6, 0.1)

    # Dynamically derive dependent features
    sim_anomaly = sim_tmax - sim_normal
    sim_dtr = sim_tmax - sim_tmin
    sim_tmean = (sim_tmax + sim_tmin) / 2.0
    sim_rh_mean = min(100.0, sim_rh + 20.0)

    # Synthesize complete 26-feature dictionary from baseline reference
    sim_dict = {
        "tmax_lag1": sim_tmax,
        "tmin_lag1": sim_tmin,
        "tmean_lag1": sim_tmean,
        "rh_mean_lag1": sim_rh_mean,
        "rh_min_lag1": sim_rh,
        "dewpoint_mean_lag1": 14.0,
        "apparent_tmax_lag1": sim_tmax + 1.2,
        "wind_mean_lag1": sim_wind,
        "pressure_mean_lag1": 968.5,
        "solar_sum_lag1": 7100.0,
        "soil_moisture_lag1": 0.15,
        "tmax_lag2": sim_tmax - 0.4,
        "tmax_lag3": sim_tmax - 0.8,
        "apparent_tmax_lag2": sim_tmax + 0.8,
        "apparent_tmax_lag3": sim_tmax + 0.4,
        "tmax_roll3": (sim_tmax * 3 - 1.2) / 3.0,
        "tmax_roll7": (sim_tmax * 7 - 3.5) / 7.0,
        "solar_roll3": 7050.0,
        "tmax_trend3": 0.8,
        "tmax_anomaly_lag1": sim_anomaly,
        "dtr_lag1": sim_dtr,
        "pressure_tendency": -0.8,
        "hot_streak_len": float(sim_streak),
        "normal_tmax_T": sim_normal,
        "day_sin": np.sin(2 * np.pi * 140 / 365.25),
        "day_cos": np.cos(2 * np.pi * 140 / 365.25),
    }

    # Standardize feature vector
    X_sim_scaled = standardize_vector(sim_dict).reshape(1, -1)

    # 1. McCulloch-Pitts Neuron Evaluation
    e1 = sim_tmax >= mp_neuron.tmax_threshold
    e2 = sim_anomaly >= mp_neuron.anomaly_threshold
    e3 = sim_rh <= mp_neuron.rh_min_threshold
    inh = sim_wind >= mp_neuron.wind_threshold
    net_mp = (float(e1) + float(e2) + float(e3)) - (mp_neuron.inhibitory_weight * float(inh))
    mp_fired = net_mp >= mp_neuron.theta

    # 2. Continuous Soft Computing Models
    sim_p_mlp = float(mlp_ensemble.predict_proba(X_sim_scaled)[0])
    sim_p_ebpa = float(ebpa_net.predict_proba(X_sim_scaled)[0])
    sim_fuzzy_risk, sim_p_fuzzy = fuzzy_ctrl.evaluate_sample(
        tmax_lag1=sim_tmax,
        tmax_anomaly_lag1=sim_anomaly,
        rh_min_lag1=sim_rh,
    )
    sim_p_hybrid = W_ANN * sim_p_mlp + W_EBPA * sim_p_ebpa + W_FUZZY * sim_p_fuzzy
    sim_alert = sim_p_hybrid >= TAU_STAR

    st.markdown("---")
    st.subheader("⚡ Soft Computing Model Inferences Under Simulated Atmospheric State")

    # M-P Neuron Synaptic Firing Breakdown
    st.markdown("#### 1. McCulloch-Pitts Biological Neuron (Exp 1 — Threshold Logic)")
    badge_e1 = '<span class="badge-synapse-active">⚡ FIRED (+1)</span>' if e1 else '<span class="badge-synapse-inactive">💤 INACTIVE (0)</span>'
    badge_e2 = '<span class="badge-synapse-active">⚡ FIRED (+1)</span>' if e2 else '<span class="badge-synapse-inactive">💤 INACTIVE (0)</span>'
    badge_e3 = '<span class="badge-synapse-active">⚡ FIRED (+1)</span>' if e3 else '<span class="badge-synapse-inactive">💤 INACTIVE (0)</span>'
    badge_inh = '<span class="badge-alert">🛑 VETO (-2)</span>' if inh else '<span class="badge-synapse-active">🛡️ CALM (0)</span>'
    badge_mp = '<span class="badge-alert">🔥 ACTION POTENTIAL</span>' if mp_fired else '<span class="badge-normal">💤 QUIESCENT</span>'

    mp_c1, mp_c2, mp_c3, mp_c4, mp_c5 = st.columns(5)
    with mp_c1:
        st.markdown(
            f"**Excitatory 1:** $T_{{max}} \\ge 42.0^\\circ\\text{{C}}$<br>{badge_e1}",
            unsafe_allow_html=True,
        )
    with mp_c2:
        st.markdown(
            f"**Excitatory 2:** $\\Delta T \\ge 3.5^\\circ\\text{{C}}$<br>{badge_e2}",
            unsafe_allow_html=True,
        )
    with mp_c3:
        st.markdown(
            f"**Excitatory 3:** $RH_{{min}} \\le 25\\%$<br>{badge_e3}",
            unsafe_allow_html=True,
        )
    with mp_c4:
        st.markdown(
            f"**Inhibitory:** $Wind \\ge 15\\text{{ km/h}}$<br>{badge_inh}",
            unsafe_allow_html=True,
        )
    with mp_c5:
        st.markdown(
            f"**Axon Hillock Action:** $\\text{{Net}}={net_mp:.1f}$ ($\\theta=2.0$)<br>{badge_mp}",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 2. Model Ensemble Comparison")
    col_pred1, col_pred2 = st.columns([3, 2])
    with col_pred1:
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4), dpi=150)
        models_label = [
            "M-P Neuron (Exp 1)",
            "Pure-NumPy EBPA (Exp 2/6)",
            "Deep MLP Ensemble (Exp 6)",
            "Mamdani Fuzzy (Exp 3)",
            "GA Neuro-Fuzzy (Exp 7)",
        ]
        probs = [
            float(mp_fired),
            sim_p_ebpa,
            sim_p_mlp,
            sim_p_fuzzy,
            sim_p_hybrid,
        ]
        colors = ["#6c757d", "#17a2b8", "#007bff", "#fd7e14", "#dc3545"]

        bars = ax_bar.barh(models_label, probs, color=colors, height=0.55)
        ax_bar.axvline(
            TAU_STAR,
            color="#dc3545",
            linestyle="--",
            linewidth=1.5,
            label=f"GA Threshold (τ* = {TAU_STAR:.4f})",
        )
        ax_bar.set_xlim(0, 1.05)
        ax_bar.set_xlabel("Predicted Heatwave Probability / Firing State", fontsize=10)
        ax_bar.set_title("Soft Computing Predictions Under Simulated Parameters", fontsize=11, fontweight="bold")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.5)
        ax_bar.legend(loc="lower right", fontsize=9)

        for bar in bars:
            width = bar.get_width()
            ax_bar.text(
                width + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{width * 100:.1f}%",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with col_pred2:
        st.markdown(
            f"""
            <div style="background-color: #f1f3f5; border-radius: 8px; padding: 20px; border-left: 5px solid {'#dc3545' if sim_alert else '#28a745'};">
                <h4 style="margin-top:0;">Ensemble Consensus Status</h4>
                <p><strong>Simulated State:</strong> Tmax={sim_tmax}°C, Anom={sim_anomaly:+.1f}°C, RH={sim_rh}%, Wind={sim_wind} km/h</p>
                <hr style="margin: 10px 0;">
                <p><strong>Deep MLP Ensemble:</strong> <code>{sim_p_mlp * 100:.1f}%</code></p>
                <p><strong>NumPy EBPA Network:</strong> <code>{sim_p_ebpa * 100:.1f}%</code></p>
                <p><strong>Mamdani Fuzzy Risk:</strong> <code>{sim_fuzzy_risk:.1f}/100</code></p>
                <p><strong>GA Neuro-Fuzzy Probability:</strong> <code>{sim_p_hybrid * 100:.1f}%</code></p>
                <h3 style="color: {'#dc3545' if sim_alert else '#28a745'}; margin-bottom: 0;">
                    {'🚨 HEATWAVE ALERT ISSUED' if sim_alert else '✅ NO ALERT (NORMAL REGIME)'}
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# TAB 3: KJSCE Soft Computing Lab Explorer
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("KJSCE Soft Computing Lab Explorer: Syllabus Alignment (Exp 1–7 & CO1–CO5)")
    st.markdown(
        "Demonstrating end-to-end alignment with the **KJSCE Soft Computing Lab Syllabus (SVU-KJSCE 2023_1.0)** "
        "applied to climate informatics and regional hazard mitigation."
    )

    exp_tab_labels = [
        "🏆 Final Model Benchmarks (CO1–CO5)",
        "📈 Exp 2: EBPA Activation Study (CO2)",
        "🎯 Exp 4 & 5: Perceptron vs ADALINE (CO2)",
        "🧬 Exp 7: GA Neuro-Fuzzy (CO5)",
        "🖼️ Publication Figure Gallery",
    ]
    t3_1, t3_2, t3_3, t3_4, t3_5 = st.tabs(exp_tab_labels)

    with t3_1:
        st.markdown("#### Test Split Generalization Benchmark (2019–2026 Out-of-Sample)")
        test_csv_path = PROJECT_ROOT / "results" / "tables" / "final_model_comparison_test.csv"
        if test_csv_path.exists():
            df_test = pd.read_csv(test_csv_path)
            st.dataframe(df_test.style.format(precision=4).highlight_max(subset=["F1 Score", "PR-AUC", "ROC-AUC"], color="#d4edda"), use_container_width=True)
        else:
            st.info("Test comparison table not found.")

        st.markdown("#### Validation Split Benchmark (2011–2018 Hyperparameter Selection)")
        val_csv_path = PROJECT_ROOT / "results" / "tables" / "final_model_comparison_val.csv"
        if val_csv_path.exists():
            df_val = pd.read_csv(val_csv_path)
            st.dataframe(df_val.style.format(precision=4), use_container_width=True)

    with t3_2:
        st.markdown("#### Experiment 2 & 6: EBPA Network Activation Function Study")
        st.markdown(
            "Evaluation of Binary Sigmoid, Bipolar Sigmoid (hyperbolic tangent scaled), and ReLU activations "
            "trained with identical learning rate (η = 0.05), momentum (α = 0.9), and class-weighted BCE loss."
        )
        act_csv_path = PROJECT_ROOT / "results" / "tables" / "exp2_activation_comparison.csv"
        if act_csv_path.exists():
            df_act = pd.read_csv(act_csv_path)
            st.dataframe(df_act.style.format(precision=4).highlight_max(subset=["val_f1", "val_pr_auc", "val_roc_auc"], color="#d4edda"), use_container_width=True)

    with t3_3:
        st.markdown("#### Experiment 4 & 5: Rosenblatt Perceptron vs Widrow-Hoff ADALINE Delta Rule")
        st.markdown(
            "- **Perceptron (Exp 4):** Discrete step activation with updates occurring only upon misclassification.\n"
            "- **ADALINE (Exp 5):** Continuous linear pre-activation gradient descent minimizing Mean Squared Error (MSE)."
        )
        fig_adaline = PROJECT_ROOT / "results" / "figures" / "fig06_adaline_learning_curve.png"
        if fig_adaline.exists():
            st.image(str(fig_adaline), caption="Figure 6: Perceptron vs ADALINE Delta Rule Convergence Dynamics", use_container_width=True)

    with t3_4:
        st.markdown("#### Experiment 7: Genetic Algorithm Optimization of Ensembling Parameters")
        st.markdown(
            "Fitness objective: $F = F_1(\\text{val}) + 0.1 \\cdot \\text{PR-AUC}(\\text{val})$.  \n"
            f"Optimal evolved weights: $w_{{\\text{{ANN}}}} = {W_ANN:.4f}, w_{{\\text{{EBPA}}}} = {W_EBPA:.4f}, w_{{\\text{{Fuzzy}}}} = {W_FUZZY:.4f}, \\tau^* = {TAU_STAR:.4f}$."
        )
        fig_ga = PROJECT_ROOT / "results" / "figures" / "fig10_ga_fitness_progression.png"
        if fig_ga.exists():
            st.image(str(fig_ga), caption="Figure 10: GA Generation-wise Fitness Progression", use_container_width=True)

    with t3_5:
        st.markdown("#### Complete Academic Figure Gallery (12 High-Resolution Plots)")
        fig_files = sorted(list((PROJECT_ROOT / "results" / "figures").glob("fig*.png")))
        fig_names = [f.name for f in fig_files]
        if fig_files:
            selected_fig_name = st.selectbox("Select Publication Figure", options=fig_names)
            selected_path = PROJECT_ROOT / "results" / "figures" / selected_fig_name
            st.image(str(selected_path), caption=selected_fig_name, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 4: Mathematical Foundations & Viva Guide
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Mathematical Foundations & Comprehensive Viva Voce Guide")
    st.markdown(
        "Formal mathematical derivations and oral defense answers structured for external evaluation."
    )

    with st.expander("1. McCulloch-Pitts Biological Neuron Model (Exp 1 — CO1)", expanded=True):
        st.markdown(
            r"""
            **Formulation (Warren McCulloch & Walter Pitts, 1943):**
            Let $X = \{x_1, x_2, \dots, x_n\}$ denote excitatory binary inputs and $Y = \{y_1, y_2, \dots, y_m\}$ denote inhibitory binary inputs.
            The net membrane potential $\text{Net}$ is:
            $$\text{Net} = \sum_{i=1}^{n} w_i x_i - w_{\text{inh}} \sum_{j=1}^{m} y_j$$
            The firing decision (axon action potential) follows an all-or-none threshold rule:
            $$y = \begin{cases} 1 & \text{if } \text{Net} \ge \theta \\ 0 & \text{if } \text{Net} < \theta \end{cases}$$
            *Nagpur Operationalization:*
            $$x_1 = [T_{\text{max}} \ge 42.0^\circ\text{C}], \quad x_2 = [\Delta T_{\text{norm}} \ge 3.5^\circ\text{C}], \quad x_3 = [RH_{\text{min}} \le 25\%]$$
            $$y_{\text{inh}} = [Wind \ge 15.0\text{ km/h}], \quad w_{\text{inh}} = 2.0, \quad \theta = 2.0$$
            """
        )

    with st.expander("2. Single-Layer Perceptron Learning Rule (Exp 4 — CO2)"):
        st.markdown(
            r"""
            **Formulation (Frank Rosenblatt, 1958):**
            Given inputs $\mathbf{x} \in \mathbb{R}^d$ and bipolar targets $t \in \{-1, +1\}$:
            $$\text{net} = \mathbf{w}^T \mathbf{x} + b, \quad y = \text{sgn}(\text{net})$$
            Weight update upon misclassification ($y \ne t$):
            $$\mathbf{w}(t+1) = \mathbf{w}(t) + \eta \, t \, \mathbf{x}, \quad b(t+1) = b(t) + \eta \, t$$
            *Convergence Theorem:* Guaranteed to converge in finite steps if and only if classes are linearly separable in feature space.
            """
        )

    with st.expander("3. ADALINE & Continuous Widrow-Hoff Delta Rule (Exp 5 — CO2)"):
        st.markdown(
            r"""
            **Formulation (Bernard Widrow & Ted Hoff, 1960):**
            ADALINE (Adaptive Linear Neuron) computes weight updates on the continuous pre-activation:
            $$y_{\text{in}} = \mathbf{w}^T \mathbf{x} + b$$
            Minimizes the Sum of Squared Errors (SSE):
            $$E = \frac{1}{2} (t - y_{\text{in}})^2$$
            Gradient descent delta rule:
            $$\Delta \mathbf{w} = -\eta \nabla_{\mathbf{w}} E = \eta (t - y_{\text{in}}) \mathbf{x}$$
            *Distinction from Perceptron:* Updates occur on continuous analog error $(t - y_{\text{in}})$ rather than discrete classification errors $(t - y)$.
            """
        )

    with st.expander("4. Error Backpropagation Algorithm (EBPA) (Exp 2 & 6 — CO2/CO3)"):
        st.markdown(
            r"""
            **Formulation (Rumelhart, Hinton, & Williams, 1986):**
            Two-layer feedforward network with hidden units $z_j$ and output unit $y_k$:
            $$z_{\text{in}, j} = v_{0j} + \sum_i x_i v_{ij}, \quad z_j = f(z_{\text{in}, j})$$
            $$y_{\text{in}, k} = w_{0k} + \sum_j z_j w_{jk}, \quad y_k = g(y_{\text{in}, k})$$
            **Output Layer Error Signal ($\delta_k$):**
            $$\delta_k = (t_k - y_k) g'(y_{\text{in}, k})$$
            For binary cross-entropy with sigmoid activation, $g'(y_{\text{in}}) = y_k (1 - y_k)$, simplifying the gradient:
            $$\frac{\partial L}{\partial w_{jk}} = -\delta_k z_j$$
            **Hidden Layer Error Signal ($\delta_j$):**
            $$\delta_{\text{in}, j} = \sum_k \delta_k w_{jk}, \quad \delta_j = \delta_{\text{in}, j} f'(z_{\text{in}, j})$$
            **Momentum Weight Update:**
            $$\Delta w_{jk}(t) = \eta \delta_k z_j + \alpha \Delta w_{jk}(t-1)$$
            $$\Delta v_{ij}(t) = \eta \delta_j x_i + \alpha \Delta v_{ij}(t-1)$$
            """
        )

    with st.expander("5. Mamdani Fuzzy Inference System (Exp 3 — CO4)"):
        st.markdown(
            r"""
            **Formulation (Lotfi Zadeh 1965, Ebrahim Mamdani 1975):**
            Antecedent membership functions:
            $$\mu_{\text{tri}}(x; a, b, c) = \max\left(0, \min\left(\frac{x-a}{b-a}, \frac{c-x}{c-b}\right)\right)$$
            $$\mu_{\text{trap}}(x; a, b, c, d) = \max\left(0, \min\left(\frac{x-a}{b-a}, 1, \frac{d-x}{d-c}\right)\right)$$
            **Rule Firing via Mamdani Min Operator:**
            $$\alpha_r = \min_{i} \mu_{A_{ir}}(x_i)$$
            **Aggregate Consequent Output:**
            $$\mu_{\text{agg}}(z) = \max_r \min(\alpha_r, \mu_{C_r}(z))$$
            **Centroid Defuzzification (Center of Gravity):**
            $$z^* = \frac{\int z \, \mu_{\text{agg}}(z) \, dz}{\int \mu_{\text{agg}}(z) \, dz}$$
            """
        )

    with st.expander("6. Genetic Algorithm for Neuro-Fuzzy Optimization (Exp 7 — CO5)"):
        st.markdown(
            r"""
            **Formulation (John Holland, 1975):**
            Real-coded chromosome: $\vec{x} = [w_{\text{ANN}}, w_{\text{EBPA}}, w_{\text{Fuzzy}}, \tau]$  
            Normalized constraint: $\sum_{i} w_i = 1, \quad w_i \ge 0$
            **Fitness Function (Class Imbalance Mitigation):**
            $$\text{Fitness}(\vec{x}) = F_1(\text{val}; \tau) + 0.1 \cdot \text{PR-AUC}(\text{val})$$
            **Evolutionary Operators:**
            - *Selection:* Roulette wheel proportional selection based on rank-transformed fitness.
            - *Crossover:* Whole arithmetic crossover: $\mathbf{c}_1 = \gamma \mathbf{p}_1 + (1-\gamma)\mathbf{p}_2$.
            - *Mutation:* Gaussian perturbation with decaying variance $\sigma(g) = \sigma_0 (1 - g/G)$.
            """
        )

    with st.expander("7. High-Yield Viva Voce Questions & Model Answers"):
        st.markdown(
            """
            **Q1: Why is Precision-Recall AUC (PR-AUC) more meaningful than ROC-AUC for Nagpur heatwaves?**  
            *A1:* In May, heatwaves occur on only ~24.3% of days. ROC-AUC incorporates the False Positive Rate ($FP / (FP + TN)$), which is heavily diluted by large numbers of true negatives, artificially inflating ROC-AUC. PR-AUC focuses exclusively on the positive minority class ($TP / (TP + FN)$ vs $TP / (TP + FP)$).

            **Q2: What is the thermodynamic justification for the wind inhibitory synapse?**  
            *A2:* In Vidarbha, extreme heatwaves develop under quasi-stagnant anticyclonic conditions where turbulent boundary-layer mixing is suppressed. Sustained surface winds ($\ge 15\text{ km/h}$) break regional nocturnal inversions and advect cooler maritime or boundary-layer air, preventing extreme cumulative thermal buildup.

            **Q3: How does the Mamdani Fuzzy System complement the Deep MLP?**  
            *A3:* While Deep MLPs excel at discovering complex non-linear interactions across high-dimensional lag matrices, they are susceptible to distribution shift and lack transparent reasoning. The Mamdani controller provides a physically grounded expert prior based on IMD meteorological criteria, guaranteeing baseline conservatism on borderline days.
            """
        )
