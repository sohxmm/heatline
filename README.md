# Nagpur Heatwave Prediction using Artificial Neural Networks

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Problem Statement

> **"Design and develop an artificial neural network to predict heat wave in May month considering climate conditions of a specific demographic area."**

This project focuses on **Nagpur, Maharashtra** (21.1458°N, 79.0882°E) in the Vidarbha region of Central India. Nagpur experiences extreme pre-monsoon heatwaves in May, where temperatures routinely exceed 42°C–45°C. The objective is to forecast next-day heatwave events using prior-day atmospheric conditions and historical climatological baselines.

---

## Overview & Approach

1. **Meteorological Data (1981–2026):** 46 years of April–May hourly weather data fetched from ECMWF ERA5-Land reanalysis via the Open-Meteo Archive API and aggregated into daily variables (Tmax, Tmin, relative humidity, surface pressure, wind speed, solar radiation, soil moisture).
2. **Climatology & Ground Truth:** WMO 1991–2020 30-year smoothed baseline normals. Ground-truth heatwave labels follow the India Meteorological Department (IMD) criteria for plains ($T_{\text{max}} \ge 40^\circ\text{C}$ with departure $\ge +4.5^\circ\text{C}$, or absolute $T_{\text{max}} \ge 43.5^\circ\text{C}$).
3. **Chronological Splits:**
   - **Training Set:** 1981–2010 (930 May days)
   - **Validation Set:** 2011–2018 (248 May days)
   - **Test Set (Held-out):** 2019–2026 (248 May days)
4. **Feature Engineering:** 26 leakage-safe predictors computed strictly from days $\le T-1$ (lags, 3-day trends, rolling means, diurnal temperature range, pressure tendency, hot streak duration, and WMO day-$T$ normal).

---

## Models Implemented

- **Connectionist Baselines:**
  - **McCulloch-Pitts Neuron:** Handcrafted threshold logic unit with excitatory thermal/humidity inputs and inhibitory wind ventilation veto.
  - **Single-Layer Perceptron:** Rosenblatt's step-activation supervised learning rule on bipolar targets $\{-1, +1\}$.
  - **ADALINE (Delta Rule):** Widrow-Hoff Least Mean Squares (LMS) continuous gradient descent minimizing mean squared error.
  - **Statistical Baselines:** Persistence ($T-1$) and Class-Weighted Logistic Regression.
- **Artificial Neural Networks:**
  - **Pure-NumPy EBPA Network:** From-scratch 2-layer Error Backpropagation Network with explicit forward pass, momentum weight updates, and activation function comparison (Binary Sigmoid, Bipolar Sigmoid, ReLU).
  - **10-Seed Deep MLP Ensemble:** Multi-layer perceptron with Batch Normalization and Dropout trained across 10 random seeds.
- **Fuzzy Logic & Genetic Algorithm:**
  - **Mamdani Fuzzy Controller:** Expert rule-based fuzzy inference system modeling Vidarbha regional thermodynamics (14 rules, centroid defuzzification).
  - **GA Neuro-Fuzzy Hybrid:** Genetic Algorithm optimizing ensemble weights ($w_{\text{ANN}}=0.4082, w_{\text{EBPA}}=0.3910, w_{\text{Fuzzy}}=0.2007$) and decision threshold ($\tau^* = 0.4979$).

---

## Performance on Out-of-Sample Test Set (2019–2026)

| Model | Accuracy | Precision | Recall | F1 Score | PR-AUC | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Persistence Baseline | 92.74% | 0.7568 | 0.7568 | 0.7568 | 0.6090 | 0.8571 |
| McCulloch-Pitts Neuron | 82.26% | 0.4521 | **0.8919** | 0.6000 | 0.4193 | 0.8512 |
| Perceptron Step Rule | 85.08% | 0.5000 | 0.7838 | 0.6105 | 0.4241 | 0.8232 |
| Delta Rule ADALINE | 93.15% | 0.7941 | 0.7297 | 0.7606 | 0.8293 | 0.9620 |
| Logistic Regression | 89.11% | 0.6000 | 0.8108 | 0.6897 | 0.8476 | 0.9636 |
| **Pure-NumPy EBPA** | **93.95%** | **0.8235** | 0.7568 | **0.7887** | 0.8382 | 0.9640 |
| **10-Seed Deep MLP Ensemble** | 90.32% | 0.6383 | 0.8108 | 0.7143 | **0.8749** | 0.9649 |
| Mamdani Fuzzy Controller | 87.90% | 0.5593 | **0.8919** | 0.6875 | 0.7584 | 0.9600 |
| **GA Neuro-Fuzzy Hybrid** | 91.53% | 0.6905 | 0.7838 | 0.7342 | 0.8583 | **0.9668** |

---

## Interactive Streamlit Demo

The repository includes an interactive web application built with Streamlit (`app/streamlit_app.py`) using pure NumPy inference:

- **Tab 1 — Historical May Replay (1981–2026):** Select any year and day of May to inspect actual temperatures, ground-truth label, and individual model predictions alongside the month's temperature trajectory against WMO normals.
- **Tab 2 — Interactive What-If Simulator:** Adjust sliders for prior-day Tmax, Tmin, humidity, wind, and hot streak length to see real-time predictions and live synaptic firing of the McCulloch-Pitts neuron.
- **Tab 3 — Results & Figures Explorer:** View benchmark comparison tables and 12 publication-quality plots (training curves, confusion matrices, ROC/PR curves).
- **Tab 4 — Mathematical Formulations & Viva Guide:** LaTeX equations and theoretical explanations for all models (M-P neuron, Perceptron, ADALINE, EBPA, Fuzzy inference, and GA).

```bash
streamlit run app/streamlit_app.py
```

---

## Getting Started

### 1. Installation

```bash
git clone https://github.com/sohxmm/heatwave-ANN.git
cd heatwave-ANN

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Windows
# source .venv/bin/activate     # On Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web App
```bash
streamlit run app/streamlit_app.py
```

### 3. Run Unit Tests
```bash
pytest
```
*Runs 31 tests verifying feature leakage prevention, data splitting, label calculations, and model inference.*

### 4. Notebooks
Explore the step-by-step notebooks in `notebooks/`:
- `notebooks/01_eda.ipynb`: Climatological analysis, decadal temperature trends, and correlation heatmaps.
- `notebooks/02_modeling.ipynb`: Training and evaluation of baselines, EBPA, Deep MLP ensemble, Fuzzy controller, and GA hybrid.

---

## Project Structure

```
heatwave-ANN/
├── app/
│   └── streamlit_app.py      # Interactive 4-tab Streamlit demo
├── data/
│   ├── processed/            # Daily aggregated weather, normals, and features
│   └── raw/                  # Cached ERA5 hourly weather files
├── models/
│   ├── ebpa_weights.npz      # Pure-NumPy EBPA trained weights
│   ├── ensemble_weights.npz  # Lightweight weights for 10-seed MLP ensemble
│   ├── scaler.json           # Feature normalization parameters (fit on train split)
│   └── threshold.json        # GA-optimized ensembling weights & threshold
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis notebook
│   └── 02_modeling.ipynb     # Modeling and evaluation notebook
├── results/
│   ├── figures/              # Evaluation plots (ROC/PR curves, loss curves, confusion matrices)
│   └── tables/               # Model comparison CSV tables
├── src/
│   ├── ann_numpy.py          # From-scratch NumPy EBPA network
│   ├── baselines.py          # Persistence and Logistic Regression baselines
│   ├── climatology.py        # WMO 1991-2020 baseline normals calculation
│   ├── daily.py              # Hourly to daily meteorological aggregation
│   ├── evaluate.py           # Evaluation scripts and plot generation
│   ├── export_model.py       # NumPy export and inference engine
│   ├── features.py           # 26 engineered meteorological features
│   ├── fetch_data.py         # Open-Meteo ERA5 downloader
│   ├── fuzzy_controller.py   # Mamdani Fuzzy Logic Controller
│   ├── ga_optimizer.py       # Genetic Algorithm ensembling optimizer
│   ├── labels.py             # IMD heatwave label computation
│   ├── sc_baselines.py       # McCulloch-Pitts, Perceptron, and ADALINE
│   ├── split.py              # Chronological train/val/test splitting
│   └── train_ann.py          # 10-seed Deep MLP training
└── tests/                    # 31 unit tests (data leakage, split, models)
```
