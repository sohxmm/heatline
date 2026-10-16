# Architectural & Scientific Decisions Log

This document records key design choices, scientific justifications, and trade-offs for the **Nagpur May Heat Wave Prediction Project** (`nagpur-heatwave-ann`), adhering to KJSCE Soft Computing Lab requirements.

---

## Decision D001: Demographics & Target Location Selection
- **Status:** Accepted
- **Context:** The course problem statement specifies: *"Design and develop an artificial neural network to predict heat wave in May month considering climate conditions of a specific demographic area."*
- **Decision:** Selected **Nagpur Urban Agglomeration (Vidarbha region, Central Maharashtra)**, target coordinates `latitude: 21.1458, longitude: 79.0882`, elevation ~310 m.
- **Rationale:** Nagpur is situated in India's Core Heatwave Zone (CHZ) on the central Deccan plateau. Unlike coastal regions (e.g. Mumbai) where maritime effects dampen extreme heat, Nagpur experiences frequent, prolonged, severe May heatwaves with dry continental advection, making it an ideal demographic domain with strong class balance for machine learning.
- **Alternatives Considered:** Mumbai (too coastal, heatwaves are rare and humidity-driven), Delhi (higher winter/summer variability, but Nagpur has more sustained May heat events).

---

## Decision D002: Reanalysis Data Source Selection (Open-Meteo ERA5-Land)
- **Status:** Accepted
- **Context:** Need 46 years (1981–2026) of consistent, gap-free hourly meteorological records without paid API keys or restrictive station NDAs.
- **Decision:** Use Open-Meteo Historical Archive API querying ECMWF ERA5-Land (0.1° resolution, ~9–11 km grid) with fallback to ERA5 (0.25° resolution).
- **Rationale:** Free, keyless, reproducible, provides all standard thermodynamic variables (2m temperature, dewpoint, humidity, wind, pressure, solar radiation, soil moisture).
- **Alternatives Considered:** IMD gridded data (complex binary format, restricted access, latency), NASA POWER (coarser resolution 0.5°).

---

## Decision D003: Pure-NumPy Inference Architecture for Production & Streamlit
- **Status:** Accepted
- **Context:** Streamlit Cloud and embedded demonstration platforms often fail or exhaust memory when loading large PyTorch/TensorFlow wheels.
- **Decision:** While training is executed with PyTorch (and pure NumPy for EBPA), the trained 10-seed ensemble weights are exported to a lightweight `models/ensemble_weights.npz` and `models/scaler.json`. Inference in `app/streamlit_app.py` is written in pure NumPy.
- **Rationale:** Instant cold-start (< 1s), zero heavy C++ dependencies, 100% reproducible forward pass across any platform.

---

## Decision D004: Strict Chronological Splitting Strategy
- **Status:** Accepted
- **Context:** Meteorological time series possess temporal autocorrelation and multi-decadal warming trends.
- **Decision:** Disallow random or k-fold CV splits. Use strict chronological partitioning:
  - **Training split:** 1981–2010 (30 years, 930 May days)
  - **Validation split:** 2011–2018 (8 years, 248 May days)
  - **Locked Test split:** 2019–2026 (8 years, 248 May days)
- **Rationale:** Simulates real-world operational forecasting where models trained on historical decades predict future seasons without temporal leakage.
