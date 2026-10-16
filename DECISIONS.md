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

---

## Decision D005: Open-Meteo ERA5-Land Grid Coordinates & Elevation
- **Status:** Accepted
- **Context:** Target coordinates for Nagpur Urban Agglomeration are `latitude: 21.1458, longitude: 79.0882`. Open-Meteo returns the nearest reanalysis grid cell center and model orography.
- **Decision:** Grid cell coordinates returned by Open-Meteo ERA5-Land are:
  - **Grid Latitude:** 21.099998° N
  - **Grid Longitude:** 79.100006° E (~5.2 km displacement from target point, well within urban footprint)
  - **Grid Elevation:** 311.0 m (matches Nagpur's plateau elevation of ~310 m)
- **Completeness:** All 8 requested hourly variables are 100% available with zero missing values across 1981–2026.

---

## Decision D006: Reanalysis Model Selection (ERA5 for Full 8-Variable Coverage)
- **Status:** Accepted
- **Context:** `PROJECT_SPEC.md` Section 2.1 specifies: *"Probe hourly variables (temperature_2m, relative_humidity_2m, dew_point_2m, apparent_temperature, surface_pressure, wind_speed_10m, shortwave_radiation, soil_moisture_0_to_7cm). If any variable is missing in ERA5-Land, fetch from ERA5 and log it in DECISIONS.md."*
- **Observation:** Probing the Open-Meteo ERA5-Land archive endpoint revealed that `apparent_temperature`, `surface_pressure`, `wind_speed_10m`, and `shortwave_radiation` return `null` in the Open-Meteo ERA5-Land dataset for this region. Conversely, Open-Meteo `era5` (0.25° resolution, grid lat 21.25°, lon 79.00°, elevation 311.0 m) provides 100% non-null coverage for all 8 variables.
- **Decision:** As prescribed in `PROJECT_SPEC.md`, switched the reanalysis model to `era5` to obtain 100% complete hourly records for all 8 variables across all 46 years (1981–2026).

---

## Decision D007: Primary Target Heat Wave Label Selection
- **Status:** Accepted
- **Context:** `PROJECT_SPEC.md` Section 2.3 defines a selection protocol among Candidate A (`imd_reanalysis_plains`), Candidate B (`climatology_p85_42c`), and Reference (`imd_strict_raw`). The primary target must yield a Training split (1981–2010) positive rate between 15% and 35%, ideally closest to 22%.
- **Evaluation Results:**
  - Candidate A (`imd_reanalysis_plains`): **24.30% positive rate (226/930 days in Train)**, 35.48% in Val, 14.92% in Test, 24.61% overall.
  - Candidate B (`climatology_p85_42c`): 14.95% positive rate in Train (below the 15% floor).
  - Reference (`imd_strict_raw`): 5.91% positive rate in Train (severe class imbalance, unsuitable for stable ANN training without heavy synthetic sampling).
- **Decision:** Selected **`imd_reanalysis_plains`** as the primary `heatwave_day` target. It satisfies all criteria, sits within 2.3% of the 22% ideal target, and provides balanced positive cases across historical decades.



