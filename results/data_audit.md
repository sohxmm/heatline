# Climatological Data Audit & Heatwave Target Selection Report
**Project:** Heat Wave Prediction in May for Nagpur (Vidarbha) using Error Backpropagation ANN, Fuzzy Logic & Genetic Optimization  
**Course:** KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP)  
**Location:** Nagpur Urban Agglomeration, Vidarbha, Central Maharashtra, India  
**Audit Date:** October 2026  

---

## 1. Executive Summary

This data audit validates 46 consecutive years (1981–2026) of April–May meteorological reanalysis data for Nagpur, retrieved from the ECMWF ERA5 reanalysis via the Open-Meteo Historical Archive API. All 2,806 calendar days (1,380 April days + 1,426 May days) have 100% data completeness with zero missing records across all 8 thermodynamic variables. 

Three candidate heatwave definitions were evaluated against the standard WMO 1991–2020 30-year climatological baseline. Based on our climatological selection criteria, **Candidate A (`imd_reanalysis_plains`)** was selected as the primary binary prediction target, producing an optimal Training set positive rate of **24.30%** (226 / 930 May days), exceptionally well-aligned with the target 22% benchmark.

---

## 2. Geographic & Model Domain

Nagpur is located on the Deccan plateau in Central India, situated within India's **Core Heatwave Zone (CHZ)**. It experiences severe continental heating with dry north-westerly advection during May.

| Parameter | Specified Target | Reanalysis Grid Cell | Concordance Notes |
|---|---|---|---|
| **Latitude** | 21.1458° N | 21.2500° N | ECMWF ERA5 0.25° grid (~11.6 km offset) |
| **Longitude** | 79.0882° E | 79.0000° E | Well within Nagpur Urban Agglomeration footprint |
| **Elevation** | ~310 m | 311.0 m | Matches physical terrain of Nagpur plateau |
| **Timezone** | Asia/Kolkata | Asia/Kolkata (UTC+05:30) | Standard Indian Standard Time (IST) |
| **Period** | 1981 – 2026 (46 yrs) | 1981 – 2026 (46 yrs) | Gap-free continuous archive |
| **April Days** | 1,380 days (30/yr) | 1,380 days (30/yr) | Reserved exclusively for lag/rolling features |
| **May Target Days** | 1,426 days (31/yr) | 1,426 days (31/yr) | Primary target forecasting universe |

---

## 3. Meteorological Variable Completeness Audit

Hourly data was retrieved and aggregated into local daily variables (midnight-to-midnight IST). Every variable achieved 100.0% completeness (0 nulls out of 2,806 daily records).

| Variable | Description | Units | Missing (%) | May Mean | May Min | May Max |
|---|---|---|---|---|---|---|
| `tmax` | Daily Maximum 2m Air Temperature | °C | 0.0% | 41.59 | 28.30 | 46.90 |
| `tmin` | Daily Minimum 2m Air Temperature | °C | 0.0% | 29.15 | 20.00 | 33.70 |
| `tmean` | Daily Mean 2m Air Temperature | °C | 0.0% | 35.15 | 24.37 | 39.80 |
| `apparent_tmax` | Daily Maximum Apparent Temperature (Heat Index) | °C | 0.0% | 41.40 | 28.70 | 47.70 |
| `rh_mean` | Daily Mean Relative Humidity | % | 0.0% | 33.24 | 14.71 | 74.04 |
| `rh_min` | Daily Minimum Relative Humidity | % | 0.0% | 15.68 | 4.00 | 56.00 |
| `dewpoint_mean` | Daily Mean Dew Point Temperature | °C | 0.0% | 14.18 | -0.19 | 24.96 |
| `wind_mean` | Daily Mean 10m Wind Speed | km/h | 0.0% | 13.91 | 5.37 | 26.65 |
| `pressure_mean` | Daily Mean Surface Pressure | hPa | 0.0% | 967.65 | 954.96 | 976.71 |
| `solar_sum` | Daily Integrated Global Shortwave Radiation | W/m² sum | 0.0% | 6,944.5 | 1,458.0 | 7,934.0 |
| `soil_moisture_mean`| Daily Mean 0–7cm Volumetric Soil Moisture | m³/m³ | 0.0% | 0.2014 | 0.1700 | 0.3204 |

---

## 4. WMO 1991–2020 Climatological Baseline Normals

Under World Meteorological Organization (WMO) standards, climatological normals are calculated over the standard 30-year period (1991–2020) using a centered ±7-calendar-day window (15 calendar days × 30 years = ~450 observations per calendar date).

### Key Baseline Normals across May:
- **May 1:** Normal $T_{\text{max}} = 41.54^\circ\text{C}$, $P_{85} = 43.60^\circ\text{C}$, $P_{90} = 44.00^\circ\text{C}$
- **May 10:** Normal $T_{\text{max}} = 41.81^\circ\text{C}$, $P_{85} = 43.80^\circ\text{C}$, $P_{90} = 44.20^\circ\text{C}$
- **May 20:** Normal $T_{\text{max}} = 42.17^\circ\text{C}$, $P_{85} = 44.10^\circ\text{C}$, $P_{90} = 44.60^\circ\text{C}$
- **May 31:** Normal $T_{\text{max}} = 42.44^\circ\text{C}$, $P_{85} = 44.70^\circ\text{C}$, $P_{90} = 45.20^\circ\text{C}$

The climatological curve exhibits the expected seasonal inland progression, with mean peak temperatures rising by nearly 1.0°C over May as solar declination and surface heating intensify before the onset of the Southwest Monsoon.

---

## 5. Decadal Climate & Temperature Trends (May 1981–2026)

To assess multi-decadal climate shifts and verify the chronological split validity, May metrics were computed across 10-year tranches:

| Decade / Period | May Mean $T_{\text{max}}$ (°C) | May Record Peak (°C) | May Mean $T_{\text{min}}$ (°C) | Diurnal Range (DTR, °C) | Mean RH (%) | Days $\ge 42^\circ\text{C}$ / yr | Days $\ge 44^\circ\text{C}$ / yr | Days $\ge 45^\circ\text{C}$ / yr | Heatwave Days / yr (`imd_reanalysis`) |
|---|---|---|---|---|---|---|---|---|---|
| **1981–1990** | 41.31 | 46.70 | 28.74 | 12.57 | 30.6% | 14.4 | 5.0 | 2.8 | 7.3 |
| **1991–2000** | 41.84 | 46.30 | 29.35 | 12.49 | 30.5% | 17.5 | 5.1 | 1.3 | 8.3 |
| **2001–2010** | 41.95 | 46.90 | 29.66 | 12.29 | 31.4% | 16.7 | 3.9 | 1.4 | 7.0 |
| **2011–2020** | 42.32 | 46.30 | 29.66 | 12.67 | 32.8% | 18.1 | 7.6 | 2.8 | 10.7 |
| **2021–2026** | 39.80 | 45.40 | 27.82 | 11.98 | 45.4% | 8.0 | 2.0 | 0.5 | 3.0 |
| **1981–2026 (All)** | **41.59** | **46.90** | **29.15** | **12.44** | **33.2%** | **15.5** | **5.0** | **1.9** | **7.6** |

### Trend Observations:
1. **Steady Warming from 1981 through 2020:** Mean May $T_{\text{max}}$ increased from 41.31°C (1980s) to 42.32°C (2010s), with heatwave frequency peaking at 10.7 days/year in 2011–2020.
2. **2021–2026 Humidity & Pre-Monsoon Anomaly:** The recent 2021–2026 period showed elevated relative humidity (45.4% vs historical ~31%) and increased convective cloud cover, resulting in cooler surface $T_{\text{max}}$ (39.80°C) and fewer dry heatwave days (3.0/yr). This provides an exceptionally rigorous, out-of-distribution stress test for the locked Test set (2019–2026).

---

## 6. Candidate Heatwave Label Evaluation

Three definitions were evaluated across all 1,426 May days:
1. **Candidate A (`imd_reanalysis_plains`):**  
   $$T_{\text{max}} \ge 43.5^\circ\text{C} \quad \lor \quad (T_{\text{max}} \ge 40.0^\circ\text{C} \land T_{\text{max}} - \text{Normal} \ge 3.5^\circ\text{C})$$
2. **Candidate B (`climatology_p85_42c`):**  
   $$T_{\text{max}} \ge \max(42.0^\circ\text{C}, P_{85,\text{calendar}})$$
3. **Reference (`imd_strict_raw`):**  
   $$T_{\text{max}} \ge 45.0^\circ\text{C} \quad \lor \quad (T_{\text{max}} \ge 40.0^\circ\text{C} \land T_{\text{max}} - \text{Normal} \ge 4.5^\circ\text{C})$$

### Split Distribution Summary:

| Split Partition | May Calendar Days | Candidate A: `imd_reanalysis_plains` | Candidate B: `climatology_p85_42c` | Reference: `imd_strict_raw` |
|---|---|---|---|---|
| **Train (1981–2010)** | 930 | **226 (24.30%)** | 139 (14.95%) | 55 (5.91%) |
| **Validation (2011–2018)**| 248 | **88 (35.48%)** | 51 (20.56%) | 23 (9.27%) |
| **Test (2019–2026)** | 248 | **37 (14.92%)** | 19 (7.66%) | 8 (3.23%) |
| **Overall (1981–2026)** | 1,426 | **351 (24.61%)** | 209 (14.66%) | 86 (6.03%) |

---

## 7. Primary Target Selection Rationale

Based on our climatological selection criteria:
- Primary criterion: Training set (1981–2010) positive rate must fall between **15% and 35%**, targeting closest to **22%**.
- **Candidate A (`imd_reanalysis_plains`)** yields **24.30%** in Train, perfectly satisfying the boundary condition and lying within 2.3% of the ideal 22% rate.
- **Candidate B (`climatology_p85_42c`)** yields **14.95%** in Train, slightly violating the 15% lower bound.
- **Reference (`imd_strict_raw`)** yields **5.91%**, which would introduce severe class imbalance ($>15:1$) leading to poor gradient signal during Error Backpropagation training without synthetic oversampling.

**Decision:** **Candidate A (`imd_reanalysis_plains`)** is officially designated as the primary target `heatwave_day`. Candidate B and Reference will be retained for sensitivity experiments.

---

## 8. Leakage Prevention Protocol

1. **Chronological Barrier:**
   - Training: 1981–2010 (30 years)
   - Validation: 2011–2018 (8 years)
   - Locked Test: 2019–2026 (8 years)
2. **Lagged Feature Formulation:**
   - For predicting heat wave state on Day $T$, all features are restricted to atmospheric observations through Day $T-1$.
   - Automated unit test `tests/test_leakage.py` will perturb all weather features on Day $T$ by $+50.0^\circ\text{C}$ and assert that feature vectors for Day $T$ remain bitwise identical.
