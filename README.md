# NCAA → NBA Stick Probability Model (2025)

This project develops a reproducible machine learning pipeline that estimates the probability an NCAA player will **stick** in the NBA.  
A player is considered to have “stuck” if they:

- Achieve a **Combined Metric ≥ 5**, and  
- Play **2+ NBA seasons** averaging **15+ MPG**

All predictive inputs come strictly from **NCAA statistics and attributes**.  
No draft position, NBA performance, team context, or post-college information is used as a feature.

---

## 1. Project Overview

The full pipeline includes:

- Clean and standardized NCAA player data  
- Strict **temporal split** (Train ≤ 2022, Test 2023–2025)  
- Median imputation  
- SMOTE oversampling (training only)  
- Tuned XGBoost classifier  
- Feature selection using LASSO  
- Deployment via an interactive dashboard (Streamlit)

The goal is to identify college profiles that historically translate to long-term NBA rotation value.

---

## 2. Final Feature Set (13 Features)

The final production model uses the following features:

- FTA_per40  
- FT_per40  
- Def_Impact  
- BLK_percent  
- Box_Production_peak  
- PER_peak  
- DRB_percent  
- PER_improvement  
- FG_per40  
- NBA_Ready_Score_v2_100  
- 2P_per40  
- WS  
- FT_rate  

### NBA_Ready_Score_v2_100 Components

A weighted developmental composite built from:

- Two_Way_Impact_peak  
- BPM_peak  
- Class Numeric  
- Conf_Strength  
- BPM_improvement  
- TOV_per40  
- WS40_peak  

All weights are derived from LASSO coefficient magnitudes.

---

## 3. Stick Metric Definition

The model is trained on a custom **NBA Stick Metric**, with per-event weights:

- FGM: 1.5  
- FGA: 1  
- 3PM: 0.4  
- FTM: 0.4  
- AST: 1.15  
- TOV: –1.9  
- OREB: 2.2  
- DREB: 2.2  
- STL: 3  
- BLK: 10  
- PF: –0.4  
- Missed FG: –0.9  
- Missed 3PA: –1  
- Missed FT: –0.4  
- Double-double: +3  
- Triple-double: +10  
- Plus/minus: +0.22  

The metric is normalized per game and adjusted to per-36 pace.

A player is labeled as **Stuck = 1** if:

1. Combined Metric ≥ 5  
2. ≥ 2 NBA seasons with ≥ 15 MPG  

---

## 4. Model Training Pipeline

Key modeling steps:

1. Median imputation  
2. SMOTE (sampling_strategy = 0.10)  
3. XGBoost classifier:
   - n_estimators = 650  
   - max_depth = 4  
   - learning_rate = 0.025  
   - subsample = 0.90  
   - colsample_bytree = 0.70  
4. Optimal threshold from the F1-maximizing point on the PR curve  
5. Evaluation on 2023–2025 holdout:
   - AUC-ROC  
   - PR-AUC  
   - Precision/Recall/F1  
   - Confusion matrix  

This produces a calibrated, temporally stable probability of sticking.

---

## 5. Dashboard Summary

The Streamlit dashboard displays:

- Season  
- Player  
- Team  
- NBA_YOS  
- Predicted NBA stick probability  

Features:

- Season filter  
- Top-N filter (10–50)  
- Smooth-color probability gradient  
- Supports all seasons ≤ 2023  

The dashboard is designed for scouting, analytics, historical comparison, and identifying undervalued NCAA archetypes.

---

## 6. Notes

- These are **not** draft rankings.  
- Probabilities are based on NCAA production only.  
- The model is optimized for **out-of-time validation**.  
- Future seasons should only be added with a preserved temporal split.

---

