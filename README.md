# NCAA to NBA Stick Probability Modeling Pipeline and Dashboard

## 1. Project Overview

This project develops a reproducible machine learning pipeline that estimates the probability that an NCAA player will “stick” in the NBA. A player is defined as having "stuck" if they achieve a Combined Metric of 5 or higher and average at least 15 minutes per game in two or more NBA seasons.

All model inputs come strictly from NCAA production and player attributes.  
No draft information or NBA performance data is used as a predictive feature.

The final system includes the following components:

1. Clean and standardized data assembly  
2. Strict temporal train/test split (train ≤ 2022, test ≥ 2023)  
3. Median imputation applied before SMOTE  
4. SMOTE oversampling to correct class imbalance  
5. XGBoost classifier tuned for recall stability  
6. Selection of the seven strongest predictive features  
7. Deployment in an interactive Dash dashboard

---

## 2. The Final Seven Feature Model

The final predictive model uses the following seven features:

1. FTA_per40  
2. FT_per40  
3. NBA_Ready_Score_100  
4. Def_Impact  
5. BLK_percent  
6. Box_Production_peak  
7. PER_peak  

The feature **NBA_Ready_Score_100** is a composite index derived from weighted components:

- Two_Way_Impact_peak (0.80)  
- BPM_peak (0.70)  
- Class Numeric (–0.65)  
- Conf_Strength (0.60)  
- BPM_improvement (–0.54)  
- TOV_per40 (0.49)  
- WS40_peak (–0.48)

No draft position or round is incorporated at any stage.

---

## 3. Stick Metric Definition

The "Stick" Metric is a pace- and usage-adjusted impact formula that applies the following per-event weights:

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

The result is normalized by games played and adjusted to per-36-minute pace.

NBA players typically score around **5** on this metric.  
To count as “sticking,” a player must achieve:

1. Combined Metric ≥ 5  
2. At least two NBA seasons with ≥ 15 MPG  

---

## 4. Model Training Pipeline

The training pipeline includes:

1. Median imputation (applied before resampling)  
2. SMOTE with sampling_strategy = 0.10 and k_neighbors = 3  
3. XGBoost classifier with:  
   - n_estimators = 600  
   - max_depth = 4  
   - learning_rate = 0.025  
   - subsample = 0.90  
   - colsample_bytree = 0.70  
   - tree_method = “hist”  

4. Threshold selection using the optimal F1 point on the precision–recall curve  
5. Evaluation using:  
   - AUC-ROC  
   - PR-AUC  
   - Precision  
   - Recall  
   - F1 score  
   - Balanced accuracy  
   - Confusion matrix  

---

## 5. Dashboard Summary

The dashboard displays:

- Season  
- Player  
- Team  
- NBA_YOS (NBA years played)  
- Predicted probability of sticking (using the final 7-feature model)  

Features:

- Color-coded probability tiers  
- Season filter  
- Adjustable Top-N ranking  
- Excludes incomplete or future seasons (e.g., 2024)  

The dashboard is designed for scouting, historical analysis, and identifying overperforming NCAA profiles that may translate into NBA longevity.

---

## 6. Usage Notes

- The dashboard does not represent draft rankings; probabilities are independent of draft data.  
- All NCAA seasons up to 2023 are eligible for prediction.  
- The model is optimized for generalization on out-of-time seasons and should not be retrained with future seasons included unless the entire temporal split is restructured.
