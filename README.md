NCAA to NBA Stick Probability Modeling Pipeline and Dashboard
1. Project Overview

This project builds a fully reproducible machine learning system that estimates the probability that an NCAA basketball player will “stick” in the NBA.
A player is considered to have “stuck” if they remain rostered for multiple seasons and meet a minimum playing time threshold.

The workflow uses:

A strict pre-2023 temporal split

Median imputation

SMOTE oversampling

A tuned XGBoost classifier

A final seven-feature model

An interactive Dash dashboard

All predictions are based entirely on NCAA production.
No draft position or NBA outcomes are used as model inputs.

2. Modeling Workflow
2.1 Step 1 — Imports and Global Configuration

This step establishes the modeling environment by:

Organizing imports into logical groups

Setting a global random state for reproducibility

Applying consistent plot styling

Outcome: A stable, reproducible environment for the entire project.

2.2 Step 2 — Data Preparation and Leakage Prevention

Key actions:

Convert season text to a numeric Season_Year

Split data temporally (train ≤ 2022, test ≥ 2023)

Remove all columns that would not be known at draft time, including:

Player name

Team

Season

NBA statistics such as minutes, plus-minus, stick outcomes

Restrict inputs to numeric NCAA features only

Outcome: A dataset aligned with real draft-time prediction conditions.

2.3 Step 3 — Missing Value Imputation

Median imputation is applied to the training data

Imputation is performed before SMOTE

Ensures synthetic samples created by SMOTE do not contain NaNs

2.4 Step 4 — Resampling with SMOTE

Class imbalance is handled using SMOTE

Sampling strategy is set to 0.10

k_neighbors=3 ensures stable behavior on sparse minority cases

2.5 Step 5 — Final Seven-Feature XGBoost Model

After evaluating more than 35 possible predictors, the final model uses:

FTA_per40

FT_per40

NBA_Ready_Score_100

Def_Impact

BLK_percent

Box_Production_peak

PER_peak

These features consistently delivered the best out-of-time performance.

2.6 Step 6 — NBA Readiness Meta-Metric

NBA_Ready_Score_100 is a composite feature built from seven weighted components:

Two_Way_Impact_peak (weight 0.80)

BPM_peak (0.70)

Class Numeric (–0.65)

Conf_Strength (0.60)

BPM_improvement (–0.54)

TOV_per40 (0.49)

WS40_peak (–0.48)

This metric serves as a high-level indicator of pro readiness.

2.7 Step 7 — Definition of "Stick" Ground Truth

A player is labeled as having “stuck” if all of the following are true:

A Combined Metric of at least 5

At least two NBA seasons played

At least 15 minutes per game on average across those seasons

The Combined Metric includes weighted components for:

Scoring

Efficiency

Creation

Defensive impact

Penalties for turnovers and fouls

Bonuses for double-doubles, triple-doubles, and plus-minus

Pace and per-minute normalization

2.8 Step 8 — Threshold Optimization

The classification threshold is determined by maximizing the F1-score:

Compute precision-recall curve

Compute F1-score at each threshold

Select the threshold corresponding to maximum F1

Use this threshold for final classification

This improves recall and precision on rare events.

3. Interactive Dashboard

An interactive Dash application is included for exploring model outputs.

3.1 Displayed Fields

Season_Year

Player

Team

NBA_YOS (actual years of NBA service for historical players)

Predicted probability of sticking

3.2 Dashboard Features

Season filter

Top-N selector

Automatic sorting by predicted probability

Color-coded probability classes (high, mid, low)

Clean and modern UI using standardized styling

3.3 Structure

The dashboard file includes:

Data loading and preprocessing

Styling and layout

Callback logic for dynamic table updates

Main execution block

4. Summary

This project provides:

A rigorously leak-free modeling pipeline

A tuned seven-feature XGBoost model

A composite NBA readiness index

A validated stick metric based on real NBA outcomes

A production-ready dashboard for exploring probabilities

A model card, diagrams, or a GitHub-friendly layout can be produced upon request.
