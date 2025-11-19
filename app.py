import streamlit as st
import pandas as pd

# =============================================================================
# PAGE CONFIG — MUST BE FIRST STREAMLIT COMMAND
# =============================================================================

st.set_page_config(
    page_title="College Prospect Dashboard — Final 7-Feature Model",
    layout="wide"
)

# =============================================================================
# LOAD DATA
# =============================================================================

CSV_URL = "https://raw.githubusercontent.com/Tlappas-23/College-Basketball-Prospect-Evaluation/main/Top_30_Per_Season_BEST_MODEL_v4.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)

    # Remove 2024 if present
    df = df[df["Season_Year"] != 2024]

    return df

df = load_data()

# =============================================================================
# PROFESSIONAL STYLING HELPERS
# =============================================================================

def color_scale(value):
    """
    Apply a smooth green gradient based on prediction probability.
    """
    # normalize probability (0 → 1)
    alpha = min(max(value, 0), 1)
    return f"background-color: rgba(46, 204, 113, {alpha * 0.35});"


def style_rows(df):
    """
    Apply gradient based on Pred_Stick_Proba.
    """
    styled = df.style.format({"Pred_Stick_Proba": "{:.2%}"})

    styled = styled.apply(
        lambda row: [color_scale(row["Pred_Stick_Proba"])] * len(row),
        axis=1
    )

    return styled


# =============================================================================
# HEADER SECTION
# =============================================================================

st.markdown(
    """
    <h1 style="font-size:38px; font-weight:700; margin-bottom:0;">
        College Prospect Dashboard — Final XGBoost Model (10 Features)
    </h1>

    <p style="font-size:16px; color:#444; max-width:900px; line-height:1.55;">
        This dashboard visualizes the final probability model that predicts whether 
        an NCAA player is likely to <strong>stick</strong> in the NBA based solely on collegiate production.
        The model uses median-imputed missing values, a strict pre-2023 temporal split to prevent leakage,
        SMOTE balancing on the training data only, and an optimized XGBoost classifier.
    </p>

    <p style="font-size:16px; color:#444; max-width:900px; line-height:1.55;">
        <strong>These probabilities do not represent draft rankings.</strong>
        They estimate the likelihood a college player will stick in the NBA 
        (play 2+ NBA seasons averaging at least 15 MPG with a Combined Metric ≥ 5).
    </p>

    <h3 style="margin-top:35px;">Final Model Inputs (10 Features)</h3>
    <ul style="font-size:16px; color:#333; line-height:1.65;">
        <li>Box_Production_peak</li>
        <li>FTA_per40</li>
        <li>Def_Impact</li>
        <li>NBA_Ready_Score</li>
        <li>FG_per40</li>
        <li>PER</li>
        <li>BLK_per40</li>
        <li>TOV_percent</li>
        <li>WS40</li>
        <li>FT_percent</li>
    </ul>

    <h3 style="margin-top:30px;">Inside the NBA_Ready_Score</h3>
    <p style="font-size:15px; color:#555; max-width:900px; line-height:1.55;">
        NBA_Ready_Score combines seven high-signal developmental indicators:
        <br><br>
        • Two_Way_Impact_peak (0.80) <br>
        • BPM_peak (0.70) <br>
        • Class Numeric (–0.65) <br>
        • Conf_Strength (0.60) <br>
        • BPM_improvement (–0.54) <br>
        • TOV_per40 (0.49) <br>
        • WS40_peak (–0.48) <br><br>

        All scaling is standardized within-season to preserve temporal integrity.
        No NBA or draft information is included during training.
    </p>

    <p style="font-size:14px; color:#444; margin-top:20px;">
        Additional context variable shown in the table: <strong>NBA_YOS</strong> (NBA seasons played).
    </p>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================

st.sidebar.markdown("## Filters")

season_list = ["All Seasons"] + sorted(df["Season_Year"].unique().tolist())
season_choice = st.sidebar.selectbox("Season", season_list)

top_n = st.sidebar.selectbox(
    "Show Top N Players",
    [10, 15, 20, 25, 30, 40, 50],
    index=2
)

# =============================================================================
# APPLY FILTERS
# =============================================================================

filtered_df = df.copy()

if season_choice != "All Seasons":
    filtered_df = filtered_df[filtered_df["Season_Year"] == season_choice]

filtered_df = filtered_df.sort_values("Pred_Stick_Proba", ascending=False).head(top_n)

# =============================================================================
# DISPLAY TABLE
# =============================================================================

st.markdown("## Top Projected Players")

styled = style_rows(filtered_df)

st.dataframe(
    styled,
    use_container_width=True
)
