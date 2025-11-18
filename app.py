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

CSV_URL = "https://raw.githubusercontent.com/Tlappas-23/College-Basketball-Prospect-Evaluation/main/Top_30_Per_Season_BEST_MODEL.csv"

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
        College Prospect Dashboard — Final 7-Feature Model
    </h1>

    <p style="font-size:16px; color:#444; max-width:900px; line-height:1.55;">
        This dashboard visualizes the final probability model that predicts whether 
        an NCAA player is likely to <strong>stick</strong> in the NBA based solely on collegiate production.
        The model uses median-imputed missing values, a strict pre-2023 temporal split,
        SMOTE balancing, and an optimized XGBoost classifier.
    </p>

    <p style="font-size:16px; color:#444; max-width:900px; line-height:1.55;">
        <strong>These probabilities do not represent draft rankings.</strong>
        They estimate the likelihood a college player will stick in the NBA.
    </p>

    <h3 style="margin-top:35px;">Final Model Inputs (7 Features)</h3>
    <ul style="font-size:16px; color:#333; line-height:1.65;">
        <li>FTA_per40</li>
        <li>FT_per40</li>
        <li>NBA_Ready_Score_100</li>
        <li>Def_Impact</li>
        <li>BLK_percent</li>
        <li>Box_Production_peak</li>
        <li>PER_peak</li>
    </ul>

    <p style="font-size:15px; color:#555; max-width:900px;">
        Additional context variable included: <strong>NBA_YOS</strong> (NBA years played)
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
