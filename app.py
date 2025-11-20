import streamlit as st
import pandas as pd

# =============================================================================
# PAGE CONFIG — MUST BE FIRST STREAMLIT COMMAND
# =============================================================================

st.set_page_config(
    page_title="College Prospect Dashboard — Final 2025 Model",
    layout="wide"
)

# =============================================================================
# LOAD DATA
# =============================================================================

CSV_URL = "https://raw.githubusercontent.com/Tlappas-23/College-Basketball-Prospect-Evaluation/main/FINAL_2025_College_to_NBA_Top50_Per_Season.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)

    # Remove incomplete season if present
    df = df[df["Season_Year"] != 2024]

    return df

df = load_data()

# =============================================================================
# PROFESSIONAL STYLING HELPERS
# =============================================================================

def color_scale(value):
    """Smooth green gradient based on predicted probability."""
    alpha = min(max(value, 0), 1)
    return f"background-color: rgba(46, 204, 113, {alpha * 0.35});"

def style_rows(df):
    """Apply gradient to entire row based on Pred_Stick_Proba."""
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
        College Prospect Dashboard — Final XGBoost Model (13 Features)
    </h1>

    <p style="font-size:16px; color:#444; max-width:900px; line-height:1.55;">
        This dashboard displays the final NBA stick probability model trained on NCAA performance only.
        The pipeline uses median-imputed missing values, a strict pre-2023 temporal split, SMOTE applied
        only to the training fold, and a tuned XGBoost classifier.
    </p>

    <p style="font-size:16px; color:#444; max-width:900px; line-height:1.55;">
        <strong>Important:</strong> These scores are not draft rankings. They measure the probability a player
        will “stick” in the NBA — defined as playing ≥2 seasons at ≥15 MPG with a combined metric ≥5.
    </p>

    <h3 style="margin-top:35px;">Final Model Inputs (13 Features)</h3>
    <ul style="font-size:16px; color:#333; line-height:1.65;">
        <li>FTA_per40</li>
        <li>FT_per40</li>
        <li>Def_Impact</li>
        <li>BLK_percent</li>
        <li>Box_Production_peak</li>
        <li>PER_peak</li>
        <li>DRB_percent</li>
        <li>PER_improvement</li>
        <li>FG_per40</li>
        <li>NBA_Ready_Score_v2_100</li>
        <li>2P_per40</li>
        <li>WS</li>
        <li>FT_rate</li>
    </ul>

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
    [10, 20, 30, 40, 50],
    index=2  # Default: Top 30
)

# =============================================================================
# APPLY FILTERS
# =============================================================================

filtered_df = df.copy()

if season_choice != "All Seasons":
    # SINGLE-SEASON VIEW
    filtered_df = filtered_df[filtered_df["Season_Year"] == season_choice]
else:
    # MULTI-SEASON VIEW → rank within season, then stack
    filtered_df = (
        filtered_df
        .sort_values(["Season_Year", "Pred_Stick_Proba"], ascending=[True, False])
        .groupby("Season_Year")
        .head(top_n)
    )

# Sort final display by probability
filtered_df = filtered_df.sort_values("Pred_Stick_Proba", ascending=False)

# =============================================================================
# DISPLAY TABLE
# =============================================================================

st.markdown("## Top Projected Players")

styled = style_rows(filtered_df)

st.dataframe(
    styled,
    use_container_width=True
)
