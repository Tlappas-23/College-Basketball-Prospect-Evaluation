import streamlit as st
import pandas as pd

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="College Prospect Dashboard — Final 2025 Model",
    layout="wide",
)

# ======================================================
# LOAD DATA
# ======================================================

CSV_URL = (
    "https://raw.githubusercontent.com/Tlappas-23/College-Basketball-Prospect-Evaluation/main/"
    "FINAL_2025_College_to_NBA_Top50_Per_Season.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)

    # Standardize column names
    df = df.rename(columns={
        "SEASON": "Season_Year",
        "STICK %": "Pred_Stick_Proba",
    })

    # Convert probability (%) → 0–1 decimal
    df["Pred_Stick_Proba"] = df["Pred_Stick_Proba"] / 100.0

    return df

df = load_data()

# ======================================================
# STYLING HELPERS
# ======================================================

def color_scale(prob):
    alpha = min(max(prob, 0), 1)
    return f"background-color: rgba(46, 204, 113, {alpha * 0.33});"

def style_rows(df):
    styled = df.style.format({"Pred_Stick_Proba": "{:.2%}"})
    styled = styled.apply(
        lambda row: [color_scale(row["Pred_Stick_Proba"])] * len(row),
        axis=1,
    )
    return styled

# ======================================================
# HEADER (Professional Style)
# ======================================================

st.markdown(
    """
    <div style="padding: 20px 0 10px 0;">
        <h1 style="font-size:40px; font-weight:700; margin-bottom:4px;">
            College Prospect Dashboard — Final 2025 Stick Probability Model
        </h1>
        <p style="font-size:16px; color:#555; max-width:850px; line-height:1.55;">
            Interactive visualization of the final NCAA-to-NBA stick probability model.
            Scores reflect model-derived probabilities that a player will 
            <strong>“stick” in the NBA</strong> based solely on NCAA production.
            Results displayed here represent the top probability players per season,
            produced by a 13-feature XGBoost model with strict temporal validation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# SIDEBAR — Filters + Navigation
# ======================================================

st.sidebar.markdown(
    """
    <h2 style="margin-top:0;">Filters</h2>
    """,
    unsafe_allow_html=True,
)

season_list = ["All Seasons"] + sorted(df["Season_Year"].unique().tolist())
season_choice = st.sidebar.selectbox("Season", season_list)

top_n = st.sidebar.selectbox(
    "Show Top N Players",
    [10, 13, 15, 20, 25, 30, 35, 40, 45, 50, 60, 100],
    index=2,
)

# ======================================================
# FILTER LOGIC — FIXED (Top N works everywhere)
# ======================================================

filtered_df = df.copy()

if season_choice != "All Seasons":
    # Single season → take top N within that season
    temp = filtered_df[filtered_df["Season_Year"] == season_choice]
    filtered_df = (
        temp.sort_values("Pred_Stick_Proba", ascending=False)
            .head(top_n)
    )
else:
    # Multi-season → independently top N per season
    filtered_df = (
        filtered_df.sort_values(["Season_Year", "Pred_Stick_Proba"], ascending=[True, False])
                    .groupby("Season_Year")
                    .head(top_n)
    )

# Always sort final output by probability
filtered_df = filtered_df.sort_values("Pred_Stick_Proba", ascending=False)

# ======================================================
# MAIN TABLE
# ======================================================

st.markdown("## Top Projected Players")

styled = style_rows(filtered_df)

st.dataframe(
    styled,
    use_container_width=True,
)

# ======================================================
# FOOTER
# ======================================================

st.markdown(
    """
    <br><hr><p style="text-align:center; color:#777; font-size:14px;">
        NCAA → NBA Stick Model (2025 Edition) • XGBoost • 13 Features • Strict Temporal Validation<br>
        Built entirely from college performance statistics — no draft information used in training.
    </p>
    """,
    unsafe_allow_html=True,
)
