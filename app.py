import streamlit as st
import pandas as pd

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NBA Stick Probability Dashboard",
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

    # Standardize columns
    df = df.rename(columns={
        "SEASON": "Season_Year",
        "STICK %": "Pred_Stick_Proba"
    })
    df["Pred_Stick_Proba"] = df["Pred_Stick_Proba"] / 100.0
    return df

df = load_data()

# ======================================================
# THEMING HELPERS
# ======================================================
def color_scale(prob):
    alpha = min(max(prob, 0), 1)
    return f"background-color: rgba(76, 175, 80, {alpha * 0.33});"

def style_rows(df):
    styled = df.style.format({"Pred_Stick_Proba": "{:.2%}"})
    styled = styled.apply(lambda row: [color_scale(row["Pred_Stick_Proba"])] * len(row), axis=1)
    return styled

# ======================================================
# TOP NAVIGATION BAR (Clean & Professional)
# ======================================================
st.markdown(
    """
    <div style="background-color:#0B2447; padding:18px 28px; border-radius:8px; margin-bottom:25px;">
        <h1 style="color:white; margin:0; font-size:32px; font-weight:700;">
            NCAA → NBA Stick Probability Dashboard (2025 Model)
        </h1>
        <p style="color:#d1d5db; margin:4px 0 0 0; font-size:15px;">
            Powered by a 13-feature XGBoost model with strict pre-2023 temporal validation.
            Probabilities estimate long-term NBA success using NCAA production only.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# SIDEBAR FILTERS (Cleaner UI)
# ======================================================
st.sidebar.markdown("### Filters")

season_list = ["All Seasons"] + sorted(df["Season_Year"].unique().tolist())
season_choice = st.sidebar.selectbox("Select Season", season_list)

top_n = st.sidebar.slider(
    "Show Top N Players",
    10, 100, 30, step=5
)

# ======================================================
# FILTERING LOGIC (Corrected & Unified)
# ======================================================
filtered_df = df.copy()

if season_choice != "All Seasons":
    # Single season → top N within that season
    temp = filtered_df[filtered_df["Season_Year"] == season_choice]
    filtered_df = temp.sort_values("Pred_Stick_Proba", ascending=False).head(top_n)
else:
    # Multi-season → independently top N per season
    filtered_df = (
        filtered_df
        .sort_values(["Season_Year", "Pred_Stick_Proba"], ascending=[True, False])
        .groupby("Season_Year")
        .head(top_n)
    )

filtered_df = filtered_df.sort_values("Pred_Stick_Proba", ascending=False)

# ======================================================
# SUMMARY METRICS (Professional Cards)
# ======================================================
col1, col2, col3 = st.columns(3)

col1.metric("Total Players Displayed", len(filtered_df))
col2.metric("Seasons Included", filtered_df["Season_Year"].nunique())
col3.metric("Top Prediction", f"{filtered_df['Pred_Stick_Proba'].max():.2%}")

# ======================================================
# MAIN TABLE
# ======================================================
st.markdown("### Top Projected Players")

styled = style_rows(filtered_df)

st.dataframe(
    styled,
    use_container_width=True,
    height=550
)

# ======================================================
# FOOTER
# ======================================================
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:#777; font-size:14px; padding-top:5px;">
        © 2025 NCAA → NBA Stick Model • 13-Feature XGBoost Pipeline • Zero Draft Leakage<br>
        Built for scouting, analytics, and long-term player projection.
    </p>
    """,
    unsafe_allow_html=True,
)
