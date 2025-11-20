import streamlit as st
import pandas as pd

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="College Prospect Dashboard — Final 2025 Model",
    layout="wide"
)

# ======================================================
# LOAD DATA
# ======================================================

CSV_URL = "https://raw.githubusercontent.com/Tlappas-23/College-Basketball-Prospect-Evaluation/main/FINAL_2025_College_to_NBA_Top50_Per_Season.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)

    # Rename for consistency
    df = df.rename(columns={"SEASON": "Season_Year", "STICK %": "Pred_Stick_Proba"})

    # Convert probability from percent → decimal 0–1
    df["Pred_Stick_Proba"] = df["Pred_Stick_Proba"] / 100.0

    return df

df = load_data()

# ======================================================
# STYLING HELPERS
# ======================================================

def color_scale(value):
    alpha = min(max(value, 0), 1)
    return f"background-color: rgba(46, 204, 113, {alpha * 0.35});"

def style_rows(df):
    styled = df.style.format({
        "Pred_Stick_Proba": "{:.2%}"
    })
    styled = styled.apply(
        lambda row: [color_scale(row["Pred_Stick_Proba"])] * len(row),
        axis=1
    )
    return styled

# ======================================================
# HEADER
# ======================================================

st.markdown("""
<h1 style="font-size:38px; font-weight:700;">
    College Prospect Dashboard — Final XGBoost Model (13 Features)
</h1>

<p style="font-size:16px; color:#444; max-width:880px;">
    This dashboard visualizes the predicted probability that an NCAA player 
    will <strong>stick</strong> in the NBA. The exported results represent the 
    <strong>Top 50 players per season</strong> according to the final 2025 model.
</p>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR FILTERS
# ======================================================

st.sidebar.markdown("## Filters")

season_list = ["All Seasons"] + sorted(df["Season_Year"].unique().tolist())
season_choice = st.sidebar.selectbox("Season", season_list)

top_n = st.sidebar.selectbox(
    "Show Top N Players",
    [10, 20, 30, 40, 50],
    index=2
)

# ======================================================
# APPLY FILTERS
# ======================================================

filtered_df = df.copy()

if season_choice != "All Seasons":
    filtered_df = filtered_df[filtered_df["Season_Year"] == season_choice]
else:
    filtered_df = (
        filtered_df.sort_values(["Season_Year", "Pred_Stick_Proba"], ascending=[True, False])
                   .groupby("Season_Year")
                   .head(top_n)
    )

filtered_df = filtered_df.sort_values("Pred_Stick_Proba", ascending=False)

# ======================================================
# TABLE OUTPUT
# ======================================================

st.markdown("## Top Projected Players")

styled = style_rows(filtered_df)

st.dataframe(
    styled,
    use_container_width=True
)
