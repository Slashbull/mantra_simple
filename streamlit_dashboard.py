# streamlit_dashboard.py (FINAL, BULLETPROOF, 100% DATA-DRIVEN)
import streamlit as st
import pandas as pd
import io
from datetime import datetime

from core_system_foundation import load_and_process
from signal_engine import run_signal_engine
from decision_engine import run_decision_engine
from anomaly_detector import run_anomaly_detector
from sector_mapper import run_sector_mapper
from watchlist_builder import build_watchlist
from edge_finder import find_edges

# --- Streamlit page config ---
st.set_page_config(
    page_title="M.A.N.T.R.A. — Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Header ---
st.title("📈 M.A.N.T.R.A. — Indian Stock Intelligence Engine")
st.caption("Decisions, Not Guesses. Data-Driven Edge Only.")

# --- Health panel in sidebar ---
with st.sidebar:
    st.markdown("### 🩺 Data Health")
    # --- Use the new tuple output (stocks_df, sector_df, summary) ---
    stocks_df, sector_df, summary = load_and_process()
    stock_df = stocks_df  # for compatibility in downstream code
    st.metric("Stocks", summary.get("total_stocks", 0))
    st.metric("Sectors", summary.get("total_sectors", 0))
    st.metric("Blanks", summary.get("blank_cells", 0))
    st.metric("Duplicates", summary.get("duplicates", 0))
    st.caption(f"Source: {summary.get('source','')}")
    st.caption(f"Data Hash: {summary.get('data_hash','')}")
    st.caption(f"Last Reload: {datetime.now():%Y-%m-%d %H:%M}")

# --- Regime selector ---
regime = st.sidebar.selectbox(
    "Market Regime",
    ["balanced", "momentum", "value", "growth", "volume"], index=0
)

# --- Data pipeline ---
with st.spinner("Loading & computing signals..."):
    df = run_signal_engine(stock_df, sector_df, regime=regime)
    df = run_decision_engine(df)
    df = run_anomaly_detector(df)
    sector_scores = run_sector_mapper(sector_df)

# --- Sidebar Filters (Category, Sector, Tag, Score, Search, Export) ---
tags = ["Buy", "Watch", "Avoid"]
selected_tag = st.sidebar.selectbox("Tag", tags, index=0)
min_score = st.sidebar.slider("Min Final Score", 0, 100, 60)

cat_col = "category"
categories = ["All"] + sorted(df[cat_col].dropna().unique()) if cat_col in df and cat_col in df.columns else ["All"]
selected_category = st.sidebar.selectbox("Category", categories, index=0)

sector_list = ["All"] + sorted(sector_scores["sector"].unique()) if "sector" in sector_scores else ["All"]
selected_sector = st.sidebar.selectbox("Sector", sector_list, index=0)

search = st.sidebar.text_input("Search Ticker or Company").upper().strip()
export_fmt = st.sidebar.radio("Export", ["CSV", "Excel"], index=0)

# --- Bulletproof Filtering Logic ---
def smart_filter(df):
    # Return empty DataFrame if missing columns
    required = ["tag", "final_score"]
    if df is None or df.empty or any(col not in df.columns for col in required):
        return df.iloc[0:0]
    q = (df["tag"] == selected_tag) & (df["final_score"] >= min_score)
    if selected_category != "All" and "category" in df.columns:
        q &= (df["category"] == selected_category)
    if selected_sector != "All" and "sector" in df.columns:
        q &= (df["sector"] == selected_sector)
    if search:
        q &= (
            df["ticker"].str.upper().str.contains(search, na=False) |
            df["company_name"].str.upper().str.contains(search, na=False)
        )
    return df[q]

filtered = smart_filter(df)

# --- Main KPIs ---
k1, k2, k3 = st.columns(3)
k1.metric("Total Stocks", len(df))
k2.metric("Buy Tags", int(df["tag"].eq("Buy").sum()) if "tag" in df else 0)
k3.metric("Anomalies", int(df["anomaly"].sum()) if "anomaly" in df else 0)

# --- Top 10 Buy Cards ---
st.subheader("🎯 Top 10 Buy Ideas")
if "tag" in filtered and "final_score" in filtered:
    top10 = filtered[filtered["tag"] == "Buy"].nlargest(10, "final_score")
else:
    top10 = pd.DataFrame()
if top10.empty:
    st.info("No 'Buy' ideas found with filters.")
else:
    cols = st.columns(5)
    for i, (_, row) in enumerate(top10.iterrows()):
        with cols[i % 5]:
            st.metric(
                label=f"{row['ticker']} — {row['company_name'][:18]}",
                value=f"{row['final_score']:.1f}",
                delta=f"₹{row.get('target_price', 0):.0f}"
            )

# --- Main Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 All Opportunities", "🔥 Sector Rotation", "🚨 Anomalies", "🪄 Edge Finder"
])

# --- Tab 1: All Opportunities ---
with tab1:
    st.subheader("All Opportunities Table")
    if filtered.empty:
        st.warning("No stocks match your filters. Adjust your criteria.")
    else:
        st.dataframe(filtered, use_container_width=True)
        if export_fmt == "CSV":
            st.download_button("Download CSV", filtered.to_csv(index=False).encode(), "opportunities.csv")
        else:
            buf = io.BytesIO()
            filtered.to_excel(buf, index=False)
            st.download_button("Download Excel", buf.getvalue(), "opportunities.xlsx")

# --- Tab 2: Sector Rotation ---
with tab2:
    st.subheader("Sector Heatmap / Rotation")
    if sector_scores.empty:
        st.warning("No sector data.")
    else:
        st.dataframe(sector_scores, use_container_width=True)
        if "sector_score" in sector_scores and "sector" in sector_scores:
            st.bar_chart(sector_scores.set_index("sector")["sector_score"])

# --- Tab 3: Anomalies ---
with tab3:
    st.subheader("Anomaly Detector")
    anomalies = df[df["anomaly"]] if "anomaly" in df else pd.DataFrame()
    if anomalies.empty:
        st.success("No anomalies detected in current regime.")
    else:
        st.dataframe(anomalies, use_container_width=True)

# --- Tab 4: Edge Finder ---
with tab4:
    st.subheader("Edge Finder — Alpha Opportunities")
    edge_df = find_edges(filtered) if not filtered.empty else pd.DataFrame()
    if edge_df.empty:
        st.info("No special edges found in current filter.")
    else:
        st.dataframe(edge_df, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M}")
st.caption("All logic is 100% data-driven. This is your personal edge.")
