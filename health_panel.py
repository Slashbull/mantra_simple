# health_panel.py (Minimalist, M.A.N.T.R.A. Edition)

import streamlit as st
from datetime import datetime

def render_health_panel(load_summary: dict):
    """
    Renders a quick health panel in the Streamlit sidebar.
    - load_summary: Dict from your core loader (should have basic stats)
    """
    st.sidebar.markdown("### 🩺 Data Health")

    total_stocks = load_summary.get("total_stocks", 0)
    total_sectors = load_summary.get("total_sectors", 0)
    blanks = load_summary.get("blank_cells", 0)
    dups = load_summary.get("duplicates", 0)
    source = load_summary.get("source", "-")
    reload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    st.sidebar.metric("Stocks", total_stocks)
    st.sidebar.metric("Sectors", total_sectors)
    st.sidebar.metric("Blanks", blanks)
    st.sidebar.metric("Duplicates", dups)
    st.sidebar.caption(f"Data Source: `{source}`")
    st.sidebar.caption(f"🕒 {reload_time}")

    if blanks > total_stocks * 5:
        st.sidebar.warning("⚠️ High blanks! Check your sheet.")
    if dups > 0:
        st.sidebar.warning("⚠️ Duplicate tickers found.")

    # Optionally add a reload button for Streamlit Cloud
    if st.sidebar.button("🔄 Reload Data", help="Clears cache and reloads"):
        st.cache_data.clear()
        st.experimental_rerun()
