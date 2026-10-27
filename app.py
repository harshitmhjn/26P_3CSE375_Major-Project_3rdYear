"""Streamlit entry point.

    streamlit run app.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

st.set_page_config(
    page_title="Thermal-aware DRL scheduler",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dashboard.styles import apply_theme         # noqa: E402
apply_theme()

from dashboard.common import init_state          # noqa: E402
from dashboard import (p_overview, p_simulator, p_algorithms, p_training,   # noqa: E402
                       p_results, p_architecture)

PAGES = {
    "Overview": p_overview,
    "Data-Center Simulator": p_simulator,
    "Scheduling Algorithms": p_algorithms,
    "DRL Training": p_training,
    "Experimental Results": p_results,
    "Architecture & Methodology": p_architecture,
}

init_state()

# Sleek command-center brand block
st.sidebar.markdown("""
<div style="padding:4px 0 16px 0; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:14px;">
  <div style="display:flex; align-items:center; gap:10px;">
    <div style="background:linear-gradient(135deg, #ea580c 0%, #f97316 100%); width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; box-shadow:0 0 16px rgba(249,115,22,0.45);">
      <span style="font-size:1.3rem;">🌡️</span>
    </div>
    <div>
      <div style="font-weight:800; font-size:1.05rem; letter-spacing:-0.02em; color:#f8fafc; font-family:'Outfit', sans-serif;">THERMAL DRL</div>
      <div style="font-size:0.7rem; color:#f97316; font-weight:700; letter-spacing:0.06em; text-transform:uppercase;">Autonomous Scheduler</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Thermal-aware scheduling")
page = st.sidebar.radio("Page", list(PAGES), key="page", label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.markdown("""
<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px 12px; margin-bottom:12px;">
  <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
    <span style="color:#94a3b8; font-size:0.75rem;">Backend Engine</span>
    <span style="color:#38bdf8; font-size:0.75rem; font-weight:600;">Gymnasium Env</span>
  </div>
  <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
    <span style="color:#94a3b8; font-size:0.75rem;">DRL Framework</span>
    <span style="color:#c084fc; font-size:0.75rem; font-weight:600;">Stable-Baselines3</span>
  </div>
  <div style="display:flex; justify-content:space-between;">
    <span style="color:#94a3b8; font-size:0.75rem;">Thermal Model</span>
    <span style="color:#34d399; font-size:0.75rem; font-weight:600;">Lumped Parameter</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.caption("IILM University, Greater Noida\nB.Tech CSE - Major Project 2026-27\n\n"
                   "Simulation-based study. The thermal model is a simplified approximation, "
                   "not a validated physical model.")
PAGES[page].render()
