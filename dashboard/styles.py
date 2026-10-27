"""Modern dark cockpit styling tokens, CSS injection, and UI components."""

import streamlit as st


DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

:root {
  --bg-deep: #090d16;
  --bg-card: rgba(17, 24, 39, 0.75);
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-glow: rgba(249, 115, 22, 0.35);
  --accent-orange: #f97316;
  --accent-amber: #f59e0b;
  --accent-cyan: #06b6d4;
  --accent-emerald: #10b981;
  --accent-red: #ef4444;
  --text-main: #f1f5f9;
  --text-muted: #94a3b8;
}

/* Global App Container */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 10% 8%, rgba(249, 115, 22, 0.06) 0%, transparent 35%),
    radial-gradient(circle at 90% 85%, rgba(6, 182, 212, 0.05) 0%, transparent 40%),
    radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.03) 0%, transparent 60%),
    #0b0f19 !important;
  color: var(--text-main) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stHeader"] {
  background: rgba(11, 15, 25, 0.85) !important;
  backdrop-filter: blur(12px) !important;
  border-bottom: 1px solid var(--border-subtle) !important;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
  background-color: #0c121e !important;
  border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
  font-family: 'Outfit', sans-serif !important;
  letter-spacing: -0.02em !important;
}

/* Sidebar radio navigation as sleek pills */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
  background: rgba(255, 255, 255, 0.02) !important;
  border: 1px solid rgba(255, 255, 255, 0.04) !important;
  border-radius: 8px !important;
  padding: 8px 12px !important;
  margin-bottom: 6px !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  cursor: pointer !important;
}

[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
  background: rgba(249, 115, 22, 0.08) !important;
  border-color: rgba(249, 115, 22, 0.3) !important;
  transform: translateX(3px) !important;
}

[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(90deg, rgba(249, 115, 22, 0.16) 0%, rgba(249, 115, 22, 0.04) 100%) !important;
  border-color: rgba(249, 115, 22, 0.5) !important;
  border-left: 3px solid #f97316 !important;
  color: #fff !important;
  font-weight: 600 !important;
}

/* Metric Cards */
[data-testid="stMetric"] {
  background: linear-gradient(135deg, rgba(21, 29, 45, 0.8) 0%, rgba(13, 19, 32, 0.9) 100%) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4) !important;
  backdrop-filter: blur(10px) !important;
  transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stMetric"]:hover {
  transform: translateY(-2px) !important;
  border-color: rgba(249, 115, 22, 0.4) !important;
  box-shadow: 0 8px 24px -2px rgba(249, 115, 22, 0.15) !important;
}

[data-testid="stMetricLabel"] {
  color: #94a3b8 !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
  color: #f8fafc !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 1.55rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em !important;
}

[data-testid="stMetricDelta"] {
  font-size: 0.76rem !important;
  font-weight: 500 !important;
}

/* Primary and Secondary Buttons */
button[kind="primary"] {
  background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  color: #ffffff !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 14px rgba(249, 115, 22, 0.35) !important;
  transition: all 0.2s ease !important;
}

button[kind="primary"]:hover {
  box-shadow: 0 6px 22px rgba(249, 115, 22, 0.55) !important;
  transform: translateY(-1px) !important;
}

button[kind="secondary"] {
  background: rgba(30, 41, 59, 0.75) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  color: #f1f5f9 !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover {
  background: rgba(51, 65, 85, 0.9) !important;
  border-color: rgba(249, 115, 22, 0.45) !important;
  color: #ffffff !important;
  transform: translateY(-1px) !important;
}

/* Tabs */
[data-testid="stTabs"] {
  margin-top: 10px !important;
}

[data-testid="stTabs"] button[role="tab"] {
  color: #94a3b8 !important;
  font-weight: 500 !important;
  border-radius: 6px !important;
  padding: 8px 16px !important;
  transition: all 0.2s ease !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: #f97316 !important;
  background: rgba(249, 115, 22, 0.08) !important;
  border-bottom: 2px solid #f97316 !important;
  font-weight: 600 !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div > div > div {
  background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 35%, #10b981 70%, #f97316 100%) !important;
  box-shadow: 0 0 12px rgba(6, 182, 212, 0.35) !important;
  border-radius: 9999px !important;
}

/* Form Expander & Input styling */
[data-testid="stExpander"] {
  background: rgba(17, 24, 39, 0.65) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 10px !important;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
  backdrop-filter: blur(8px) !important;
}

/* Streamlit Alerts / Callouts */
div[data-testid="stAlert"] {
  background: rgba(17, 24, 39, 0.75) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 10px !important;
  backdrop-filter: blur(8px) !important;
}

/* Custom scrollbars */
::-webkit-scrollbar {
  width: 7px;
  height: 7px;
}
::-webkit-scrollbar-track {
  background: rgba(11, 15, 25, 0.8);
}
::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.18);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(249, 115, 22, 0.5);
}

/* Modern Typography Accent */
h1 {
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em !important;
  background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

h2, h3 {
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: -0.02em !important;
  color: #f8fafc !important;
}

/* Custom HUD Badges & Indicators */
.hud-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(17, 24, 39, 0.8);
}

.hud-pill-green {
  border-color: rgba(16, 185, 129, 0.4);
  background: rgba(16, 185, 129, 0.12);
  color: #34d399;
}

.hud-pill-amber {
  border-color: rgba(245, 158, 11, 0.4);
  background: rgba(245, 158, 11, 0.12);
  color: #fbbf24;
}

.hud-pill-red {
  border-color: rgba(239, 68, 68, 0.4);
  background: rgba(239, 68, 68, 0.14);
  color: #f87171;
}

.hud-pill-cyan {
  border-color: rgba(6, 182, 212, 0.4);
  background: rgba(6, 182, 212, 0.12);
  color: #38bdf8;
}

.hud-pill-orange {
  border-color: rgba(249, 115, 22, 0.4);
  background: rgba(249, 115, 22, 0.12);
  color: #fb923c;
}

/* Pulse animation for live indicators */
@keyframes pulse-ring {
  0% { transform: scale(0.95); opacity: 0.8; }
  50% { transform: scale(1.15); opacity: 1; }
  100% { transform: scale(0.95); opacity: 0.8; }
}

.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  animation: pulse-ring 2s infinite ease-in-out;
}
.pulse-green { background-color: #10b981; box-shadow: 0 0 8px #10b981; }
.pulse-amber { background-color: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
.pulse-red { background-color: #ef4444; box-shadow: 0 0 8px #ef4444; }
.pulse-cyan { background-color: #06b6d4; box-shadow: 0 0 8px #06b6d4; }
.pulse-slate { background-color: #94a3b8; }

/* Cockpit Banner Card */
.cockpit-banner {
  background: linear-gradient(135deg, rgba(22, 30, 49, 0.7) 0%, rgba(13, 19, 33, 0.85) 100%);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-left: 4px solid #f97316;
  border-radius: 12px;
  padding: 14px 20px;
  margin-bottom: 16px;
  backdrop-filter: blur(10px);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
</style>
"""


def apply_theme():
    """Injects high-fidelity dark cockpit styling into the Streamlit app."""
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def status_badge(state: str, label: str = None) -> str:
    """Return an HTML snippet for a sleek HUD status badge."""
    state_clean = state.lower()
    if state_clean in ("finished", "idle", "paused"):
        color_cls = "hud-pill-cyan"
        dot_cls = "pulse-cyan"
    elif state_clean in ("running", "active", "normal"):
        color_cls = "hud-pill-green"
        dot_cls = "pulse-green"
    elif state_clean in ("warm", "warning"):
        color_cls = "hud-pill-amber"
        dot_cls = "pulse-amber"
    elif state_clean in ("hotspot", "critical", "terminated"):
        color_cls = "hud-pill-red"
        dot_cls = "pulse-red"
    else:
        color_cls = "hud-pill-orange"
        dot_cls = "pulse-slate"

    txt = label or state.upper()
    return f"""<span class="hud-pill {color_cls}">
        <span class="pulse-dot {dot_cls}"></span>{txt}
    </span>"""


def header_banner(title: str, subtitle: str, badge_html: str = ""):
    """Renders a sleek top-level header banner with optional status pill."""
    badge_part = f"<div>{badge_html}</div>" if badge_html else ""
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px; flex-wrap:wrap; gap:12px;">
      <div>
        <h1 style="margin:0; font-size:1.85rem; line-height:1.2;">{title}</h1>
        <p style="margin:4px 0 0 0; color:#94a3b8; font-size:0.875rem;">{subtitle}</p>
      </div>
      {badge_part}
    </div>
    """, unsafe_allow_html=True)
