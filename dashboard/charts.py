"""Plotly figures for the dashboard. Every figure is built from data passed in;
nothing is hardcoded. Policy colours come from dcsched.palette so a scheduler
keeps the same colour on every page.
Styled with a high-fidelity dark cockpit telemetry aesthetic."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dcsched.palette import color_for, label_for, sort_policies, LIGHT, DARK

# Modern dark telemetry FLIR-inspired heat scale:
# Deep cool indigo (idle) -> Cyan/Teal (cool) -> Emerald (nominal) -> Amber (warming) -> Neon Red (hotspot) -> Bright Yellow-White (critical)
HEAT = [
    [0.0, "#0c1b33"],    # 25°C ambient idle
    [0.22, "#0284c7"],   # 35°C nominal cool
    [0.45, "#059669"],   # 48°C normal operating
    [0.68, "#f59e0b"],   # 58°C safe threshold boundary
    [0.85, "#ef4444"],   # 68°C hotspot
    [1.0, "#fef08a"]     # 75°C peak/critical incandescent
]

RED = DARK["threshold"]


def _base(fig, height=320, title=None):
    """Applies modern dark cockpit styling across all Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.45)",
        height=height,
        margin=dict(l=14, r=14, t=44 if title else 14, b=14),
        title=dict(
            text=title,
            x=0.01,
            font=dict(family="Inter, -apple-system, sans-serif", size=14, color="#f1f5f9", weight="bold")
        ) if title else None,
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0,
            bgcolor="rgba(17, 24, 39, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
            font=dict(family="Inter, sans-serif", size=11, color="#cbd5e1")
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1f2937",
            bordercolor="rgba(255, 255, 255, 0.15)",
            font=dict(family="Inter, sans-serif", size=12, color="#f8fafc")
        )
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(color="#94a3b8", size=11, family="Inter, sans-serif")
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(color="#94a3b8", size=11, family="Inter, sans-serif")
    )
    return fig


def rack_heatmap(snapshot, thermal_cfg, n_racks, per_rack):
    """Servers laid out as racks (rows) x slots (columns), coloured by temperature."""
    z = np.zeros((n_racks, per_rack))
    txt = [[""] * per_rack for _ in range(n_racks)]
    hov = [[""] * per_rack for _ in range(n_racks)]
    for r in snapshot:
        i, j = r["rack"], r["server"] % per_rack
        z[i, j] = r["temperature_c"]
        flag = " ⚠" if r["status"] in ("hotspot", "critical") else ""
        txt[i][j] = f"<b>S{r['server']}</b>{flag}<br><b>{r['temperature_c']:.0f}°C</b><br>{r['cpu_util']:.0%}"
        hov[i][j] = (f"Server {r['server']} (rack {r['rack']})<br>Temp {r['temperature_c']:.1f} °C"
                     f"<br>CPU {r['cpu_util']:.0%}<br>Power {r['power_kw']:.2f} kW"
                     f"<br>Tasks {r['n_tasks']}<br>Status {r['status']}")

    fig = go.Figure(go.Heatmap(
        z=z, text=txt, texttemplate="%{text}", hovertext=hov, hoverinfo="text",
        colorscale=HEAT, zmin=thermal_cfg.ambient_temp, zmax=thermal_cfg.critical_temp,
        xgap=5, ygap=5,
        colorbar=dict(
            title=dict(text="°C", font=dict(color="#cbd5e1", size=12)),
            tickfont=dict(color="#94a3b8"),
            thickness=14,
            len=0.92,
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)"
        )
    ))
    fig.update_yaxes(
        tickvals=list(range(n_racks)),
        ticktext=[f"Rack {i}" for i in range(n_racks)],
        autorange="reversed",
        tickfont=dict(color="#cbd5e1", size=12, family="Inter, sans-serif"),
        gridcolor="rgba(255, 255, 255, 0.05)"
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(220, 95 * n_racks),
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Inter, sans-serif")
    )
    return fig


def temperature_history(history, safe_temp):
    h = pd.DataFrame(history)
    fig = go.Figure()
    if not h.empty:
        fig.add_trace(go.Scatter(
            x=h["step"], y=h["max_temp"], name="Hottest server",
            line=dict(color="#f97316", width=2.4),
            fill="tonexty", fillcolor="rgba(249, 115, 22, 0.07)"
        ))
        fig.add_trace(go.Scatter(
            x=h["step"], y=h["mean_temp"], name="Mean temperature",
            line=dict(color="#38bdf8", width=1.8, dash="dot")
        ))
    fig.add_hline(
        y=safe_temp, line=dict(color=RED, dash="dash", width=1.4),
        annotation_text=f"safe threshold ({safe_temp:.0f} °C)",
        annotation_position="top left",
        annotation_font=dict(color="#f87171", size=11, family="Inter, sans-serif")
    )
    fig.update_yaxes(title="Temperature (°C)")
    fig.update_xaxes(title="Simulation step")
    return _base(fig)


def queue_and_completed(history):
    h = pd.DataFrame(history)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not h.empty:
        fig.add_trace(go.Scatter(
            x=h["step"], y=h["queue"], name="Queue length",
            line=dict(color="#38bdf8", width=1.8),
            fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.12)"
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=h["step"], y=h["completed"], name="Completed (cum.)",
            line=dict(color="#10b981", width=2.2)
        ), secondary_y=True)
    fig.update_yaxes(title_text="Queue length", secondary_y=False, title_font=dict(color="#38bdf8"))
    fig.update_yaxes(title_text="Tasks completed", secondary_y=True, title_font=dict(color="#10b981"))
    fig.update_xaxes(title="Simulation step")
    return _base(fig)


def power_history(history):
    h = pd.DataFrame(history)
    fig = go.Figure()
    if not h.empty:
        fig.add_trace(go.Scatter(
            x=h["step"], y=h["it_power_kw"], name="IT power", stackgroup="p",
            line=dict(color="#818cf8", width=0.8),
            fillcolor="rgba(99, 102, 241, 0.45)"
        ))
        fig.add_trace(go.Scatter(
            x=h["step"], y=h["cooling_power_kw"], name="Cooling power (proxy)",
            stackgroup="p",
            line=dict(color="#fbbf24", width=0.8),
            fillcolor="rgba(245, 158, 11, 0.45)"
        ))
    fig.update_yaxes(title="Power (kW)")
    fig.update_xaxes(title="Simulation step")
    return _base(fig)


def server_bars(snapshot, safe_temp):
    df = pd.DataFrame(snapshot)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Temperature per server (°C)", "CPU utilisation per server"))
    colors = ["#ef4444" if t > safe_temp else ("#f59e0b" if t > safe_temp - 5 else "#06b6d4") for t in df["temperature_c"]]
    fig.add_trace(go.Bar(
        x=df["server"], y=df["temperature_c"],
        marker_color=colors, showlegend=False,
        marker=dict(line=dict(width=0))
    ), 1, 1)
    fig.add_hline(
        y=safe_temp, line=dict(color=RED, dash="dash", width=1.2), row=1, col=1,
        annotation_text="safe limit", annotation_position="top left",
        annotation_font=dict(color="#f87171", size=10)
    )
    fig.add_trace(go.Bar(
        x=df["server"], y=df["cpu_util"],
        marker_color="#8b5cf6", showlegend=False,
        marker=dict(line=dict(width=0))
    ), 1, 2)
    fig.update_yaxes(range=[0, 1.05], row=1, col=2, tickformat=".0%")
    fig.update_xaxes(title="Server ID", dtick=1)
    fig.for_each_annotation(lambda a: a.update(font=dict(color="#f1f5f9", size=13, family="Inter, sans-serif")))
    return _base(fig, height=300)


def metric_bars(summary: pd.DataFrame, metric: str, title: str, ylabel: str, fmt="{:.3f}"):
    """Bar chart of ``<metric>_mean`` with ``<metric>_std`` error bars, one bar per policy."""
    names = sort_policies(list(summary["policy"]))
    rows = summary.set_index("policy").loc[names]
    fig = go.Figure(go.Bar(
        x=[label_for(n) for n in names], y=rows[f"{metric}_mean"].values,
        error_y=dict(type="data", array=rows[f"{metric}_std"].values, color="#94a3b8", thickness=1.4),
        marker_color=[color_for(n) for n in names],
        text=[fmt.format(v) for v in rows[f"{metric}_mean"].values], textposition="outside",
        textfont=dict(color="#f1f5f9", size=11, family="JetBrains Mono, monospace")
    ))
    fig.update_yaxes(title=ylabel)
    fig.update_layout(showlegend=False, hovermode="closest")
    return _base(fig, height=300, title=title)


def trace_comparison(traces: dict, safe_temp: float, key="max_temp", title="Hottest-server temperature over one episode"):
    fig = go.Figure()
    for name in sort_policies(list(traces)):
        y = traces[name][key]
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(y) + 1), y=y, name=label_for(name),
            line=dict(color=color_for(name), width=1.6)
        ))
    fig.add_hline(
        y=safe_temp, line=dict(color=RED, dash="dash", width=1.4),
        annotation_text="safe threshold", annotation_position="top left",
        annotation_font=dict(color="#f87171", size=11)
    )
    fig.update_yaxes(title="Temperature (°C)")
    fig.update_xaxes(title="Simulation step")
    return _base(fig, height=360, title=title)


def learning_curves(curves: dict, window: int = 10):
    """curves: label -> {'timesteps': array, 'returns': array}; shows a rolling mean."""
    fig = go.Figure()
    for label, c in curves.items():
        r = pd.Series(c["returns"]).rolling(window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=c["timesteps"], y=r, name=label,
            line=dict(color=color_for(label), width=2.0)
        ))
    fig.update_xaxes(title="Training timesteps")
    fig.update_yaxes(title=f"Episode return (rolling mean of {window})")
    return _base(fig, height=360)


def radar_comparison(summary: pd.DataFrame):
    """Spider/radar comparison of policy trade-offs (all dimensions 0-100, higher is better)."""
    categories = [
        "Thermal Safety",
        "Energy Efficiency",
        "Completion Rate",
        "Throughput",
        "Low Latency"
    ]
    fig = go.Figure()

    # Normalize metrics to 0-100 score where higher is always better
    max_hotspot = max(summary["hotspot_frequency_mean"].max(), 1e-4)
    max_energy = max(summary["it_energy_kwstep_mean"].max(), 1.0)
    max_wait = max(summary["mean_wait_time_mean"].max(), 0.1)
    max_tp = max(summary["throughput_mean"].max(), 0.1)

    for _, row in summary.iterrows():
        pname = row["policy"]
        # Inverse metrics (lower raw = higher score)
        thermal_score = max(0.0, 100.0 * (1.0 - row["hotspot_frequency_mean"] / max_hotspot))
        energy_score = max(0.0, 100.0 * (1.0 - row["it_energy_kwstep_mean"] / max_energy))
        latency_score = max(0.0, 100.0 * (1.0 - row["mean_wait_time_mean"] / max_wait))
        # Direct metrics (higher raw = higher score)
        comp_score = min(100.0, max(0.0, row["completion_rate_mean"] * 100.0))
        tp_score = min(100.0, max(0.0, (row["throughput_mean"] / max_tp) * 100.0))

        values = [thermal_score, energy_score, comp_score, tp_score, latency_score]
        values.append(values[0])  # close loop

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=label_for(pname),
            line=dict(color=color_for(pname), width=2.0),
            fill="toself",
            opacity=0.15
        ))

    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 105], showticklabels=False, linecolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(tickfont=dict(size=11, color="#cbd5e1", family="Inter, sans-serif"),
                             linecolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(15, 23, 42, 0.45)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", y=-0.15, x=0, bgcolor="rgba(17,24,39,0.7)"),
        showlegend=True
    )
    return fig
