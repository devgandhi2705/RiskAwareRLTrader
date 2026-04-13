"""
demo_dashboard.py
-----------------
Streamlit dashboard for the AI Portfolio Manager live demo.
Reads demo_state.json written by demo_live_portfolio.py and
refreshes automatically every 3 seconds.

Run:
    streamlit run demo_dashboard.py
"""

import json
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh   # pip install streamlit-autorefresh

# ── Config ─────────────────────────────────────────────────────────────────────
ASSETS      = ["BTC", "ETH", "SPY", "GLD", "Silver", "Nifty50", "Sensex"]
STATE_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_state.json")
REFRESH_MS  = 3_000   # match STEP_SLEEP in demo script

REGIME_COLOUR = {
    "BULL":    "#00d26a",
    "NEUTRAL": "#f5a623",
    "BEAR":    "#e84040",
}

ASSET_COLOURS = [
    "#f7931a",  # BTC  — orange
    "#627eea",  # ETH  — indigo
    "#1f77b4",  # SPY  — blue
    "#ffdd57",  # GLD  — gold
    "#c0c0c0",  # Silver — silver
    "#ff7f0e",  # Nifty  — orange-ish
    "#2ca02c",  # Sensex — green
]

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Portfolio Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-refresh every REFRESH_MS
count = st_autorefresh(interval=REFRESH_MS, key="data_refresh")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark background */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] { background: #161b22; }

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
}
.metric-value.positive { color: #00d26a; }
.metric-value.negative { color: #e84040; }

/* Regime badge */
.regime-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.12em;
}
.regime-BULL    { background:#003d1c; color:#00d26a; border:1px solid #00d26a; }
.regime-NEUTRAL { background:#3d2a00; color:#f5a623; border:1px solid #f5a623; }
.regime-BEAR    { background:#3d0000; color:#e84040; border:1px solid #e84040; }

/* Section headers */
.section-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin: 16px 0 12px 0;
}

/* Progress bars for allocations */
.alloc-row { display: flex; align-items: center; margin: 5px 0; }
.alloc-label { width: 70px; font-size: 0.85rem; color: #c9d1d9; }
.alloc-bar-bg {
    flex: 1; height: 10px;
    background: #21262d;
    border-radius: 5px; overflow: hidden;
    margin: 0 10px;
}
.alloc-bar-fill { height: 100%; border-radius: 5px; }
.alloc-pct { width: 42px; text-align: right; font-size: 0.85rem; color: #8b949e; }

/* Step indicator */
.step-indicator {
    font-size: 0.75rem; color: #8b949e;
    text-align: right; margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── State loader ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=2)
def load_state(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def empty_state():
    return {
        "step": 0,
        "date": "—",
        "regime": "NEUTRAL",
        "portfolio_value": 100_000,
        "daily_return": 0.0,
        "weights": {a: 1 / len(ASSETS) for a in ASSETS},
        "portfolio_history": [100_000],
        "returns_history":   [],
        "sharpe":      0.0,
        "max_drawdown": 0.0,
        "volatility":  0.0,
        "agent":       "—",
        "total_return": 0.0,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def drawdown_series(values):
    v    = np.array(values, dtype=float)
    peak = np.maximum.accumulate(v)
    dd   = (v - peak) / np.maximum(peak, 1e-9) * 100
    return dd.tolist()


def colour_sign(val: float) -> str:
    if val > 0:
        return "positive"
    elif val < 0:
        return "negative"
    return ""


def fmt_sign(val: float, decimals=2) -> str:
    s = f"{val:+.{decimals}f}"
    return s


# ── Chart builders ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8b949e", size=11),
    margin=dict(l=8, r=8, t=36, b=8),
)


def chart_portfolio_value(history: list) -> go.Figure:
    n  = len(history)
    xs = list(range(n))
    ys = history

    color_line = "#00d26a" if ys[-1] >= ys[0] else "#e84040"
    fill_color = "rgba(0,210,106,0.08)" if ys[-1] >= ys[0] else "rgba(232,64,64,0.08)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color=color_line, width=2),
        fill="tozeroy",
        fillcolor=fill_color,
        name="Portfolio",
        hovertemplate="Step %{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=100_000, line_dash="dot",
                  line_color="#30363d", line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Portfolio Value", font=dict(size=13, color="#c9d1d9")),
        xaxis=dict(showgrid=False, zeroline=False, title="Step"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=False,
                   tickprefix="$", tickformat=",.0f"),
        showlegend=False,
        height=260,
    )
    return fig


def chart_drawdown(history: list) -> go.Figure:
    dd = drawdown_series(history)
    xs = list(range(len(dd)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=dd,
        mode="lines",
        line=dict(color="#e84040", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(232,64,64,0.12)",
        name="Drawdown",
        hovertemplate="Step %{x}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Drawdown", font=dict(size=13, color="#c9d1d9")),
        xaxis=dict(showgrid=False, zeroline=False, title="Step"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=True,
                   ticksuffix="%"),
        showlegend=False,
        height=200,
    )
    return fig


def chart_allocation_pie(weights: dict) -> go.Figure:
    labels = list(weights.keys())
    values = [weights[a] * 100 for a in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        marker=dict(colors=ASSET_COLOURS,
                    line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e6edf3"),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Asset Allocation", font=dict(size=13, color="#c9d1d9")),
        showlegend=False,
        height=280,
    )
    return fig


def chart_rolling_returns(returns: list) -> go.Figure:
    if len(returns) < 2:
        returns = [0.0, 0.0]
    xs = list(range(len(returns)))
    ys = [r * 100 for r in returns]

    colors = ["#00d26a" if v >= 0 else "#e84040" for v in ys]

    fig = go.Figure()
    fig.add_bar(
        x=xs, y=ys,
        marker_color=colors,
        name="Daily Return",
        hovertemplate="Step %{x}<br>%{y:.3f}%<extra></extra>",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Daily Returns", font=dict(size=13, color="#c9d1d9")),
        xaxis=dict(showgrid=False, zeroline=False, title="Step"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=True,
                   ticksuffix="%"),
        showlegend=False,
        height=200,
    )
    return fig


# ── Allocation HTML bars ────────────────────────────────────────────────────────

def alloc_bars_html(weights: dict) -> str:
    rows = ""
    for i, (asset, w) in enumerate(weights.items()):
        pct   = w * 100
        color = ASSET_COLOURS[i % len(ASSET_COLOURS)]
        rows += f"""
        <div class="alloc-row">
          <div class="alloc-label">{asset}</div>
          <div class="alloc-bar-bg">
            <div class="alloc-bar-fill"
                 style="width:{min(pct,100):.1f}%; background:{color};"></div>
          </div>
          <div class="alloc-pct">{pct:.1f}%</div>
        </div>"""
    return rows


# ── Main dashboard ─────────────────────────────────────────────────────────────

def main():
    raw = load_state(STATE_FILE)
    s   = raw if raw else empty_state()

    regime  = s.get("regime", "NEUTRAL")
    history = s.get("portfolio_history", [100_000])
    returns = s.get("returns_history", [])
    weights = s.get("weights", {a: 1/len(ASSETS) for a in ASSETS})
    step    = s.get("step", 0)
    total_r = s.get("total_return", 0.0)
    daily_r = s.get("daily_return", 0.0)

    # ── Title row ──────────────────────────────────────────────────────────────
    t_col, badge_col, step_col = st.columns([3, 1.2, 1])
    with t_col:
        st.markdown("## 📈 AI Portfolio Manager — Live Demo")
    with badge_col:
        st.markdown(
            f'<div style="padding-top:12px">'
            f'<span class="regime-badge regime-{regime}">{regime}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with step_col:
        st.markdown(
            f'<div class="step-indicator" style="padding-top:18px">'
            f'Agent: <b>{s.get("agent","—")}</b><br>'
            f'Step {step}/365 &nbsp;|&nbsp; {s.get("date","—")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Metric row ─────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)

    def _metric(col, label, value_str, css_class=""):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value {css_class}">{value_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _metric(m1, "Portfolio Value",
            f"${s['portfolio_value']:,.0f}")
    _metric(m2, "Total Return",
            f"{fmt_sign(total_r)}%",
            colour_sign(total_r))
    _metric(m3, "Today's Return",
            f"{fmt_sign(daily_r)}%",
            colour_sign(daily_r))
    _metric(m4, "Sharpe Ratio",
            f"{s.get('sharpe', 0):.2f}",
            "positive" if s.get("sharpe", 0) > 1 else "")
    _metric(m5, "Max Drawdown",
            f"-{s.get('max_drawdown', 0):.1f}%",
            "negative" if s.get("max_drawdown", 0) > 5 else "")

    st.markdown("")

    # ── Main charts row ────────────────────────────────────────────────────────
    left, right = st.columns([2.4, 1])

    with left:
        st.plotly_chart(
            chart_portfolio_value(history),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            chart_drawdown(history),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        st.plotly_chart(
            chart_allocation_pie(weights),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Allocation bars
        st.markdown(
            '<div class="section-header">Allocation Detail</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            alloc_bars_html(weights),
            unsafe_allow_html=True,
        )

    # ── Daily returns bar ──────────────────────────────────────────────────────
    st.plotly_chart(
        chart_rolling_returns(returns),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── Performance table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Performance Metrics</div>',
                unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    perf = {
        "Volatility (ann.)": f"{s.get('volatility', 0):.1f}%",
        "Sharpe Ratio":       f"{s.get('sharpe', 0):.2f}",
        "Max Drawdown":       f"-{s.get('max_drawdown', 0):.1f}%",
        "Total Return":       f"{fmt_sign(total_r)}%",
    }
    for col, (label, val) in zip([p1, p2, p3, p4], perf.items()):
        _metric(col, label, val)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    ts = s.get("timestamp", time.time())
    st.markdown(
        f'<div style="text-align:center; color:#484f58; font-size:0.72rem;">'
        f'Last updated: {time.strftime("%H:%M:%S", time.localtime(ts))} &nbsp;|&nbsp; '
        f'Auto-refresh every {REFRESH_MS//1000}s &nbsp;|&nbsp; '
        f'Run <code>python demo_live_portfolio.py --agent {s.get("agent","ppo").lower()}</code> to start simulation'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Waiting state if no simulation running
    if step == 0 and raw is None:
        st.info(
            "⏳ Waiting for simulation to start…  \n"
            "Run: `python demo_live_portfolio.py --agent ppo` in your terminal."
        )


if __name__ == "__main__":
    main()