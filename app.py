"""
ETF Seasonality Dashboard
=========================
A professional, web-based ETF seasonality analysis tool built with Streamlit + Plotly.
Run: streamlit run etf_seasonality_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ETF Seasonality Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — dark, refined, financial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f14;
    color: #d4d8e2;
}

/* Main background */
.stApp { background-color: #0d0f14; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #12151c !important;
    border-right: 1px solid #1e2330;
}

/* Title area */
.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #e8ecf5;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    color: #5a6278;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Info box */
.info-box {
    background: linear-gradient(135deg, #141826 0%, #1a1f2e 100%);
    border: 1px solid #2a3045;
    border-left: 3px solid #4a9eff;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.85rem;
    color: #8d9ab0;
    line-height: 1.6;
}
.info-box strong { color: #a8b8cc; }

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: #12151c;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
}
.metric-label {
    font-size: 0.7rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #e2e8f0;
}
/* Updated KPI text colors for minimalist look */
.metric-pos { color: #FFFFFF; } 
.metric-neg { color: #BBBBBB; } 

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #4a9eff;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
    border-bottom: 1px solid #1e2330;
    padding-bottom: 0.4rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #12151c;
    border-bottom: 1px solid #1e2330;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #5a6278;
    padding: 0.6rem 1.4rem;
    border-radius: 0;
    letter-spacing: 0.05em;
}
.stTabs [aria-selected="true"] {
    color: #4a9eff !important;
    border-bottom: 2px solid #4a9eff !important;
    background: transparent !important;
}

/* Input widgets */
.stTextInput input, .stNumberInput input {
    background: #12151c !important;
    border: 1px solid #1e2330 !important;
    color: #d4d8e2 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 4px !important;
}
.stRadio label, .stCheckbox label { color: #8d9ab0 !important; font-size: 0.85rem !important; }
.stRadio [data-testid="stRadio"] > div { gap: 0.5rem; }

/* Download button */
.stDownloadButton button {
    background: #1a2540 !important;
    border: 1px solid #2a4070 !important;
    color: #4a9eff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 4px !important;
    letter-spacing: 0.05em;
}
.stDownloadButton button:hover {
    background: #223060 !important;
    border-color: #4a9eff !important;
}

/* Spinner */
.stSpinner { color: #4a9eff; }

/* Plotly chart container */
.js-plotly-plot { border-radius: 6px; }

/* Error / warning */
.stAlert { border-radius: 6px; font-size: 0.85rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #1e2330; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2a3045; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS & THEME
# ─────────────────────────────────────────────
CURRENT_YEAR = datetime.today().year
PLOTLY_TEMPLATE = "plotly_dark"

# --- OUR CUSTOM MINIMALIST COLOR PALETTE ---
COLORS = {
    "pos_bar":    "#555555",   # dark grey for positive
    "neg_bar":    "#BBBBBB",   # light grey for negative
    "cur_year":   "#FFFFFF",   # bold white for current year
    "avg_line":   "#00E5FF",   # neon blue cumulative average
    "spaghetti":  "rgba(255, 255, 255, 0.15)", # pure white transparent ghost lines
    "vline":      "#FF4444",   # red vertical line for current period
    "text_annot": "#cbd5e1",
    "bg":         "#0d0f14",
    "grid":       "#1a1f2e",
    "border":     "#1e2330",
}

# ─────────────────────────────────────────────
# DATA FETCHING — cached so UI changes are instant
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str, start_year: int) -> pd.DataFrame | None:
    """Download adjusted close prices via yfinance. Returns daily close series."""
    try:
        start_str = f"{start_year}-01-01"
        df = yf.download(ticker, start=start_str, auto_adjust=True, progress=False)
        if df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        close = close.dropna()
        close.index = pd.to_datetime(close.index)
        return close.to_frame(name="close")
    except Exception:
        return None


def compute_seasonality(
    df: pd.DataFrame,
    timeframe: str,
    start_year: int,
) -> dict:
    close = df["close"].copy()

    if timeframe == "Weekly":
        resampled = close.resample("W-FRI").last().dropna()
        roc = resampled.pct_change().dropna() * 100
        roc.index = pd.to_datetime(roc.index)
        roc_df = roc.to_frame(name="roc")
        roc_df["year"] = roc_df.index.isocalendar().year.astype(int)
        roc_df["period"] = roc_df.index.isocalendar().week.astype(int)
        w53_years = roc_df[roc_df["period"] == 53]["year"].nunique()
        if w53_years < 3:
            roc_df = roc_df[roc_df["period"] != 53]
    else:
        resampled = close.resample("ME").last().dropna()
        roc = resampled.pct_change().dropna() * 100
        roc.index = pd.to_datetime(roc.index)
        roc_df = roc.to_frame(name="roc")
        roc_df["year"] = roc_df.index.year
        roc_df["period"] = roc_df.index.month

    today = datetime.today()
    if timeframe == "Weekly":
        current_period = today.isocalendar().week
    else:
        current_period = today.month

    cur_year_data = roc_df[roc_df["year"] == CURRENT_YEAR].copy()
    hist_data = roc_df[roc_df["year"] < CURRENT_YEAR].copy()

    completed_years = sorted(hist_data["year"].unique())

    pivot = hist_data.pivot_table(index="year", columns="period", values="roc")
    periods = sorted(pivot.columns.tolist())

    def _avg(pivot_subset):
        return pivot_subset.mean()

    def _winrate(pivot_subset):
        return (pivot_subset > 0).sum() / pivot_subset.notna().sum() * 100

    def _window(pivot, n):
        years = sorted(pivot.index.tolist())
        subset = years[-n:] if len(years) >= n else years
        return pivot.loc[subset]

    pv5   = _window(pivot, 5)
    pv10  = _window(pivot, 10)
    pvmax = pivot 

    avg_5   = _avg(pv5)
    avg_10  = _avg(pv10)
    avg_max = _avg(pvmax)

    wr_5   = _winrate(pv5)
    wr_10  = _winrate(pv10)
    wr_max = _winrate(pvmax)

    cur_pivot = cur_year_data.pivot_table(index="year", columns="period", values="roc")
    cur_roc = cur_pivot.iloc[0] if not cur_pivot.empty else pd.Series(dtype=float)

    return {
        "periods":         periods,
        "avg_5":           avg_5,
        "avg_10":          avg_10,
        "avg_max":         avg_max,
        "wr_5":            wr_5,
        "wr_10":           wr_10,
        "wr_max":          wr_max,
        "cur_roc":         cur_roc,
        "pivot":           pivot,
        "completed_years": completed_years,
        "current_period":  current_period,
        "start_year":      start_year,
    }

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def _base_layout(title: str, height: int = 380) -> dict:
    return dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        title=dict(text=title, font=dict(family="IBM Plex Mono", size=13, color="#8d9ab0"), x=0.01),
        height=height,
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis=dict(
            gridcolor=COLORS["grid"],
            linecolor=COLORS["border"],
            showline=True,
            tickfont=dict(family="IBM Plex Mono", size=10, color="#5a6278"),
            title_font=dict(family="IBM Plex Mono", size=10, color="#5a6278"),
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            linecolor=COLORS["border"],
            showline=False,
            zeroline=True,
            zerolinecolor="#2a3045",
            zerolinewidth=1,
            tickfont=dict(family="IBM Plex Mono", size=10, color="#5a6278"),
            ticksuffix="%",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", size=10, color="#8d9ab0"),
            orientation="h",
            y=1.08, x=0,
        ),
        hovermode="x unified",
    )


def make_bar_chart(
    data: dict,
    window_key: str, 
    show_winrate: bool,
    timeframe: str,
    title: str,
) -> go.Figure:
    avg   = data[f"avg_{window_key}"]
    wr    = data[f"wr_{window_key}"]
    cur   = data["cur_roc"]
    periods = data["periods"]
    cur_period = data["current_period"]

    bar_colors = [COLORS["pos_bar"] if v >= 0 else COLORS["neg_bar"]
                  for v in avg.reindex(periods).fillna(0)]

    annotations = []
    if show_winrate:
        for p in periods:
            v = avg.get(p, np.nan)
            w = wr.get(p, np.nan)
            if pd.isna(v) or pd.isna(w):
                continue
            y_offset = abs(v) * 0.07 + 0.15
            ay_offset = 18 if v >= 0 else -18
            annotations.append(dict(
                x=p, y=v + (y_offset if v >= 0 else -y_offset),
                text=f"{w:.0f}%",
                showarrow=False,
                font=dict(family="IBM Plex Mono", size=9, color="#FFFFFF"),
                textangle=-90 if timeframe == "Weekly" else 0,
                xanchor="center",
                yanchor="bottom" if v >= 0 else "top",
            ))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=periods,
        y=avg.reindex(periods).values,
        marker_color=bar_colors,
        marker_line_width=0,
        opacity=1.0, # Removed opacity drop so greys pop nicely
        name="Hist. Avg",
        hovertemplate="Period %{x}<br>Avg ROC: %{y:.2f}%<extra></extra>",
    ))

    cur_x = [p for p in periods if p in cur.index]
    cur_y = [cur[p] for p in cur_x]
    if cur_x:
        fig.add_trace(go.Scatter(
            x=cur_x, y=cur_y,
            mode="lines+markers",
            line=dict(color=COLORS["cur_year"], width=3),
            marker=dict(size=6, color=COLORS["cur_year"]),
            name=f"{CURRENT_YEAR} Actual",
            hovertemplate="Period %{x}<br>Actual: %{y:.2f}%<extra></extra>",
            zorder=10,
        ))

    if cur_period in periods:
        fig.add_vline(
            x=cur_period, line_dash="dash",
            line_color=COLORS["vline"], line_width=1.5,
            annotation_text=f"Now: {cur_period}",
            annotation_font=dict(family="IBM Plex Mono", size=10, color=COLORS["vline"]),
            annotation_position="top right",
        )

    layout = _base_layout(title)
    layout["annotations"] = annotations
    layout["xaxis"]["title"] = "Week" if timeframe == "Weekly" else "Month"
    layout["xaxis"]["dtick"] = 1
    fig.update_layout(**layout)
    return fig


def make_cumulative_chart(
    data: dict,
    window_key: str,
    show_spaghetti: bool,
    timeframe: str,
    title: str,
) -> go.Figure:
    avg     = data[f"avg_{window_key}"]
    cur     = data["cur_roc"]
    pivot   = data["pivot"]
    periods = data["periods"]
    cur_period = data["current_period"]
    completed_years = data["completed_years"]

    def _cum(series, periods_list):
        vals = [series.get(p, np.nan) for p in periods_list]
        cum = [0.0]
        running = 100.0
        for v in vals:
            if pd.isna(v):
                cum.append(cum[-1])
            else:
                running *= (1 + v / 100)
                cum.append(running - 100.0)
        return
