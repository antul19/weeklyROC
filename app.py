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
from datetime import datetime
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
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f14;
    color: #d4d8e2;
}

.stApp { background-color: #0d0f14; }

section[data-testid="stSidebar"] {
    background: #12151c !important;
    border-right: 1px solid #1e2330;
}

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
.metric-pos { color: #FFFFFF; } 
.metric-neg { color: #BBBBBB; } 

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

.stTextInput input, .stNumberInput input {
    background: #12151c !important;
    border: 1px solid #1e2330 !important;
    color: #d4d8e2 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 4px !important;
}
.stRadio label, .stCheckbox label { color: #8d9ab0 !important; font-size: 0.85rem !important; }
.stRadio [data-testid="stRadio"] > div { gap: 0.5rem; }

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

.stSpinner { color: #4a9eff; }
.js-plotly-plot { border-radius: 6px; }
.stAlert { border-radius: 6px; font-size: 0.85rem; }

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

COLORS = {
    "pos_bar":    "#555555",   
    "neg_bar":    "#BBBBBB",   
    "cur_year":   "#FFFFFF",                   
    "cur_year_bar": "rgba(255, 255, 255, 0.4)",
    "avg_line":   "#00E5FF",   
    "spaghetti":  "rgba(255, 255, 255, 0.15)", 
    "vline":      "#FF4444",   
    "text_annot": "#cbd5e1",
    "bg":         "#0d0f14",
    "grid":       "#1a1f2e",
    "border":     "#1e2330",
    
    # Presidential Cycle Colors
    "cycle_post": "#FF9900",  
    "cycle_mid":  "#B026FF",  
    "cycle_pre":  "#39FF14",  
    "cycle_elec": "#00E5FF",  
    
    # Global Macro Colors
    "us":         "#00E5FF",  # Neon Blue
    "canada":     "#FF3333",  # Crimson Red
    "india":      "#FFA500",  # Orange
    "gold":       "#FFD700",  # True Gold
    "btc":        "#F7931A",  # Bitcoin Orange
    "vix":        "#FF00FF",  # Magenta
    "oil":        "#00FF00",  # Lime Green
    "tnx":        "#B0C4DE",  # Light Steel Blue
    
    "macro_zone": "rgba(255, 68, 68, 0.12)"
}

MACRO_EVENTS = [
    {"start": "1929-08-01", "end": "1933-03-01", "name": "Great Depression"},
    {"start": "1939-09-01", "end": "1945-09-02", "name": "World War II"},
    {"start": "1973-01-01", "end": "1974-12-01", "name": "1970s Oil/Inflation"},
    {"start": "2000-03-01", "end": "2002-10-01", "name": "Dot-Com Crash"},
    {"start": "2007-10-01", "end": "2009-03-01", "name": "Global Fin Crisis"},
    {"start": "2020-02-01", "end": "2020-04-01", "name": "COVID-19 Crash"}
]

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_seasonality_data_v4(ticker: str, start_year: int, timeframe: str) -> pd.DataFrame | None:
    try:
        start_str = f"{start_year - 1}-11-01"
        interval = "1wk" if timeframe == "Weekly" else "1mo"
        
        df = yf.download(ticker, start=start_str, interval=interval, auto_adjust=False, progress=False)
        if df.empty: return None
        
        close = df["Close"]
        if isinstance(close, pd.DataFrame): close = close.squeeze()
        close = close.dropna()
        
        roc = close.pct_change() * 100
        roc_df = roc.to_frame(name="roc")
        roc_df.index = pd.to_datetime(roc_df.index)
        
        if timeframe == "Weekly":
            roc_df["year"] = roc_df.index.isocalendar().year.astype(int)
            roc_df["period"] = roc_df.index.isocalendar().week.astype(int)
            w53_years = roc_df[roc_df["period"] == 53]["year"].nunique()
            if w53_years < 3: roc_df = roc_df[roc_df["period"] != 53]
        else:
            roc_df["year"] = roc_df.index.year
            roc_df["period"] = roc_df.index.month
            
        roc_df = roc_df[roc_df["year"] >= start_year]
        return roc_df
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_presidential_cycle_data() -> pd.DataFrame | None:
    try:
        df = yf.download("^GSPC", start="1980-12-01", interval="1mo", auto_adjust=False, progress=False)
        if df.empty: return None
        
        close = df["Close"]
        if isinstance(close, pd.DataFrame): close = close.squeeze()
        close = close.dropna()
        
        roc = close.pct_change() * 100
        roc_df = roc.to_frame(name="roc")
        roc_df.index = pd.to_datetime(roc_df.index)
        
        roc_df["year"] = roc_df.index.year
        roc_df["period"] = roc_df.index.month
        
        roc_df = roc_df[roc_df["year"] >= 1981]
        return roc_df
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_global_macro_data() -> dict:
    tickers = {
        "US (^GSPC)": "^GSPC",
        "Canada (^GSPTSE)": "^GSPTSE",
        "India (^NSEI)": "^NSEI",
        "Gold (GC=F)": "GC=F",
        "Bitcoin (BTC-USD)": "BTC-USD",
        "Volatility (^VIX)": "^VIX",
        "Crude Oil (CL=F)": "CL=F",
        "10Yr Yield (^TNX)": "^TNX"
    }
    data_dict = {}
    
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start="1927-12-01", interval="1d", auto_adjust=False, progress=False)
            if not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame): close = close.squeeze()
                close = close.dropna().resample("ME").last().dropna()
                data_dict[name] = close
        except Exception:
            pass
    return data_dict

def compute_seasonality(roc_df: pd.DataFrame, timeframe: str, start_year: int) -> dict:
    periods = list(range(1, 53)) if timeframe == "Weekly" else list(range(1, 13))
    today = datetime.today()
    current_period = today.isocalendar().week if timeframe == "Weekly" else today.month

    cur_year_data = roc_df[roc_df["year"] == CURRENT_YEAR].copy()
    hist_data = roc_df[roc_df["year"] < CURRENT_YEAR].copy()

    completed_years = sorted(hist_data["year"].unique())
    pivot = hist_data.pivot_table(index="year", columns="period", values="roc")

    def _avg(pivot_subset): return pivot_subset.mean()
    def _winrate(pivot_subset): return (pivot_subset > 0).sum() / pivot_subset.notna().sum() * 100
    def _window(pivot, n):
        years = sorted(pivot.index.tolist())
        subset = years[-n:] if len(years) >= n else years
        return pivot.loc[subset]

    pv5, pv10, pvmax = _window(pivot, 5), _window(pivot, 10), pivot 
    avg_5, avg_10, avg_max = _avg(pv5), _avg(pv10), _avg(pvmax)
    wr_5, wr_10, wr_max = _winrate(pv5), _winrate(pv10), _winrate(pvmax)

    cur_pivot = cur_year_data.pivot_table(index="year", columns="period", values="roc")
    cur_roc = cur_pivot.iloc[0] if not cur_pivot.empty else pd.Series(dtype=float)

    return {
        "periods": periods, "avg_5": avg_5, "avg_10": avg_10, "avg_max": avg_max,
        "wr_5": wr_5, "wr_10": wr_10, "wr_max": wr_max, "cur_roc": cur_roc,
        "pivot": pivot, "completed_years": completed_years, 
        "current_period": current_period, "start_year": start_year,
    }

def compute_cycle_seasonality(roc_df: pd.DataFrame) -> dict:
    def get_cycle_month(y, m):
        rem = y % 4
        if rem == 1: offset = 0
        elif rem == 2: offset = 12
        elif rem == 3: offset = 24
        elif rem == 0: offset = 36
        return offset + m
        
    roc_df["cycle_month"] = roc_df.apply(lambda r: get_cycle_month(r["year"], r["period"]), axis=1)
    current_cycle_start = CURRENT_YEAR - ((CURRENT_YEAR - 1) % 4)
    
    hist_data = roc_df[roc_df["year"] < current_cycle_start]
    avg_roc = hist_data.groupby("cycle_month")["roc"].mean()
    
    cur_cycle_data = roc_df[roc_df["year"] >= current_cycle_start]
    cur_roc = cur_cycle_data.set_index("cycle_month")["roc"] if not cur_cycle_data.empty else pd.Series(dtype=float)
    
    return {
        "avg_roc": avg_roc, "cur_roc": cur_roc, "current_cycle_start": current_cycle_start
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
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", size=10, color="#8d9ab0"),
            orientation="h",
            y=1.08, x=0,
        ),
        hovermode="x unified",
    )

def make_bar_chart(data: dict, window_key: str, show_winrate: bool, timeframe: str, title: str) -> go.Figure:
    avg, wr, cur = data[f"avg_{window_key}"], data[f"wr_{window_key}"], data["cur_roc"]
    periods, cur_period = data["periods"], data["current_period"]

    bar_colors = [COLORS["pos_bar"] if v >= 0 else COLORS["neg_bar"] for v in avg.reindex(periods).fillna(0)]
    
    wr_text = []
    for p in periods:
        w = wr.get(p, np.nan)
        if show_winrate and not pd.isna(w): wr_text.append(f"{w:.0f}%")
        else: wr_text.append("")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=periods, y=avg.reindex(periods).values,
        text=wr_text, textposition='outside',
        textfont=dict(family="IBM Plex Mono", size=10, color="#FFFFFF"),
        textangle=-90 if timeframe == "Weekly" else 0,
        marker_color=bar_colors, marker_line_width=0, opacity=1.0,
        name="Hist. Avg", hovertemplate="Period %{x}<br>Avg ROC: %{y:.2f}%<extra></extra>",
        cliponaxis=False 
    ))

    cur_x = [p for p in periods if p in cur.index]
    if cur_x:
        fig.add_trace(go.Scatter(
            x=cur_x, y=[cur[p] for p in cur_x],
            mode="lines+markers", line=dict(color=COLORS["cur_year_bar"], width=2),
            marker=dict(size=7, color="#FFFFFF", line=dict(color="#000000", width=1.5)), 
            name=f"{CURRENT_YEAR} Actual", hovertemplate="Period %{x}<br>Actual: %{y:.2f}%<extra></extra>",
            zorder=10,
        ))

    if cur_period in periods:
        fig.add_vline(x=cur_period, line_dash="dash", line_color=COLORS["vline"], line_width=1.5,
                      annotation_text=f"Now: {cur_period}", annotation_font=dict(family="IBM Plex Mono", size=10, color=COLORS["vline"]),
                      annotation_position="top right")

    layout = _base_layout(title)
    layout["xaxis"]["title"] = "Week" if timeframe == "Weekly" else "Month"
    layout["xaxis"]["dtick"] = 1
    layout["yaxis"]["ticksuffix"] = "%"
    
    max_p = 52 if timeframe == "Weekly" else 12
    fig.update_layout(**layout)
    fig.update_xaxes(range=[0.5, max_p + 0.5]) 
    fig.update_yaxes(autorange=True) 
    
    return fig

def make_cumulative_chart(data: dict, window_key: str, show_spaghetti: bool, timeframe: str, title: str) -> go.Figure:
    avg, cur, pivot = data[f"avg_{window_key}"], data["cur_roc"], data["pivot"]
    periods, cur_period = data["periods"], data["current_period"]

    def _cum(series, periods_list):
        vals = [series.get(p, np.nan) for p in periods_list]
        cum, running = [0.0], 100.0
        for v in vals:
            if pd.isna(v): cum.append(cum[-1])
            else:
                running *= (1 + v / 100)
                cum.append(running - 100.0)
        return cum  

    x_anchor = [0] + periods  
    fig = go.Figure()

    if show_spaghetti:
        all_years = sorted(pivot.index.tolist())
        yrs = all_years[-5:] if window_key == "5" else (all_years[-10:] if window_key == "10" else all_years)
        for i, yr in enumerate(yrs):
            yr_series = pivot.loc[yr] if yr in pivot.index else pd.Series(dtype=float)
            fig.add_trace(go.Scatter(
                x=x_anchor, y=_cum(yr_series, periods),
                mode="lines", line=dict(color=COLORS["spaghetti"], width=1.5),
                showlegend=(i == 0), name="Past Years" if i == 0 else None,
                legendgroup="spaghetti", hoverinfo="skip",
            ))

    fig.add_trace(go.Scatter(
        x=x_anchor, y=_cum(avg, periods),
        mode="lines", line=dict(color=COLORS["avg_line"], width=3.5),
        name="Hist. Avg (Cum.)", hovertemplate="Period %{x}<br>Cum. Return: %{y:.2f}%<extra></extra>",
    ))

    cur_x_available = [p for p in periods if p in cur.index]
    if cur_x_available:
        n = len(cur_x_available) + 1
        fig.add_trace(go.Scatter(
            x=x_anchor[:n], y=_cum(pd.Series({p: cur.get(p, np.nan) for p in periods}), periods)[:n],
            mode="lines+markers", line=dict(color=COLORS["cur_year"], width=3), 
            marker=dict(size=6, color=COLORS["cur_year"]),
            name=f"{CURRENT_YEAR} Actual (Cum.)", hovertemplate="Period %{x}<br>Cum. Return: %{y:.2f}%<extra></extra>",
        ))

    if cur_period in x_anchor:
        fig.add_vline(x=cur_period, line_dash="dash", line_color=COLORS["vline"], line_width=1.5,
                      annotation_text=f"Now: {cur_period}", annotation_font=dict(family="IBM Plex Mono", size=10, color=COLORS["vline"]),
                      annotation_position="top right")

    layout = _base_layout(title)
    layout["xaxis"]["title"] = "Week" if timeframe == "Weekly" else "Month"
    layout["xaxis"]["dtick"] = 1
    layout["yaxis"]["ticksuffix"] = "%"
    
    max_p = 52 if timeframe == "Weekly" else 12
    fig.update_layout(**layout)
    fig.update_xaxes(range=[-0.5, max_p + 0.5]) 
    
    return fig

def make_presidential_cycle_chart(cycle_data: dict) -> go.Figure:
    avg_roc = cycle_data["avg_roc"]
    cur_roc = cycle_data["cur_roc"]
    start_yr = cycle_data["current_cycle_start"]
    end_yr = start_yr + 3
    
    fig = go.Figure()
    periods = list(range(1, 49))
    x_anchor = [0] + periods
    
    def _cum(series):
        cum, running = [0.0], 100.0
        for p in periods:
            v = series.get(p, np.nan)
            if pd.isna(v): cum.append(cum[-1])
            else:
                running *= (1 + v / 100)
                cum.append(running - 100.0)
        return cum
        
    fig.add_trace(go.Scatter(
        x=x_anchor, y=_cum(avg_roc),
        mode="lines", line=dict(color=COLORS["avg_line"], width=3.5),
        name="Historical Avg (48-Month Cycle)", hovertemplate="Month %{x}<br>Avg Cum. Return: %{y:.2f}%<extra></extra>"
    ))
            
    cur_x_available = [p for p in periods if p in cur_roc.index]
    if cur_x_available:
        n = len(cur_x_available) + 1
        cur_series_full = pd.Series({p: cur_roc.get(p, np.nan) for p in periods})
        fig.add_trace(go.Scatter(
            x=x_anchor[:n], y=_cum(cur_series_full)[:n],
            mode="lines+markers", line=dict(color=COLORS["cur_year"], width=3.5),
            marker=dict(size=8, color=COLORS["cur_year"], line=dict(color="#000000", width=1.5)),
            name=f"Current Cycle ({start_yr}-{end_yr}) Actual", hovertemplate="Month %{x}<br>Actual: %{y:.2f}%<extra></extra>"
        ))
        
    for m, label in [(12, "Year 1 (Post)"), (24, "Year 2 (Mid)"), (36, "Year 3 (Pre)")]:
        fig.add_vline(x=m, line_dash="dot", line_color="#4a5568", line_width=1.5)
        fig.add_annotation(x=m - 6, y=1.0, yref="paper", text=label, showarrow=False, font=dict(family="IBM Plex Mono", color="#8d9ab0", size=10))
    fig.add_annotation(x=42, y=1.0, yref="paper", text="Year 4 (Elec)", showarrow=False, font=dict(family="IBM Plex Mono", color="#8d9ab0", size=10))

    layout = _base_layout("S&P 500: 48-Month Presidential Cycle (Since 1981)", height=550)
    layout["margin"]["t"] = 120    
    layout["legend"]["y"] = 1.15  
    
    layout["xaxis"].update(title="Months Since Cycle Start (1-48)", dtick=4, range=[-0.5, 48.5])
    layout["yaxis"]["ticksuffix"] = "%"
    
    fig.update_layout(**layout)
    return fig

def make_rebased_macro_chart(data_dict: dict) -> go.Figure:
    fig = go.Figure()
    
    # Merge all series into a single dataframe to find the common start date
    df_combined = pd.DataFrame(data_dict).dropna()
    
    if df_combined.empty:
        return fig
        
    # Rebase all to 100 at the start of the overlapping timeframe
    df_rebased = df_combined / df_combined.iloc[0] * 100
    
    line_colors = {
        "US (^GSPC)": COLORS["us"], 
        "Canada (^GSPTSE)": COLORS["canada"], 
        "India (^NSEI)": COLORS["india"],
        "Gold (GC=F)": COLORS["gold"],
        "Bitcoin (BTC-USD)": COLORS["btc"],
        "Volatility (^VIX)": COLORS["vix"],
        "Crude Oil (CL=F)": COLORS["oil"],
        "10Yr Yield (^TNX)": COLORS["tnx"]
    }
    
    for col in df_rebased.columns:
        fig.add_trace(go.Scatter(
            x=df_rebased.index, y=df_rebased[col],
            mode="lines", line=dict(color=line_colors.get(col, "#FFFFFF"), width=2),
            name=col, hovertemplate="Date: %{x|%b %Y}<br>Rebased Value: %{y:,.1f}<extra></extra>"
        ))

    # Add historical shaded crisis zones
    for ev in MACRO_EVENTS:
        fig.add_vrect(
            x0=ev["start"], x1=ev["end"],
            fillcolor=COLORS["macro_zone"], opacity=0.8, layer="below", line_width=0,
            annotation_text=ev["name"], annotation_position="top left",
            annotation_font=dict(family="IBM Plex Mono", size=10, color="#FFFFFF"),
            annotation_textangle=-90
        )

    start_yr = df_rebased.index[0].year
    title_str = f"Global Macro Race (Rebased to 100 in {start_yr}) — Log Scale"
    layout = _base_layout(title_str, height=600)
    layout["margin"]["t"] = 60
    layout["yaxis"]["type"] = "log"
    layout["yaxis"]["title"] = "Index/Asset Value (Log Scale, Base 100)"

    fig.update_layout(**layout)
    return fig

# ─────────────────────────────────────────────
# CSV EXPORT & SIDEBAR
# ─────────────────────────────────────────────
def build_csv(data: dict, timeframe: str) -> bytes:
    periods, label, rows = data["periods"], "Week" if timeframe == "Weekly" else "Month", []
    for p in periods:
        rows.append({
            label: p, "Avg_5yr_%": round(data["avg_5"].get(p, np.nan), 4),
            "Avg_10yr_%": round(data["avg_10"].get(p, np.nan), 4), "Avg_Max_%": round(data["avg_max"].get(p, np.nan), 4),
            "WinRate_5yr": round(data["wr_5"].get(p, np.nan), 1), "WinRate_10yr": round(data["wr_10"].get(p, np.nan), 1),
            "WinRate_Max": round(data["wr_max"].get(p, np.nan), 1), f"{CURRENT_YEAR}_Actual_%": round(data["cur_roc"].get(p, np.nan), 4),
        })
    buf = io.BytesIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()

with st.sidebar:
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker Symbol", value="QQQ", max_chars=10).upper().strip()
    start_year = st.number_input("Start Year (Max Lookback)", min_value=1950, max_value=CURRENT_YEAR - 1, value=2010, step=1)
    st.markdown('<div class="section-header">Timeframe</div>', unsafe_allow_html=True)
    timeframe = st.radio("", ["Weekly", "Monthly"], horizontal=True, label_visibility="collapsed")
    st.markdown('<div class="section-header">Display Options</div>', unsafe_allow_html=True)
    show_winrate, show_spaghetti = st.checkbox("Show Win Rate %", value=True), st.checkbox("Show All Past Years", value=True)
    st.markdown("---")
    st.markdown('<span style="font-family:IBM Plex Mono;font-size:0.7rem;color:#3a4258;">Data via yfinance · Cached 1hr</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
col_title, col_gap = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-title">📈 ETF Seasonality Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{ticker} · {timeframe} · Since {start_year}</div>', unsafe_allow_html=True)

with st.spinner(f"Loading {ticker} data…"):
    roc_df = fetch_seasonality_data_v4(ticker, start_year, timeframe)

if roc_df is None or roc_df.empty:
    st.error(f"❌ Could not retrieve data for **{ticker}**. Please check the ticker symbol and try again.")
    st.stop()

data = compute_seasonality(roc_df, timeframe, start_year)

def _wr_badge(val):
    return "—" if pd.isna(val) else f'<span style="color:{"#FFFFFF" if val >= 50 else "#BBBBBB"};font-family:IBM Plex Mono">{val:.0f}%</span>'

cur_p = data["current_period"]
avg_max_cur, wr_max_cur, actual_cur = data["avg_max"].get(cur_p, np.nan), data["wr_max"].get(cur_p, np.nan), data["cur_roc"].get(cur_p, np.nan)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Return — Current {("Week" if timeframe=="Weekly" else "Month")}</div><div class="metric-value {"metric-pos" if (not pd.isna(avg_max_cur) and avg_max_cur >= 0) else "metric-neg"}">{f"{avg_max_cur:+.2f}%" if not pd.isna(avg_max_cur) else "—"}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Win Rate — Current {("Week" if timeframe=="Weekly" else "Month")}</div><div class="metric-value">{_wr_badge(wr_max_cur)}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{CURRENT_YEAR} Actual — Current {("Week" if timeframe=="Weekly" else "Month")}</div><div class="metric-value {"metric-pos" if (not pd.isna(actual_cur) and actual_cur >= 0) else "metric-neg"}">{f"{actual_cur:+.2f}%" if not pd.isna(actual_cur) else "N/A yet"}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Completed Years in Dataset</div><div class="metric-value" style="color:#00E5FF">{len(data["completed_years"])}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊  Average Returns", "〰️  Cumulative Trend", "🇺🇸  Presidential Cycle", "🌍  Macro Events"])

with tab1:
    for wk, label in [("5", "Last 5 Years"), ("10", "Last 10 Years"), ("max", f"Max (Since {start_year})")]:
        st.plotly_chart(make_bar_chart(data, wk, show_winrate, timeframe, label), use_container_width=True, config={"displayModeBar": False})

with tab2:
    for wk, label in [("5", "Last 5 Years — Cumulative"), ("10", "Last 10 Years — Cumulative"), ("max", f"Max (Since {start_year}) — Cumulative")]:
        st.plotly_chart(make_cumulative_chart(data, wk, show_spaghetti, timeframe, label), use_container_width=True, config={"displayModeBar": False})

with tab3:
    st.markdown("""
    <div style="background-color: #12151c; border: 1px solid #1e2330; border-radius: 6px; padding: 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #8d9ab0;">
    <strong>Note:</strong> This view overrides sidebar settings. It strictly analyzes the <strong>S&P 500 (^GSPC)</strong> mapping the continuous 48-month journey of the US Presidential term (starting from 1981).
    </div>
    """, unsafe_allow_html=True)
    with st.spinner("Loading S&P 500 48-month cycle data..."):
        spx_df = fetch_presidential_cycle_data()
        if spx_df is not None and not spx_df.empty:
            cycle_data = compute_cycle_seasonality(spx_df)
            st.plotly_chart(make_presidential_cycle_chart(cycle_data), use_container_width=True, config={"displayModeBar": False})

with tab4:
    st.markdown("""
    <div style="background-color: #12151c; border: 1px solid #1e2330; border-left: 3px solid #FF4444; border-radius: 6px; padding: 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #8d9ab0;">
    <strong>Global Macro Context:</strong> Tracking market resilience across different asset classes. Select the assets below to race them on a <strong>Logarithmic Scale</strong> (rebased to 100 at the earliest shared date).
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading Global Macro Data..."):
        global_data = fetch_global_macro_data()
        
        if global_data:
            st.markdown('<div class="section-header">Select Assets to Compare</div>', unsafe_allow_html=True)
            cols = st.columns(4)
            
            selected_assets = []
            asset_names = list(global_data.keys())
            
            for i, asset_name in enumerate(asset_names):
                # Default to checking US and Canada on load
                default_val = True if "US" in asset_name or "Canada" in asset_name else False
                with cols[i % 4]:
                    if st.checkbox(asset_name, value=default_val):
                        selected_assets.append(asset_name)
            
            if selected_assets:
                filtered_data = {k: global_data[k] for k in selected_assets}
                st.plotly_chart(make_rebased_macro_chart(filtered_data), use_container_width=True, config={"displayModeBar": False})
            else:
                st.warning("Please select at least one asset to display.")
        else:
            st.error("Failed to load global macro data.")

st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
st.download_button(label=f"⬇️  Download {ticker} {timeframe} Seasonality Data (.csv)", data=build_csv(data, timeframe), file_name=f"{ticker}_{'weekly' if timeframe == 'Weekly' else 'monthly'}_seasonality_{start_year}-{CURRENT_YEAR}.csv", mime="text/csv")
