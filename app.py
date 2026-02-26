import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# --- PAGE SETUP ---
st.set_page_config(page_title="ETF Seasonality", layout="centered")

st.title("📈 ETF Seasonality Dashboard")
st.markdown("Analyze historical averages, win rates, and current year tracking.")

st.info("**💡 What is Win Rate?** The Win Rate shows the percentage of time this specific week or month historically ended with a positive return. *Example: A high average return but a low Win Rate (e.g., 20%) means a single freak outlier year skewed the data, making it a low-probability trade.*")

# --- USER CONTROLS ---
col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("Enter Ticker:", value="QQQ").upper()

with col2:
    period_type = st.radio("Timeframe:", ["Weekly", "Monthly"], horizontal=True)

with col3:
    st.write("") 
    show_win_rate = st.checkbox("Show Win Rate %", value=True)

if period_type == "Weekly":
    yf_interval = "1wk"
    time_col = "Week"
    x_label = "Week Number of the Year (1-52)"
    current_time_val = datetime.now().isocalendar()[1]
else:
    yf_interval = "1mo"
    time_col = "Month"
    x_label = "Month of the Year (1-12)"
    current_time_val = datetime.now().month

# --- NEW: LIGHTNING-FAST CACHING ---
# ttl=3600 tells Streamlit to hold this data in memory for 1 hour (3600 seconds).
# After an hour, it will automatically clear the cache and fetch fresh market prices.
@st.cache_data(ttl=3600)
def get_historical_data(ticker_symbol, interval):
    return yf.download(ticker_symbol, start="2010-01-01", interval=interval)

# --- DATA FETCHING ---
with st.spinner(f"Fetching {period_type.lower()} data for {ticker}..."):
    try:
        # Instead of calling yf.download directly, we call our new cached function!
        data = get_historical_data(ticker, yf_interval)
        
        if data.empty:
            st.error("No data found. Please check the ticker symbol.")
        else:
            df = pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex):
                df['Close'] = data['Close'].iloc[:, 0]
            else:
                df['Close'] = data['Close']

            df['ROC'] = df['Close'].pct_change() * 100
            
            if period_type == "Weekly":
                df['Year'] = df.index.isocalendar().year
                df[time_col] = df.index.isocalendar().week
            else:
                df['Year'] = df.index.year
                df[time_col] = df.index.month

            current_year = datetime.now().year
            df_5yr = df[df['Year'] > (current_year - 5)]
            df_10yr = df[df['Year'] > (current_year - 10)]
            df_max = df

            # --- PLOTTING LOGIC ---
            plt.style.use('dark_background')
            background_color = '#1E1E1E'

            def plot_roc_bars(ax, dataset, title):
                grid = dataset.pivot_table(values='ROC', index=time_col, columns='Year')
                
                if period_type == "Weekly" and 53 in grid.index:
                    if grid.loc[53].isna().sum() > (len(grid.columns) / 2):
                        grid = grid.drop(index=53)
                
                win_rate_series = (grid > 0).sum(axis=1) / grid.notna().sum(axis=1) * 100
                grid['Average_ROC'] = grid.mean(axis=1)
                
                x_vals = np.array(grid.index.astype(int))
                y_vals = np.array(grid['Average_ROC'])
                win_rates = np.array(win_rate_series)
                
                colors = ['#555555' if val > 0 else '#BBBBBB' for val in y_vals]
                ax.bar(x_vals, y_vals, color=colors, edgecolor='none', label='Historical Avg', zorder=1)
                ax.axhline(0, color='white', linewidth=0.8, alpha=0.5, zorder=2)
                
                if show_win_rate:
                    offset = max(abs(y_vals[~np.isnan(y_vals)])) * 0.08 
                    for i, x in enumerate(x_vals):
                        if not np.isnan(y_vals[i]):
                            y_pos = y_vals[i]
                            wr_text = f"{int(win_rates[i])}%"
                            rot = 90 if period_type == "Weekly" else 0
                            
                            if y_pos > 0:
                                ax.text(x, y_pos + offset, wr_text, ha='center', va='bottom', fontsize=7, color='white', rotation=rot, zorder=4)
                            else:
                                ax.text(x, y_pos - offset, wr_text, ha='center', va='top', fontsize=7, color='white', rotation=rot, zorder=4)

                if current_year in grid.columns:
                    current_year_data = grid[current_year]
                    ax.plot(x_vals, current_year_data, color='#FFFFFF', marker='o', markersize=4, 
                            linestyle='-', linewidth=2, label=f'{current_year} Actual ROC', zorder=3)
                
                ax.axvline(x=current_time_val, color='#FF4444', linestyle='--', linewidth=1.5, 
                           alpha=0.8, label=f'Current {time_col}', zorder=0)
                
                ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=15)
                ax.set_ylabel('ROC (%)', fontsize=9, color='lightgray')
                
                y_min, y_max = ax.get_ylim()
                ax.set_ylim(y_min * 1.25, y_max * 1.25)
                
                ax.set_xticks(x_vals)
                ax.tick_params(axis='x', labelsize=8, colors='lightgray')
                ax.tick_params(axis='y', labelsize=8, colors='lightgray')
                ax.grid(axis='y', linestyle=':', color='gray', alpha=0.3)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('gray')
                ax.spines['bottom'].set_color('gray')
                ax.set_facecolor(background_color)
                
                ax.legend(loc='upper left', fontsize=8, framealpha=0.2, facecolor=background_color)

            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11, 14), facecolor=background_color, dpi=300)

            title_suffix = "| Win Rate % | Current Year" if show_win_rate else "| Current Year"
            plot_roc_bars(axes[0], df_5yr, f"{ticker} 5-Year Average {title_suffix}")
            plot_roc_bars(axes[1], df_10yr, f"{ticker} 10-Year Average {title_suffix}")
            plot_roc_bars(axes[2], df_max, f"{ticker} Max (Since 2010) Average {title_suffix}")

            axes[2].set_xlabel(x_label, fontsize=10, color='lightgray', labelpad=10)
            plt.tight_layout(pad=3.0)

            st.pyplot(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
