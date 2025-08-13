# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Dividend Yield Charts (Forward & TTM)", page_icon="📈", layout="wide")

BRAND = {
    "primary": "#26947d",
    "primary_dark": "#1f7765",
    "bg": "#eef8f7",
    "ink": "#222222",
    "muted": "#818181"
}

def _style():
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .stApp {{
            background: {BRAND["bg"]};
            color: {BRAND["ink"]};
        }}
        .metric-label {{ color: {BRAND["ink"]} !important; }}
        </style>
        """,
        unsafe_allow_html=True
    )


def classify_frequency(spacings_days):
    """Infer expected payments per year from spacing of dividends"""
    if len(spacings_days) == 0 or np.isnan(spacings_days).all():
        return 1
    med = float(np.nanmedian(spacings_days))
    if med <= 45:
        return 12
    if med <= 135:
        return 4
    if med <= 240:
        return 2
    return 1


def adjust_dividends_for_splits(divs, splits):
    """Adjust dividends for splits to align with auto-adjusted prices"""
    if divs is None or divs.empty:
        return divs
    if splits is None or splits.empty:
        return divs
    s = splits.copy().sort_index().replace(0, 1.0).astype(float)
    cum = s.cumprod()
    total = float(cum.iloc[-1])
    cum_at_div = cum.reindex(divs.index, method='ffill').fillna(1.0)
    factors = total / cum_at_div
    return divs.astype(float) / factors


def estimate_forward_annual(div_window, min_obs=3):
    """Estimate forward annual dividend from a window of split-adjusted dividends"""
    if div_window is None or len(div_window) < min_obs:
        return np.nan
    amounts = div_window.values.astype(float)
    dates = div_window.index.values
    if len(dates) >= 2:
        spacings = np.diff(pd.to_datetime(div_window.index).values).astype('timedelta64[D]').astype(int)
        expected_count = classify_frequency(spacings)
    else:
        expected_count = 1
    med = np.median(amounts)
    if np.isnan(med) or med <= 0:
        return np.nan
    band = 1.75 if len(amounts) >= 6 else 2.25
    mask_regular = (amounts >= med / band) & (amounts <= med * band)
    regular = amounts[mask_regular]
    if len(regular) < max(2, min_obs - 1):
        cap = np.percentile(amounts, 70)
        regular = amounts[amounts <= cap]
    if len(regular) == 0:
        return np.nan
    typical = float(np.median(regular))
    return typical * float(expected_count)


@st.cache_data(ttl=60*60)
def fetch_series(ticker, years=25):
    """Fetch price data, compute forward and TTM yields, and return dividend series"""
    t_use = ticker.strip().upper()
    tried = [t_use]
    def _pull(sym):
        tk = yf.Ticker(sym)
        hist = tk.history(period="max", auto_adjust=True)
        divs = tk.dividends
        splits = tk.splits
        return hist, divs, splits
    try:
        hist, divs, splits = _pull(t_use)
        if hist is None or hist.empty:
            raise ValueError
    except Exception:
        if not t_use.endswith(".TO"):
            t_alt = t_use + ".TO"
            tried.append(t_alt)
            try:
                hist, divs, splits = _pull(t_alt)
                t_use = t_alt
            except Exception:
                return None
        else:
            return None
    px = hist["Close"].resample("M").last().dropna()
    divs_adj = adjust_dividends_for_splits(divs, splits)
    fwd_annual = []
    ttm_annual = []
    for dt in px.index:
        start24 = dt - pd.DateOffset(months=24)
        window24 = divs_adj[(divs_adj.index > start24) & (divs_adj.index <= dt)]
        fwd_annual.append(estimate_forward_annual(window24))
        start12 = dt - pd.DateOffset(months=12)
        window12 = divs_adj[(divs_adj.index > start12) & (divs_adj.index <= dt)]
        ttm_annual.append(window12.sum())
    fwd_annual = pd.Series(fwd_annual, index=px.index)
    ttm_annual = pd.Series(ttm_annual, index=px.index)
    fwd_y = (fwd_annual / px) * 100.0
    ttm_y = (ttm_annual / px) * 100.0
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    px = px[px.index >= cutoff]
    fwd_y = fwd_y[fwd_y.index >= cutoff]
    ttm_y = ttm_y[ttm_y.index >= cutoff]
    meta = {"ticker_used": t_use, "tried": tried}
    return px, fwd_y, ttm_y, divs_adj, meta


def compute_thresholds(series):
    ys = series.dropna().values
    if len(ys) < 12:
        return None
    return {"p5": float(np.nanpercentile(ys, 5)), "p95": float(np.nanpercentile(ys, 95))}


def build_chart_single(indexes, y_series, label, color):
    return go.Scatter(
        x=indexes, y=y_series, name=label,
        mode="lines", line=dict(width=2, color=color),
        fill="tozeroy",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>"+label+": %{y:.2f}%<extra></extra>"
    )


def build_chart(px, fwd_y, ttm_y, view_type, thresholds_fwd, thresholds_ttm, ticker_used):
    fig = go.Figure()
    if view_type in ("Forward Yield","Both"):
        fig.add_trace(build_chart_single(fwd_y.index, fwd_y, "Forward Yield (est.)", BRAND["primary"]))
    if view_type in ("TTM Yield","Both"):
        fig.add_trace(build_chart_single(ttm_y.index, ttm_y, "TTM Yield", BRAND["primary_dark"]))
    if view_type in ("Forward Yield","Both") and thresholds_fwd:
        fig.add_hline(y=thresholds_fwd["p95"], line_width=2, line_dash="dot", line_color=BRAND["primary_dark"],
                      annotation_text=f"Forward 95th pct: {thresholds_fwd['p95']:.2f}%", annotation_position="top left")
        fig.add_hline(y=thresholds_fwd["p5"], line_width=2, line_dash="dot", line_color=BRAND["muted"],
                      annotation_text=f"Forward 5th pct: {thresholds_fwd['p5']:.2f}%", annotation_position="bottom left")
    if view_type in ("TTM Yield","Both") and thresholds_ttm:
        fig.add_hline(y=thresholds_ttm["p95"], line_width=2, line_dash="dot", line_color=BRAND["primary_dark"],
                      annotation_text=f"TTM 95th pct: {thresholds_ttm['p95']:.2f}%", annotation_position="top right")
        fig.add_hline(y=thresholds_ttm["p5"], line_width=2, line_dash="dot", line_color=BRAND["muted"],
                      annotation_text=f"TTM 5th pct: {thresholds_ttm['p5']:.2f}%", annotation_position="bottom right")
    if view_type in ("Forward Yield","Both") and not fwd_y.dropna().empty:
        fy_val = float(fwd_y.dropna().iloc[-1])
        fig.add_trace(go.Scatter(x=[fwd_y.dropna().index[-1]], y=[fy_val], name="Current Forward", mode="markers+text",
            marker=dict(size=10, color=BRAND["primary"]), text=[f"{fy_val:.2f}%"], textposition="middle right",
            hovertemplate="<b>Latest Forward</b><br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"))
    if view_type in ("TTM Yield","Both") and not ttm_y.dropna().empty:
        ty_val = float(ttm_y.dropna().iloc[-1])
        fig.add_trace(go.Scatter(x=[ttm_y.dropna().index[-1]], y=[ty_val], name="Current TTM", mode="markers+text",
            marker=dict(size=10, color=BRAND["primary_dark"]), text=[f"{ty_val:.2f}%"], textposition="middle right",
            hovertemplate="<b>Latest TTM</b><br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"))
    fig.update_layout(
        title=f"Dividend Yield History — {ticker_used}",
        xaxis_title="Date",
        yaxis_title="Dividend Yield (%)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True))
    )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(count=25, label="25Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    return fig


def build_calendar(divs_adj, as_of_date, months=24):
    if divs_adj is None or divs_adj.empty:
        return pd.DataFrame(columns=["Date","Dividend","Type"])
    end = as_of_date
    start = end - pd.DateOffset(months=months)
    window = divs_adj[(divs_adj.index > start) & (divs_adj.index <= end)]
    if window.empty:
        return pd.DataFrame(columns=["Date","Dividend","Type"])
    amounts = window.values.astype(float)
    med = np.median(amounts)
    def classify(amount):
        if med <= 0:
            return "Unknown"
        if amount > med * 1.5 or amount < med * 0.5:
            return "Special"
        return "Regular"
    data = {
        "Date": window.index.date,
        "Dividend": amounts,
        "Type": [classify(a) for a in amounts]
    }
    df = pd.DataFrame(data).sort_values(by="Date", ascending=False)
    return df


def helper_text():
    st.markdown(
        '''
**What this shows:**
- **Forward Yield** uses recent dividends to estimate the next year's dividends (adjusting for special dividends, frequency changes, and splits).
- **TTM Yield** uses the sum of the last 12 months of dividends divided by price.
- **Use .TO for TSX/Canadian stocks** (e.g., `BNS.TO`, `ENB.TO`).
- **U.S. stocks**: just enter the ticker (e.g., `JNJ`, `PG`).
        '''
    )


_style()

st.title("📈 Dividend Yield Charts (Forward & TTM)")

st.caption("Data via Yahoo Finance (`yfinance`). Educational only — not investment advice.")

with st.sidebar:
    st.header("Settings")
    default_ticker = "BNS.TO"
    tick = st.session_state.get("ticker_input", default_ticker)
    st.subheader("Quick Picks (Canada)")
    colq1, colq2, colq3, colq4 = st.columns(4)
    if colq1.button("ENB.TO"):
        tick = "ENB.TO"
    if colq2.button("BNS.TO"):
        tick = "BNS.TO"
    if colq3.button("TD.TO"):
        tick = "TD.TO"
    if colq4.button("BMO.TO"):
        tick = "BMO.TO"
    ticker_input = st.text_input("Ticker", value=tick, help="Example: BNS.TO (Canada) or JNJ (U.S.)").strip()
    st.session_state["ticker_input"] = ticker_input
    period_years = st.slider("History window (years)", 10, 30, 25)
    view_type = st.radio("View Type", options=["Forward Yield","TTM Yield","Both"], index=0)
    st.markdown("")
    helper_text()
    st.write("---")
    st.write("Dividend Growth Investing & Retirement")

if not ticker_input:
    st.info("Enter a ticker to begin.")
    st.stop()

res = fetch_series(ticker_input, years=period_years)
if res is None:
    st.error("Sorry — couldn't find data for that ticker. Try adding `.TO` for Canadian stocks.")
    st.stop()

px, fwd_y, ttm_y, divs_adj, meta = res
if fwd_y.dropna().empty and ttm_y.dropna().empty:
    st.warning("Insufficient dividend history to compute yields. Try another ticker.")
    st.stop()

thresholds_fwd = compute_thresholds(fwd_y) if view_type in ("Forward Yield","Both") else None
thresholds_ttm = compute_thresholds(ttm_y) if view_type in ("TTM Yield","Both") else None

fig = build_chart(px, fwd_y, ttm_y, view_type, thresholds_fwd, thresholds_ttm, meta["ticker_used"])

cols = st.columns(3)
if view_type in ("Forward Yield","Both") and not fwd_y.dropna().empty:
    cols[0].metric("Current Forward Yield (est.)", f"{float(fwd_y.dropna().iloc[-1]):.2f}%")
if view_type in ("TTM Yield","Both") and not ttm_y.dropna().empty:
    cols[1].metric("Current TTM Yield", f"{float(ttm_y.dropna().iloc[-1]):.2f}%")
cols[2].metric("Ticker used", meta["ticker_used"])

st.plotly_chart(fig, use_container_width=True)

if divs_adj is not None and not divs_adj.empty:
    calendar = build_calendar(divs_adj, divs_adj.index.max())
    st.subheader("Dividend Calendar Preview (last 24 months)")
    st.dataframe(calendar)

st.caption("Forward yield estimated using last 24 months (split-adjusted dividends) and inferred payment frequency, trimming special dividends; TTM yield uses last 12 months' dividends. Percentile lines show potential under/over valuation zones.")
