import zipfile
from io import BytesIO

import certifi
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# -----------------------------
# ① CFTC COT（官方 CSV）
# -----------------------------

@st.cache_data(ttl=86400)
def load_cot_gold():
    url = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2024.zip"

    resp = requests.get(url, verify=certifi.where(), timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(BytesIO(resp.content)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f)

    # 日期列
    date_cols = [c for c in df.columns if "Date" in c or "DATE" in c.upper()]
    if not date_cols:
        raise ValueError("找不到日期列")
    date_col = date_cols[0]

    # Managed Money 多空列
    long_cols = [c for c in df.columns if "M_Money_Long" in c]
    short_cols = [c for c in df.columns if "M_Money_Short" in c]
    if not long_cols or not short_cols:
        raise ValueError("找不到 Managed Money 多空列")

    gold = df[df["Market_and_Exchange_Names"].str.contains(
        "GOLD - COMMODITY EXCHANGE INC", na=False
    )].copy()

    gold["Date"] = pd.to_datetime(gold[date_col], errors='coerce')

    # 关键改动：把字符串转数字再做减法
    gold["Net_Spec"] = (
            pd.to_numeric(gold[long_cols[0]].astype(str).str.replace(",", ""), errors="coerce") -
            pd.to_numeric(gold[short_cols[0]].astype(str).str.replace(",", ""), errors="coerce")
    )

    return gold[["Date", "Net_Spec"]].sort_values("Date")

# -----------------------------
# ② 黄金期货价格 + 持仓
# -----------------------------
@st.cache_data(ttl=3600)
def load_futures(symbol="GC=F", period="6mo"):
    # symbol: 黄金期货
    fut = yf.Ticker(symbol)
    hist = fut.history(period=period)
    hist = hist.reset_index()  # 把日期从 index 变成列

    # 自动匹配列名
    date_col = [c for c in hist.columns if "date" in c.lower()][0]
    close_col = [c for c in hist.columns if "close" in c.lower()][0]

    # Open Interest 列可能不存在
    oi_col_candidates = [c for c in hist.columns if "openinterest" in c.lower() or "oi" == c.lower()]
    if oi_col_candidates:
        oi_col = oi_col_candidates[0]
        df = hist[[date_col, close_col, oi_col]].copy()
        df.rename(columns={date_col: "Date", close_col: "Close", oi_col: "Open Interest"}, inplace=True)
    else:
        # 如果没有 OI，则只返回日期和收盘价
        df = hist[[date_col, close_col]].copy()
        df.rename(columns={date_col: "Date", close_col: "Close"}, inplace=True)
        df["Open Interest"] = pd.NA  # 补一列空值，方便后续处理

    return df
# -----------------------------
# ③ 期权情绪（GLD Put/Call Proxy）
# -----------------------------
@st.cache_data(ttl=3600)
def load_option_sentiment():
    gld = yf.Ticker("GLD")
    opt_dates = gld.options[-3:]  # 最近几期
    rows = []

    for d in opt_dates:
        opt = gld.option_chain(d)
        calls = opt.calls["volume"].sum()
        puts = opt.puts["volume"].sum()
        rows.append({
            "date": pd.to_datetime(d),
            "put_call_ratio": puts / max(calls, 1)
        })

    return pd.DataFrame(rows)

# -----------------------------
# Dashboard
# -----------------------------
def show_dashboard():
    # === CFTC ===
    st.subheader("① CFTC 投机资金（Managed Money）")
    cot = load_cot_gold()
    st.line_chart(cot.set_index("Date"))

    latest = cot["Net_Spec"].iloc[-1]
    high = cot["Net_Spec"].quantile(0.9)

    if latest > high:
        st.warning("⚠️ 投机净多处于历史高位 → 易洗仓")
    else:
        st.success("🟢 投机仓位健康")

    st.divider()

    # === Futures ===
    st.subheader("② 价格 vs 持仓（是否去杠杆）")
    fut = load_futures()
    fut["price_chg"] = fut["Close"].pct_change()
    fut["oi_chg"] = fut["Open Interest"].pct_change()

    fut["signal"] = "Normal"
    mask = (fut["price_chg"] < -0.03) & (fut["oi_chg"] < -0.1)
    fut.loc[mask, "signal"] = "Forced Deleveraging"

    st.dataframe(fut.tail(10), use_container_width=True)

    if mask.iloc[-1]:
        st.error("❗ 当前下跌属于：去杠杆 / 洗仓")

    st.divider()

    # === Options ===
    st.subheader("③ 期权情绪（Put / Call）")
    opt = load_option_sentiment()
    st.line_chart(opt.set_index("date"))

    pcr = opt["put_call_ratio"].iloc[-1]
    st.metric("Put / Call Ratio", f"{pcr:.2f}")

    if pcr < 0.7:
        st.warning("📉 Call 拥挤 → 易被砸")
    elif pcr > 1.2:
        st.success("🟢 防御情绪重 → 下行空间有限")


st.title("🧠 黄金 · 大资金真实行为监控（CFTC + Futures + Options）")
show_dashboard()