import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧠 大资金结构监控仪表盘（黄金 / 白银）")

# =====================
# 参数
# =====================
symbol_map = {
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F"
}

asset = st.sidebar.selectbox("选择品种", list(symbol_map.keys()))
symbol = symbol_map[asset]

# =====================
# 获取数据
# =====================
data = yf.download(symbol, period="6mo", interval="1d")
data.dropna(inplace=True)

# =====================
# 计算指标
# =====================
data["Return"] = data["Close"].pct_change()
data["Volatility"] = data["Return"].rolling(10).std() * np.sqrt(252)

# 用成交量 proxy 未平仓变化（教学版）
data["OI_proxy"] = data["Volume"].rolling(3).mean()

# =====================
# 结构判断逻辑
# =====================
latest = data.iloc[-1]
prev = data.iloc[-5]

price_change = float((latest["Close"] - prev["Close"]) / prev["Close"])
oi_change = float((latest["OI_proxy"] - prev["OI_proxy"]) / prev["OI_proxy"])
vol_change = latest["Volatility"] - prev["Volatility"]

if price_change < -0.03 and oi_change < -0.15:
    structure = "🔴 去杠杆 / 被迫平仓"
elif price_change < -0.03 and oi_change > 0:
    structure = "🟠 新空进场（趋势型）"
elif price_change > 0 and oi_change < 0:
    structure = "🟡 空头回补反弹"
else:
    structure = "🟢 正常交易 / 无明显结构风险"

# =====================
# 展示结构判断
# =====================
st.subheader("📌 当前结构判断")
st.metric(
    label="市场状态",
    value=structure,
    delta=f"价格变化 {price_change:.2%} | OI变化 {oi_change:.2%}"
)

# =====================
# 图表 1：价格 + OI
# =====================
st.subheader("📉 价格 & OI 结构")

fig, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(data.index, data["Close"], label="Price")
ax1.set_ylabel("Price")

ax2 = ax1.twinx()
ax2.plot(data.index, data["OI_proxy"], color="orange", alpha=0.6, label="OI proxy")
ax2.set_ylabel("OI proxy")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
st.pyplot(fig)

# =====================
# 图表 2：波动率
# =====================
st.subheader("⚡ 波动率（CTA 风控触发风险）")

fig2, ax = plt.subplots(figsize=(10,3))
ax.plot(data.index, data["Volatility"])
ax.axhline(data["Volatility"].quantile(0.75), linestyle="--", color="red", alpha=0.5)
ax.set_ylabel("Volatility")
st.pyplot(fig2)

# =====================
# 解释说明
# =====================
with st.expander("📖 如何解读？"):
    st.markdown("""
- **价格跌 + OI 暴跌**：不是看空，是被迫去杠杆  
- **价格跌 + OI 上升**：新空在进场（要小心趋势反转）  
- **波动率急升**：CTA / 风控资金可能继续卖  
- **OI 稳定 + 波动率回落**：结构性下跌接近尾声  
""")
