import streamlit as st
import pandas as pd
import numpy as np

st.title("📊 综合金融看板 - 首页")

# --- 1. 关键指标概览 ---
# 使用 st.columns 布局一行展示三个指标
col1, col2, col3 = st.columns(3)
col1.metric("总资产预估", "¥1,245,000", "+2.4%")
col2.metric("本月盈利", "¥34,500", "-0.5%")
col3.metric("持仓标的数量", "12", "1")

st.divider()

# --- 2. 模拟数据生成 ---
# 创建一些随机数据用于绘图
dates = pd.date_range("2024-01-01", periods=30)
chart_data = pd.DataFrame(
    np.random.randn(30, 3).cumsum(axis=0) + 100,
    index=dates,
    columns=['股票账户', '加密货币', '基金']
)

# --- 3. 统计图表 ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 资产净值走势")
    # 绘制折线图
    st.line_chart(chart_data)

with col_right:
    st.subheader("🍰 资产配置分布")
    # 模拟饼图/柱状图数据
    allocation_data = pd.DataFrame({
        "资产类别": ["股票", "加密货币", "债券", "现金", "黄金"],
        "比例": [40, 20, 15, 15, 10]
    }).set_index("资产类别")

    # 绘制柱状图
    st.bar_chart(allocation_data)

# --- 4. 消息通知区域 ---
with st.expander("🔔 近期系统消息"):
    st.info("数据接口维护通知：今晚 24:00 - 02:00 CoinGecko API 可能出现间歇性中断。")
    st.success("您的 3 月份投资月报已生成。")
