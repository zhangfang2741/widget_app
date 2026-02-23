import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
import zipfile
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 核心配置：精确对接 CFTC 官方字段名 ---
MARKETS = {
    "Gold": {"ticker": "GC=F", "name": "GOLD - COMMODITY EXCHANGE INC."},
    "Silver": {"ticker": "SI=F", "name": "SILVER - COMMODITY EXCHANGE INC."},
    "DXY": {"ticker": "DX-Y.NYB", "name": "U.S. DOLLAR INDEX - ICE FUTURES U.S."}
}


@st.cache_data(ttl=3600 * 12)
def fetch_cftc_active_data():
    """
    抓取当前活跃年份 (2026) 和上一年度 (2025) 的物理文件
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    # 在 2026 年初，最新数据存储在 deafut.zip 中
    urls = [
        "https://www.cftc.gov/files/dea/history/deafut.zip",
        "https://www.cftc.gov/files/dea/history/dea_fut_hist_2025.zip"
    ]

    all_dfs = []
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    for fname in z.namelist():
                        if fname.endswith(('.txt', '.csv')):
                            with z.open(fname) as f:
                                df = pd.read_csv(f, low_memory=False)
                                # 清洗列名中的空格
                                df.columns = [c.strip() for c in df.columns]
                                all_dfs.append(df)
        except Exception as e:
            st.warning(f"无法获取数据源 {url}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    # 统一日期格式并去重
    combined['Report_Date_as_MM_DD_YYYY'] = pd.to_datetime(combined['Report_Date_as_MM_DD_YYYY'])
    return combined.drop_duplicates(subset=['Market_and_Exchange_Names', 'Report_Date_as_MM_DD_YYYY'])


# --- 2. 界面与交互 ---
st.title("🛡️ 2026 专家级：全真实 CFTC 筹码雷达")
st.info("数据说明：本系统抓取 CFTC 官方周五发布的最新持仓（数据截至当周二）。")

with st.sidebar:
    asset_key = st.selectbox("分析资产", list(MARKETS.keys()))
    window_size = st.slider("滚动分析周期 (周)", 26, 104, 52)

with st.spinner('正在同步全球筹码数据...'):
    # 此处已修正函数名调用错误
    raw_data = fetch_cftc_active_data()

if not raw_data.empty:
    config = MARKETS[asset_key]
    # 过滤品种并计算净持仓
    df = raw_data[raw_data['Market_and_Exchange_Names'] == config['name']].copy()
    df = df.sort_values('Report_Date_as_MM_DD_YYYY')

    # 净持仓计算逻辑: 多头 - 空头
    df['nc_net'] = df['NonComm_Positions_Long_All'] - df['NonComm_Positions_Short_All']
    df['c_net'] = df['Comm_Positions_Long_All'] - df['Comm_Positions_Short_All']

    # 计算归一化信号：COT Index
    rmin = df['nc_net'].rolling(window_size).min()
    rmax = df['nc_net'].rolling(window_size).max()
    df['cot_index'] = (df['nc_net'] - rmin) / (rmax - rmin) * 100

    # 抓取 yfinance 真实历史行情
    prices = yf.download(config['ticker'], start=df['Report_Date_as_MM_DD_YYYY'].min(), interval="1d")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    # 合并数据：前向填充周报持仓
    final = prices[['Close']].rename(columns={'Close': 'price'}).join(
        df.set_index('Report_Date_as_MM_DD_YYYY')[['nc_net', 'c_net', 'cot_index']],
        how='left'
    ).ffill()

    # --- 3. 可视化绘制 ---
    fig, ax1 = plt.subplots(figsize=(14, 7))
    plt.style.use('dark_background')
    ax2 = ax1.twinx()

    # 价格与筹码分布 (2026 真实价格)
    price_color = '#FFD700' if asset_key == "Gold" else '#FFFFFF'
    ax2.plot(final.index, final['price'], color=price_color, linewidth=2, label="Market Price")
    ax1.fill_between(final.index, final['nc_net'], 0, color='red', alpha=0.3, label="投机大户(Non-Comm)净仓")
    ax1.plot(final.index, final['c_net'], color='cyan', alpha=0.6, label="商业机构(Comm)对冲仓")

    # 标注超买反转点 (COT Index > 90)
    signals = final[final['cot_index'] > 90]
    ax2.scatter(signals.index, signals['price'], color='red', marker='v', s=120, label="信号: 筹码过热")

    st.pyplot(fig)

    # 4. 数据汇总
    latest = final.iloc[-1]
    st.write(f"### 实时统计概览 ({final.index[-1].date()})")
    c1, c2, c3 = st.columns(3)
    c1.metric("市场真实价格", f"${latest['price']:.2f}")
    c2.metric("COT Index", f"{latest['cot_index']:.1f}%")
    c3.metric("当前风险状态", "⚠️ 建议减仓" if latest['cot_index'] > 90 else "✅ 正常博弈")
else:
    st.error("无法解析数据。请确保本地可正常访问 https://www.cftc.gov")