import streamlit as st
import pandas as pd
import requests
import io
import zipfile
import matplotlib.pyplot as plt
from datetime import datetime

# --- 1. 配置：资产代码映射 (Stooq 格式) ---
# Gold Futures = GC.F, Silver Futures = SI.F
# GLD ETF = GLD.US, SLV ETF = SLV.US
ASSET_CONFIG = {
    "Gold": {"ticker": "GC.F", "etf": "GLD.US", "kw": "GOLD", "ex": "COMMODITY"},
    "Silver": {"ticker": "SI.F", "etf": "SLV.US", "kw": "SILVER", "ex": "COMMODITY"}
}


# --- 2. 核心：Stooq 数据抓取引擎 ---
def fetch_stooq_data(ticker):
    """
    通过 Stooq 接口获取真实历史价格，规避 yfinance 限流问题
    """
    url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df['Date'] = pd.to_datetime(df['Date'])
            return df.set_index('Date')
    except Exception as e:
        st.error(f"Stooq 数据同步失败: {e}")
    return pd.DataFrame()


# --- 3. 核心：CFTC 物理包解析 ---
@st.cache_data(ttl=43200)
def fetch_cftc_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    # 合并 2025 和 2026 年度物理包
    urls = [
        "https://www.cftc.gov/files/dea/history/deacot2026.zip",
        "https://www.cftc.gov/files/dea/history/deacot2025.zip"
    ]
    all_dfs = []
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    for fname in z.namelist():
                        with z.open(fname) as f:
                            df = pd.read_csv(f, low_memory=False)
                            df.columns = [c.strip() for c in df.columns]
                            all_dfs.append(df)
        except:
            continue

    if not all_dfs: return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)

    # 模糊识别关键字段 [Image of silver commitments of traders chart]
    def find_col(kws, cols):
        for c in cols:
            if all(k.lower() in c.lower() for k in kws): return c
        return None

    d_col = find_col(['As of Date', 'YYMMDD'], combined.columns)
    nc_l, nc_s = find_col(['NonComm', 'Long'], combined.columns), find_col(['NonComm', 'Short'], combined.columns)
    m_col = find_col(['Market', 'Exchange', 'Names'], combined.columns)

    combined['report_date'] = pd.to_datetime(combined[d_col], errors='coerce').dt.normalize()
    combined['nc_net'] = combined[nc_l] - combined[nc_s]
    combined['m_name'] = combined[m_col].astype(str)
    return combined.dropna(subset=['report_date', 'nc_net'])


# --- 4. 主程序逻辑 ---
def main():
    st.set_page_config(page_title="2026 筹码真相手册", layout="wide")
    st.title("🛡️ 专家级筹码监控 (Stooq 稳健版)")
    st.markdown("---")

    asset_key = st.sidebar.selectbox("分析目标", list(ASSET_CONFIG.keys()))
    window = st.sidebar.slider("分析窗口 (周)", 26, 104, 52)

    with st.spinner('同步 CFTC 原始持仓包...'):
        raw_data = fetch_cftc_data()

    if not raw_data.empty:
        conf = ASSET_CONFIG[asset_key]
        # 过滤品种
        df_cftc = raw_data[raw_data['m_name'].str.contains(conf['kw'], case=False) &
                           raw_data['m_name'].str.contains(conf['ex'], case=False)].copy()

        if df_cftc.empty:
            st.error("未找到对应 CFTC 品种。")
            return

        df_cftc = df_cftc.sort_values('report_date').drop_duplicates('report_date')

        # 计算 COT Index
        df_cftc['rmin'] = df_cftc['nc_net'].rolling(window).min()
        df_cftc['rmax'] = df_cftc['nc_net'].rolling(window).max()
        df_cftc['cot_index'] = (df_cftc['nc_net'] - df_cftc['rmin']) / (df_cftc['rmax'] - df_cftc['rmin']) * 100

        # 使用 Stooq 获取行情
        with st.spinner('同步 Stooq 实盘行情...'):
            prices = fetch_stooq_data(conf['ticker'])
            etf_data = fetch_stooq_data(conf['etf'])

        if not prices.empty:
            # 数据归一化对齐
            prices.index = pd.to_datetime(prices.index).normalize()
            if not etf_data.empty:
                etf_data.index = pd.to_datetime(etf_data.index).normalize()

            # 数据大合并
            final = prices[['Close']].rename(columns={'Close': 'price'}).join(
                df_cftc.set_index('report_date')[['nc_net', 'cot_index']], how='left'
            ).ffill()

            if not etf_data.empty:
                final = final.join(etf_data[['Close', 'Volume']].rename(
                    columns={'Close': 'etf_price', 'Volume': 'etf_vol'}), how='left').ffill()

            final = final.dropna()

            # --- 5. 可视化展现 (Matplotlib 专家模式) ---
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            plt.style.use('dark_background')

            # 顶部图表：期货筹码与价格
            ax2 = ax1.twinx()
            ax2.plot(final.index, final['price'], color='#FFD700', linewidth=2, label="Price (Stooq)")
            ax1.fill_between(final.index, final['nc_net'], 0, color='red', alpha=0.3, label="大户投机净持仓")
            ax1.set_ylabel("期货净持仓 (张)", color='red')
            ax2.set_ylabel("价格 (USD)", color='#FFD700')

            # 底部图表：ETF 活跃度
            ax3.bar(final.index, final.get('etf_vol', 0), color='cyan', alpha=0.4, label="ETF 成交量")
            ax4 = ax3.twinx()
            ax4.plot(final.index, final.get('etf_price', 0), color='lime', linewidth=1, label="ETF Price")
            ax3.set_ylabel("ETF 活跃度", color='cyan')

            # 信号：COT Index > 90 标记红色倒三角
            high_idx = final[final['cot_index'] > 90]
            ax2.scatter(high_idx.index, high_idx['price'], color='red', marker='v', s=120, label="超买预警")

            ax1.legend(loc='upper left');
            ax2.legend(loc='upper right')
            st.pyplot(fig)

            # 数据汇总
            latest = final.iloc[-1]
            st.write(f"### 2026 实时快报 (截止: {final.index[-1].date()})")
            c1, c2, c3 = st.columns(3)
            c1.metric("市场现价", f"${latest['price']:.2f}")
            c2.metric("COT Index", f"{latest['cot_index']:.1f}%")
            # 真实行情揭示：2026年1月27日后，白银处于剧烈抛售后的低位盘整期
            st.warning(f"筹码状态: {'⚠️ 极端超买' if latest['cot_index'] > 90 else '✅ 风险已大幅释放'}")
        else:
            st.error("Stooq 行情获取失败，请检查网络连接。")


if __name__ == "__main__":
    main()