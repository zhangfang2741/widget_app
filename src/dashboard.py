import streamlit as st
import pandas as pd
import requests
import io
import zipfile
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import time

# --- 1. 全局配置与资产映射 ---
# Finviz 映射用于实时行情，CFTC 关键词用于历史持仓匹配
ASSET_CONFIG = {
    "Silver (白银)": {
        "fv_ticker": "silver",
        "cftc_kw": ["SILVER", "COMMODITY"],
        "color": "#C0C0C0"
    },
    "Gold (黄金)": {
        "fv_ticker": "gold",
        "cftc_kw": ["GOLD", "COMMODITY"],
        "color": "#FFD700"
    },
    "DXY (美元指数)": {
        "fv_ticker": "us-dollar-index",
        "cftc_kw": ["U.S. DOLLAR INDEX", "ICE"],
        "color": "#1E90FF"
    }
}


# --- 2. Finviz 抓取引擎 (替代 yfinance) ---
def fetch_finviz_data(asset_ticker):
    """
    抓取 Finviz 期货详情页的 Snapshot 数据
    """
    url = f"https://finviz.com/futures_details.ashx?t={asset_ticker}&p=d1"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Referer': 'https://finviz.com/'
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        data = {}
        # Finviz 的快照数据存储在 snapshot-table2 中
        tables = soup.find_all('table', class_='snapshot-table2')
        for table in tables:
            for row in table.find_all('tr'):
                cols = row.find_all('td')
                for i in range(0, len(cols), 2):
                    key = cols[i].text.strip()
                    val = cols[i + 1].text.strip()
                    data[key] = val
        return data
    except Exception as e:
        st.sidebar.error(f"Finviz 抓取异常: {e}")
        return None


# --- 3. CFTC 物理包解析引擎 (历史筹码) ---
@st.cache_data(ttl=43200)
def fetch_cftc_historical_data():
    """
    直接从 CFTC 官网下载并合并 2025-2026 年度 Legacy 压缩包
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
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
                            df.columns = [str(c).strip() for c in df.columns]
                            all_dfs.append(df)
        except:
            continue

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # 模糊识别列名 (兼容空格、下划线、大小写)
    def find_col(kws, cols):
        for c in cols:
            if all(k.lower() in c.lower() for k in kws): return c
        return None

    d_col = find_col(['As of Date', 'YYMMDD'], combined.columns)
    nc_l = find_col(['NonComm', 'Long'], combined.columns)
    nc_s = find_col(['NonComm', 'Short'], combined.columns)
    m_col = find_col(['Market', 'Exchange', 'Names'], combined.columns)

    if not d_col or not nc_l:
        return pd.DataFrame()

    combined['report_date'] = pd.to_datetime(combined[d_col], errors='coerce').dt.normalize()
    combined['nc_net'] = combined[nc_l] - combined[nc_s]
    combined['m_name'] = combined[m_col].astype(str)

    return combined.dropna(subset=['report_date', 'nc_net'])


# --- 4. 主程序界面 ---
def main():
    st.set_page_config(page_title="2026 Finviz/CFTC 筹码雷达", layout="wide")

    st.title("🛡️ 专家级筹码监控：Finviz 实时感官 + CFTC 历史底牌")
    st.markdown("---")

    asset_label = st.sidebar.selectbox("选择监控资产", list(ASSET_CONFIG.keys()))
    window = st.sidebar.slider("分析窗口 (周)", 26, 104, 52)
    conf = ASSET_CONFIG[asset_label]

    # 4.1 获取 Finviz 实时快照
    with st.spinner('正在透视 Finviz 实时情绪...'):
        fv_snapshot = fetch_finviz_data(conf['fv_ticker'])

    if fv_snapshot:
        # 展示 Finviz 核心指标卡
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("当前成交价", fv_snapshot.get('Price', 'N/A'), fv_snapshot.get('Change', 'N/A'))
        c2.metric("52周波动区间", fv_snapshot.get('52W Range', 'N/A'))
        # Finviz COT 指数：显示 Speculators 的相对强度
        c3.metric("Finviz COT (Spec)", fv_snapshot.get('COT Speculator', 'N/A'))
        c4.metric("Finviz COT (Comm)", fv_snapshot.get('COT Commercial', 'N/A'))

    # 4.2 获取 CFTC 历史趋势
    with st.spinner('正在解压 CFTC 历史持仓包...'):
        raw_data = fetch_cftc_historical_data()

    if not raw_data.empty:
        # 资产过滤
        df = raw_data[raw_data['m_name'].str.contains(conf['cftc_kw'][0], case=False) &
                      raw_data['m_name'].str.contains(conf['cftc_kw'][1], case=False)].copy()

        if df.empty:
            st.error("CFTC 数据匹配失败，请检查关键词。")
            return

        df = df.sort_values('report_date').drop_duplicates('report_date')

        # 计算 COT Index (52周归一化)
        df['rmin'] = df['nc_net'].rolling(window).min()
        df['rmax'] = df['nc_net'].rolling(window).max()
        df['cot_index'] = (df['nc_net'] - df['rmin']) / (df['rmax'] - df['rmin']) * 100

        # --- 5. 绘图逻辑 (Matplotlib) ---

        fig, ax1 = plt.subplots(figsize=(14, 6))
        plt.style.use('dark_background')

        # 绘制投机大户净持仓 (左轴)
        ax1.fill_between(df['report_date'], df['nc_net'], 0, color='red', alpha=0.3, label="大户(Non-Comm)净持仓")
        ax1.set_ylabel("净持仓张数 (Net Positions)", color='red', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='red')

        # 绘制 COT Index (右轴)
        ax2 = ax1.twinx()
        ax2.plot(df['report_date'], df['cot_index'], color='cyan', linewidth=1.5, label="COT Index (信号线)")
        ax2.axhline(80, color='yellow', linestyle='--', alpha=0.5, label="超买阈值 (80)")
        ax2.axhline(20, color='lime', linestyle='--', alpha=0.5, label="超卖阈值 (20)")
        ax2.set_ylabel("COT Index (%)", color='cyan', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='cyan')

        plt.title(f"{asset_label} 历史筹码动能分析 (2025-2026)", fontsize=16, pad=20)
        ax1.grid(alpha=0.1)

        st.pyplot(fig)

        # 4.3 专家风险识别
        st.markdown("---")
        latest_idx = df['cot_index'].iloc[-1]

        # 结合 2026 年 1 月市场真实逻辑：白银从 $120 跌至 $84
        st.subheader("🧠 筹码风险哨兵")
        if latest_idx > 80:
            st.warning(f"🚨 预警：当前 {asset_label} 处于【极端拥挤】状态（COT Index: {latest_idx:.1f}%）。"
                       "Finviz 数据显示大户情绪过热，警惕高位获利了结引发的闪崩。")
        elif latest_idx < 20:
            st.success(f"✅ 机会：当前 {asset_label} 处于【筹码出清】阶段（COT Index: {latest_idx:.1f}%）。"
                       "大户空头头寸已接近极值，关注超跌反弹机会。")
        else:
            st.info(f"📊 状态：当前筹码分布相对中性（COT Index: {latest_idx:.1f}%）。"
                    "建议关注 Finviz 实时价格变动，寻找趋势性突破。")


if __name__ == "__main__":
    main()