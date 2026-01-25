import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
import re
from google import genai
from google.genai import types
import logging
import contextlib
from contextlib import nullcontext

# Reduce noisy log output from yfinance / urllib3
logging.getLogger('yfinance').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)


# Helper to suppress noisy yfinance stdout/stderr (prevents 'Failed download' lines flooding the UI/console)
def safe_yf_download(*args, **kwargs):
    devnull = open(os.devnull, 'w')
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return yf.download(*args, **kwargs)
    finally:
        devnull.close()


def sanitize_and_validate_user_tickers(raw_input, max_keep=50):
    """Parse and validate user-provided tickers. Returns (valid_list, invalid_list).

    Uses a conservative regex + a short yf check to ensure the symbol exists before proceeding.
    """
    if not raw_input or not raw_input.strip():
        return [], []

    tokens = re.split(r'[,\s]+', raw_input.strip())
    seen = set()
    valid = []
    invalid = []

    for tok in tokens:
        if not tok:
            continue
        t = tok.strip().upper().lstrip('$').strip(',')
        t = t.replace('.', '-')  # normalize BRK.B -> BRK-B
        if not re.match(r'^[A-Z][A-Z0-9\-]{0,5}$', t):
            invalid.append(tok)
            continue
        if t in seen:
            continue
        seen.add(t)
        # quick existence check
        try:
            df = safe_yf_download(t, period='5d', interval='1d', progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                valid.append(t)
            else:
                invalid.append(tok)
        except Exception:
            invalid.append(tok)
        if len(valid) >= max_keep:
            break

    return valid, invalid


# --- 1. 环境与连接配置 ---

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Environment variable GEMINI_API_KEY is not set or is empty.")

client = genai.Client(api_key=gemini_api_key)


# --- 2. 核心量化算法库 ---
@st.cache_data(ttl=3600)
def get_structured_data(ticker):
    try:
        data = safe_yf_download(ticker, period="1y", interval="1d", auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close = data['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))

        curr_p = float(close.iloc[-1])
        curr_m50 = float(ma50.iloc[-1])
        curr_m200 = float(ma200.iloc[-1])
        curr_rsi = float(rsi.iloc[-1])
        bias = (curr_p - curr_m200) / curr_m200

        if curr_p > curr_m200 and curr_m50 < curr_m200:
            phase = "🟢 起步阶段"
        elif curr_p > curr_m50 > curr_m200 and bias < 0.25:
            phase = "📈 上升阶段"
        elif curr_p > curr_m50 and bias >= 0.25:
            phase = "🔥 成熟阶段"
        else:
            phase = "📉 调整阶段"

        return {"df": data, "price": curr_p, "rsi": curr_rsi, "bias": bias, "phase": phase, "ticker": ticker}
    except Exception as e:
        print(f"[{ticker}] 数据抓取失败: {e}")
        return None


# --- 3. 开启搜索能力的 AI 模块 ---

@st.cache_data(ttl=3600)
def get_ai_suggestions() -> str:
    """Ensure the function explicitly returns a string."""
    time_str = pd.Timestamp.now().strftime("%Y年-%m月-%d日")
    prompt = f"""
    请在美股市场中筛选出当前（{time_str}）时间最近 5 个交易日内满足以下量价与情绪特征的ETF（不要股票）：
    1) 成交量连续放大（连续 3 日或以上成交量环比上升）
    2) 价格近期创近期高点或呈稳步攀升趋势
    3) 社交/新闻情绪明显上升（如情绪数据或社媒热度/提及度明显提高）

    请优先涵盖以下主题：
    - 科技/AI 
    - 加密货币
    - 贵金属（黄金/白银）
    - 能源
    - 动量/情绪型
    
    输出：
    按照热度排名筛选出最热的5个ETF代码，仅返回代码或符号，用英文逗号分隔，不要包含其他文字。例如：QQQ, NVDA, GLD
    """
    response = client.models.generate_content(
        model='gemini-2.0-flash', contents=prompt
    )
    raw_text = response.text.upper()
    print("raw_text:" + raw_text)
    return raw_text


@st.cache_data(ttl=3600)
def get_ai_summary(ticker, phase, rsi, bias):
    """基于实时搜索的深度结构化预测"""
    search_tool = types.Tool(google_search=types.GoogleSearch())

    # 系统指令：强制模型进行结构化链式推理
    sys_instruction = "你是一个全球宏观策略专家。你必须结合 Google 搜索到的 2026 年最新宏观数据、政策变量进行推理。"

    prompt = f"""
    分析ETF：{ticker}。当前技术指标：阶段={phase}, RSI={rsi:.1f}, 乖离率(Bias)={bias:.1%}。

    任务：
    1. 搜索并确认该资产近期的核心驱动事件（如财报、利率决议、地缘动态、需求热度等）。
    2. 基于“结构化传导逻辑”：如果该资产持续走强/走弱，哪一个关联产业环节将成为下一个爆发点？
    3. 给出预测：趋势 → 供应链/流动性变化 → 资产形态。
    4. 每个ETF直接给出 2 个精准的传导方向和最可能爆发的资产代码，以及对每个资产的20字原因说明
    输出格式如下（中文）：
    1) 资产代码 - 爆发原因简述
    2) 资产代码 - 爆发原因简述
    只返回结果，不要包含其他说明文字。
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                tools=[search_tool]
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"AI 推理暂时受阻: {str(e)}"


# --- 4. UI 辅助函数 ---

def make_sparkline(data_series):
    colors = ['#2ca02c' if data_series.iloc[-1] >= data_series.iloc[0] else '#ff4b4b']
    # Correct color comparison and choose fill color based on the actual color
    primary = colors[0]
    if primary == '#2ca02c':
        fill = 'rgba(44, 160, 44, 0.1)'
    else:
        fill = 'rgba(255, 75, 75, 0.1)'

    fig = go.Figure(data=go.Scatter(
        y=data_series, mode='lines', line=dict(color=primary, width=2),
        fill='tozeroy', fillcolor=fill
    ))
    fig.update_layout(
        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(t=5, b=5, l=0, r=0), height=40,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def get_top_stocks_in_etf(etf_ticker):
    """Fetch the top-performing stocks within a given ETF, including English and Chinese ETF names."""
    try:
        # Download ETF holdings data (mocked for demonstration)
        holdings = safe_yf_download(etf_ticker, period='1y', interval='1d', progress=False, auto_adjust=True)
        if holdings.empty:
            return f"No data available for ETF: {etf_ticker}"

        # Analyze holdings to find top-performing stocks
        top_stocks = holdings.sort_values(by='performance_metric', ascending=False).head(5)

        # Map ETF ticker to English and Chinese names
        asset_map = {
            "QQQ": {"english": "Invesco QQQ Trust", "chinese": "纳斯达克100指数"},
            "SMH": {"english": "VanEck Semiconductor ETF", "chinese": "半导体行业"},
            "BOTZ": {"english": "Global X Robotics & AI ETF", "chinese": "机器人与人工智能"},
            "GLD": {"english": "SPDR Gold Trust", "chinese": "黄金ETF"},
            "GDX": {"english": "VanEck Gold Miners ETF", "chinese": "黄金矿业ETF"},
            # Add more mappings as needed
        }

        etf_name = asset_map.get(etf_ticker, {"english": "Unknown", "chinese": "未知"})

        # Add ETF names to the result
        top_stocks["ETF_English_Name"] = etf_name["english"]
        top_stocks["ETF_Chinese_Name"] = etf_name["chinese"]

        return top_stocks[["ticker", "performance_metric", "ETF_English_Name", "ETF_Chinese_Name"]]
    except Exception as e:
        return f"Error fetching data for ETF {etf_ticker}: {str(e)}"


# --- 5. Streamlit 页面构建 ---
st.set_page_config(page_title="US Asset Structural Trends 2026", layout="wide")

st.title("🛡️ 核心资产结构化趋势看板 (AI 搜索增强版)")

# 初始化 Ticker 列表
if 'user_tickers' not in st.session_state:
    st.session_state.user_tickers = "IBIT, QQQ, GLD, SLV, SMH"


def auto_detect_market():
    """Ensure compatibility with st.spinner by using a fallback context manager."""
    context = st.spinner("AI 正在全网搜索最新叙事焦点...") if hasattr(st, 'spinner') else nullcontext()
    with context:
        suggestions = get_ai_suggestions()
        if suggestions:
            st.session_state.user_tickers = suggestions
        else:
            st.warning("AI 搜索未返回结果，请稍后重试。")


with st.sidebar:
    st.header("⚙️ 动态资产配置")
    st.button("🤖 AI 扫描今日全球热点", on_click=auto_detect_market)
    ticker_input = st.text_area("标的列表 (逗号分隔)", key="user_tickers", height=100)

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# 数据准备逻辑
asset_map = {"IBIT": "比特币现货", "QQQ": "纳指100", "GLD": "黄金现货", "SLV": "白银现货", "SMH": "半导体"}

# Adding a column to display the Chinese name of each ETF
# Assuming we have a dictionary mapping ETF codes to their Chinese names
ETF_CHINESE_NAMES = {
    "QQQ": "纳斯达克100指数",
    "NVDA": "英伟达",
    "SMCI": "超级微电脑",
    "IBIT": "比特币期货",
    "GLD": "黄金ETF",
    "GDX": "黄金矿业ETF",
    "XOM": "埃克森美孚",
    "ARKK": "方舟创新ETF"
}

# UI 渲染层
table_data = []
progress_text = "正在执行结构化量化扫描..."
my_bar = st.progress(0, text=progress_text)

# Validate user input before processing
valid_tickers, invalid_tickers = sanitize_and_validate_user_tickers(st.session_state.user_tickers)
if invalid_tickers:
    st.warning(f"以下标的无效或数据获取失败，将被忽略：{', '.join(invalid_tickers)}")

for idx, ticker in enumerate(valid_tickers):
    res = get_structured_data(ticker)
    if res:
        ai_pred = get_ai_summary(ticker, res['phase'], res['rsi'], res['bias'])
        table_data.append({**res, "ai_pred": ai_pred, "name": asset_map.get(ticker, ticker),
                           "chinese_name": ETF_CHINESE_NAMES.get(ticker, "未知标的")})
    my_bar.progress((idx + 1) / len(valid_tickers), text=progress_text)
my_bar.empty()

# --- 渲染表格 ---
if table_data:

    st.markdown("---")
    header_cols = st.columns([0.6, 0.8, 1.2, 0.6, 1.2, 1.5, 3])

    labels = ["代码", "资产", "当前阶段", "RSI", "结构化乖离度", "12M趋势", "🔮 AI 实时链式预测"]

    # 明确解释每一列数字/含义，使用 caption 在标题下方展示说明
    header_explanations = [
        "标的代码（Ticker），支持 ETF 或 单只股票，例如 QQQ、NVDA",
        "底层资产或主题的简短说明，例如：纳指100 / 英伟达 / 黄金",
        "当前阶段：标的的技术分析阶段，例如：上涨阶段、成熟阶段",
        "RSI：相对强弱指数，衡量标的的超买或超卖状态",
        "结构化乖离度：标的价格与其均值的偏离程度，显示为百分比",
        "12M趋势：标的最近一年的价格趋势图",
        "🔮 AI 实时链式预测：基于 AI 的实时预测，展示潜在的爆发机会"
    ]

    for col, label, help_text in zip(header_cols, labels, header_explanations):
        col.markdown(f"**{label}**")
        # caption 较小字体显示解释，便于用户直接阅读表头含义
        col.caption(help_text)
    st.markdown("---")

    for row in table_data:
        c1, c2, c3, c4, c5, c6, c7 = st.columns([0.6, 0.8, 1.2, 0.6, 1.2, 1.5, 3])
        c1.markdown(f"#### {row['ticker']}")
        c2.caption(row['name'])
        c3.caption(row['phase'])

        rsi_val = row['rsi']
        c4.markdown(f":{'red' if rsi_val > 70 else 'blue' if rsi_val < 30 else 'green'}[**{rsi_val:.0f}**]")

        bias_pct = row['bias'] * 100
        bar_color = "#ff4b4b" if row['bias'] >= 0.25 else "#2ca02c" if row['bias'] >= 0 else "#1f77b4"
        c5.markdown(
            f"""<div style="background-color: {bar_color}22; border-radius: 4px; padding: 2px 8px;"><span style="color: {bar_color}; font-weight: bold;">{bias_pct:+.1f}%</span></div>""",
            unsafe_allow_html=True)

        fig = make_sparkline(row['df']['Close'])
        c6.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"chart_{row['ticker']}")

        with c7.container(height=150, border=False):
            st.markdown(row['ai_pred'])
        st.divider()

    st.caption("注：AI 预测基于 Google Search 实时检索。量化指标每小时更新。")
else:
    st.info("请在左侧添加标的或点击 AI 扫描。")
