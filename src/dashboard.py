import streamlit as st
import pandas as pd
import yfinance as yf
import talib
import numpy as np
import json
import os
import io
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import TypedDict, Dict, List, Optional, Any
from langgraph.graph import StateGraph, START, END
from google import genai
from dotenv import load_dotenv
from urllib.request import Request, urlopen
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()


# --- 1. 结构化数据定义 ---
class SentimentResult(BaseModel):
    score: float = Field(description="情绪分数，范围从 -1.0 (利空) 到 1.0 (利好)")
    reason: str = Field(description="简短分析理由，限 20 字")


class BatchSentiment(BaseModel):
    results: Dict[str, SentimentResult] = Field(description="以 Ticker 为键，SentimentResult 为值的字典")


class GraphState(TypedDict):
    dynamic_etf_list: List[str]
    etf_news_sentiment: Dict[str, int]
    etf_news_reasons: Dict[str, str]
    etf_highlights: Optional[pd.DataFrame]
    raw_sectors: Optional[pd.DataFrame]
    raw_industries: Optional[pd.DataFrame]
    hierarchy_db: Any
    etf_cn_map: Dict[str, str]  # ✅ 新增：ETF ticker -> 中文名
    error: Optional[str]



# --- 2. 核心辅助函数 ---
def get_rss_news(ticker: str) -> List[str]:
    """获取标的近 7 天实时新闻标题"""
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock+when:7d&hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = Request(url, headers=headers)
        with urlopen(req, timeout=10) as response:
            root = ET.fromstring(response.read())
            return [t.text for item in root.findall('.//item')[:5] if (t := item.find('title')) is not None]
    except:
        return []


# --- 3. 节点逻辑 ---

def discover_etf_node(state: GraphState):
    """节点 1: 扫描最活跃 ETF 列表"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = "https://finviz.com/screener.ashx?v=111&f=ind_exchangetradedfund&o=-volume"
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as resp:
            df = pd.read_html(io.StringIO(resp.read().decode('utf-8')))[-2]
            return {"dynamic_etf_list": df['Ticker'].tolist()[:25]}
    except:
        return {"dynamic_etf_list": ['SPY', 'QQQ', 'IWM', 'SMH', 'XLK']}


def sentiment_node(state: GraphState):
    """节点 2: AI 实时舆情评分 (40% 权重)"""
    etf_pool = state.get("dynamic_etf_list", [])
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    parser = JsonOutputParser(pydantic_object=BatchSentiment)

    sentiment_map, reason_map = {}, {}
    status_placeholder = st.empty()

    # 分批处理提高稳定性
    for i in range(0, min(len(etf_pool), 12), 4):
        batch = etf_pool[i:i + 4]
        status_placeholder.text(f"🧪 AI 深度解析舆情中: {batch}...")
        news_payload = [{"ticker": t, "news": get_rss_news(t)} for t in batch]

        prompt = f"分析标的最新情绪：\n{parser.get_format_instructions()}\n数据：{json.dumps(news_payload)}"
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt,
                                                      config={'response_mime_type': 'application/json'})
            parsed = parser.parse(response.text)
            for t, res in parsed['results'].items():
                sentiment_map[t] = int((float(res['score']) + 1) * 50)
                reason_map[t] = res['reason']
        except:
            continue
    status_placeholder.empty()
    return {"etf_news_sentiment": sentiment_map, "etf_news_reasons": reason_map}


def etf_scanner_node(state: GraphState):
    """节点 3: TA-Lib 量价指标 + AI 情绪融合 (60:40)"""
    etf_pool = state.get("dynamic_etf_list", [])
    sent_scores = state.get("etf_news_sentiment", {})
    sent_reasons = state.get("etf_news_reasons", {})
    results = []

    for ticker in etf_pool:
        try:
            df = yf.download(ticker, start=datetime.now() - timedelta(days=60), progress=False)
            if df is None or len(df) < 20: continue

            # 处理 MultiIndex 确保取出一维数组
            closes = df['Close'].iloc[:, 0].values if isinstance(df['Close'], pd.DataFrame) else df['Close'].values
            volumes = df['Volume'].iloc[:, 0].values if isinstance(df['Volume'], pd.DataFrame) else df['Volume'].values
            closes, volumes = closes.flatten().astype(float), volumes.flatten().astype(float)

            # 技术面分：基于 OBV 斜率与价格高位
            obv = talib.OBV(closes, volumes)
            slope = talib.LINEARREG_SLOPE(obv, timeperiod=5)[-1]
            tech_score = int((closes[-1] / np.max(closes[-20:])) * 75 + (15 if slope > 0 else 0))

            # 舆情分：获取 AI 评分
            news_score = sent_scores.get(ticker, 50)

            # 综合强度
            comp_score = int(tech_score * 0.6 + news_score * 0.4)

            # 多头决策建议
            if comp_score >= 82 and slope > 0:
                rec, reason = "🌟 强烈推荐", "量价舆情强力共振"
            elif tech_score >= 75 and slope > 0:
                rec, reason = "✅ 建议买入", "技术趋势多头占优"
            elif news_score >= 80 and slope <= 0:
                rec, reason = "⚠️ 警惕诱多", "情绪亢奋但资金面背离"
            else:
                rec, reason = "❌ 暂不推荐", "合力不足或趋势偏弱"

            results.append({
                "代码": ticker, "现价": f"${closes[-1]:.2f}",
                "技术分": tech_score, "舆情分": news_score, "综合强度": comp_score,
                "决策建议": rec, "多头理由": reason, "AI解读": sent_reasons.get(ticker, "无")
            })
        except:
            continue
    return {"etf_highlights": pd.DataFrame(results).sort_values("综合强度", ascending=False)}


def fetch_market_node(state: GraphState):
    """节点 4: 板块行情原始数据"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        def get_data(g):
            url = f"https://finviz.com/groups.ashx?g={g}&v=140&o=-perf1m"
            req = Request(url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                df = pd.read_html(io.StringIO(resp.read().decode('utf-8')))[-2]
                for col in ['Perf Week', 'Perf Month']:
                    df[col] = df[col].astype(str).str.replace('%', '').replace('-', '0').astype(float)
                return df

        return {"raw_sectors": get_data('sector'), "raw_industries": get_data('industry')}
    except:
        return {"error": "板块抓取失败"}


def ai_modeling_node(state: GraphState):
    """节点 5: AI 自动化层级树建模 + ETF 中文名映射"""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    sectors = state["raw_sectors"]["Name"].tolist() if state.get("raw_sectors") is not None else []
    industries = state["raw_industries"]["Name"].tolist() if state.get("raw_industries") is not None else []
    etfs = state.get("dynamic_etf_list", [])[:25]

    # 要求模型返回固定结构：market_hierarchy + etf_cn_map
    prompt = (
        "请输出 JSON，包含两个字段：\n"
        "1) market_hierarchy: 以 Sector 英文名为 key，value 包含 cn(中文名) 与 sub(子行业数组)，"
        "sub 元素包含 en/cn/desc。\n"
        "2) etf_cn_map: 以 ETF ticker 为 key，value 为中文名（无法确定则给出简短中文或原 ticker）。\n"
        f"Sectors: {sectors}\n"
        f"Industries: {industries}\n"
        f"ETFs: {etfs}\n"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        h_data = json.loads(response.text) if response and getattr(response, "text", None) else {}

        # 兜底规范化
        market_h = h_data.get("market_hierarchy") if isinstance(h_data, dict) else {}
        etf_cn_map = h_data.get("etf_cn_map") if isinstance(h_data, dict) else {}

        if not isinstance(market_h, dict):
            market_h = {}
        if not isinstance(etf_cn_map, dict):
            etf_cn_map = {}

        payload = {"market_hierarchy": market_h, "etf_cn_map": etf_cn_map}
        with open("market_hierarchy.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)

        return {"hierarchy_db": market_h, "etf_cn_map": etf_cn_map}
    except:
        return {}


# --- 4. 构建工作流 ---
def build_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("discover", discover_etf_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("scanner", etf_scanner_node)
    workflow.add_node("fetcher", fetch_market_node)
    workflow.add_node("ai", ai_modeling_node)

    workflow.add_edge(START, "discover")
    workflow.add_edge("discover", "sentiment")
    workflow.add_edge("sentiment", "scanner")
    workflow.add_edge("scanner", "fetcher")
    workflow.add_edge("fetcher", "ai")
    workflow.add_edge("ai", END)
    return workflow.compile()


# --- 5. 渲染 UI ---

def render_ui():
    st.set_page_config(page_title="AI 量化决策系统", layout="wide")
    st.title("🦅 智能多头量化与行业解析看板")

    if st.button("🚀 启动全流程深度扫描", type="primary"):
        app = build_workflow()
        current_state = {
            "dynamic_etf_list": [],
            "etf_news_sentiment": {},
            "etf_news_reasons": {},
            "etf_highlights": pd.DataFrame(),
            "raw_sectors": None,
            "raw_industries": None,
            "hierarchy_db": {},
            "etf_cn_map": {},  # ✅ 新增
            "error": None,
        }
        with st.status("正在进行多维交叉分析...", expanded=True) as status:
            for event in app.stream(current_state):
                for node_name, output in event.items():
                    st.write(f"✅ 节点 `{node_name}` 处理完毕")
                    if output:
                        current_state.update(output)
            status.update(label="扫描完毕!", state="complete")
        st.session_state.final_state = current_state

    if "final_state" in st.session_state:
        state = st.session_state.final_state

        # --- ETF 中文名映射（用于主表 + 行业透视树展示） ---
        etf_cn_map = state.get("etf_cn_map") or {}
        if not isinstance(etf_cn_map, dict):
            etf_cn_map = {}

        # 1. 多头决策主表：增加「中文名」列
        if state.get("etf_highlights") is not None and not state["etf_highlights"].empty:
            st.subheader("🔥 实时量化与舆情共振榜单")
            df_show = state["etf_highlights"].copy()
            if "代码" in df_show.columns:
                df_show.insert(0, "中文名", df_show["代码"].map(lambda x: etf_cn_map.get(str(x), str(x))))

            st.dataframe(
                df_show,
                width="stretch",
                hide_index=True,
                column_config={
                    "综合强度": st.column_config.ProgressColumn(min_value=0, max_value=100),
                    "多头理由": st.column_config.TextColumn(width="large"),
                },
            )

        # 2. 行业透视树：在标题处展示 ETF 中文名列表（来自动态 ETF 池）
        if state.get("raw_sectors") is not None:
            st.divider()
            st.subheader("🌳 行业透视层级树 (AI 归类)")

            # 展示 ETF 中文名概览（放在行业树上方）
            etf_pool = state.get("dynamic_etf_list", [])[:25]
            if etf_pool:
                cn_list = [etf_cn_map.get(t, t) for t in etf_pool]
                st.caption("本次扫描 ETF: " + " / ".join(cn_list))

            s_df, i_df = state["raw_sectors"], state["raw_industries"]

            # 关键修复：把 hierarchy_db 规范化为 dict，避免 list.get 报错
            h_db = state.get("hierarchy_db") or {}
            if isinstance(h_db, list):
                h_db = h_db[0] if (len(h_db) > 0 and isinstance(h_db[0], dict)) else {}
            elif not isinstance(h_db, dict):
                h_db = {}

            for _, s_row in s_df.sort_values("Perf Month", ascending=False).iterrows():
                s_en = s_row["Name"]
                s_meta = h_db.get(s_en, {"cn": s_en, "sub": []})
                icon = "🔴" if s_row["Perf Month"] > 0 else "🟢"

                with st.expander(f"{icon} {s_row['Perf Month']}% | {s_meta.get('cn', s_en)}"):
                    sub_list = s_meta.get("sub", [])
                    sub_names = [item.get("en") for item in sub_list if isinstance(item, dict)]
                    sub_data = i_df[i_df["Name"].isin(sub_names)].copy()

                    if not sub_data.empty:
                        map_dict = {
                            item["en"]: (item.get("cn", item["en"]), item.get("desc", ""))
                            for item in sub_list
                            if isinstance(item, dict) and "en" in item
                        }
                        sub_data["中文名"] = sub_data["Name"].apply(lambda x: map_dict.get(x, (x, ""))[0])
                        sub_data["月涨幅%"] = sub_data["Perf Month"]
                        st.dataframe(
                            sub_data[["中文名", "Name", "月涨幅%"]].rename(columns={"Name": "原名"}).style.map(
                                lambda x: (
                                    "color: #ff4b4b; font-weight: bold"
                                    if isinstance(x, float) and x > 0
                                    else "color: #09ab3b; font-weight: bold"
                                    if isinstance(x, float) and x < 0
                                    else ""
                                ),
                                subset=["月涨幅%"],
                            ),
                            width="stretch",
                            hide_index=True,
                        )
if __name__ == "__main__":
    render_ui()
