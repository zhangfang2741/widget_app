import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pickle
import os
import datetime
from pathlib import Path

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# --- 1. 配置 ---
TIINGO_API_KEY = "302c6b2a5781f2b0831b324870f217944ced68e6"
CACHE_DIR = Path("tiingo_ticker_cache")
CACHE_DIR.mkdir(exist_ok=True)
CHINESE_NAMES = {
    # 01 信息技术
    "XLK": "科技行业精选指数ETF-SPDR",
    "VGT": "先锋信息技术ETF-Vanguard",
    "SMH": "半导体指数ETF-VanEck",
    "IGV": "软件服务指数ETF-iShares",
    # 02 医疗保健
    "XLV": "医疗保健行业精选指数ETF-SPDR",
    "IBB": "纳斯达克生物技术ETF-iShares",
    "XBI": "标普生物技术ETF-SPDR",
    "IHI": "医疗器械指数ETF-iShares",
    # 03 金融
    "XLF": "金融行业精选指数ETF-SPDR",
    "KBE": "标普银行ETF-SPDR",
    "KRE": "标普地区银行ETF-SPDR",
    "IAI": "证券经纪商指数ETF-iShares",
    # 04 可选消费
    "XLY": "可选消费行业精选指数ETF-SPDR",
    "XRT": "标普零售指数ETF-SPDR",
    "PEJ": "休闲娱乐指数ETF-Invesco",
    # 05 必需消费
    "XLP": "必需消费行业精选指数ETF-SPDR",
    "VDC": "先锋必需消费ETF-Vanguard",
    "COST": "开市客(个股)",
    # 06 工业
    "XLI": "工业行业精选指数ETF-SPDR",
    "ITA": "航空国防指数ETF-iShares",
    "JETS": "全球航空业ETF-US Global",
    "PAVE": "基础建设指数ETF-Global X",
    # 07 能源
    "XLE": "能源行业精选指数ETF-SPDR",
    "XOP": "标普油气开采ETF-SPDR",
    "ICLN": "全球清洁能源ETF-iShares",
    # 08 原材料
    "XLB": "原材料行业精选指数ETF-SPDR",
    "GLD": "黄金ETF-SPDR Gold",
    "SLV": "白银ETF-iShares Silver",
    "GDX": "金矿股指数ETF-VanEck",
    "COPX": "铜矿股指数ETF-Global X",
    # 09 通信服务
    "XLC": "通信服务行业精选指数ETF-SPDR",
    "VOX": "先锋通信服务ETF-Vanguard",
    "SOCL": "社交媒体指数ETF-Global X",
    # 10 房地产
    "XLRE": "房地产行业精选指数ETF-SPDR",
    "VNQ": "先锋房地产REITs ETF-Vanguard",
    "REZ": "住宅房地产指数ETF-iShares",
    # 11 公用事业
    "XLU": "公用事业行业精选指数ETF-SPDR",
    "VPU": "先锋公用事业ETF-Vanguard",
    "NEE": "新纪元能源(个股)",
    # 12 另类/跨行业
    "ARKK": "方舟创新ETF-ARK Invest",
    "BITO": "比特币策略ETF-ProShares",
    "MSOS": "大麻核心ETF-AdvisorShares",
}

# 行业层级定义 (涵盖100+细分)
ETF_LIBRARY = {
    "01 信息技术": ["XLK", "VGT", "SMH", "IGV"],
    "02 医疗保健": ["XLV", "IBB", "XBI", "IHI"],
    "03 金融": ["XLF", "KBE", "KRE", "IAI"],
    "04 可选消费": ["XLY", "XRT", "PEJ"],
    "05 必需消费": ["XLP", "VDC", "COST"],
    "06 工业": ["XLI", "ITA", "JETS", "PAVE"],
    "07 能源": ["XLE", "XOP", "ICLN"],
    "08 原材料": ["XLB", "GLD", "SLV", "GDX", "COPX"],
    "09 通信服务": ["XLC", "VOX", "SOCL"],
    "10 房地产": ["XLRE", "VNQ", "REZ"],
    "11 公用事业": ["XLU", "VPU", "NEE"],
    "12 跨行业/另类": ["ARKK", "BITO", "MSOS"],
}

TICKER_TO_SECTOR = {t: s for s, ts in ETF_LIBRARY.items() for t in ts}
ALL_TICKERS = list(TICKER_TO_SECTOR.keys())

st.set_page_config(layout="wide", page_title="Market_Foldable_Tree")
st.title("🌲 美国行业资金流热力图")

# --- 2. 缓存与数据抓取 (CLV算法) ---
def fetch_ticker_data(ticker: str) -> pd.DataFrame | None:
    cache_path = CACHE_DIR / f"{ticker}.pkl"

    # 1) 读缓存：必须包含 Flow/Date/Ticker，否则视为无效缓存，走重新拉取
    if cache_path.exists() and (time.time() - os.path.getmtime(cache_path)) < 86400:
        try:
            with open(cache_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, pd.DataFrame):
                required = {"Flow", "Date", "Ticker"}
                if required.issubset(set(obj.columns)):
                    return obj
                # 缓存是旧结构或坏数据：忽略，继续走网络拉取覆盖缓存
        except Exception:
            pass

    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&token={TIINGO_API_KEY}"

    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None

        df = pd.DataFrame(r.json())
        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        denom = df["adjHigh"] - df["adjLow"]
        clv = ((df["adjClose"] - df["adjLow"]) - (df["adjHigh"] - df["adjClose"])) / (denom + 1e-9)
        df["Flow"] = clv * (df["adjClose"] * df["volume"])

        res = df[["date", "Flow"]].rename(columns={"date": "Date"})
        res["Ticker"] = ticker

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(res, f)
        except Exception:
            pass

        return res
    except Exception:
        return None


# --- 3. 构建可折叠汇总数据表 ---
with st.spinner("数据处理中..."):
    all_dfs: list[pd.DataFrame] = []

    total = len(ALL_TICKERS)
    progress = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(ALL_TICKERS, start=1):
        status.markdown(f"正在加载：`{ticker}`（{i}/{total}）")
        res = fetch_ticker_data(ticker)
        if isinstance(res, pd.DataFrame) and not res.empty:
            all_dfs.append(res)
        progress.progress(int(i * 100 / total))

    status.markdown("")
    progress.progress(100)

    if not all_dfs:
        st.error("无法加载数据，请检查缓存文件夹或 API 权限。")
        st.stop()

    full_df = pd.concat(all_dfs, ignore_index=True)
    required_cols = {"Flow", "Date", "Ticker"}
    missing = required_cols - set(full_df.columns)
    if missing:
        st.error(f"数据列缺失：{sorted(missing)}。请删除 `tiingo_ticker_cache` 下旧缓存后重试。")
        st.stop()

    full_df["板块"] = full_df["Ticker"].map(TICKER_TO_SECTOR)

    flow_mean = full_df["Flow"].mean()
    flow_std = full_df["Flow"].std()
    if not np.isfinite(flow_std) or flow_std == 0:
        flow_std = 1e-9

    full_df["Intensity"] = (full_df["Flow"] - flow_mean) / flow_std

    # 侧边栏：频率切换（天/周/月）
    freq = st.sidebar.radio("统计频率", ["天", "周", "月"], index=0, horizontal=True)

    # 统一日期列为 datetime，便于按周/月聚合
    full_df["Date"] = pd.to_datetime(full_df["Date"], errors="coerce")
    full_df = full_df.dropna(subset=["板块", "Date"])

    if freq == "周":
        # 用每周周一作为“周”标签（周频聚合）
        full_df["日期"] = full_df["Date"].dt.to_period("W-MON").dt.start_time.dt.strftime("%Y-%m-%d")
        recent_units_label = "显示最近周数"
    elif freq == "月":
        # 用每月月初作为“月”标签（月频聚合）
        full_df["日期"] = full_df["Date"].dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d")
        recent_units_label = "显示最近月数"
    else:
        full_df["日期"] = full_df["Date"].dt.strftime("%Y-%m-%d")
        recent_units_label = "显示最近交易日数"

    pivot = (
        full_df.pivot_table(index=["板块", "Ticker"], columns="日期", values="Intensity", aggfunc="mean")
        .fillna(0)
    )

    # 列按倒序（最近在左）
    pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)

    # 最近 N 个周期（天/周/月）
    recent_units = st.sidebar.slider(recent_units_label, 5, 30, 15)
    if pivot.shape[1] > recent_units:
        pivot = pivot.iloc[:, :recent_units]
    st.sidebar.markdown(
        r"""
    ### 📖 交互指南：
    1\. \*\*折叠查看概况\*\*：左侧“板块/ETF”列可折叠/展开分组。  

    2\. \*\*展开看细分\*\*：每个板块下展示对应 ETF 明细。  

    3\. \*\*颜色解读\*\*：本页“资金流强度”来自 CLV\+成交额的量化计算，并做标准化后着色。  
       - \*\*资金流（Flow）计算\*\*：先计算 CLV（Close Location Value，收盘价在当日区间的位置）  
         $$CLV=\frac{(C-L)-(H-C)}{H-L}=\frac{2C-H-L}{H-L}$$
         其中 $C=adjClose$、$H=adjHigh$、$L=adjLow$。为避免 $H=L$ 的除零，代码用 $H-L+1e-9$ 做平滑。  
         然后用“CLV \* 价格 \* 成交量”近似当日资金流强弱：  
         $$Flow=CLV\times(adjClose\times volume)$$
       - \*\*强度（Intensity）计算\*\*：对全样本的 Flow 做 Z\-score 标准化：  
         $$Intensity=\frac{Flow-mean(Flow)}{std(Flow)}$$
         若标准差为 0（或非有限值）则用极小值替代以避免除零。  
       - \*\*聚合与颜色\*\*：按“天/周/月”对 Intensity 取均值聚合。Intensity \> 0 显示红色，\< 0 显示绿色；颜色越深表示 $|Intensity|$ 越大（越“强”）。
    """,
        unsafe_allow_html=True,
    )
    # --- 4. 使用 AgGrid 渲染（分组折叠 + 热力） ---
    grid_df = pivot.reset_index()

    cellstyle_jscode = JsCode(
        """
        function(params) {
            const v = params.value;
            if (v === null || v === undefined) return {};
            const x = Number(v);
            if (isNaN(x)) return {};

            const scale = 1.6;
            const minAlpha = 0.20;
            const maxAlpha = 0.95;

            let a = Math.min(Math.abs(x) / scale, 1.0);
            a = minAlpha + (maxAlpha - minAlpha) * a;

            const textColor = (a >= 0.60) ? "white" : "black";

            if (x > 0) {
                return { backgroundColor: `rgba(255,0,0,${a})`, color: textColor };
            } else if (x < 0) {
                return { backgroundColor: `rgba(0,160,0,${a})`, color: textColor };
            } else {
                return { backgroundColor: "white", color: "black" };
            }
        }
        """
    )

    # 不使用 HTML，直接拼接纯文本：中文名 (Ticker)
    def _format_name(row: pd.Series) -> str:
        t = str(row["Ticker"])
        cn = CHINESE_NAMES.get(t, t)
        return f"{t}-{cn}"

    grid_df["名称"] = grid_df.apply(_format_name, axis=1)

    # --- 配置 AgGrid：用“名称”替代原来左侧的 Ticker 列展示 ---
    gb = GridOptionsBuilder.from_dataframe(grid_df)

    gb.configure_column("板块", rowGroup=True, hide=True)
    gb.configure_column("Ticker", hide=True)

    gb.configure_column(
        "名称",
        header_name="Ticker",
        pinned="left",
        width=260,
        minWidth=200,
        maxWidth=420,
    )

    date_cols = [c for c in grid_df.columns if c not in ("板块", "Ticker", "名称")]
    default_sort_col = date_cols[0] if date_cols else None

    for c in date_cols:
        gb.configure_column(
            c,
            headerName=str(c),
            type=["numericColumn"],
            aggFunc="avg",
            suppressAggFuncInHeader=True,
            valueFormatter="(params.value==null)?'':Number(params.value).toFixed(2)",
            cellStyle=cellstyle_jscode,
            suppressSizeToFit=True,
            sort="desc" if c == default_sort_col else None,  # \u2190 新增：默认按最新列倒序
        )
    gb.configure_default_column(sortable=True, filter=True, resizable=True)

    if "_ag_grid_ver" not in st.session_state:
        st.session_state["_ag_grid_ver"] = 0

    c1, c2, c3 = st.columns([1, 1, 8])
    with c1:
        if st.button("展开所有", use_container_width=True):
            st.session_state["_ag_expand_mode"] = "expand"
            st.session_state["_ag_grid_ver"] += 1
    with c2:
        if st.button("折叠所有", use_container_width=True):
            st.session_state["_ag_expand_mode"] = "collapse"
            st.session_state["_ag_grid_ver"] += 1

    expand_mode = st.session_state.get("_ag_expand_mode", "expand")  # "expand" / "collapse"

    on_grid_ready = JsCode(
        f"""
        function(params) {{
            try {{
                const mode = {repr(expand_mode)};

                if (mode) {{
                    params.api.forEachNode(function(node) {{
                        if (node.group) {{
                            node.setExpanded(mode === "expand");
                        }}
                    }});
                }}

                setTimeout(function () {{
                    try {{
                        const allColIds = [];
                        params.columnApi.getAllColumns().forEach(function (col) {{
                            allColIds.push(col.getColId());
                        }});
                        params.columnApi.autoSizeColumns(allColIds, false);
                    }} catch (e) {{}}
                }}, 50);
            }} catch (e) {{}}
        }}
        """
    )
    gb.configure_grid_options(
        groupDisplayType="singleColumn",
        groupIncludeFooter=True,
        groupIncludeTotalFooter=True,
        autoGroupColumnDef={
            "headerName": "板块/ETF",
            "minWidth": 130,
            "pinned": "left",
            "cellRendererParams": {"suppressCount": False},
        },
        domLayout="normal",
        rowHeight=32,
        onGridReady=on_grid_ready,
    )

    grid_options = gb.build()

    AgGrid(
        grid_df,
        gridOptions=grid_options,
        height=1500,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        theme="streamlit",
        fit_columns_on_grid_load=False,
        key=f"market_intensity_grid_{st.session_state['_ag_grid_ver']}",
    )

    # --- 执行后清除触发标记，避免下次重建重复动作 ---
    if "_ag_expand_mode" in st.session_state:
        del st.session_state["_ag_expand_mode"]
