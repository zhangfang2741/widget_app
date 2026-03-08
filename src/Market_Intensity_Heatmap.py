# python
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pickle
import os
import datetime
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import plotly.express as px

# --- 1. 配置 ---
TIINGO_API_KEY = "302c6b2a5781f2b0831b324870f217944ced68e6"
CACHE_DIR = Path("tiingo_ticker_cache")
CACHE_DIR.mkdir(exist_ok=True)

CHINESE_NAMES = {
    "XLK": "科技行业精选指数ETF-SPDR",
    "SOXX": "iShares半导体指数ETF",
    "AIQ": "Global X人工智能与科技ETF",
    "SKYY": "First Trust云计算指数ETF",
    "QTUM": "Defiance量子计算与机器学习ETF",
    "BUG": "Global X网络安全指数ETF",
    "IGV": "iShares扩张科技软件行业ETF",
    "XLV": "医疗保健行业精选指数ETF-SPDR",
    "XHE": "SPDR标普健康医疗设备ETF",
    "IHF": "iShares美国医疗保健提供商ETF",
    "XBI": "SPDR标普生物技术ETF",
    "PJP": "Invesco动力制药ETF",
    "XLF": "金融行业精选指数ETF-SPDR",
    "KBE": "SPDR标普银行指数ETF",
    "IYG": "iShares美国金融服务ETF",
    "KIE": "SPDR标普保险ETF",
    "BLOK": "Amplify转型数据共享ETF(区块链)",
    "KCE": "SPDR标普资本 market ETF",
    "REM": "iShares安硕抵押贷款地产投资信托ETF",
    "XLY": "可选消费行业精选指数ETF-SPDR",
    "CARZ": "First Trust纳斯达克全球汽车指数ETF",
    "XRT": "SPDR标普零售业ETF",
    "XHB": "SPDR标普家居建设ETF",
    "PEJ": "Invesco休闲娱乐ETF",
    "XLP": "必需消费行业精选指数ETF-SPDR",
    "PBJ": "Invesco动力食品饮料ETF",
    "MOO": "VanEck全球农产品ETF",
    "XLI": "工业行业精选指数ETF-SPDR",
    "ITA": "iShares美国航空航天与国防ETF",
    "PKB": "Invesco动力住宅建设ETF",
    "PAVE": "Global X美国基础设施发展ETF",
    "IYT": "iShares交通运输ETF",
    "JETS": "U.S. Global Jets 航空业ETF",
    "BOAT": "SonicShares全球航运ETF",
    "IFRA": "iShares美国基础设施ETF",
    "UFO": "Procure太空ETF",
    "SHLD": "Strive美国国防与航空航天ETF",
    "XLE": "能源行业精选指数ETF-SPDR",
    "IEZ": "iShares美国石油设备与服务ETF",
    "XOP": "SPDR标普石油天然气开采ETF",
    "FAN": "First Trust全球风能ETF",
    "TAN": "Invesco太阳能ETF",
    "NLR": "VanEck铀及核能ETF",
    "XLB": "原材料行业精选指数ETF-SPDR",
    "XME": "SPDR标普金属与采矿ETF",
    "WOOD": "iShares全球林业ETF",
    "COPX": "Global X铜矿股ETF",
    "GLD": "SPDR黄金ETF",
    "GLTR": "Aberdeen标准实物贵金属篮子ETF",
    "SLV": "iShares白银ETF",
    "SLX": "VanEck矢量钢铁ETF",
    "BATT": "Amplify锂电池及关键材料ETF",
    "XLC": "通信服务行业精选指数ETF-SPDR",
    "IYZ": "iShares美国电信ETF",
    "PNQI": "Invesco纳斯达克互联网ETF",
    "XLRE": "房地产行业精选指数ETF-SPDR",
    "INDS": "Pacer工业地产ETF",
    "REZ": "iShares住宅与多户家庭地产投资信托ETF",
    "SRVR": "Pacer数据基础设施与房地产ETF",
    "XLU": "公用事业行业精选指数ETF-SPDR",
    "ICLN": "iShares全球清洁能源ETF",
    "PHO": "Invesco水资源ETF",
    "GRID": "First Trust纳斯达克智能电网基础设施ETF",
    "QQQ": "Invesco纳斯达克100指数ETF",
    "SPY": "SPDR标普500指数ETF",
    "TLT": "iShares 20年期以上美国国债ETF",
    "EEM": "iShares MSCI新兴市场ETF",
    "VEA": "Vanguard FTSE发达市场ETF",
    "FXI": "iShares中国大盘股ETF",
    "ARKK": "ARK创新ETF",
    "BITO": "ProShares比特币策略ETF",
    "MSOS": "AdvisorShares纯大麻ETF",
    "IPO": "Renaissance IPO ETF",
    "GBTC": "灰度比特币现货ETF",
    "ETHE": "灰度以太坊现货ETF"
}

ETF_LIBRARY = {
    "01 信息技术": ["XLK", "SOXX", "AIQ", "SKYY", "QTUM", "BUG", "IGV"],
    "02 医疗保健": ["XLV", "XHE", "IHF", "XBI", "PJP"],
    "03 金融": ["XLF", "KBE", "IYG", "KIE", "BLOK", "KCE", "REM"],
    "04 可选消费": ["XLY", "CARZ", "XRT", "XHB", "PEJ"],
    "05 必需消费": ["XLP", "PBJ", "MOO"],
    "06 工业": ["XLI", "ITA", "PKB", "PAVE", "IYT", "JETS", "BOAT","IFRA","UFO","SHLD"],
    "07 能源": ["XLE", "IEZ", "XOP", "FAN", "TAN", "NLR"],
    "08 原材料": ["XLB", "PKB", "XME", "WOOD", "COPX", "GLD", "GLTR", "SLV", "SLX", "BATT"],
    "09 通信服务": ["XLC", "IYZ", "PNQI"],
    "10 房地产": ["XLRE", "INDS", "REZ", "SRVR"],
    "11 公用事业": ["XLU", "ICLN", "PHO", "GRID"],
    "12 全球宏观/另类": ["TLT", "EEM", "VEA", "FXI", "ARKK", "BITO", "MSOS", "IPO", "UFO","GBTC", "ETHE"]
}

TICKER_TO_SECTOR = {t: s for s, ts in ETF_LIBRARY.items() for t in ts}
ALL_TICKERS = list(TICKER_TO_SECTOR.keys())

st.set_page_config(layout="wide", page_title="Market_Foldable_Tree")
st.title("🌲 美国行业资金流热力图")

# --- 2. 缓存与数据抓取 (CLV算法) ---
def fetch_ticker_data(ticker: str) -> pd.DataFrame | None:
    cache_path = CACHE_DIR / f"{ticker}.pkl"
    if cache_path.exists() and (time.time() - os.path.getmtime(cache_path)) < 86400:
        try:
            with open(cache_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, pd.DataFrame):
                required = {"Flow", "Date", "Ticker"}
                if required.issubset(set(obj.columns)):
                    return obj
        except Exception as e:
            st.warning(f"无法读取缓存 {cache_path}: {e}")
            pass

    # 其余情况向 API 请求最新数据
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&token={TIINGO_API_KEY}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            st.error(f"API 请求失败 {ticker}: {r.status_code} {r.text}")
            return None

        df = pd.DataFrame(r.json())
        if df.empty:
            st.warning(f"API 返回空数据 {ticker}")
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
        except Exception as e:
            st.warning(f"无法写入缓存 {cache_path}: {e}")
            pass

        return res
    except Exception as e:
        st.warning(f"无法获取数据 {ticker}: {e}")
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

    freq = st.sidebar.radio("统计频率", ["天", "周", "月"], index=0, horizontal=True)

    full_df["Date"] = pd.to_datetime(full_df["Date"], errors="coerce")
    full_df = full_df.dropna(subset=["板块", "Date"])

    if freq == "周":
        full_df["日期"] = full_df["Date"].dt.to_period("W-MON").dt.start_time.dt.strftime("%Y-%m-%d")
        recent_units_label = "显示最近周数"
    elif freq == "月":
        full_df["日期"] = full_df["Date"].dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d")
        recent_units_label = "显示最近月数"
    else:
        full_df["日期"] = full_df["Date"].dt.strftime("%Y-%m-%d")
        recent_units_label = "显示最近交易日数"

    pivot = (
        full_df.pivot_table(index=["板块", "Ticker"], columns="日期", values="Intensity", aggfunc="mean")
        .fillna(0)
    )

    pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)

    recent_units = st.sidebar.slider(recent_units_label, 5, 30, 15)
    if pivot.shape[1] > recent_units:
        pivot = pivot.iloc[:, :recent_units]

    st.sidebar.markdown(
        r"""
    ### 📖 交互指南：
    1\. \*\*折叠查看概况\*\*：左侧\`板块/ETF\`列可折叠/展开分组。

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

    def _format_name(row: pd.Series) -> str:
        t = str(row["Ticker"])
        cn = CHINESE_NAMES.get(t, t)
        return f"{t}-{cn}"

    grid_df["名称"] = grid_df.apply(_format_name, axis=1)

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
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
            sort="desc" if c == default_sort_col else None,
        )
    gb.configure_default_column(sortable=True, filter=True, resizable=True)

    on_grid_ready = JsCode(
        f"""
        function(params) {{
            try {{
                const mode = {repr(st.session_state.get("_ag_expand_mode", "expand"))};

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

    if "_ag_grid_ver" not in st.session_state:
        st.session_state["_ag_grid_ver"] = 0

    # 折叠/展开/全屏 控制按钮
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("展开所有", use_container_width=True):
            st.session_state["_ag_expand_mode"] = "expand"
            st.session_state["_ag_grid_ver"] += 1
    with c2:
        if st.button("折叠所有", use_container_width=True):
            st.session_state["_ag_expand_mode"] = "collapse"
            st.session_state["_ag_grid_ver"] += 1
    with c3:
        # 全屏开关
        if st.session_state.get("_ag_fullscreen"):
            if st.button("退出全屏", use_container_width=True, key=f"exit_full_{st.session_state['_ag_grid_ver']}"):
                st.session_state.pop("_ag_fullscreen", None)
                st.session_state["_ag_grid_ver"] += 1
        else:
            if st.button("全屏", use_container_width=True, key=f"enter_full_{st.session_state['_ag_grid_ver']}"):
                st.session_state["_ag_fullscreen"] = True
                st.session_state["_ag_grid_ver"] += 1

    # 若处于全屏模式，只渲染表格（独占页面）
    if st.session_state.get("_ag_fullscreen"):
        st.info("全屏模式：表格占据页面。点击「退出全屏」返回。")
        grid_response = AgGrid(
            grid_df,
            gridOptions=grid_options,
            height=900,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=True,
            theme="streamlit",
            fit_columns_on_grid_load=True,
            key=f"market_intensity_grid_full_{st.session_state['_ag_grid_ver']}",
            update_mode=GridUpdateMode.SELECTION_CHANGED,
        )

        selected = grid_response.get("selected_rows", [])
        if isinstance(selected, pd.DataFrame):
            selected = selected.to_dict(orient="records")
        elif isinstance(selected, pd.Series):
            selected = [selected.to_dict()]
        elif selected is None:
            selected = []

        if len(selected) > 0:
            sel = selected[0]
            sel_ticker = sel.get("Ticker")
            if sel_ticker:
                st.session_state["_selected_ticker"] = sel_ticker
                st.session_state["_open_ticker_modal"] = True

        # 额外提供一个退出全屏按钮（冗余但便捷）
        if st.button("退出全屏 (下方)"):
            st.session_state.pop("_ag_fullscreen", None)
            st.session_state["_ag_grid_ver"] += 1

    else:
        # Grid 与 右侧面板并列显示（正常布局）
        col_grid, col_panel = st.columns([8, 4])

        with col_grid:
            grid_response = AgGrid(
                grid_df,
                gridOptions=grid_options,
                height=1500,
                allow_unsafe_jscode=True,
                enable_enterprise_modules=True,
                theme="streamlit",
                fit_columns_on_grid_load=False,
                key=f"market_intensity_grid_{st.session_state['_ag_grid_ver']}",
                update_mode=GridUpdateMode.SELECTION_CHANGED,
            )

            selected = grid_response.get("selected_rows", [])

            if isinstance(selected, pd.DataFrame):
                selected = selected.to_dict(orient="records")
            elif isinstance(selected, pd.Series):
                selected = [selected.to_dict()]
            elif selected is None:
                selected = []

            if len(selected) > 0:
                sel = selected[0]
                sel_ticker = sel.get("Ticker")
                if sel_ticker:
                    st.session_state["_selected_ticker"] = sel_ticker
                    st.session_state["_open_ticker_modal"] = True

        with col_panel:
            if st.session_state.get("_open_ticker_modal") and st.session_state.get("_selected_ticker"):
                ticker = st.session_state["_selected_ticker"]
                panel_title = f"{ticker} 资金流 (Flow) — 可交互"

                with st.expander(panel_title, expanded=True):
                    with st.spinner(f"加载 {ticker} 数据..."):
                        df_t = fetch_ticker_data(ticker)

                    if isinstance(df_t, pd.DataFrame) and not df_t.empty:
                        df_plot = df_t.sort_values("Date")
                        fig = px.line(df_plot, x="Date", y="Flow", title=panel_title, markers=True)
                        fig.update_layout(autosize=True, height=640, xaxis_title="Date", yaxis_title="Flow")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"无法获取 {ticker} 的资金流数据。")

                    if st.button("关闭右侧面板", use_container_width=True, key=f"close_ticker_right_{ticker}"):
                        st.session_state.pop("_open_ticker_modal", None)
                        st.session_state.pop("_selected_ticker", None)
            else:
                st.info("在表格中选择一行以在右侧查看该 Ticker 的资金流图表。")

    # 清理一次性控制变量
    if "_ag_expand_mode" in st.session_state:
        st.session_state.pop("_ag_expand_mode", None)