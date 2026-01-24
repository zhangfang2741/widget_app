import streamlit as st
import pandas as pd
import requests

# 页面配置
st.set_page_config(page_title="🔥 当前热门资产追踪", layout="wide")

st.title("🔥 全球热门加密资产实时看板")
st.markdown("数据来源于 CoinGecko 实时热门搜索榜单")


# 获取数据的函数
def get_trending_assets():
    url = "https://api.coingecko.com/api/v3/search/trending"
    try:
        response = requests.get(url)
        data = response.json()

        # 提取热门币种
        coins = []
        for coin in data['coins']:
            item = coin['item']
            coins.append({
                "排名": item['score'] + 1,
                "名称": item['name'],
                "符号": item['symbol'],
                "市值排名": item['market_cap_rank'],
                "价格 (BTC)": f"{item['price_btc']:.10f}",
                "图标": item['small']
            })
        return pd.DataFrame(coins)
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return None


# 侧边栏与刷新按钮
if st.button('点击刷新数据'):
    st.rerun()

# 展示数据
df = get_trending_assets()

if df is not None:
    # 使用 columns 布局增加视觉效果
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 热门列表")
        # 隐藏索引并美化表格
        st.dataframe(
            df,
            column_config={
                "图标": st.column_config.ImageColumn("图标")
            },
            hide_index=True,
            use_container_width=True
        )

    with col2:
        st.subheader("📈 资产详情 (前3名)")
        top_3 = df.head(3)
        for _, row in top_3.iterrows():
            with st.expander(f"No.{row['排名']} - {row['名称']} ({row['符号']})"):
                st.write(f"该资产当前在 CoinGecko 上的市值排名为第 **{row['市值排名']}** 位。")
                st.image(row['图标'], width=50)
else:
    st.warning("暂时无法加载数据，请检查网络或 API 限制。")