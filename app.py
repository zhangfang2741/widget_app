import os
# 禁用当前进程的代理设置，直接连接互联网
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['no_proxy'] = '*'

import streamlit as st
from dotenv import load_dotenv as load_env
load_env()
# 页面配置放在最前面
# layout="wide": 在手机上尽量占满屏幕宽度，减少白边
# initial_sidebar_state="auto": 在手机端自动折叠菜单，PC端默认展开，适配移动设备操作逻辑
st.set_page_config(
    page_title="综合金融看板",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- 主应用入口 (使用 st.navigation 实现菜单式导航) ---
def main():
    # 定义页面列表，指向独立的文件路径
    # 确保 trending.py 和 portfolio.py 与 app.py 在同一目录下
    pages = [
        st.Page("src/Market_Intensity_Heatmap.py", title="美国行业资金流热力图", icon="💼",default=True),
        # st.Page("src/dashboard.py", title="首页概览", icon="🏠"),
        # st.Page("src/trending.py", title="热门资产", icon="🔥"),
        # st.Page("src/portfolio.py", title="投资组合 (示例)", icon="💼"),
        # st.Page("src/cot_index.py", title="COT 庄家筹码雷达", icon="💼"),
    ]

    # 创建导航栏
    # 在手机端，这会自动渲染为左上角的折叠菜单
    pg = st.navigation(pages)

    # 运行选中的页面
    pg.run()


if __name__ == "__main__":
    main()
