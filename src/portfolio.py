import pandas as pd
from google import genai
from google.genai import types

API_KEY = "AIzaSyAHv7J2ukKTfMCrIXjFF-PE_fJdBBEzGZs"
client = genai.Client(api_key=API_KEY)

# 必须声明这个工具
search_tool = types.Tool(google_search=types.GoogleSearch())
time_str = pd.Timestamp.now().strftime("%Y年-%m月-%d日")
response = client.models.generate_content(
    model='gemini-2.0-flash',  # 推荐 2.0 Flash
    contents=f"""
    请在美股市场中筛选出当前（{time_str}）时间最近两周内满足以下量价与情绪特征的5只ETF：
    1) 成交量连续放大（连续 3 日或以上成交量环比上升）
    2) 价格近期创近期高点或呈稳步攀升趋势
    3) 社交/新闻情绪明显上升（如情绪数据或社媒热度/提及度明显提高）

    请优先涵盖以下主题：
    - 科技/AI 
    - 加密货币
    - 贵金属（黄金/白银）
    - 能源
    - 动量/情绪型
    
    输出格式：
    以JSON数组个水输如:[{{"code": "ETF code","name":"EFT中文名称", "performance": "涨跌幅"}}]
    """,
    config=types.GenerateContentConfig(
        tools=[search_tool]  # 这里是关键！
    )
)

# 获取带搜索溯源的内容
print(response.text)
