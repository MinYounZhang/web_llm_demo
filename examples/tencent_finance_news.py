import asyncio
from pathlib import Path
from src.agent import Agent



async def main():
    """腾讯新闻财经类新闻搜索任务"""
    # 设置数据保存目录
    data_dir = Path("data/tencent_news")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    task_description = """
任务：在腾讯新闻网站浏览财经类新闻并保存重要新闻信息

具体步骤：
1. 打开腾讯新闻网站
2. 找到并点击"财经"频道或栏目
3. 浏览今日财经新闻列表
4. 从新闻列表中筛选出较为重要的2篇新闻（优先选择头条、热点或重大财经事件相关的新闻）
5. 对于每篇选中的新闻，点击进入详情页面，收集以下信息：
   - 新闻标题
   - 发布时间
   - 新闻来源/作者
   - 新闻正文内容（前1000字）
   - 新闻摘要
   - 新闻分类标签
   - 阅读量（如果有）
   - 新闻链接
6. 将收集到的信息整理成结构化数据
7. 保存新闻信息到本地文件

注意事项：
- 重点关注财经类别的新闻，如股市、经济政策、企业动态、金融市场等
- 选择新闻时优先考虑重要性：头条新闻 > 热点新闻 > 一般新闻
- 确保收集的新闻是今日发布的
- 仔细提取新闻内容，确保数据的准确性和完整性
- 每处理完一篇新闻后要返回新闻列表页面继续处理下一篇
- 如果遇到需要登录或其他阻碍，使用ask_user寻求帮助
- 保存数据时使用如下格式的参数：{"news_data": {"标题": "...", "内容": "..."}, "news_index": 0}
"""
    
    # 初始化Agent
    agent = Agent()
    
    # 执行任务
    await agent.run_task(task_description)
    
    # 关闭浏览器
    await agent.browser_manager.close_browser()


if __name__ == "__main__":
    asyncio.run(main()) 