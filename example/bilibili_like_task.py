import asyncio
from src.agent import Agent

async def main():
    """B站视频搜索和点赞任务"""
    task_description = """
任务：在哔哩哔哩（B站）搜索视频并进行点赞操作

具体步骤：
1. 打开B站网站 (www.bilibili.com)
2. 在搜索框中输入关键词："炉石传说"
3. 执行搜索操作
4. 在搜索结果中定位并点击第3个视频
5. 在视频页面中：
   - 等待视频加载完成
   - 找到点赞按钮
   - 点击点赞按钮
6. 确认点赞操作是否成功

注意事项：
- 优先使用element_id参数来操作页面元素，这样更准确可靠
- 如果遇到需要登录的情况，使用ask_user询问如何处理
- 确保正确识别视频序号，避免点击错误的视频
- 在点赞前确保页面已完全加载
- 如果遇到任何异常情况（如验证码、弹窗等），使用ask_user寻求帮助
"""
    
    # 初始化Agent
    agent = Agent()
    
    # 执行任务
    await agent.run_task(task_description)
    
    # 关闭浏览器
    await agent.browser_manager.close_browser()

if __name__ == "__main__":
    asyncio.run(main()) 