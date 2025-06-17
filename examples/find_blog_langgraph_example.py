#!/usr/bin/env python3
"""
LangGraph Agent 示例：查找 Lilian Weng 的博客并总结最新播客内容

这个示例演示了如何使用 LangGraphAgent 来：
1. 搜索并找到 Lilian Weng 的博客
2. 浏览博客内容
3. 找到并分析她的最新播客/博文
4. 总结播客的主要内容

"""

import asyncio
from src.agent.agent_langgraph import LangGraphAgent
import sys
import traceback

def global_exception_handler(exc_type, exc_value, exc_traceback):
    filename = exc_traceback.tb_frame.f_code.co_filename
    lineno = exc_traceback.tb_lineno
    print(f"🌐 [全局异常] {filename}:{lineno} - {exc_type.__name__}: {exc_value}")
    traceback.print_tb(exc_traceback)  # 打印完整堆栈

sys.excepthook = global_exception_handler  # 注入全局钩子

async def main():
    """Lilian Weng 博客分析任务"""
    task_description = """
任务：找到 Lilian Weng 的博客,并总结她最新的博文内容

"""
    
    # 初始化 LangGraph Agent
    agent = LangGraphAgent()
    
    # 执行任务
    await agent.run_task(task_description)
    
    # 关闭浏览器
    await agent.browser_manager.close_browser()

if __name__ == "__main__":
    asyncio.run(main()) 


