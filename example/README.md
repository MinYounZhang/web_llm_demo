# Agent 示例代码

## 概述

本目录包含了三个简洁的Agent使用示例，展示了如何使用统一的代码结构来执行不同类型的Web自动化任务。

## 示例文件

### 1. bilibili_like_task.py

B站视频搜索和点赞任务示例

**功能**：

- 打开B站网站
- 搜索"炉石传说"关键词
- 点击第3个搜索结果视频
- 执行点赞操作

**运行方式**：

```bash
python example/bilibili_like_task.py
```

### 2. boss_deep_learning_search.py

Boss直聘深度学习岗位搜索示例

**功能**：

- 打开Boss直聘网站
- 搜索"深度学习算法"岗位
- 收集前5个岗位的详细信息
- 保存岗位数据到本地JSON文件

**运行方式**：

```bash
python example/boss_deep_learning_search.py
```

### 3. tencent_finance_news.py

腾讯新闻财经类新闻搜索示例

**功能**：

- 打开腾讯新闻网站
- 浏览财经频道
- 筛选2篇重要的今日财经新闻
- 保存新闻内容到本地JSON文件

**运行方式**：

```bash
python example/tencent_finance_news.py
```

## 代码结构

所有示例都采用统一的简洁结构：

```python
import asyncio
from src.agent import Agent

async def main():
    """任务描述"""
    task_description = """
    详细的任务描述...
    """
    
    # 初始化Agent
    agent = Agent()
    
    # 可选：添加自定义Action
    # agent.browser_manager.add_action(custom_action)
    
    # 执行任务
    await agent.run_task(task_description)
    
    # 关闭浏览器
    await agent.browser_manager.close_browser()

if __name__ == "__main__":
    asyncio.run(main())
```

## Agent Memory 改进

根据最新的修改，Agent现在只存储与LLM交互的关键信息到Memory中：

- **LLM输入**：发送给LLM的prompt存储在Memory中
- **LLM输出**：
  - `content`: LLM返回的原始文本
  - `meta_content`: 解析后的结构化数据（包括分析结果和动作序列）

其他日志信息只通过logger记录，不存储到Memory中，保持Memory的简洁性。

## 运行前准备

1. 确保已配置LLM API密钥（在.env文件中配置GEMINI_API_KEY或DEEPSEEK_API_KEY）
2. 确保系统已安装所需依赖
3. 根据需要修改任务描述中的具体参数

## 注意事项

- 所有示例都不包含复杂的错误处理逻辑，保持代码简洁
- 如果任务执行过程中遇到问题，Agent会通过ask_user动作寻求帮助
- 自定义Action（如保存数据）可以很容易地添加到任何示例中
