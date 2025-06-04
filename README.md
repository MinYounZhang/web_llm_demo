# 项目名称

一个基于Python的自动化工具，用于执行网站私信回复和周期性任务（如点赞）。项目采用LLM（大语言模型）和浏览器自动化技术实现核心功能。

## 技术栈

- Python 3.8+
- Playwright (Patchright)
- APScheduler
- DeepSeek
- Gemini

## 快速开始

1. 克隆仓库
2. 安装依赖: `pip install -r requirements.txt`
3. 配置环境变量 (参考 `src/config.py`)
4. 运行示例 (待补充)

## 项目结构

```
project/
├── src/
│   ├── agent/                # agent类，根据任务、当前浏览器状态，通过LLM操作浏览器工作
│   ├── browser/              # 浏览器自动化相关代码
│   ├── llm/                  # LLM模型集成
│   ├── schedule/             # 任务调度器 
│   ├── memory/               # LLM模型的对话历史和上下文信息
│   └── config.py            # 配置文件
├── tests/                   # 测试代码
├── requirements.txt
└── README.md
```
