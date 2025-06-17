import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.agent import Agent
from src.browser.actions import Action
from patchright.async_api import Page


class SaveJobDataAction(Action):
    """保存岗位数据到本地文件的自定义Action"""
    
    def __init__(self, data_dir: Path):
        super().__init__(
            name="save_job_data",
            description="保存岗位数据到本地文件。参数：job_data(dict): 岗位信息字典, job_index(int): 岗位序号"
        )
        self.data_dir = data_dir
    
    async def execute(self, page: Page, **kwargs: Any) -> Dict[str, Any]:
        """保存岗位数据的实现"""
        job_data = kwargs.get('job_data', {})
        job_index = kwargs.get('job_index', 0)
        
        if not job_data:
            return {"success": False, "message": "没有岗位数据需要保存"}
        
        # 添加抓取时间
        job_data["抓取时间"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deep_learning_job_{job_index+1}_{timestamp}.json"
        filepath = self.data_dir / filename
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(job_data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True, 
            "message": f"岗位数据已保存到 {filepath}",
            "filepath": str(filepath)
        }


async def main():
    """深度学习算法岗位搜索任务"""
    # 设置数据保存目录
    data_dir = Path("data/boss_jobs")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    task_description = """
任务：在Boss直聘网站搜索深度学习算法岗位并保存详细信息

具体步骤：
1. 打开Boss直聘网站 (https://www.zhipin.com)
2. 在搜索框中输入关键词："深度学习算法"
3. 执行搜索操作
4. 从搜索结果中选择前5个岗位
5. 对于每个岗位，点击进入详情页面，收集以下信息：
   - 岗位名称
   - 公司名称
   - 薪资范围
   - 工作地点
   - 工作经验要求
   - 学历要求
   - 岗位描述/职责要求
   - 公司规模
   - 公司行业
   - 发布时间
6. 将收集到的信息整理成结构化数据
7. 调用save_job_data动作保存岗位信息到本地文件

注意事项：
- 如果遇到需要登录的情况，可以尝试游客模式浏览或使用ask_user询问如何处理
- 仔细识别页面元素，确保点击正确的岗位链接
- 在收集信息时要仔细提取，确保数据的准确性和完整性
- 每处理完一个岗位后要返回搜索结果页面继续处理下一个
- 如果遇到反爬虫验证或其他阻碍，使用ask_user寻求帮助
- 保存数据时使用如下格式的参数：{"job_data": {"岗位名称": "...", "公司名称": "..."}, "job_index": 0}
"""
    
    # 初始化Agent
    agent = Agent()
    
    # 添加自定义保存动作
    save_action = SaveJobDataAction(data_dir)
    agent.browser_manager.add_action(save_action)
    
    # 执行任务
    await agent.run_task(task_description)
    
    # 关闭浏览器
    await agent.browser_manager.close_browser()


if __name__ == "__main__":
    asyncio.run(main()) 