from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Union
import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

from src.config import config, logger
from src.agent import Agent # Scheduler 会用到 Agent 来执行具体任务
from src.llm import LLMFactory
import src.llm

class TaskContext:
    """用于传递给调度任务的上下文信息。"""
    def __init__(self, agent: Agent, task_description: str, task_params: Dict[str, Any]):
        self.agent = agent
        self.task_description = task_description
        self.task_params = task_params # 例如，对于重复任务，可以包含次数等

async def scheduled_agent_task_wrapper(context: TaskContext):
    """
    APScheduler 实际调用的包装函数，它会调用 Agent 的 run_task。
    """
    logger.info(f"调度任务开始: {context.task_description}, 参数: {context.task_params}")
    try:
        await context.agent.run_task(task_description=context.task_description)
        logger.info(f"调度任务 '{context.task_description}' 执行完成。")
    except Exception as e:
        logger.error(f"调度任务 '{context.task_description}' 执行失败: {e}", exc_info=True)
    finally:
        # 对于只执行一次的任务，或达到次数限制的任务，可以在这里考虑移除
        # 但APScheduler的DateTrigger和设置了end_date的IntervalTrigger/CronTrigger会自动处理
        pass

class SchedulerManager:
    """
    管理 APScheduler 实例和任务调度。
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        jobstores = {
            'default': SQLAlchemyJobStore(url=config.scheduler_config.db_url)
        }
        # executors = {
        #     'default': {'type': 'threadpool', 'max_workers': 20}, # 默认执行器
        #     'processpool': ProcessPoolExecutor(max_workers=5) # 如果有CPU密集型任务
        # }
        job_defaults = {
            'coalesce': False, # 是否合并同一任务的多次触发
            'max_instances': 3 # 同一任务的最大并发实例数
        }
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            # executors=executors, 
            job_defaults=job_defaults,
            timezone='Asia/Shanghai' # 或者其他时区，或者使用 UTC
        )
        logger.info(f"SchedulerManager 初始化完成，使用数据库: {config.scheduler_config.db_url}")

    def start(self):
        """启动调度器。"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("APScheduler 已启动。")
        else:
            logger.info("APScheduler 已在运行中。")

    def shutdown(self, wait: bool = True):
        """关闭调度器。"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("APScheduler 已关闭。")

    async def add_job(
        self, 
        task_description: str, 
        trigger_type: str, 
        trigger_args: Dict[str, Any],
        job_id: str | None = None,
        task_params: Dict[str, Any] | None = None,
        **job_kwargs: Any
    ) -> str | None:
        """
        添加一个新的调度任务，该任务将由 Agent 执行。

        Args:
            task_description: 要传递给 Agent 的任务描述。
            trigger_type: 触发器类型 ("date", "interval", "cron").
            trigger_args: 传递给触发器构造函数的参数。
                - for "date": {"run_date": "YYYY-MM-DD HH:MM:SS" or datetime_object}
                - for "interval": {"weeks": 0, "days": 0, "hours": 0, "minutes": 0, "seconds": 5, "start_date": ..., "end_date": ...}
                - for "cron": {"year": "*", "month": "*", "day": "*", "week": "*", "day_of_week": "*", "hour": "0", "minute": "5", "second": "0", "start_date": ..., "end_date": ...}
            job_id: 任务的唯一ID。如果为 None，APScheduler 会自动生成。
            task_params: 传递给任务执行上下文的额外参数 (e.g., {"max_runs": 5}).
            **job_kwargs: 其他传递给 scheduler.add_job 的参数 (e.g., name, misfire_grace_time).

        Returns:
            成功添加的任务的 ID，如果失败则为 None。
        """
        context = TaskContext(agent=self.agent, task_description=task_description, task_params=task_params or {})
        
        trigger: Union[DateTrigger, IntervalTrigger, CronTrigger | None] = None
        if trigger_type == "date":
            trigger = DateTrigger(**trigger_args)
        elif trigger_type == "interval":
            trigger = IntervalTrigger(**trigger_args)
        elif trigger_type == "cron":
            trigger = CronTrigger(**trigger_args)
        else:
            logger.error(f"不支持的触发器类型: {trigger_type}")
            return None

        try:
            job_name = job_kwargs.pop("name", f"agent_task_{job_id or task_description[:20]}")
            added_job = self.scheduler.add_job(
                scheduled_agent_task_wrapper, 
                trigger=trigger, 
                args=[context], 
                id=job_id,
                name=job_name,
                **job_kwargs
            )
            logger.info(f"任务 '{added_job.name}' (ID: {added_job.id}) 已成功添加到调度器，触发器: {trigger_type} {trigger_args}")
            return added_job.id
        except Exception as e:
            logger.error(f"添加任务 '{task_description}' 到调度器失败: {e}", exc_info=True)
            return None

    def remove_job(self, job_id: str):
        """移除指定的调度任务。"""
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"任务 ID '{job_id}' 已从调度器移除。")
        except Exception as e: # JobLookupError if job_id does not exist
            logger.error(f"移除任务 ID '{job_id}' 失败: {e}")

    def get_job(self, job_id: str):
        """获取指定ID的任务信息。"""
        return self.scheduler.get_job(job_id)

    def get_all_jobs(self):
        """获取所有已调度的任务信息。"""
        return self.scheduler.get_jobs()
    
    # "监听浏览器某种事件并执行的任务" 的实现会复杂得多，
    # 可能需要 BrowserManager 能够发出事件，或者 Agent 能够轮询特定状态。
    # 这超出了APScheduler的典型范围，可能需要自定义逻辑或与其他事件系统集成。
    # 初步版本将主要关注基于时间的调度。

async def main():
    # --- 配置 Agent (确保API Key等已在 .env 中设置) ---
    # 尝试创建LLM实例，如果API Key未配置则不继续测试
    llm_instance = None
    try:
        llm_instance = LLMFactory.create_llm()
    except ValueError as e:
        logger.error(f"创建LLM实例失败: {e}，请检查API Key配置。")
        print(f"创建LLM实例失败: {e}，请检查API Key配置。")
        return

    if (isinstance(llm_instance, src.llm.GeminiLLM) and not config.llm_config.gemini_api_key) or \
       (isinstance(llm_instance, src.llm.DeepSeekLLM) and not config.llm_config.deepseek_api_key):
        logger.error("所选LLM的API Key未配置，无法运行Scheduler测试。")
        print("所选LLM的API Key未配置，无法运行Scheduler测试。")
        return
    
    test_agent = Agent(llm=llm_instance)
    scheduler_manager = SchedulerManager(agent=test_agent)
    scheduler_manager.start()

    # --- 添加调度任务示例 ---
    try:
        # 1. DateTrigger: 在特定时间执行一次
        run_at = datetime.now() + timedelta(seconds=10)
        date_job_id = await scheduler_manager.add_job(
            task_description="访问example.com并获取标题 (定时一次)",
            trigger_type="date",
            trigger_args={"run_date": run_at},
            job_id="one_time_example_task"
        )
        if date_job_id:
            logger.info(f"添加了一次性任务，ID: {date_job_id}, 将在 {run_at} 执行。")

        # 2. IntervalTrigger: 每隔一段时间执行
        interval_job_id = await scheduler_manager.add_job(
            task_description="每15秒检查 playwright.dev 的主页标题",
            trigger_type="interval",
            trigger_args={"seconds": 15, "start_date": datetime.now() + timedelta(seconds=3)}, # 3秒后开始
            job_id="interval_playwright_title_check",
            misfire_grace_time=5 # 5秒内未执行则跳过
        )
        if interval_job_id:
            logger.info(f"添加了间隔任务，ID: {interval_job_id}, 每15秒执行。")

        # 3. CronTrigger: 按CRON表达式执行 (例如每天早上8点)
        # cron_job_id = await scheduler_manager.add_job(
        #     task_description="每天早上8点执行一次特定任务",
        #     trigger_type="cron",
        #     trigger_args={"hour": 8, "minute": 0},
        #     job_id="daily_morning_task"
        # )
        # if cron_job_id:
        #     logger.info(f"添加了CRON任务，ID: {cron_job_id}。")

        logger.info("调度器运行中...等待任务执行。按 Ctrl+C 退出。")
        logger.info(f"当前所有任务: {scheduler_manager.get_all_jobs()}")
        
        # 保持主线程运行一段时间以观察调度任务
        # 在实际应用中，调度器可能在后台长时间运行
        await asyncio.sleep(35) # 运行35秒，足够 interval_task 执行两次

    except KeyboardInterrupt:
        logger.info("接收到中断信号，准备关闭调度器...")
    except Exception as e:
        logger.error(f"SchedulerManager main 测试中发生错误: {e}", exc_info=True)
    finally:
        if interval_job_id: # 尝试移除，避免下次运行时冲突 (如果数据库持久化)
             scheduler_manager.remove_job(interval_job_id)
        scheduler_manager.shutdown()
        if test_agent and test_agent.browser_manager:
            await test_agent.browser_manager.close_browser()
        logger.info("Scheduler 测试结束。")

if __name__ == "__main__":
    # 需要 APScheduler, SQLAlchemy
    # pip install apscheduler sqlalchemy
    # 创建一个 .env 文件并填入你的 API KEY, e.g., GEMINI_API_KEY=xxx
    asyncio.run(main()) 