import logging
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

# 创建logs目录
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 基本日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(os.path.join(log_dir, "app.log"), encoding='utf-8')  # 输出到文件
    ]
)

@dataclass
class LLMConfig:
    """LLM相关配置"""
    provider: str
    gemini_api_key: str | None
    deepseek_api_key: str | None
    deepseek_base_url: str
    temperature: float
    max_tokens: int
    model_name: str

@dataclass
class BrowserConfig:
    """浏览器相关配置"""
    timeout: int
    user_agent: str | None
    navigation_timeout: int
    headless: bool
    # Cookie管理相关配置
    enable_cookie_management: bool  # 是否启用cookie管理
    cookie_save_path: str  # cookie保存路径
    auto_save_cookies: bool  # 是否自动保存cookies
    cookie_domains: list[str] | None  # 需要保存cookie的域名列表，None表示保存所有域名
    # 智能Cookie保存配置
    enable_smart_cookie_save: bool  # 是否启用智能Cookie保存（检测认证相关cookie）
    smart_cookie_save_threshold: int  # 智能保存的最小cookie数量变化阈值

@dataclass
class AgentConfig:
    """Agent相关配置"""
    max_iterations: int
    action_delay_ms: int
    # 错误管理相关配置
    max_action_retries: int  # Action执行最大重试次数
    action_retry_delay_ms: int  # Action重试间隔
    max_llm_retries: int  # LLM调用最大重试次数
    llm_retry_delay_ms: int  # LLM重试间隔
    action_timeout_multiplier: float  # Action超时时间倍数
    human_intervention_timeout_s: int  # 人工干预超时时间(秒)
    enable_fallback_analysis: bool  # 启用LLM分析失败原因

@dataclass
class SchedulerConfig:
    """调度器相关配置"""
    db_url: str

class Config:
    """项目配置类，按模块分组封装所有参数"""

    def __init__(self):
        provider = "deepseek" #"gemini"
        self._llm = LLMConfig(
            provider=provider,
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_base_url="https://api.deepseek.com/v1",
            temperature=0.1,
            max_tokens=8192,
            model_name="gemini-pro" if provider == "gemini" else "deepseek-chat"
        )

        self._browser = BrowserConfig(
            timeout=30000,
            user_agent=None,
            navigation_timeout=60000,
            headless=False,
            # Cookie管理相关配置
            enable_cookie_management=True,
            cookie_save_path="cookies/browser_state.json",
            auto_save_cookies=True,
            cookie_domains=None,  # None表示保存所有域名的cookies
            # 智能Cookie保存配置
            enable_smart_cookie_save=True,
            smart_cookie_save_threshold=10
        )

        self._agent = AgentConfig(
            max_iterations=10,
            action_delay_ms=1000,
            # 错误管理相关配置
            max_action_retries=3,
            action_retry_delay_ms=2000,
            max_llm_retries=3,
            llm_retry_delay_ms=1000,
            action_timeout_multiplier=1.5,
            human_intervention_timeout_s=60,
            enable_fallback_analysis=True
        )

        self._scheduler = SchedulerConfig(
            db_url="sqlite:///tasks.sqlite"
        )

        self._log_level = logging.INFO

    @property
    def llm_config(self) -> LLMConfig:
        return self._llm

    @property
    def browser_config(self) -> BrowserConfig:
        return self._browser

    @property
    def agent_config(self) -> AgentConfig:
        return self._agent

    @property
    def scheduler_config(self) -> SchedulerConfig:
        return self._scheduler

    @property
    def log_level(self) -> int:
        return self._log_level


config = Config()
logger = logging.getLogger(__name__)

# 实例化Config后，检查并提示API密钥配置情况
if config.llm_config.provider == "gemini" and not config.llm_config.gemini_api_key:
    logger.warning("LLM Provider 设置为 Gemini，但 GEMINI_API_KEY 未在环境变量中配置。")
if config.llm_config.provider == "deepseek" and not config.llm_config.deepseek_api_key:
    logger.warning("LLM Provider 设置为 DeepSeek，但 DEEPSEEK_API_KEY 未在环境变量中配置。")


if __name__ == "__main__":
    os.environ["GEMINI_API_KEY"] = "AIzaSyB0000000000000000000000000000000"
    os.environ["DEEPSEEK_API_KEY"] = "sk-00000000000000000000000000000000"

    logger.info("配置已加载并通过Config类实例化。")
    logger.info(f"当前日志级别: {logging.getLevelName(logger.getEffectiveLevel())}")
    logger.info(f"LLM Provider: {config.llm_config.provider}")
    logger.info(f"Gemini API Key available: {bool(config.llm_config.gemini_api_key)}")
    logger.info(f"DeepSeek API Key available: {bool(config.llm_config.deepseek_api_key)}")
    logger.info(f"DeepSeek Base URL: {config.llm_config.deepseek_base_url}")
    logger.info(f"LLM Temperature: {config.llm_config.temperature}")
    logger.info(f"LLM Max Tokens: {config.llm_config.max_tokens}")
    logger.info(f"Playwright Timeout: {config.browser_config.timeout}ms")
    logger.info(f"Headless Mode: {config.browser_config.headless}")
    logger.info(f"Cookie Management Enabled: {config.browser_config.enable_cookie_management}")
    logger.info(f"Cookie Save Path: {config.browser_config.cookie_save_path}")
    logger.info(f"Auto Save Cookies: {config.browser_config.auto_save_cookies}")
    logger.info(f"Cookie Domains Filter: {config.browser_config.cookie_domains}")
    logger.info(f"Agent Max Iterations: {config.agent_config.max_iterations}")
    logger.info(f"Scheduler DB URL: {config.scheduler_config.db_url}") 