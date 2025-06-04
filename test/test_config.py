import os
import pytest
from src.config import Config, LLMConfig, BrowserConfig, AgentConfig, SchedulerConfig

@pytest.fixture
def config():
    """创建配置对象的fixture"""
    return Config()

def test_llm_config(config):
    """测试LLM配置"""
    assert isinstance(config.llm_config, LLMConfig)
    assert config.llm_config.provider in ["gemini", "deepseek"]
    assert config.llm_config.temperature == 0.1
    assert config.llm_config.max_tokens == 51200
    assert config.llm_config.deepseek_base_url == "https://api.deepseek.com"

def test_browser_config(config):
    """测试浏览器配置"""
    assert isinstance(config.browser_config, BrowserConfig)
    assert config.browser_config.timeout == 30000
    assert config.browser_config.navigation_timeout == 60000
    assert isinstance(config.browser_config.headless, bool)

def test_agent_config(config):
    """测试Agent配置"""
    assert isinstance(config.agent_config, AgentConfig)
    assert config.agent_config.max_iterations == 10
    assert config.agent_config.action_delay_ms == 1000

def test_scheduler_config(config):
    """测试调度器配置"""
    assert isinstance(config.scheduler_config, SchedulerConfig)
    assert config.scheduler_config.db_url == "sqlite:///tasks.sqlite"

def test_api_key_environment_variables():
    """测试API密钥环境变量设置"""
    test_gemini_key = "test_gemini_key"
    test_deepseek_key = "test_deepseek_key"
    
    os.environ["GEMINI_API_KEY"] = test_gemini_key
    os.environ["DEEPSEEK_API_KEY"] = test_deepseek_key
    
    config = Config()
    assert config.llm_config.gemini_api_key == test_gemini_key
    assert config.llm_config.deepseek_api_key == test_deepseek_key 