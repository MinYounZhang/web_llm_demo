import pytest
import os
from src.config import Config

@pytest.fixture(scope="session")
def config():
    """全局配置fixture"""
    # 设置测试环境变量
    os.environ["GEMINI_API_KEY"] = "test_gemini_key"
    os.environ["DEEPSEEK_API_KEY"] = "test_deepseek_key"
    return Config()

@pytest.fixture(autouse=True)
def setup_test_env():
    """自动设置测试环境"""
    # 在每个测试前设置
    os.environ["PYTEST_RUNNING"] = "1"
    
    yield
    
    # 在每个测试后清理
    if "PYTEST_RUNNING" in os.environ:
        del os.environ["PYTEST_RUNNING"] 