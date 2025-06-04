import pytest
from unittest.mock import Mock, patch
from src.agent.agent import Agent
from src.config import Config
from src.memory.memory import Memory
from src.memory.message import Message, Role
from src.llm.base_llm import BaseLLM

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def memory():
    return Memory()

@pytest.fixture
def mock_llm():
    mock = Mock(spec=BaseLLM)
    mock.chat = Mock()
    return mock

@pytest.fixture
def agent(config, memory, mock_llm):
    return Agent(config.agent_config, memory, mock_llm)

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """测试Agent初始化"""
    assert agent._memory is not None
    assert agent._llm is not None
    assert agent._config is not None

@pytest.mark.asyncio
async def test_agent_process_message(agent, mock_llm):
    """测试Agent处理消息"""
    test_message = "测试消息"
    mock_llm.chat.return_value = "AI回复"
    
    response = await agent.process_message(test_message)
    
    assert response == "AI回复"
    mock_llm.chat.assert_called_once()
    assert len(agent._memory.messages) == 2  # 用户消息和AI回复

@pytest.mark.asyncio
async def test_agent_memory_management(agent):
    """测试Agent记忆管理"""
    test_message = "测试消息"
    await agent.process_message(test_message)
    
    messages = agent._memory.messages
    assert len(messages) == 2
    assert messages[0].role == Role.USER
    assert messages[0].content == test_message
    assert messages[1].role == Role.ASSISTANT

@pytest.mark.asyncio
async def test_agent_error_handling(agent, mock_llm):
    """测试Agent错误处理"""
    mock_llm.chat.side_effect = Exception("LLM错误")
    
    with pytest.raises(Exception) as exc_info:
        await agent.process_message("测试消息")
    
    assert str(exc_info.value) == "LLM错误"

@pytest.mark.asyncio
async def test_agent_conversation_flow(agent, mock_llm):
    """测试Agent对话流程"""
    messages = [
        "你好",
        "第二条消息",
        "第三条消息"
    ]
    
    mock_llm.chat.side_effect = ["回复1", "回复2", "回复3"]
    
    for msg in messages:
        response = await agent.process_message(msg)
        assert response is not None
    
    assert len(agent._memory.messages) == 6  # 3个用户消息和3个AI回复
    assert mock_llm.chat.call_count == 3 