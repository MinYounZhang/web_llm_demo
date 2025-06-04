import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.llm.base_llm import BaseLLM
from src.llm.gemini_llm import GeminiLLM
from src.llm.deepseek_llm import DeepSeekLLM
from src.llm.llm_factory import LLMFactory
from src.config import Config, LLMConfig
from src.memory.message import Message

@pytest.fixture
def mock_config():
    """创建一个带有测试配置的 Config 实例"""
    with patch.dict('os.environ', {
        'GEMINI_API_KEY': 'test_gemini_key',
        'DEEPSEEK_API_KEY': 'test_deepseek_key'
    }):
        config = Config()
        # 确保配置有合适的测试值
        config._llm = LLMConfig(
            provider="gemini",
            gemini_api_key="test_gemini_key",
            deepseek_api_key="test_deepseek_key",
            deepseek_base_url="https://api.deepseek.com",
            temperature=0.1,
            max_tokens=51200,
            model_name="gemini-pro"
        )
        return config

def test_llm_factory(mock_config):
    """测试LLM工厂类"""
    # 测试Gemini LLM创建
    llm = LLMFactory.create_llm(
        llm_provider="gemini",
        api_key=mock_config.llm_config.gemini_api_key,
        model_name="gemini-pro"
    )
    assert isinstance(llm, GeminiLLM)
    
    # 测试Deepseek LLM创建
    llm = LLMFactory.create_llm(
        llm_provider="deepseek",
        api_key=mock_config.llm_config.deepseek_api_key,
        base_url=mock_config.llm_config.deepseek_base_url,
        model_name="deepseek-chat"
    )
    assert isinstance(llm, DeepSeekLLM)
    
    # 测试无效provider
    with pytest.raises(ValueError, match="不支持的 LLM 提供商"):
        LLMFactory.create_llm(llm_provider="invalid")

    # 测试缺少API key
    with pytest.raises(ValueError, match="API Key未配置"):
        LLMFactory.create_llm(llm_provider="gemini")
    
    # 测试Deepseek缺少base_url
    with pytest.raises(ValueError, match="DeepSeek base URL未配置"):
        LLMFactory.create_llm(
            llm_provider="deepseek",
            api_key=mock_config.llm_config.deepseek_api_key
        )

@pytest.mark.asyncio
async def test_gemini_llm(mock_config):
    """测试Gemini LLM"""
    with patch('google.generativeai.GenerativeModel') as mock_model_class:
        mock_model = AsyncMock()
        mock_response = Mock()
        mock_response.parts = [Mock(text="测试回复")]
        mock_model.generate_content_async.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        llm = LLMFactory.create_llm(
            llm_provider="gemini",
            api_key=mock_config.llm_config.gemini_api_key,
            model_name="gemini-pro"
        )
        
        messages = [Message(role="user", content="测试问题")]
        response = await llm.chat_completion(messages)
        
        assert response == "测试回复"
        mock_model.generate_content_async.assert_called_once()

@pytest.mark.asyncio
async def test_deepseek_llm(mock_config):
    """测试Deepseek LLM"""
    with patch('openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="测试回复"))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        llm = LLMFactory.create_llm(
            llm_provider="deepseek",
            api_key=mock_config.llm_config.deepseek_api_key,
            base_url=mock_config.llm_config.deepseek_base_url,
            model_name="deepseek-chat"
        )
        
        messages = [Message(role="user", content="测试问题")]
        response = await llm.chat_completion(messages)
        
        assert response == "测试回复"
        mock_client.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_llm_streaming(mock_config):
    """测试LLM流式输出"""
    # 测试Gemini流式输出
    with patch('google.generativeai.GenerativeModel') as mock_model_class:
        mock_model = AsyncMock()
        async def mock_stream():
            chunks = [Mock(parts=[Mock(text="测试"), Mock(text="回复")])]
            for chunk in chunks:
                yield chunk
        mock_model.generate_content_async.return_value = mock_stream()
        mock_model_class.return_value = mock_model
        
        llm = LLMFactory.create_llm(
            llm_provider="gemini",
            api_key=mock_config.llm_config.gemini_api_key,
            model_name="gemini-pro"
        )
        messages = [Message(role="user", content="测试问题")]
        response_stream = await llm.chat_completion(messages, stream=True)
        
        chunks = []
        async for chunk in response_stream:
            for part in chunk.parts:
                chunks.append(part.text)

        assert "".join(chunks) == "测试回复"
    
    # 测试Deepseek流式输出
    with patch('openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        async def mock_stream():
            chunks = [
                Mock(choices=[Mock(delta=Mock(content="测试"))]),
                Mock(choices=[Mock(delta=Mock(content="回复"))])
            ]
            for chunk in chunks:
                yield chunk
        mock_client.chat.completions.create.return_value = mock_stream()
        mock_client_class.return_value = mock_client
        
        llm = LLMFactory.create_llm(
            llm_provider="deepseek",
            api_key=mock_config.llm_config.deepseek_api_key,
            base_url=mock_config.llm_config.deepseek_base_url,
            model_name="deepseek-chat"
        )
        messages = [Message(role="user", content="测试问题")]
        response_stream = await llm.chat_completion(messages, stream=True)
        
        chunks = []
        async for chunk in response_stream:
            for part in chunk.parts:
                chunks.append(part.text)

        assert "".join(chunks) == "测试回复"

@pytest.mark.asyncio
async def test_llm_with_system_message(mock_config):
    """测试带有系统消息的LLM调用"""
    # 测试Gemini处理系统消息
    with patch('google.generativeai.GenerativeModel') as mock_model_class:
        mock_model = AsyncMock()
        mock_response = Mock()
        mock_response.parts = [Mock(text="遵循系统指令的回复")]
        mock_model.generate_content_async.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        llm = LLMFactory.create_llm(
            llm_provider="gemini",
            api_key=mock_config.llm_config.gemini_api_key,
            model_name="gemini-pro"
        )
        messages = [
            Message(role="system", content="你是一个有帮助的助手"),
            Message(role="user", content="测试问题")
        ]
        response = await llm.chat_completion(messages)
        
        assert response == "遵循系统指令的回复"
        mock_model.generate_content_async.assert_called_once()
    
    # 测试Deepseek处理系统消息
    with patch('openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="遵循系统指令的回复"))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        llm = LLMFactory.create_llm(
            llm_provider="deepseek",
            api_key=mock_config.llm_config.deepseek_api_key,
            base_url=mock_config.llm_config.deepseek_base_url,
            model_name="deepseek-chat"
        )
        messages = [
            Message(role="system", content="你是一个有帮助的助手"),
            Message(role="user", content="测试问题")
        ]
        response = await llm.chat_completion(messages)
        
        assert response == "遵循系统指令的回复"
        mock_client.chat.completions.create.assert_called_once() 