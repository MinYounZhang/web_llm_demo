import os
from typing import Any, AsyncGenerator, Dict, List, Union, AsyncIterator
from src.llm.base_llm import BaseLLM
from src.memory.message import Message
from src.config import config, logger

# 使用 LangChain 的 ChatDeepSeek 来实现 Langsmith 追踪
try:
    from langchain_deepseek import ChatDeepSeek
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_DEEPSEEK_AVAILABLE = True
except ImportError:
    logger.warning("langchain-deepseek 未安装。请运行: pip install langchain-deepseek")
    LANGCHAIN_DEEPSEEK_AVAILABLE = False
    # 备用方案：继续使用 openai 客户端
    import openai

class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM 模型的实现，使用 LangChain ChatDeepSeek 以支持 Langsmith 追踪。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model_name: str = "deepseek-chat"):
        """
        初始化 DeepSeekLLM。

        Args:
            api_key: DeepSeek API 密钥。如果为 None，则从配置中读取。
            base_url: DeepSeek API 的基础 URL。如果为 None，则从配置中读取。
            model_name: 要使用的 DeepSeek 模型名称。
        """
        self.api_key = api_key or config.llm_config.deepseek_api_key
        self.base_url = base_url or config.llm_config.deepseek_base_url
        if not self.api_key:
            logger.error("DeepSeek API Key未配置。")
            raise ValueError("DeepSeek API Key未配置。")
        
        self.model_name = model_name
        
        # 设置 Langsmith 追踪环境变量
        if not os.getenv("LANGSMITH_TRACING"):
            os.environ["LANGSMITH_TRACING"] = "true"
            logger.info("已启用 Langsmith 追踪")
        
        try:
            if LANGCHAIN_DEEPSEEK_AVAILABLE:
                # 使用 LangChain 的 ChatDeepSeek（推荐方式）
                self.client = ChatDeepSeek(
                    model=self.model_name,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=config.llm_config.temperature,
                    max_tokens=config.llm_config.max_tokens,
                )
                self.use_langchain = True
                logger.info(f"DeepSeekLLM 初始化成功（使用 LangChain ChatDeepSeek），模型：{self.model_name}, Base URL: {self.base_url}")
            else:
                # 备用方案：使用 OpenAI 客户端
                self.client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                self.use_langchain = False
                logger.warning(f"DeepSeekLLM 初始化成功（使用 OpenAI 客户端备用方案），模型：{self.model_name}, Base URL: {self.base_url}")
                logger.warning("建议安装 langchain-deepseek 以获得更好的 Langsmith 集成：pip install langchain-deepseek")
        except Exception as e:
            logger.error(f"初始化 DeepSeek 客户端失败: {e}")
            raise

    async def chat_completion(self, messages: List[Message], stream: bool = False, **kwargs: Any) -> Union[str, AsyncGenerator[str, None]]:
        """
        使用 DeepSeek 生成聊天回复。

        Args:
            messages: 对话消息列表。
            stream: 是否启用流式输出。
            **kwargs: 其他特定于模型的参数，如 temperature, max_tokens。

        Returns:
            如果 stream 为 False，则返回字符串形式的回复。
            如果 stream 为 True，则返回一个异步生成器，逐块生成回复。
        """
        if self.use_langchain:
            return await self._chat_completion_langchain(messages, stream, **kwargs)
        else:
            return await self._chat_completion_openai(messages, stream, **kwargs)

    async def _chat_completion_langchain(self, messages: List[Message], stream: bool = False, **kwargs: Any) -> Union[str, AsyncGenerator[str, None]]:
        """使用 LangChain ChatDeepSeek 进行聊天补全"""
        langchain_messages = self._convert_messages_to_langchain_format(messages)
        
        # 合并参数
        request_params = {
            "temperature": kwargs.get("temperature", config.llm_config.temperature),
            "max_tokens": kwargs.get("max_tokens", config.llm_config.max_tokens),
        }
        # 过滤掉值为 None 的参数
        final_params = {k: v for k, v in request_params.items() if v is not None}

        logger.debug(f"向 DeepSeek (LangChain) 发送请求: {final_params}")

        try:
            if stream:
                async def stream_generator() -> AsyncGenerator[str, None]:
                    async for chunk in self.client.astream(langchain_messages, **final_params):
                        if chunk.content:
                            yield chunk.content
                return stream_generator()
            else:
                response = await self.client.ainvoke(langchain_messages, **final_params)
                if response and response.content:
                    return response.content
                logger.warning("DeepSeek (LangChain) 响应中没有找到有效的 content。")
                return ""
        except Exception as e:
            logger.error(f"DeepSeek (LangChain) chat completion 失败: {e}")
            raise

    async def _chat_completion_openai(self, messages: List[Message], stream: bool = False, **kwargs: Any) -> Union[str, AsyncGenerator[str, None]]:
        """使用 OpenAI 客户端进行聊天补全（备用方案）"""
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        request_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", config.llm_config.temperature),
            "max_tokens": kwargs.get("max_tokens", config.llm_config.max_tokens),
            "stream": stream,
        }
        # 过滤掉值为 None 的参数 (除了 stream，因为它必须存在)
        final_params = {k: v for k, v in request_params.items() if v is not None or k == 'stream'}

        logger.debug(f"向 DeepSeek (OpenAI 备用) 发送请求: {final_params}")

        try:
            response = await self.client.chat.completions.create(**final_params)

            if stream:
                async def stream_generator(async_iter: AsyncIterator[openai.types.chat.ChatCompletionChunk]) -> AsyncGenerator[str, None]:
                    async for chunk in async_iter:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator(response)
            else:
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    return response.choices[0].message.content
                logger.warning("DeepSeek (OpenAI 备用) 响应中没有找到有效的 content。")
                return ""
        except Exception as e:
            logger.error(f"DeepSeek (OpenAI 备用) chat completion 失败: {e}")
            raise

    def _convert_messages_to_langchain_format(self, messages: List[Message]) -> List:
        """将 Message 对象列表转换为 LangChain 消息格式。"""
        langchain_messages = []
        for msg in messages:
            if msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            else:
                # 默认处理为 HumanMessage
                langchain_messages.append(HumanMessage(content=msg.content))
        return langchain_messages

    def _convert_messages_to_openai_format(self, messages: List[Message]) -> List[Dict[str, str]]:
        """将 Message 对象列表转换为 OpenAI API 所需的格式。"""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def get_config(self) -> Dict[str, Any]:
        """获取当前 DeepSeekLLM 的配置信息。"""
        return {
            "provider": "deepseek",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
            "default_temperature": config.llm_config.temperature,
            "default_max_tokens": config.llm_config.max_tokens,
            "use_langchain": self.use_langchain,
            "langsmith_tracing_enabled": os.getenv("LANGSMITH_TRACING") == "true",
        }

if __name__ == '__main__':
    import asyncio

    async def main():
        # 请确保 .env 文件中配置了 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL
        if not config.llm_config.deepseek_api_key:
            print("请在 .env 文件中配置 DEEPSEEK_API_KEY")
            return

        deepseek_llm = None
        try:
            deepseek_llm = DeepSeekLLM()
        except ValueError as e:
            print(e)
            return
        except Exception as e:
            print(f"初始化DeepSeekLLM时发生错误: {e}")
            return

        # 显示配置信息
        config_info = deepseek_llm.get_config()
        print(f"配置信息: {config_info}")

        test_messages = [
            Message(role="user", content="你好，请用中文介绍一下你自己。"),
        ]

        print("\n--- 非流式调用 (DeepSeek) ---")
        try:
            response = await deepseek_llm.chat_completion(test_messages, temperature=0.7, max_tokens=150)
            print(f"DeepSeek: {response}")
        except Exception as e:
            print(f"DeepSeek 调用出错: {e}")

        print("\n--- 流式调用 (DeepSeek) ---")
        try:
            async_generator = await deepseek_llm.chat_completion(test_messages, stream=True, temperature=0.7, max_tokens=150)
            print("DeepSeek (stream): ", end="")
            async for chunk in async_generator:
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"DeepSeek 流式调用出错: {e}")
        
        print("\n--- System Message Test (非流式, DeepSeek) ---")
        test_messages_with_system = [
            Message(role="system", content="你是一个乐于助人的AI助手，请用中文回答，并且只说'好的'。"),
            Message(role="user", content="太阳为什么从东边升起？请详细解释。"),
        ]
        try:
            response_system = await deepseek_llm.chat_completion(test_messages_with_system, temperature=0.1, max_tokens=50)
            print(f"DeepSeek: {response_system}")
        except Exception as e:
            print(f"DeepSeek 调用出错 (system message): {e}")

    if config.llm_config.deepseek_api_key: # 仅在配置了API Key时运行测试
        asyncio.run(main())
    else:
        logger.warning("未配置 DEEPSEEK_API_KEY，跳过 DeepSeekLLM 的 main 测试。")
        print("未配置 DEEPSEEK_API_KEY，跳过 DeepSeekLLM 的 main 测试。请在 .env 文件中设置。") 