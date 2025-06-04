import openai
from typing import Any, AsyncGenerator, Dict, List, Union, AsyncIterator
from src.llm.base_llm import BaseLLM
from src.memory.message import Message
from src.config import config, logger

class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM 模型的实现，使用 OpenAI SDK。"""

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
        try:
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"DeepSeekLLM 初始化成功，使用模型：{self.model_name}, Base URL: {self.base_url}")
        except Exception as e:
            logger.error(f"初始化 DeepSeek (AsyncOpenAI) 客户端失败: {e}")
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

        logger.debug(f"向 DeepSeek 发送请求: {final_params}")

        try:
            response = await self.client.chat.completions.create(**final_params)

            if stream:
                async def stream_generator(async_iter: AsyncIterator[openai.types.chat.ChatCompletionChunk]) -> AsyncGenerator[str, None]:
                    async for chunk in async_iter:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                        # 可以在这里添加对 finish_reason 的处理
                        # if chunk.choices and chunk.choices[0].finish_reason == "stop":
                        #     break
                return stream_generator(response)
            else:
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    return response.choices[0].message.content
                logger.warning("DeepSeek 响应中没有找到有效的 content。")
                return "" # 或抛出异常
        except openai.APIConnectionError as e:
            logger.error(f"DeepSeek API 连接错误: {e}")
            raise
        except openai.RateLimitError as e:
            logger.error(f"DeepSeek API 请求频率超出限制: {e}")
            raise
        except openai.APIStatusError as e:
            logger.error(f"DeepSeek API 状态错误: {e.status_code} - {e.response}")
            raise
        except Exception as e:
            logger.error(f"DeepSeek chat completion 失败: {e}")
            raise

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
        }

if __name__ == '__main__':
    import asyncio
    import os

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