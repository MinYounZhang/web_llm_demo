import google.generativeai as genai
import os
from typing import Any, AsyncGenerator, Dict, List, Union
from src.llm.base_llm import BaseLLM
from src.memory.message import Message
from src.config import config, logger

class GeminiLLM(BaseLLM):
    """Gemini LLM 模型的实现。"""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-pro"):
        """
        初始化 GeminiLLM。

        Args:
            api_key: Gemini API 密钥。如果为 None，则从配置中读取。
            model_name: 要使用的 Gemini 模型名称。
        """
        self.api_key = api_key or config.llm_config.gemini_api_key
        if not self.api_key:
            logger.error("Gemini API Key未配置。")
            raise ValueError("Gemini API Key未配置。")
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        try:
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"GeminiLLM 初始化成功，使用模型：{self.model_name}")
        except Exception as e:
            logger.error(f"初始化 Gemini GenerativeModel 失败: {e}")
            raise

    async def chat_completion(self, messages: List[Message], stream: bool = False, **kwargs: Any) -> Union[str, AsyncGenerator[str, None]]:
        """
        使用 Gemini 生成聊天回复。

        Args:
            messages: 对话消息列表。
            stream: 是否启用流式输出。
            **kwargs: 其他特定于模型的参数，如 temperature, max_output_tokens。

        Returns:
            如果 stream 为 False，则返回字符串形式的回复。
            如果 stream 为 True，则返回一个异步生成器，逐块生成回复。
        """
        gemini_messages = self._convert_messages_to_gemini_format(messages)
        
        generation_config_params = {
            "temperature": kwargs.get("temperature", config.llm_config.temperature),
            "max_output_tokens": kwargs.get("max_tokens", config.llm_config.max_tokens),
        }
        # 过滤掉值为 None 的参数
        generation_config = genai.types.GenerationConfig(**{k: v for k, v in generation_config_params.items() if v is not None})

        logger.debug(f"向 Gemini 发送请求: messages={gemini_messages}, stream={stream}, config={generation_config}")

        try:
            if stream:
                async def stream_generator():
                    response = await self.model.generate_content_async(
                        gemini_messages,
                        stream=True,
                        generation_config=generation_config
                    )
                    async for chunk in response:
                        if chunk.parts:
                            yield chunk.parts[0].text
                        # 可以在这里添加对 finish_reason 的处理，例如 if chunk.candidates[0].finish_reason...
                return stream_generator()
            else:
                response = await self.model.generate_content_async(
                    gemini_messages,
                    stream=False,
                    generation_config=generation_config
                )
                # logger.debug(f"Gemini 原始响应: {response}")
                if response.parts:
                    return response.parts[0].text
                elif response.candidates and response.candidates[0].content.parts: # 有时内容在 candidate 中
                     return response.candidates[0].content.parts[0].text
                logger.warning("Gemini 响应中没有找到有效的 parts。")
                return ""  # 或抛出异常
        except Exception as e:
            logger.error(f"Gemini chat completion 失败: {e}")
            # 可以根据具体错误类型进行更细致的处理
            # 例如：genai.types.generation_types.StopCandidateException
            # 或者 google.api_core.exceptions.PermissionDenied
            raise

    def _convert_messages_to_gemini_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """将 Message 对象列表转换为 Gemini API 所需的格式。"""
        gemini_messages = []
        for msg in messages:
            # Gemini 的角色是 'user' 或 'model'
            role = "user" if msg.role == "user" else "model"
            # 对于 system 消息，Gemini 通常将其作为聊天的初始指令或上下文，
            # 可以考虑将其内容合并到第一条 user/model 消息中，或者作为单独的 'user' role text part.
            # 这里简单处理，将 system 消息也转为 model 角色（或 user，取决于具体策略）
            # 或者在 generate_content 的 system_instruction 参数中提供 (如果模型支持)
            if msg.role == "system":
                # Gemini 没有显式的 system 角色，通常做法是将其内容放在第一个 'user' 消息之前，
                # 或者作为 `GenerativeModel.start_chat` 的 `history` 的一部分，或者用作 `system_instruction`
                # 这里我们暂时将其转换为一个 'user' role 的消息，或者可以由调用者处理
                # 此处为了简单，我们先将其作为 'user' message.
                # 也可以考虑让LLM的system prompt在Agent中处理，不传入这里
                gemini_messages.append({"role": "user", "parts": [{"text": f"[System Prompt] {msg.content}"}]})
            else:
                 gemini_messages.append({"role": role, "parts": [{"text": msg.content}]})
        return gemini_messages

    def get_config(self) -> Dict[str, Any]:
        """获取当前 GeminiLLM 的配置信息。"""
        return {
            "provider": "gemini",
            "model_name": self.model_name,
            "api_key_configured": bool(self.api_key),
            "default_temperature": config.llm_config.temperature,
            "default_max_tokens": config.llm_config.max_tokens,
        }

if __name__ == '__main__':
    import asyncio

    async def main():
        # 请确保 .env 文件中配置了 GEMINI_API_KEY
        if not config.llm_config.gemini_api_key:
            print("请在 .env 文件中配置 GEMINI_API_KEY")
            return

        gemini_llm = None
        try:
            gemini_llm = GeminiLLM()
        except ValueError as e:
            print(e)
            return

        test_messages = [
            Message(role="user", content="你好，介绍一下你自己。"),
        ]
        
        print("\n--- 非流式调用 ---")
        try:
            response = await gemini_llm.chat_completion(test_messages, temperature=0.7, max_tokens=100)
            print(f"Gemini: {response}")
        except Exception as e:
            print(f"Gemini 调用出错: {e}")

        print("\n--- 流式调用 ---")
        try:
            async_generator = await gemini_llm.chat_completion(test_messages, stream=True, temperature=0.7, max_tokens=100)
            print("Gemini (stream): ", end="")
            async for chunk in async_generator:
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"Gemini 流式调用出错: {e}")

        print("\n--- System Message Test (非流式) ---")
        test_messages_with_system = [
            Message(role="system", content="你是一个乐于助人的AI助手，请用中文回答。"),
            Message(role="user", content="太阳为什么从东边升起？"),
        ]
        try:
            response_system = await gemini_llm.chat_completion(test_messages_with_system, temperature=0.7, max_tokens=150)
            print(f"Gemini: {response_system}")
        except Exception as e:
            print(f"Gemini 调用出错 (system message): {e}")


    if os.getenv("GEMINI_API_KEY"): # 仅在配置了API Key时运行测试
        asyncio.run(main())
    else:
        logger.warning("未配置 GEMINI_API_KEY，跳过 GeminiLLM 的 main 测试。")
        print("未配置 GEMINI_API_KEY，跳过 GeminiLLM 的 main 测试。请在 .env 文件中设置。") 