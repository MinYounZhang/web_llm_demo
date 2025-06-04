from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Union
from src.memory.message import Message # 确保可以引用 Message 类

class BaseLLM(ABC):
    """LLM 模型的抽象基类。"""

    @abstractmethod
    async def chat_completion(self, messages: List[Message], stream: bool = False, **kwargs: Any) -> Union[str, AsyncGenerator[str, None]]:
        """
        生成聊天回复。

        Args:
            messages: 对话消息列表。
            stream: 是否启用流式输出。
            **kwargs: 其他特定于模型的参数。

        Returns:
            如果 stream 为 False，则返回字符串形式的回复。
            如果 stream 为 True，则返回一个异步生成器，逐块生成回复。
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """获取当前 LLM 的配置信息。"""
        pass 