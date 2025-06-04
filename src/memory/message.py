import datetime
from enum import Enum
from typing import Any, Dict, Literal, Optional

class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

class Message:
    """
    消息类，用于包裹用户信息、系统信息和LLM输出的信息。
    """
    def __init__(self, role: str, content: str, meta_content: Optional[Dict[str, Any]] = None, timestamp: Optional[str] = None):
        """
        初始化 Message 对象。

        Args:
            role: 消息的角色 (user, system, assistant)。
            content: 消息的主要内容。
            timestamp: 消息时间戳
            meta_content: 消息的元数据，例如时间戳、来源等。
        """
        if role not in ["user", "system", "assistant"]:
            raise ValueError(f"Role={role}, must be one of 'user', 'system', or 'assistant'")
        self.role = role
        self.content = content
        self.timestamp = datetime.datetime.now().isoformat() if timestamp is None else timestamp
        self.meta_content = meta_content if meta_content is not None else {}

    def __str__(self) -> str:
        return f"{self.role.capitalize()}: {self.content}"

    def to_dict(self) -> Dict[str, Any]:
        """将消息对象转换为字典，方便序列化或传递。"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "meta_content": self.meta_content
        }

if __name__ == '__main__':
    user_message = Message(role="user", content="你好")
    system_message = Message(role="system", content="系统提示", meta_content={"timestamp": "2024-07-27"})
    assistant_message = Message(role="assistant", content="你好，有什么可以帮您的吗？")

    print(user_message)
    print(system_message)
    print(assistant_message.to_dict()) 