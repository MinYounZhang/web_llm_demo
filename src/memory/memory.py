from typing import List, Any, Dict
from .message import Message

class Memory:
    """
    记忆类，用于存储和管理对话历史和上下文信息。
    """
    def __init__(self):
        """初始化 Memory 对象。"""
        self.messages: List[Message] = []

    def add_message(self, message: Message):
        """
        向记忆中添加一条消息。

        Args:
            message: 要添加的 Message 对象。
        """
        self.messages.append(message)

    def get_all_messages(self) -> List[Message]:
        """获取所有存储的消息。"""
        return self.messages

    def get_messages_by_role(self, role: str) -> List[Message]:
        """
        根据角色筛选消息。

        Args:
            role: 要筛选的角色 (e.g., 'user', 'assistant', 'system').

        Returns:
            符合指定角色的消息列表。
        """
        return [msg for msg in self.messages if msg.role == role]

    def get_last_n_messages(self, n: int) -> List[Message]:
        """
        获取最后 N 条消息。

        Args:
            n: 要获取的消息数量。

        Returns:
            最后 N 条消息的列表。
        """
        return self.messages[-n:]

    def clear_memory(self):
        """清空所有记忆。"""
        self.messages = []

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        将所有消息转换为字典列表。

        Returns:
            包含所有消息字典的列表。
        """
        return [msg.to_dict() for msg in self.messages]

if __name__ == '__main__':
    from .message import Message # 确保在直接运行时可以找到 Message

    memory = Memory()
    memory.add_message(Message(role="user", content="你好！"))
    memory.add_message(Message(role="assistant", content="你好，有什么可以帮您？"))
    memory.add_message(Message(role="system", content="系统已启动。", meta_content={"timestamp": "2024-07-28"}))

    print("所有消息:")
    for msg in memory.get_all_messages():
        print(msg)

    print("\n用户消息:")
    for msg in memory.get_messages_by_role("user"):
        print(msg)

    print("\n最后两条消息:")
    for msg in memory.get_last_n_messages(2):
        print(msg)

    print("\n消息字典列表:")
    print(memory.to_dict_list())

    memory.clear_memory()
    print("\n清空后消息数量:", len(memory.get_all_messages())) 