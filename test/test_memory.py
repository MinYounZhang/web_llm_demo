from datetime import datetime
import pytest
from src.memory.memory import Memory
from src.memory.message import Message, Role

@pytest.fixture
def memory():
    return Memory()

def test_message_creation():
    """测试消息创建"""
    content = "测试消息"
    role = Role.USER
    
    message = Message(role=role, content=content)
    
    assert message.role == role
    assert message.content == content
    assert isinstance(message.timestamp, datetime)

def test_memory_add_message(memory):
    """测试添加消息到记忆"""
    message = Message(role=Role.USER, content="测试消息")
    memory.add_message(message)
    
    assert len(memory.messages) == 1
    assert memory.messages[0] == message

def test_memory_clear(memory):
    """测试清空记忆"""
    message = Message(role=Role.USER, content="测试消息")
    memory.add_message(message)
    memory.clear_memory()
    
    assert len(memory.messages) == 0

def test_memory_get_last_n_messages(memory):
    """测试获取最后N条消息"""
    messages = [
        Message(role=Role.USER, content="消息1"),
        Message(role=Role.ASSISTANT, content="消息2"),
        Message(role=Role.USER, content="消息3")
    ]
    
    for msg in messages:
        memory.add_message(msg)
    
    last_two = memory.get_last_n_messages(2)
    assert len(last_two) == 2
    assert last_two[0].content == "消息2"
    assert last_two[1].content == "消息3"