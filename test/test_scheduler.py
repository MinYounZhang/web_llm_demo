import pytest
from unittest.mock import Mock, patch
import sqlite3
from datetime import datetime, timedelta
from src.schedule.scheduler_manager import SchedulerManager
from src.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def scheduler(config):
    scheduler = SchedulerManager(config.scheduler_config)
    scheduler.initialize()
    yield scheduler
    scheduler.close()

def test_scheduler_initialization(scheduler):
    """测试调度器初始化"""
    assert scheduler._conn is not None
    assert scheduler._cursor is not None

def test_add_task(scheduler):
    """测试添加任务"""
    task_data = {
        "name": "测试任务",
        "description": "这是一个测试任务",
        "schedule_time": datetime.now() + timedelta(hours=1),
        "status": "pending"
    }
    
    task_id = scheduler.add_task(
        task_data["name"],
        task_data["description"],
        task_data["schedule_time"],
        task_data["status"]
    )
    
    assert task_id is not None
    task = scheduler.get_task(task_id)
    assert task["name"] == task_data["name"]
    assert task["description"] == task_data["description"]
    assert task["status"] == task_data["status"]

def test_update_task(scheduler):
    """测试更新任务"""
    # 添加任务
    task_id = scheduler.add_task(
        "原始任务",
        "原始描述",
        datetime.now() + timedelta(hours=1),
        "pending"
    )
    
    # 更新任务
    new_name = "更新后的任务"
    new_description = "更新后的描述"
    new_status = "completed"
    
    scheduler.update_task(
        task_id,
        name=new_name,
        description=new_description,
        status=new_status
    )
    
    updated_task = scheduler.get_task(task_id)
    assert updated_task["name"] == new_name
    assert updated_task["description"] == new_description
    assert updated_task["status"] == new_status

def test_delete_task(scheduler):
    """测试删除任务"""
    task_id = scheduler.add_task(
        "要删除的任务",
        "这个任务将被删除",
        datetime.now() + timedelta(hours=1),
        "pending"
    )
    
    scheduler.delete_task(task_id)
    
    with pytest.raises(Exception):
        scheduler.get_task(task_id)

def test_get_pending_tasks(scheduler):
    """测试获取待处理任务"""
    # 添加多个任务
    future_time = datetime.now() + timedelta(hours=1)
    scheduler.add_task("任务1", "描述1", future_time, "pending")
    scheduler.add_task("任务2", "描述2", future_time, "completed")
    scheduler.add_task("任务3", "描述3", future_time, "pending")
    
    pending_tasks = scheduler.get_pending_tasks()
    assert len(pending_tasks) == 2
    assert all(task["status"] == "pending" for task in pending_tasks)

def test_error_handling(scheduler):
    """测试错误处理"""
    # 测试无效的任务ID
    with pytest.raises(Exception):
        scheduler.get_task(999999)
    
    # 测试无效的状态更新
    task_id = scheduler.add_task(
        "测试任务",
        "描述",
        datetime.now() + timedelta(hours=1),
        "pending"
    )
    with pytest.raises(ValueError):
        scheduler.update_task(task_id, status="invalid_status") 