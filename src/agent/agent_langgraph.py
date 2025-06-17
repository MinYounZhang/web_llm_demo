import asyncio
import json
import logging
from typing import Any, Dict, List, Tuple, TypedDict, Annotated, Optional, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, ValidationError, TypeAdapter
from pydantic.types import conint, confloat

from src.config import config, logger
from src.memory import Memory, Message
from src.llm import LLMFactory, BaseLLM
from src.browser import BrowserManager
from src.error_management import (
    error_manager, 
    HumanInterventionError, 
    ActionTimeoutError,
    detect_human_intervention_needed,
    get_error_type,
    ErrorType
)

# 超参数配置
LANGGRAPH_CONFIG = {
    "max_iterations": 50,           # 最大迭代次数
    "max_retry_count": 3,           # 最大重试次数
    "max_elements_display": 200,    # 最大页面元素显示数量
    "max_element_text_length": 300, # 元素文本最大长度
    "max_history_actions": 5,       # 最大历史动作记录数
    "enable_light_reflection": True, # 启用轻度反思
    "max_light_reflection_per_node": 3, # 单次节点运行时最大轻度反思次数
}

# 错误分类枚举
class ErrorCategory(Enum):
    """错误分类"""
    ELEMENT_ERROR = "element_error"          # 元素相关错误
    NETWORK_ERROR = "network_error"          # 网络相关错误
    PERMISSION_ERROR = "permission_error"    # 权限相关错误
    BROWSER_ERROR = "browser_error"          # 浏览器相关错误
    TASK_PLANNING_ERROR = "task_planning_error"  # 任务规划错误
    UNKNOWN_ERROR = "unknown_error"          # 未知错误

# 可以通过轻度反思处理的错误类型
LIGHT_REFLECTION_ERRORS = [
    'element_not_found', 'timeout', 'stale_element', 
    'element_not_clickable', 'element_not_visible',
    'page_not_loaded', 'url_mismatch'
]

# ==================== Pydantic数据模型 ====================

class SubTaskModel(BaseModel):
    """子任务数据模型"""
    id: int = Field(default=1, description="子任务ID")
    description: str = Field(default="", description="子任务描述")
    status: str = Field(default="pending", description="任务状态: pending, in_progress, completed, failed")
    priority: int = Field(default=1, ge=1, le=10, description="优先级 1-10")
    dependencies: List[int] = Field(default_factory=list, description="依赖的子任务ID列表")
    expected_outcome: str = Field(default="", description="预期结果")
    max_retries: int = Field(default=3, ge=0, description="最大重试次数")
    current_retries: int = Field(default=0, ge=0, description="当前重试次数")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    def update(self, **kwargs) -> 'SubTaskModel':
        """更新子任务数据"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        valid_statuses = ["pending", "in_progress", "completed", "failed"]
        if v not in valid_statuses:
            raise ValueError(f"状态必须是 {valid_statuses} 中的一个")
        return v

class ActionModel(BaseModel):
    """动作数据模型"""
    action: str = Field(description="动作名称")
    args: Dict[str, Any] = Field(default_factory=dict, description="动作参数")
    timestamp: Optional[datetime] = Field(default=None, description="执行时间")
    result: Optional[Dict[str, Any]] = Field(default=None, description="执行结果")
    
    def update(self, **kwargs) -> 'ActionModel':
        """更新动作数据"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)

class BrowserStateModel(BaseModel):
    """浏览器状态数据模型"""
    current_url: str = Field(default="", description="当前URL")
    current_title: str = Field(default="", description="当前页面标题")
    available_tabs: List[Dict[str, Any]] = Field(default_factory=list, description="可用标签页")
    page_elements: Optional[Any] = Field(default=None, description="页面元素")
    
    def update(self, **kwargs) -> 'BrowserStateModel':
        """更新浏览器状态"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)

class ErrorInfoModel(BaseModel):
    """错误信息数据模型"""
    type: str = Field(default="unknown", description="错误类型")
    message: str = Field(default="", description="错误消息")
    category: str = Field(default="unknown_error", description="错误分类")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误发生时间")
    context: Dict[str, Any] = Field(default_factory=dict, description="错误上下文")
    
    def update(self, **kwargs) -> 'ErrorInfoModel':
        """更新错误信息"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)

class LightReflectionResultModel(BaseModel):
    """轻度反思结果数据模型"""
    action: str = Field(default="continue", description="建议的动作")
    message: str = Field(default="", description="反思消息")
    new_args: Dict[str, Any] = Field(default_factory=dict, description="新的参数")
    wait_time: int = Field(default=0, ge=0, description="等待时间(秒)")
    confidence: confloat(ge=0.0, le=1.0) = Field(default=0.5, description="置信度")
    
    def update(self, **kwargs) -> 'LightReflectionResultModel':
        """更新轻度反思结果"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)

class ReflectionResultModel(BaseModel):
    """反思结果数据模型"""
    next_decision: str = Field(default="continue_current", description="下一步决策")
    analysis: str = Field(default="", description="分析结果")
    confidence: confloat(ge=0.0, le=1.0) = Field(default=0.5, description="置信度")
    suggestions: List[str] = Field(default_factory=list, description="建议列表")
    subtask_completed: bool = Field(default=False, description="子任务是否完成")
    
    def update(self, **kwargs) -> 'ReflectionResultModel':
        """更新反思结果"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)
    
    @field_validator('next_decision')
    @classmethod
    def validate_next_decision(cls, v):
        valid_decisions = ["continue_current", "next_subtask", "exception_handling", "task_completed"]
        if v not in valid_decisions:
            raise ValueError(f"决策必须是 {valid_decisions} 中的一个")
        return v

class NodeResultModel(BaseModel):
    """节点执行结果数据模型"""
    node_type: str = Field(description="节点类型")
    result: str = Field(description="执行结果描述")
    success: bool = Field(default=True, description="是否成功")
    data: Dict[str, Any] = Field(default_factory=dict, description="附加数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="执行时间")
    
    def update(self, **kwargs) -> 'NodeResultModel':
        """更新节点结果"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)

class ExecutionStateModel(BaseModel):
    """执行状态数据模型"""
    actions_to_execute: List[ActionModel] = Field(default_factory=list, description="待执行的动作列表")
    executed_actions: List[ActionModel] = Field(default_factory=list, description="已执行的动作历史")
    last_action_result: Dict[str, Any] = Field(default_factory=dict, description="最后一个动作的执行结果")
    
    def update(self, **kwargs) -> 'ExecutionStateModel':
        """更新执行状态"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)
    
    def add_executed_action(self, action: ActionModel):
        """添加已执行的动作"""
        self.executed_actions.append(action)
    
    def extend_executed_actions(self, actions: List[ActionModel]):
        """批量添加已执行的动作"""
        self.executed_actions.extend(actions)

class ReflectionStateModel(BaseModel):
    """反思状态数据模型"""
    light_reflection_count: int = Field(default=0, ge=0, description="当前节点轻度反思次数")
    light_reflection_result: LightReflectionResultModel = Field(default_factory=LightReflectionResultModel, description="轻度反思结果")
    subtask_completion_status: str = Field(default="in_progress", description="子任务完成状态")
    reflection_result: ReflectionResultModel = Field(default_factory=ReflectionResultModel, description="反思结果")
    
    def update(self, **kwargs) -> 'ReflectionStateModel':
        """更新反思状态"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)
    
    @field_validator('subtask_completion_status')
    @classmethod
    def validate_completion_status(cls, v):
        valid_statuses = ["in_progress", "completed", "needs_llm_reflection"]
        if v not in valid_statuses:
            raise ValueError(f"完成状态必须是 {valid_statuses} 中的一个")
        return v

class ErrorHandlingStateModel(BaseModel):
    """错误处理状态数据模型"""
    error_info: ErrorInfoModel = Field(default_factory=ErrorInfoModel, description="错误信息")
    error_category: str = Field(default="", description="错误分类")
    retry_count: int = Field(default=0, ge=0, description="重试次数")
    
    def update(self, **kwargs) -> 'ErrorHandlingStateModel':
        """更新错误处理状态"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)

class ControlFlowStateModel(BaseModel):
    """控制流状态数据模型"""
    previous_node: str = Field(default="", description="前一个执行的节点")
    previous_node_result: NodeResultModel = Field(default_factory=lambda: NodeResultModel(node_type="init", result="初始化"), description="前一个节点的执行结果")
    next_node: str = Field(default="task_decomposition", description="下一个要执行的节点")
    is_completed: bool = Field(default=False, description="任务是否完成")
    need_human_intervention: bool = Field(default=False, description="是否需要人工干预")
    
    def update(self, **kwargs) -> 'ControlFlowStateModel':
        """更新控制流状态"""
        update_data = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=update_data)


# ==================== 主要的AgentState模型 ====================

class AgentStateModel(BaseModel):
    """Agent状态数据模型 - 使用Pydantic进行数据验证"""
    # 任务相关
    original_task: str = Field(description="原始任务描述")
    subtasks: List[SubTaskModel] = Field(default_factory=list, description="子任务列表")
    current_subtask_index: int = Field(default=0, ge=0, description="当前执行的子任务索引")
    current_subtask: SubTaskModel = Field(default_factory=lambda: SubTaskModel(description="初始化"), description="当前子任务")
    
    # 浏览器状态
    browser_state: BrowserStateModel = Field(default_factory=BrowserStateModel, description="浏览器状态")
    
    # 执行状态
    execution_state: ExecutionStateModel = Field(default_factory=ExecutionStateModel, description="执行状态")
    
    # 反思状态
    reflection_state: ReflectionStateModel = Field(default_factory=ReflectionStateModel, description="反思状态")
    
    # 错误处理状态
    error_handling_state: ErrorHandlingStateModel = Field(default_factory=ErrorHandlingStateModel, description="错误处理状态")
    
    # 控制流状态
    control_flow_state: ControlFlowStateModel = Field(default_factory=ControlFlowStateModel, description="控制流状态")
    
    
    def update_subtasks(self, subtasks_data: List[Dict[str, Any]]):
        """更新子任务列表"""
        subtasks = [SubTaskModel(**task) for task in subtasks_data]
        self.subtasks = subtasks
        if subtasks:
            self.current_subtask = subtasks[0]
            self.current_subtask_index = 0
    
    def move_to_next_subtask(self) -> bool:
        """移动到下一个子任务，返回是否成功"""
        if self.current_subtask_index + 1 < len(self.subtasks):
            self.current_subtask_index += 1
            self.current_subtask = self.subtasks[self.current_subtask_index]
            return True
        return False
    
    def update_browser_state(self, **kwargs):
        """更新浏览器状态"""
        self.browser_state = self.browser_state.update(**kwargs)
    
    def update_execution_state(self, **kwargs):
        """更新执行状态"""
        self.execution_state = self.execution_state.update(**kwargs)
    
    def add_executed_action(self, action: ActionModel):
        """添加已执行的动作"""
        self.execution_state.add_executed_action(action)
    
    def extend_executed_actions(self, actions: List[ActionModel]):
        """批量添加已执行的动作"""
        self.execution_state.extend_executed_actions(actions)
    
    def update_reflection_state(self, **kwargs):
        """更新反思状态"""
        self.reflection_state = self.reflection_state.update(**kwargs)
    
    def update_error_handling_state(self, **kwargs):
        """更新错误处理状态"""
        self.error_handling_state = self.error_handling_state.update(**kwargs)
    
    def update_control_flow_state(self, **kwargs):
        """更新控制流状态"""
        self.control_flow_state = self.control_flow_state.update(**kwargs)
    
    def reset_for_new_subtask(self):
        """为新子任务重置相关状态"""
        self.update_reflection_state(
            light_reflection_count=0,
            light_reflection_result=LightReflectionResultModel(),
            subtask_completion_status="in_progress"
        )
        self.update_error_handling_state(retry_count=0)
    
    def set_error(self, error_type: str, message: str, category: str = ""):
        """设置错误信息"""
        error_info = ErrorInfoModel(type=error_type, message=message, category=category)
        self.update_error_handling_state(
            error_info=error_info,
            error_category=category or classify_error({"type": error_type, "message": message})
        )
    
    def set_node_result(self, node_type: str, result: str, success: bool = True, **data):
        """设置节点执行结果"""
        node_result = NodeResultModel(
            node_type=node_type,
            result=result,
            success=success,
            data=data
        )
        self.update_control_flow_state(
            previous_node=node_type,
            previous_node_result=node_result
        )

# Prompt模板
TASK_DECOMPOSITION_TEMPLATE = """你是浏览器自动化任务拆分专家。你的职责是将复杂的用户任务拆分为具体的、可执行的子任务序列。

=== 任务类型说明 ===
这是一个浏览器自动化任务，系统将通过Web浏览器执行各种操作：
- 页面导航、元素点击、文本输入、内容获取等
- 与网页交互获取信息、填写表单、下载文件等
- 支持多标签页操作、文件保存、搜索等复杂行为

=== 你的节点职责 ===
作为任务拆分节点(A节点)，你需要：
1. 理解用户的原始任务意图
2. 将复杂任务拆分为逻辑清晰的子任务序列
3. 确保每个子任务都是具体可操作的
4. 考虑任务间的依赖关系和执行顺序

原始任务: {original_task}

{previous_info}

{previous_subtasks_info}

=== 执行状态 ===
错误信息: {error_summary}
最近执行的动作: {execution_summary}

=== 任务拆分原则 ===
1. **适度粒度**: 避免过度拆分，每个子任务应该包含一个完整的操作流程
   - 单页面操作尽量合并为一个子任务（如"登录并进入个人中心"）
   - 信息获取任务包含导航、定位、提取全过程
   - 表单填写包含填写所有字段并提交的完整流程

2. **逻辑完整性**: 每个子任务应该有明确的开始条件和结束状态
   - 明确指出从哪个页面开始、到哪个页面结束
   - 包含必要的验证步骤（如确认登录成功、确认信息获取完整）
   - 考虑异常情况的处理（如页面跳转、弹窗处理）

3. **操作连贯性**: 相关操作应该组合在同一个子任务中
   - 搜索操作包含输入关键词、点击搜索、等待结果加载
   - 文章阅读包含点击标题、等待页面加载、获取完整内容
   - 文件下载包含点击下载按钮、等待下载、确认保存成功

4. **实际可执行性**: 确保每个子任务都能通过浏览器自动化完成
   - 避免需要人工判断的模糊任务
   - 明确具体的操作对象（按钮、链接、输入框等）
   - 提供清晰的完成标准

=== 拆分策略和注意事项 ===
**良好拆分示例**:
- ✅ "在百度搜索'人工智能'并获取前3个搜索结果的标题和链接"
- ✅ "访问指定博客文章页面，获取完整文章内容并保存为文件"
- ✅ "填写完整的用户注册表单并提交，确认注册成功"

**避免的过度拆分**:
- ❌ 不要拆分为："打开网站" → "找到搜索框" → "输入关键词" → "点击搜索按钮" → "等待结果"
- ❌ 不要拆分为："点击文章标题" → "等待页面加载" → "滚动页面" → "找到正文" → "复制内容"

**特殊情况处理**:
- 对于复杂的多步骤流程，可以按照功能模块拆分（如：注册 → 登录 → 操作 → 数据导出）
- 对于需要在不同网站间操作的任务，按网站进行拆分
- 对于需要等待用户确认或外部条件的任务，在等待点进行拆分

**重新规划时的要求**:
- 分析之前失败的根本原因（是拆分过细、操作不当、还是理解错误）
- 调整拆分粒度，减少不必要的中间步骤
- 确保每个新的子任务都有明确的业务意义

请返回JSON格式的子任务列表：
{{
  "subtasks": [
    {{
      "id": 1,
      "description": "具体的子任务描述，包含明确的操作步骤和预期结果"
    }},
    {{
      "id": 2,
      "description": "下一个子任务描述"
    }}
  ]
}}

要求：子任务描述要详细具体，让执行节点能够清楚理解要执行什么操作。"""

ACTION_EXECUTION_TEMPLATE = """你是浏览器自动化执行专家。你的职责是分析当前页面状态，根据子任务需求生成具体的浏览器操作动作。

=== 任务类型说明 ===
这是浏览器自动化任务，你需要控制浏览器完成用户指定的操作：
- 导航到指定网页、点击页面元素、输入文本内容
- 获取页面信息、填写表单、下载文件
- 处理弹窗、切换标签页、滚动页面等

=== 你的节点职责 ===
作为动作执行节点(B节点)，你需要：
1. **分析页面状态**: 理解当前页面的内容和可用操作
2. **生成执行动作**: 根据子任务要求制定具体的操作步骤
3. **完成信号**: 当你认为子任务已经完成时，输出finish动作

当前子任务: {subtask_info}

{previous_info}

=== 当前页面状态 ===
URL: {current_url}
标题: {current_title}

{reflection_summary}

=== 页面元素信息 ===
{page_info}

=== 可用的浏览器操作 ===
{actions_list}

=== 最近的执行历史 ===
{recent_actions_summary}

{memory_solution}

=== 执行指导原则 ===
1. **元素选择**: 
   - 页面元素有两个关键字段：`uniqueId`（用作element_id）和`selector`
   - 优先使用element_id参数传递uniqueId，这是最准确的定位方式
   - 如果element_id不可用，使用selector参数传递CSS选择器
   - 每个元素都有对应的selector，通常格式为[data-agent-id="uniqueId"]
   - 注意区分相似元素，选择最相关的目标元素

2. **操作策略**:
   - 对于文章/博客标题，如果任务需要完整内容，应该点击进入详细页面
   - 对于表单填写，先点击输入框获得焦点，再输入内容
   - 对于下拉菜单，先点击展开，再选择选项
   - 操作前考虑页面是否需要滚动到目标元素

3. **信息获取**:
   - 获取文本时确保获取的是完整有用的信息
   - 对于列表数据，考虑是否需要翻页或展开更多内容
   - 截图操作用于保存重要的视觉信息

4. **任务完成判断**:
   - 当你认为当前子任务已经完成时，输出finish动作
   - finish动作表示：已经完成了子任务的所有必要操作，可以进入下一个子任务
   - 只有在确实完成子任务目标后才使用finish动作

=== 注意事项 ===
- 每次执行小于5个相关的操作，避免操作序列过长
- 如果页面正在加载，优先等待页面稳定
- 遇到弹窗或确认对话框，要相应处理
- 文件下载操作要确认下载成功
- 不要重复执行相同的无效操作
- 当子任务完成时，一定要输出finish动作来表示完成

请根据以上信息生成执行动作，返回JSON格式：
{{
  "actions": [
    {{
      "action": "操作名称",
      "args": {{"参数名": "参数值"}}
    }}
  ]
}}

特别提醒：
- 如果你认为当前子任务已经完成，请在actions数组中添加finish动作：{{"action": "finish", "args": {{}}}}
- finish动作应该是最后一个动作，表示子任务完成
- 只有真正完成子任务目标时才使用finish动作，不要过早使用

记住：你的目标是高效准确地完成子任务，当子任务完成时通过finish动作明确表示。"""

REFLECTION_TEMPLATE = """你是浏览器自动化反思分析专家。你的职责是深度分析动作执行结果，准确判断子任务完成状态和整体任务进展，为下一步执行提供明智决策。

=== 任务类型说明 ===
这是浏览器自动化任务的反思阶段，你需要：
- 分析浏览器操作的执行效果和页面变化
- 判断获取的信息是否完整、操作是否达到预期目标
- 评估整体任务进展，决定后续执行策略

=== 你的节点职责 ===
作为反思节点(C节点)，你需要：
1. **结果验证**: 检查执行动作是否达到了预期效果
2. **信息完整性分析**: 评估获取的信息是否满足任务要求
3. **进度评估**: 判断当前子任务和整体任务的完成状态
4. **决策制定**: 为下一步执行提供明确的行动方向

当前子任务: {current_task_info} ({progress_info})

=== 执行情况分析 ===
最近执行的动作: {recent_actions_summary}
最后动作结果: {last_result_summary}
轻度反思处理: {light_reflection_summary}

=== 当前页面状态 ===
URL: {current_url}
标题: {current_title}

=== 页面内容分析 ===
{page_info}

=== 反思分析原则 ===
1. **完成度验证**:
   - 检查子任务的预期目标是否已经达成
   - 验证获取的信息是否完整、准确、有用
   - 确认操作是否产生了预期的页面变化

2. **质量评估**:
   - 对于信息获取任务，检查信息的质量和完整性
   - 对于操作任务，验证操作是否成功执行
   - 评估是否需要补充或改进已完成的工作

3. **进度判断**:
   - 当前子任务是否已经完全完成
   - 是否可以安全地进入下一个子任务
   - 整体任务是否已经全部完成

4. **问题识别**:
   - 识别执行过程中的问题和障碍
   - 判断问题的严重程度和解决难度
   - 确定是否需要异常处理介入

=== 决策指导原则 ===
- **continue_current**: 子任务未完成或需要补充工作时选择
- **next_subtask**: 当前子任务确实已完成，可以进入下一个
- **task_completed**: 所有子任务都完成且整体目标达成
- **exception_handling**: 遇到严重问题、多次失败或需要重新规划时选择

=== 注意事项 ===
- 不要过于乐观，确保子任务真正完成后再进入下一步
- 对于信息获取任务，要验证信息的完整性和准确性
- 如果页面出现异常状态，及时转入异常处理
- 考虑用户的原始任务意图，确保最终目标能够达成

请基于以上分析提供决策，返回JSON格式：
{{
  "subtask_completed": true/false,
  "next_decision": "continue_current|next_subtask|exception_handling|task_completed",
  "analysis": "详细的分析说明，包括当前状态评估、完成度判断、存在的问题和选择该决策的理由"
}}

记住：你的判断直接影响整个任务的执行流程，请基于客观事实做出准确的决策。"""

EXCEPTION_HANDLING_TEMPLATE = """你是浏览器自动化异常处理专家。你的职责是分析执行过程中遇到的问题，提供智能的解决方案，帮助系统克服障碍继续执行任务。

=== 任务类型说明 ===
这是浏览器自动化任务的异常处理阶段，你需要：
- 分析浏览器操作执行失败的根本原因
- 评估问题的严重程度和解决可能性
- 提供针对性的解决方案，优先考虑修复而非重新规划

=== 你的节点职责 ===
作为异常处理节点(D节点)，你需要：
1. **问题诊断**: 深入分析错误的根本原因和影响范围
2. **解决方案评估**: 评估不同解决方案的可行性和成本
3. **智能决策**: 优先选择成本较低、成功率较高的解决方案
4. **避免过度重构**: 谨慎使用replan选项，优先考虑修复和重试

当前子任务: {subtask_info}

{previous_info}

=== 错误信息分析 ===
错误详情: {error_summary}
错误分类: {error_category}
重试次数: {retry_count}

=== 执行历史回顾 ===
{execution_summary}

{memory_error_solution}

=== 异常处理原则 ===
1. **问题分析优先级**:
   - 元素定位问题 > 网络连接问题 > 页面加载问题 > 权限问题
   - 临时性问题 > 持续性问题 > 结构性问题
   - 可修复问题 > 需要规避问题 > 需要重新规划问题

2. **解决方案选择策略**:
   - **retry**: 问题可能是临时的、可通过重试解决的
   - **replan**: 仅在遇到根本性结构问题或任务理解错误时使用

3. **retry适用场景**:
   - 网络超时、页面加载缓慢
   - 元素暂时不可见或不可点击
   - 页面结构轻微变化但核心功能未变
   - 浏览器临时响应问题
   - 重试次数未达到上限且有合理的成功预期

4. **replan适用场景**（慎重使用）:
   - 网站结构发生根本性改变，原有策略完全不可行
   - 任务理解出现重大偏差，需要重新分析需求
   - 权限或访问策略发生重大变更
   - 连续多次不同类型的失败，表明整体策略有问题

=== 重要约束条件 ===
- **不轻易replan**: replan会重置所有进度，应当是最后的选择
- **优先修复**: 大多数问题都可以通过调整参数、等待、重试来解决
- **重试限制**: 考虑当前重试次数，避免无限循环
- **成本考虑**: 重试的成本远低于重新规划的成本

=== 决策指导 ===
在做决策时请考虑：
1. 这个错误是临时的还是持续的？
2. 是否可以通过简单的调整来解决？
3. 重试是否有合理的成功预期？
4. 是否已经尝试了所有可能的修复方案？
5. 问题是否真的严重到需要完全重新规划？

请基于以上原则分析问题并提供解决方案，返回JSON格式：
{{
  "action": "retry|replan",
  "analysis": "详细的错误分析，包括问题原因、严重程度评估和影响范围",
  "solution": "具体的解决方案描述，说明为什么选择这个方案以及预期效果"
}}

记住：你的决策应该帮助系统高效地克服障碍，避免不必要的重新规划。优先考虑修复和重试，只有在确实无法解决时才考虑重新规划。"""

class LightReflection:
    """轻度反思类，用于处理简单的错误和状态检查"""
    
    def __init__(self, browser_manager: BrowserManager):
        self.browser_manager = browser_manager
        self.logger = logger
        
    async def should_trigger_light_reflection(self, action_result: Dict[str, Any]) -> bool:
        """判断是否应该触发轻度反思"""
        if not LANGGRAPH_CONFIG["enable_light_reflection"]:
            return False
            
        if not action_result.get("success", True):
            error_type = action_result.get("error_type", "unknown")
            return error_type in LIGHT_REFLECTION_ERRORS
            
        return False
        
    async def perform_light_reflection(self, 
                                     action_result: Dict[str, Any],
                                     current_action: Dict[str, Any],
                                     page_elements: Any) -> Dict[str, Any]:
        """执行轻度反思"""
        self.logger.info("开始轻度反思")
        
        error_type = action_result.get("error_type", "unknown")
        
        # 根据错误类型进行不同的处理
        if error_type == "element_not_found":
            return await self._handle_element_not_found(current_action, page_elements)
        elif error_type == "timeout":
            return await self._handle_timeout(current_action)
        elif error_type == "stale_element":
            return await self._handle_stale_element(current_action, page_elements)
        elif error_type == "element_not_clickable":
            return await self._handle_element_not_clickable(current_action, page_elements)
        elif error_type == "element_not_visible":
            return await self._handle_element_not_visible(current_action, page_elements)
        elif error_type == "page_not_loaded":
            return await self._handle_page_not_loaded()
        elif error_type == "url_mismatch":
            return await self._handle_url_mismatch(current_action)
        else:
            return {"action": "continue", "message": "未知错误类型，继续执行"}
    
    async def _handle_element_not_found(self, action: Dict[str, Any], page_elements: Any) -> Dict[str, Any]:
        """处理元素未找到错误"""
        element_id = action.get("args", {}).get("element_id")
        
        if element_id and page_elements:
            # 检查是否有相似的元素
            similar_elements = await self._find_similar_elements(element_id, page_elements)
            if similar_elements:
                return {
                    "action": "modify_args",
                    "new_args": {"element_id": similar_elements[0]},
                    "message": f"找到相似元素: {similar_elements[0]}"
                }
        
        # 检查页面是否完全加载
        if await self._is_page_loading():
            return {
                "action": "wait",
                "wait_time": 2,
                "message": "页面正在加载，等待后重试"
            }
            
        return {"action": "re_execute", "message": "重新执行动作"}
    
    async def _handle_timeout(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """处理超时错误"""
        # 检查网络状态
        if await self._check_network_connection():
            return {
                "action": "wait",
                "wait_time": 3,
                "message": "网络连接正常，延长等待时间"
            }
        else:
            return {
                "action": "re_execute",
                "message": "网络连接异常，重新执行"
            }
    
    async def _handle_stale_element(self, action: Dict[str, Any], page_elements: Any) -> Dict[str, Any]:
        """处理元素过期错误"""
        # 刷新页面元素信息
        try:
            await self.browser_manager.get_page_elements_with_fallback(force_refresh=True)
            return {
                "action": "re_execute", 
                "message": "页面元素已刷新，重新执行"
            }
        except Exception as e:
            return {
                "action": "wait",
                "wait_time": 1,
                "message": f"刷新元素失败: {str(e)}，等待后重试"
            }
    
    async def _handle_element_not_clickable(self, action: Dict[str, Any], page_elements: Any) -> Dict[str, Any]:
        """处理元素不可点击错误"""
        # 检查元素是否被遮挡
        element_id = action.get("args", {}).get("element_id")
        if element_id:
            # 尝试滚动到元素位置
            return {
                "action": "scroll_to_element",
                "new_args": {"element_id": element_id},
                "message": "尝试滚动到元素位置"
            }
        
        return {"action": "wait", "wait_time": 1, "message": "等待元素变为可点击状态"}
    
    async def _handle_element_not_visible(self, action: Dict[str, Any], page_elements: Any) -> Dict[str, Any]:
        """处理元素不可见错误"""
        return {
            "action": "wait",
            "wait_time": 2,
            "message": "等待元素变为可见"
        }
    
    async def _handle_page_not_loaded(self) -> Dict[str, Any]:
        """处理页面未加载完成错误"""
        return {
            "action": "wait",
            "wait_time": 5,
            "message": "等待页面加载完成"
        }
    
    async def _handle_url_mismatch(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """处理URL不匹配错误 - 改进版本，处理URL跳转情况"""
        try:
            current_url, _, _ = await self.browser_manager.get_page_state()
            expected_url = action.get("args", {}).get("expected_url")
            
            if expected_url:
                # 检查URL是否包含预期的关键部分
                from urllib.parse import urlparse
                
                current_parsed = urlparse(current_url)
                expected_parsed = urlparse(expected_url)
                
                # 检查域名是否匹配
                if current_parsed.netloc != expected_parsed.netloc:
                    # 域名不匹配，可能需要重新导航
                    return {
                        "action": "navigate",
                        "new_args": {"url": expected_url},
                        "message": f"域名不匹配，重新导航到: {expected_url}"
                    }
                else:
                    # 域名匹配但路径不同，可能是正常的页面跳转
                    return {
                        "action": "continue",
                        "message": f"URL已跳转但在同一域名下: {current_url}"
                    }
            
            return {"action": "continue", "message": "URL匹配正常"}
            
        except Exception as e:
            return {
                "action": "continue",
                "message": f"URL检查失败，继续执行: {str(e)}"
            }
    
    async def _find_similar_elements(self, target_element_id: str, page_elements: Any) -> List[str]:
        """查找相似的元素"""
        try:
            elements = page_elements.get("elements", [])
            target_tag = None
            target_text = None
            
            # 从target_element_id中提取信息
            for element in elements:
                if element.get("id") == target_element_id:
                    target_tag = element.get("tag")
                    target_text = element.get("text", "")
                    break
            
            if not target_tag:
                return []
            
            # 查找相似元素
            similar_elements = []
            for element in elements:
                if (element.get("tag") == target_tag and 
                    element.get("id") != target_element_id):
                    # 如果有文本，优先匹配相似文本
                    if target_text and element.get("text"):
                        if target_text.lower() in element.get("text", "").lower():
                            similar_elements.append(element.get("id"))
                    else:
                        similar_elements.append(element.get("id"))
            
            return similar_elements[:3]  # 最多返回3个相似元素
            
        except Exception as e:
            self.logger.warning(f"查找相似元素失败: {e}")
            return []
    
    async def _is_page_loading(self) -> bool:
        """检查页面是否正在加载"""
        try:
            # 这里可以调用browser_manager的相关方法检查页面状态
            # 暂时返回False，可以根据实际需要实现
            return False
        except Exception:
            return True
    
    async def _check_network_connection(self) -> bool:
        """检查网络连接状态"""
        try:
            # 这里可以实现网络连接检查
            # 暂时返回True，可以根据实际需要实现
            return True
        except Exception:
            return False

# ==================== 兼容性TypedDict（用于LangGraph） ====================

# ==================== TypeAdapter for AgentState ====================

AgentStateAdapter = TypeAdapter(AgentStateModel)

class AgentState(TypedDict):
    """Agent状态定义 - 兼容LangGraph的TypedDict格式"""
    # 使用单一的model字段来存储所有状态
    model: AgentStateModel

def classify_error(error_info: Dict[str, Any]) -> str:
    """错误分类函数"""
    error_type = error_info.get('type', 'unknown')
    error_message = error_info.get('message', '').lower()
    
    if error_type in LIGHT_REFLECTION_ERRORS:
        return ErrorCategory.ELEMENT_ERROR.value
    elif 'permission' in error_message or 'access' in error_message:
        return ErrorCategory.PERMISSION_ERROR.value
    elif 'network' in error_message or 'connection' in error_message:
        return ErrorCategory.NETWORK_ERROR.value
    elif 'browser' in error_message or 'crash' in error_message:
        return ErrorCategory.BROWSER_ERROR.value
    elif 'task' in error_message or 'planning' in error_message:
        return ErrorCategory.TASK_PLANNING_ERROR.value
    else:
        return ErrorCategory.UNKNOWN_ERROR.value

class LangGraphAgent:
    """基于LangGraph的Agent类，使用状态图来管理任务执行流程"""

    def __init__(self, memory: Memory | None = None, browser_manager: BrowserManager | None = None, llm: BaseLLM | None = None):
        self.memory = memory or Memory()
        self.browser_manager = browser_manager or BrowserManager()
        self.llm = llm or LLMFactory.create_llm()
        self.light_reflection = LightReflection(self.browser_manager)
        self.graph = self._create_graph()
        logger.info(f"LangGraph Agent 初始化完成。LLM Provider: {self.llm.get_config().get('provider')}")

    def _create_graph(self) -> StateGraph:
        """创建状态图"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("task_decomposition", self._task_decomposition_node)  # A节点：任务拆分
        workflow.add_node("action_execution", self._action_execution_node)      # B节点：动作执行
        workflow.add_node("reflection", self._reflection_node)                  # C节点：反思
        workflow.add_node("exception_handling", self._exception_handling_node)  # D节点：异常处理
        
        # 设置入口点
        workflow.set_entry_point("task_decomposition")
        
        # 添加边
        workflow.add_edge("task_decomposition", "action_execution")  # A->B
        
        # 条件边：从动作执行节点出发
        workflow.add_conditional_edges(
            "action_execution",
            self._action_execution_router,
            {
                "continue_execution": "action_execution",  # B->B (持续执行)
                "need_reflection": "reflection",           # B->C (需要反思)
                "need_exception": "exception_handling"    # B->D (需要异常处理)
            }
        )
        
        # 条件边：从反思节点出发 
        workflow.add_conditional_edges(
            "reflection",
            self._reflection_router,
            {
                "continue_action": "action_execution",     # C->B
                "exception": "exception_handling",         # C->D
                "end": END                                 # 结束
            }
        )
        
        # 条件边：从异常处理节点出发
        workflow.add_conditional_edges(
            "exception_handling",
            self._exception_router,
            {
                "retry_action": "action_execution",        # D->B
                "replan_task": "task_decomposition",       # D->A (慎重)
                "end": END                                 # 结束
            }
        )
        
        return workflow.compile()

    def _action_execution_router(self, state: AgentState) -> str:
        """动作执行节点的路由函数"""
        model = state["model"]
        if model.control_flow_state.need_human_intervention:
            return "end"
        elif model.control_flow_state.next_node == "exception_handling":
            return "need_exception"
        elif model.control_flow_state.next_node == "reflection":
            return "need_reflection"
        else:  # next_node == "action_execution" 或其他情况
            return "continue_execution"

    def _reflection_router(self, state: AgentState) -> str:
        """反思节点的路由函数"""
        model = state["model"]
        if model.control_flow_state.need_human_intervention:
            return "end"
        elif model.control_flow_state.next_node == "exception_handling":
            return "exception"
        elif model.control_flow_state.next_node == "end" or model.control_flow_state.is_completed:
            return "end"
        else:
            return "continue_action"

    def _exception_router(self, state: AgentState) -> str:
        """异常处理节点的路由函数"""
        model = state["model"]
        if model.control_flow_state.need_human_intervention:
            return "end"
        elif model.control_flow_state.next_node == "task_decomposition":
            return "replan_task"
        elif model.control_flow_state.next_node == "end" or model.control_flow_state.is_completed:
            return "end"
        else:  # action_execution
            return "retry_action"

    async def _task_decomposition_node(self, state: AgentState) -> AgentState:
        """
        A节点：任务拆分节点
        将输入任务拆分成子任务
        """
        logger.info("step.1:执行任务拆分节点")
        
        model = state["model"]
        
        # 构建任务拆分的prompt
        decomposition_prompt = self._build_task_decomposition_prompt(
            model.original_task, 
            model.error_handling_state.error_info.model_dump(),
            [action.model_dump() for action in model.execution_state.executed_actions],
            model.control_flow_state.previous_node_result.model_dump(),
            [task.model_dump() for task in model.subtasks]
        )
        
        try:
            # 调用LLM进行任务拆分
            response = await self._call_llm([decomposition_prompt])
            subtasks_data = await self._parse_task_decomposition_response(response)
            logger.info(f"step.1:任务拆分完成，{subtasks_data}")
            
            # 更新状态 - 使用Pydantic模型的update方法
            model.update_subtasks(subtasks_data)
            model.reset_for_new_subtask()
            model.update_control_flow_state(next_node="action_execution")
            model.set_node_result(
                node_type="task_decomposition",
                result=f"成功拆分为{len(subtasks_data)}个子任务",
                subtasks_count=len(subtasks_data)
            )
            
            # 存储LLM交互到memory
            self.memory.add_message(decomposition_prompt)
            self.memory.add_message(Message(role="assistant", content=response))
            
        except Exception as e:
            logger.error(f"任务拆分失败: {e}")
            # 任务拆分失败时直接返回错误状态，不导航到异常处理
            model.set_error("task_decomposition_error", str(e))
            model.update_control_flow_state(is_completed=True, next_node="end")
        
        return state

    async def _action_execution_node(self, state: AgentState) -> AgentState:
        """
        B节点：动作执行节点 + 轻度反思
        接收当前网页上下文信息和子任务，输出后续执行动作列表
        如果执行失败，进行轻度反思处理
        可以持续执行 (B->B)，只有在特定条件下才转到其他节点
        """
        logger.info("step.2:执行动作执行节点")
        
        model = state["model"]
        
        # 初始化轻度反思计数
        if model.control_flow_state.previous_node != "action_execution":
            model.update_reflection_state(
                light_reflection_count=0,
                light_reflection_result=LightReflectionResultModel()
            )
        
        # 获取当前页面状态  
        try:
            current_url, current_title, available_tabs = await self.browser_manager.get_page_state()
            page_elements = await self.browser_manager.get_page_elements_with_fallback()
            
            # 更新状态中的页面信息
            model.update_browser_state(
                current_url=current_url,
                current_title=current_title,
                available_tabs=available_tabs,
                page_elements=page_elements
            )
            
        except Exception as e:
            logger.error(f"获取页面状态失败: {e}")
            model.set_error("page_state_error", str(e))
            model.update_control_flow_state(next_node="exception_handling")
            return state
        
        # 构建动作执行的prompt
        action_prompt = self._build_action_execution_prompt(
            model.current_subtask.model_dump(),
            current_url, current_title, available_tabs, page_elements,
            [action.model_dump() for action in model.execution_state.executed_actions],
            model.reflection_state.reflection_result.model_dump(),
            model.control_flow_state.previous_node_result.model_dump(),
            self.browser_manager.get_all_actions_description()
        )
        
        try:
            # 调用LLM获取动作列表和状态判断
            response = await self._call_llm([action_prompt])
            actions_data, status_info = await self._parse_action_execution_response(response)
            logger.info(f"step.2:将执行动作，{actions_data}")
            logger.info(f"step.2:LLM判断的子任务状态: {status_info}")
            
            # 执行动作
            executed_actions = []
            last_action_result = {}
            has_finish_action = status_info.get("has_finish_action", False)
            
            for action_detail in actions_data:
                action_name = action_detail.get("action")
                action_args = action_detail.get("args", {})
                
                # 创建ActionModel
                action_model = ActionModel(
                    action=action_name,
                    args=action_args,
                    timestamp=datetime.now()
                )
                
                # 执行动作
                last_action_result = await self.browser_manager.execute_action(action_name, **action_args)
                
                # 更新ActionModel的结果
                action_model = action_model.update(result=last_action_result)
                executed_actions.append(action_model)
                
                # 检查是否需要轻度反思
                if not last_action_result.get("success", True):
                    should_reflect = await self.light_reflection.should_trigger_light_reflection(last_action_result)
                    
                    if (should_reflect and 
                        model.reflection_state.light_reflection_count < LANGGRAPH_CONFIG["max_light_reflection_per_node"]):
                        
                        logger.info(f"触发轻度反思，当前次数: {model.reflection_state.light_reflection_count}")
                        
                        # 执行轻度反思
                        light_reflection_result = await self.light_reflection.perform_light_reflection(
                            last_action_result, action_detail, page_elements
                        )
                        
                        # 更新轻度反思状态
                        model.update_reflection_state(
                            light_reflection_count=model.reflection_state.light_reflection_count + 1,
                            light_reflection_result=LightReflectionResultModel(**light_reflection_result)
                        )
                        
                        # 根据轻度反思结果进行处理
                        reflection_action = light_reflection_result.get("action", "continue")
                        
                        if reflection_action == "wait":
                            wait_time = light_reflection_result.get("wait_time", 2)
                            logger.info(f"轻度反思建议等待 {wait_time} 秒")
                            await asyncio.sleep(wait_time)
                            
                        elif reflection_action == "modify_args":
                            new_args = light_reflection_result.get("new_args", {})
                            logger.info(f"轻度反思建议修改参数: {new_args}")
                            # 使用新参数重新执行
                            last_action_result = await self.browser_manager.execute_action(action_name, **new_args)
                            
                        elif reflection_action == "re_execute":
                            logger.info("轻度反思建议重新执行")
                            # 重新执行原动作
                            last_action_result = await self.browser_manager.execute_action(action_name, **action_args)
                            
                        elif reflection_action == "navigate":
                            new_args = light_reflection_result.get("new_args", {})
                            logger.info(f"轻度反思建议导航: {new_args}")
                            # 执行导航
                            last_action_result = await self.browser_manager.execute_action("navigate_to_url", **new_args)
                            
                        elif reflection_action == "scroll_to_element":
                            new_args = light_reflection_result.get("new_args", {})
                            logger.info(f"轻度反思建议滚动到元素: {new_args}")
                            # 执行滚动操作
                            scroll_result = await self.browser_manager.execute_action("scroll", **{"direction": "down", "distance": 300})
                            # 然后重新执行原动作
                            last_action_result = await self.browser_manager.execute_action(action_name, **action_args)
                        
                        # 如果轻度反思后仍然失败，且达到最大反思次数，标记需要LLM反思
                        if (not last_action_result.get("success", True) and 
                            model.reflection_state.light_reflection_count >= LANGGRAPH_CONFIG["max_light_reflection_per_node"]):
                            model.update_reflection_state(subtask_completion_status="needs_llm_reflection")
                            logger.info("轻度反思达到最大次数，需要LLM反思")
                        
                    else:
                        # 不符合轻度反思条件或达到最大次数，需要LLM反思
                        model.update_reflection_state(subtask_completion_status="needs_llm_reflection")
                        
                else:
                    # 执行成功
                    model.update_reflection_state(subtask_completion_status="in_progress")
            
            # 更新状态
            action_models = [ActionModel(**action) for action in actions_data]
            model.update_execution_state(
                actions_to_execute=action_models,
                last_action_result=last_action_result
            )
            model.extend_executed_actions(executed_actions)
            
            # 根据LLM的判断和执行结果决定下一步
            llm_subtask_status = status_info.get("subtask_status", "in_progress")
            has_finish_action = status_info.get("has_finish_action", False)
            
            # 优先检查finish动作
            if has_finish_action:
                # 检测到finish动作，直接跳转到反思节点进行任务完成确认
                model.update_control_flow_state(next_node="reflection")
                logger.info("检测到finish动作，转入反思节点进行任务完成确认")
            elif (not last_action_result.get("success", True) and 
                model.reflection_state.subtask_completion_status == "needs_llm_reflection"):
                # 执行失败且轻度反思无效，需要深度反思
                model.update_control_flow_state(next_node="reflection")
                logger.info("执行失败且轻度反思无效，转入深度反思")
            elif llm_subtask_status == "completed":
                # LLM判断子任务已完成，需要反思确认
                model.update_control_flow_state(next_node="reflection")
                completion_reason = status_info.get("completion_reason", "")
                logger.info(f"LLM判断子任务已完成，转入反思确认。完成原因: {completion_reason}")
            elif llm_subtask_status == "needs_more_info":
                # LLM判断需要更多信息，转入反思
                model.update_control_flow_state(next_node="reflection")
                logger.info("LLM判断需要更多信息，转入反思")
            else:  # llm_subtask_status == "in_progress"
                # LLM判断子任务仍在进行中，继续执行 (B->B)
                model.update_control_flow_state(next_node="action_execution")
                logger.info("LLM判断子任务仍在进行中，继续执行")
            
            # 设置节点结果
            model.set_node_result(
                node_type="action_execution",
                result=f"执行了{len(executed_actions)}个动作",
                actions_count=len(executed_actions),
                last_action_success=last_action_result.get("success", False),
                light_reflection_count=model.reflection_state.light_reflection_count
            )
            
            # 存储LLM交互到memory
            self.memory.add_message(action_prompt)
            self.memory.add_message(Message(role="assistant", content=response))
            
        except Exception as e:
            logger.error(f"动作执行节点失败: {e}")
            model.set_error("action_execution_error", str(e))
            model.update_control_flow_state(next_node="exception_handling")
            model.set_node_result(
                node_type="action_execution",
                result=f"执行失败: {str(e)}",
                success=False,
                problem_details=str(e)
            )
        
        return state
    


    async def _reflection_node(self, state: AgentState) -> AgentState:
        """C节点：总结性反思节点"""
        logger.info("step.3:执行总结性反思节点")
        
        model = state["model"]
        
        # 获取最新的页面状态
        try:
            current_url, current_title, available_tabs = await self.browser_manager.get_page_state()
            page_elements = await self.browser_manager.get_page_elements_with_fallback()
            
            # 更新浏览器状态
            model.update_browser_state(
                current_url=current_url,
                current_title=current_title,
                available_tabs=available_tabs,
                page_elements=page_elements
            )
            
        except Exception as e:
            logger.error(f"反思节点获取页面状态失败: {e}")
            model.set_error("reflection_page_error", str(e))
            model.update_control_flow_state(next_node="exception_handling")
            model.set_node_result(
                node_type="reflection",
                result=f"获取页面状态失败: {str(e)}",
                success=False,
                problem_details=str(e)
            )
            return state
        
        # 构建反思prompt
        reflection_prompt = self._build_reflection_prompt(
            model.current_subtask.model_dump(),
            [action.model_dump() for action in model.execution_state.executed_actions],
            model.execution_state.last_action_result,
            current_url, current_title, page_elements,
            [task.model_dump() for task in model.subtasks],
            model.current_subtask_index,
            model.reflection_state.light_reflection_result.model_dump()
        )
        
        try:
            # 调用LLM进行反思
            response = await self._call_llm([reflection_prompt])
            reflection_result = await self._parse_reflection_response(response)
            logger.info(f"step.3:反思结果，{reflection_result}")
            
            # 更新反思结果
            reflection_model = ReflectionResultModel(**reflection_result)
            model.update_reflection_state(reflection_result=reflection_model)
            
            # 存储LLM交互到memory
            self.memory.add_message(reflection_prompt)
            self.memory.add_message(Message(role="assistant", content=response))
            
            # 根据反思结果的决策决定下一步
            next_decision = reflection_result.get("next_decision", "continue_current")
            
            if next_decision == "next_subtask":
                # 当前子任务已完成，移动到下一个子任务
                if model.move_to_next_subtask():
                    model.reset_for_new_subtask()
                    model.update_control_flow_state(next_node="action_execution")
                    model.set_node_result(
                        node_type="reflection",
                        result=f"子任务{model.current_subtask_index}已完成，移动到下一个子任务",
                        subtask_index=model.current_subtask_index
                    )
                    logger.info(f"子任务完成，移动到下一个子任务: {model.current_subtask.description}")
                else:
                    # 这是最后一个子任务且已完成，整个任务完成
                    model.update_control_flow_state(is_completed=True, next_node="end")
                    model.set_node_result(
                        node_type="reflection",
                        result="最后一个子任务已完成，整个任务结束"
                    )
                    logger.info("最后一个子任务已完成，整个任务结束")
                    
            elif next_decision == "task_completed":
                # 所有子任务完成
                model.update_control_flow_state(is_completed=True, next_node="end")
                model.set_node_result(
                    node_type="reflection",
                    result="所有子任务已完成"
                )
                logger.info("所有子任务已完成")
                
            elif next_decision == "exception_handling":
                # 需要异常处理
                model.update_control_flow_state(next_node="exception_handling")
                model.set_node_result(
                    node_type="reflection",
                    result="检测到严重问题，需要异常处理"
                )
                logger.info("检测到严重问题，转入异常处理")
                
            else:  
                # 继续执行当前子任务
                model.update_control_flow_state(next_node="action_execution")
                model.update_reflection_state(
                    light_reflection_count=0,  # 重置轻度反思计数，给B节点新的机会
                    subtask_completion_status="in_progress"
                )
                model.set_node_result(
                    node_type="reflection",
                    result="当前子任务未完成，继续执行",
                    analysis=reflection_result.get("analysis", "")
                )
                logger.info("继续执行当前子任务")
                    
        except Exception as e:
            logger.error(f"反思节点失败: {e}")
            model.set_error("reflection_error", str(e))
            model.update_control_flow_state(next_node="exception_handling")
            model.set_node_result(
                node_type="reflection",
                result=f"反思过程失败: {str(e)}",
                success=False,
                problem_details=str(e)
            )
        
        return state

    async def _exception_handling_node(self, state: AgentState) -> AgentState:
        """D节点：深度异常处理节点（减少replan，优先尝试解决问题）"""
        logger.info("step.4:执行深度异常处理节点")
        
        model = state["model"]
        
        # 如果错误信息中没有分类，进行分类
        if not model.error_handling_state.error_category:
            error_dict = model.error_handling_state.error_info.model_dump()
            error_category = classify_error(error_dict)
            model.update_error_handling_state(error_category=error_category)
        
        error_category = model.error_handling_state.error_category
        error_info = model.error_handling_state.error_info.model_dump()
        logger.info(f"错误分类: {error_category}")
        
        # 首先尝试自动解决问题，而不是立即replan
        try:
            problem_solved = await self._try_solve_problem(error_category, error_info, state)
            if problem_solved:
                logger.info("问题已自动解决，继续执行")
                model.update_control_flow_state(next_node="action_execution")
                model.update_error_handling_state(retry_count=0)
                model.update_reflection_state(light_reflection_count=0)
                model.set_node_result(
                    node_type="exception_handling",
                    result="问题已自动解决，继续执行"
                )
                return state
        except Exception as e:
            logger.warning(f"自动解决问题失败: {e}")
        
        # 根据错误分类进行不同的处理策略
        if error_category == ErrorCategory.PERMISSION_ERROR.value:
            # 权限错误，检查是否可以通过其他方式解决
            if self._can_solve_permission_error(error_info):
                model.update_control_flow_state(next_node="action_execution")
                model.update_error_handling_state(
                    retry_count=model.error_handling_state.retry_count + 1
                )
                model.set_node_result(
                    node_type="exception_handling",
                    result="尝试通过其他方式解决权限问题"
                )
                logger.info("尝试通过其他方式解决权限问题")
            else:
                logger.warning("权限错误无法解决，需要人工干预")
                model.update_control_flow_state(
                    need_human_intervention=True,
                    is_completed=True,
                    next_node="end"
                )
                model.set_node_result(
                    node_type="exception_handling",
                    result="权限错误无法解决，需要人工干预",
                    success=False
                )
            return state
            
        elif error_category == ErrorCategory.BROWSER_ERROR.value:
            # 浏览器错误，尝试重启浏览器
            logger.warning("检测到浏览器错误，尝试重启浏览器")
            try:
                await self.browser_manager.restart_browser()
                model.update_control_flow_state(next_node="action_execution")
                model.update_error_handling_state(retry_count=0)
                model.update_reflection_state(light_reflection_count=0)
                model.set_node_result(
                    node_type="exception_handling",
                    result="浏览器已重启，重新执行任务"
                )
                return state
            except Exception as e:
                logger.error(f"重启浏览器失败: {e}")
                model.update_control_flow_state(
                    need_human_intervention=True,
                    is_completed=True,
                    next_node="end"
                )
                model.set_node_result(
                    node_type="exception_handling",
                    result=f"重启浏览器失败: {str(e)}",
                    success=False
                )
                return state
        
        # 对于其他错误类型，使用LLM进行分析
        exception_prompt = self._build_exception_handling_prompt(
            model.current_subtask.model_dump(),
            error_info,
            [action.model_dump() for action in model.execution_state.executed_actions],
            model.error_handling_state.retry_count,
            model.control_flow_state.previous_node_result.model_dump(),
            error_category
        )
        
        try:
            # 调用LLM进行异常分析
            response = await self._call_llm([exception_prompt])
            exception_result = await self._parse_exception_handling_response(response)
            logger.info(f"step.4:异常处理分析，{exception_result}")
            
            # 存储LLM交互到memory
            self.memory.add_message(exception_prompt)
            self.memory.add_message(Message(role="assistant", content=response))
            
            # 根据异常处理结果决定下一步（优先重试，减少replan）
            if exception_result.get("action") == "retry":
                # 重试当前动作
                new_retry_count = model.error_handling_state.retry_count + 1
                if new_retry_count < LANGGRAPH_CONFIG["max_retry_count"]:
                    model.update_control_flow_state(next_node="action_execution")
                    model.update_error_handling_state(retry_count=new_retry_count)
                    model.update_reflection_state(light_reflection_count=0)  # 重置轻度反思计数
                    model.set_node_result(
                        node_type="exception_handling",
                        result=f"决定重试，第 {new_retry_count} 次",
                        analysis=exception_result.get("analysis", ""),
                        solution=exception_result.get("solution", "")
                    )
                    logger.info(f"重试当前子任务，第 {new_retry_count} 次")
                else:
                    # 重试次数过多，但仍优先尝试其他解决方案而非replan
                    if self._has_alternative_solutions(error_category, error_info):
                        model.update_control_flow_state(next_node="action_execution")
                        model.update_error_handling_state(retry_count=0)  # 重置重试计数，尝试新方案
                        model.set_node_result(
                            node_type="exception_handling",
                            result="重试次数过多，尝试替代解决方案"
                        )
                        logger.info("重试次数过多，尝试替代解决方案")
                    elif self._is_critical_error_requiring_replan(error_category, error_info):
                        # 只有在关键错误且无其他解决方案时才replan
                        model.update_control_flow_state(next_node="task_decomposition")
                        model.update_error_handling_state(retry_count=0)
                        model.set_node_result(
                            node_type="exception_handling",
                            result="关键错误且无替代方案，重新规划任务",
                            analysis=f"错误类型: {error_category}, 已重试{model.error_handling_state.retry_count}次"
                        )
                        logger.info("关键错误且无替代方案，重新规划任务")
                    else:
                        # 其他情况需要人工干预
                        logger.warning("重试次数过多且无有效解决方案，需要人工干预")
                        model.update_control_flow_state(
                            need_human_intervention=True,
                            is_completed=True,
                            next_node="end"
                        )
                        model.set_node_result(
                            node_type="exception_handling",
                            result="重试次数过多且无有效解决方案，需要人工干预",
                            success=False
                        )
                        
            elif exception_result.get("action") == "replan":
                # 只有在明确需要时才replan
                if self._is_critical_error_requiring_replan(error_category, error_info):
                    model.update_control_flow_state(next_node="task_decomposition")
                    model.update_error_handling_state(retry_count=0)
                    model.update_reflection_state(light_reflection_count=0)
                    model.set_node_result(
                        node_type="exception_handling",
                        result="确认需要重新规划任务",
                        analysis=exception_result.get("analysis", ""),
                        solution=exception_result.get("solution", "")
                    )
                    logger.info("确认需要重新规划任务")
                else:
                    # 不是关键错误，继续尝试执行
                    model.update_control_flow_state(next_node="action_execution")
                    model.update_reflection_state(light_reflection_count=0)
                    model.set_node_result(
                        node_type="exception_handling",
                        result="错误不需要重新规划，继续尝试执行"
                    )
                    logger.info("错误不需要重新规划，继续尝试执行")
                    
            else:
                # 默认重试
                model.update_control_flow_state(next_node="action_execution")
                model.update_reflection_state(light_reflection_count=0)
                model.set_node_result(
                    node_type="exception_handling",
                    result="使用默认策略重试"
                )
                
        except Exception as e:
            logger.error(f"异常处理节点失败: {e}")
            # 异常处理失败时，根据错误严重程度决定下一步
            if self._is_critical_error_requiring_replan(error_category, error_info):
                model.update_control_flow_state(next_node="task_decomposition")
                model.set_node_result(
                    node_type="exception_handling",
                    result=f"异常处理失败: {str(e)}，但错误可能通过重新规划解决",
                    success=False
                )
                logger.info("异常处理失败，尝试重新规划")
            else:
                # 继续尝试执行，给系统一个机会
                model.update_control_flow_state(next_node="action_execution")
                model.set_node_result(
                    node_type="exception_handling",
                    result=f"异常处理失败: {str(e)}，继续尝试执行",
                    success=False
                )
                logger.info("异常处理失败，继续尝试执行")
        
        return state
    
    async def _try_solve_problem(self, error_category: str, error_info: Dict[str, Any], state: AgentState) -> bool:
        """尝试自动解决问题（TODO: 实现具体的问题解决逻辑）"""
        # TODO: 根据错误类型实现具体的自动解决逻辑
        logger.info(f"TODO: 尝试自动解决问题，错误类型: {error_category}")
        
        # 示例：对于元素错误，可以尝试刷新页面元素
        if error_category == ErrorCategory.ELEMENT_ERROR.value:
            try:
                # 刷新页面元素
                await self.browser_manager.get_page_elements_with_fallback(force_refresh=True)
                return True
            except Exception:
                return False
        
        # 示例：对于网络错误，可以尝试等待一段时间
        if error_category == ErrorCategory.NETWORK_ERROR.value:
            try:
                await asyncio.sleep(3)  # 等待3秒
                return True
            except Exception:
                return False
        
        return False
    
    def _can_solve_permission_error(self, error_info: Dict[str, Any]) -> bool:
        """判断权限错误是否可以通过其他方式解决"""
        # TODO: 实现具体的权限错误解决判断逻辑
        error_message = error_info.get('message', '').lower()
        
        # 如果是简单的访问限制，可能可以通过其他方式解决
        if any(keyword in error_message for keyword in ['cookies', 'session', 'token']):
            return True
            
        return False
    
    def _has_alternative_solutions(self, error_category: str, error_info: Dict[str, Any]) -> bool:
        """判断是否有替代解决方案"""
        # TODO: 实现具体的替代方案判断逻辑
        
        # 对于元素错误，通常有替代方案
        if error_category == ErrorCategory.ELEMENT_ERROR.value:
            return True
            
        # 对于网络错误，可能有替代方案
        if error_category == ErrorCategory.NETWORK_ERROR.value:
            return True
            
        return False
    
    def _is_critical_error_requiring_replan(self, error_category: str, error_info: Dict[str, Any]) -> bool:
        """判断是否是需要重新规划的关键错误"""
        # 只有少数关键错误才需要replan
        
        # 任务规划错误，需要重新规划
        if error_category == ErrorCategory.TASK_PLANNING_ERROR.value:
            return True
            
        # 严重的权限错误（如网站结构完全改变）
        if error_category == ErrorCategory.PERMISSION_ERROR.value:
            error_message = error_info.get('message', '').lower()
            if any(keyword in error_message for keyword in ['site structure changed', '网站结构改变', 'page not found', '404']):
                return True
                
        # 严重的网络错误（如网站不存在）
        if error_category == ErrorCategory.NETWORK_ERROR.value:
            error_message = error_info.get('message', '').lower()
            if any(keyword in error_message for keyword in ['site not found', '网站不存在', 'domain not found']):
                return True
                
        return False

    async def _parse_task_decomposition_response(self, response_text: str) -> List[Dict]:
        """解析任务拆分响应"""
        try:
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            elif response_text.strip().startswith("```"):
                response_text = response_text.strip()[3:-3].strip()
            
            response_dict = json.loads(response_text)
            subtasks = response_dict.get("subtasks", [])
            
            # 验证子任务格式
            valid_subtasks = []
            for task in subtasks:
                if isinstance(task, dict) and "description" in task:
                    valid_subtasks.append({
                        "id": task.get("id", len(valid_subtasks) + 1),
                        "description": task.get("description", ""),
                        "status": "pending"
                    })
            
            return valid_subtasks
            
        except Exception as e:
            logger.error(f"解析任务拆分响应失败: {e}")
            return []

    async def _parse_action_execution_response(self, response_text: str) -> Tuple[List[Dict], Dict]:
        """解析动作执行响应，返回动作列表"""
        try:
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            elif response_text.strip().startswith("```"):
                response_text = response_text.strip()[3:-3].strip()
            
            response_dict = json.loads(response_text)
            actions = response_dict.get("actions", [])
            
            # 验证动作格式
            valid_actions = []
            has_finish_action = False
            
            for action in actions:
                if isinstance(action, dict) and "action" in action:
                    args = action.get("args", {})
                    if not isinstance(args, dict):
                        args = {}
                    
                    valid_action = {
                        "action": action["action"],
                        "args": args
                    }
                    
                    # 检测finish动作
                    if action["action"] == "finish":
                        has_finish_action = True
                        continue
                    
                    valid_actions.append(valid_action)
            
            # 构建状态信息，基于是否有finish动作
            status_info = {
                "has_finish_action": has_finish_action,
                "subtask_status": "completed" if has_finish_action else "in_progress"
            }
            
            return valid_actions, status_info
            
        except Exception as e:
            logger.error(f"解析动作执行响应失败: {e}")
            logger.error(f"响应内容: {response_text}")
            return [], {"has_finish_action": False, "subtask_status": "in_progress"}

    async def _parse_reflection_response(self, response_text: str) -> Dict:
        """解析反思响应"""
        try:
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            elif response_text.strip().startswith("```"):
                response_text = response_text.strip()[3:-3].strip()
            
            response_dict = json.loads(response_text)
            
            return {
                "subtask_completed": response_dict.get("subtask_completed", False),
                "next_decision": response_dict.get("next_decision", "continue_current")
            }
            
        except Exception as e:
            logger.error(f"解析反思响应失败: {e}")
            return {
                "subtask_completed": False,
                "next_decision": "exception_handling"
            }

    async def _parse_exception_handling_response(self, response_text: str) -> Dict:
        """解析异常处理响应"""
        try:
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            elif response_text.strip().startswith("```"):
                response_text = response_text.strip()[3:-3].strip()
            
            response_dict = json.loads(response_text)
            
            return {
                "action": response_dict.get("action", "retry")
            }
            
        except Exception as e:
            logger.error(f"解析异常处理响应失败: {e}")
            return {"action": "retry"}

    def _build_task_decomposition_prompt(self, original_task: str, error_info: Dict, 
                                        executed_actions: List, previous_node_result: Dict, 
                                        previous_subtasks: List = None) -> Message:
        """构建任务拆分的prompt"""
        previous_info = ""
        if previous_node_result:
            previous_info = f"""
=== 前置节点信息 ===
节点类型: {previous_node_result.get('node_type', '未知')}
执行结果: {previous_node_result.get('result', '无结果')}
"""

        previous_subtasks_info = ""
        if previous_subtasks and len(previous_subtasks) > 0:
            subtasks_summary = [task.get("description") for task in previous_subtasks]
            previous_subtasks_info = f"""
=== 上次任务拆分 ===
上次拆分: {subtasks_summary}
请避免重复相同的拆分策略。
"""

        execution_summary = "无"
        if executed_actions:
            recent_actions = executed_actions[-3:]
            execution_summary = " -> ".join([action.get('action', 'unknown') for action in recent_actions])

        error_summary = "无"
        if error_info:
            error_summary = f"{error_info.get('type', 'unknown')}: {error_info.get('message', 'no details')}"

        content = TASK_DECOMPOSITION_TEMPLATE.format(
            original_task=original_task,
            previous_info=previous_info,
            previous_subtasks_info=previous_subtasks_info,
            error_summary=error_summary,
            execution_summary=execution_summary
        )
        
        return Message(role="system", content=content)
    
    def _build_action_execution_prompt(self, current_subtask: Dict, current_url: str, 
                                     current_title: str, available_tabs: List, 
                                     page_elements: Any, executed_actions: List,
                                     reflection_result: Dict, previous_node_result: Dict,
                                     browser_actions: List[Dict]) -> Message:
        """构建动作执行的prompt"""
        previous_info = ""
        if previous_node_result:
            previous_info = f"""
=== 前置节点信息 ===
节点类型: {previous_node_result.get('node_type', '未知')}
执行结果: {previous_node_result.get('result', '无结果')}
"""

        page_info = "无法获取页面元素信息"
        if page_elements and isinstance(page_elements, dict):
            page_info = page_elements.get("markdown", "页面元素信息格式错误")

        subtask_info = current_subtask.get('description', 'no description')

        reflection_summary = "无"
        if reflection_result:
            next_decision = reflection_result.get('next_decision', 'continue_current')
            reflection_summary = f"反思建议: {next_decision}"

        recent_actions_summary = "无"
        if executed_actions:
            recent_actions = executed_actions[-3:]
            recent_actions_summary = " -> ".join([action.get('action', 'unknown') for action in recent_actions])

        memory_solution = self._get_memory_solution(current_subtask, executed_actions)

        content = ACTION_EXECUTION_TEMPLATE.format(
            subtask_info=subtask_info,
            previous_info=previous_info,
            current_url=current_url,
            current_title=current_title,
            reflection_summary=reflection_summary,
            page_info=page_info,
            actions_list=browser_actions,
            recent_actions_summary=recent_actions_summary,
            memory_solution=memory_solution
        )
        
        return Message(role="system", content=content)
    
    def _build_reflection_prompt(self, current_subtask: Dict, executed_actions: List,
                               last_action_result: Dict, current_url: str, 
                               current_title: str, page_elements: Any, 
                               subtasks_list: List, current_subtask_index: int,
                               light_reflection_result: Dict) -> Message:
        """构建反思的prompt"""
        page_info = "无法获取页面元素信息"
        if page_elements and isinstance(page_elements, dict):
            page_info = page_elements.get("markdown", "页面元素信息格式错误")
        
        current_task_info = current_subtask.get('description', 'no description')
        progress_info = f"第 {current_subtask_index + 1}/{len(subtasks_list)} 个子任务"
        
        recent_actions_summary = "无"
        if executed_actions:
            recent_actions = executed_actions[-3:]
            recent_actions_summary = " -> ".join([action.get('action', 'unknown') for action in recent_actions])
        
        last_result_summary = "无结果"
        if last_action_result:
            action_name = last_action_result.get('action', 'unknown')
            success = last_action_result.get('success', False)
            last_result_summary = f"{action_name}: {'成功' if success else '失败'}"
        
        light_reflection_summary = "无轻度反思"
        if light_reflection_result:
            light_reflection_summary = f"轻度反思结果: {light_reflection_result.get('message', '无消息')}"

        content = REFLECTION_TEMPLATE.format(
            current_task_info=current_task_info,
            progress_info=progress_info,
            recent_actions_summary=recent_actions_summary,
            last_result_summary=last_result_summary,
            light_reflection_summary=light_reflection_summary,
            current_url=current_url,
            current_title=current_title,
            page_info=page_info
        )
        
        return Message(role="system", content=content)
    
    def _build_exception_handling_prompt(self, current_subtask: Dict, error_info: Dict,
                                       executed_actions: List, retry_count: int,
                                       previous_node_result: Dict, error_category: str) -> Message:
        """构建异常处理的prompt"""
        previous_info = ""
        if previous_node_result:
            previous_info = f"""
=== 前置节点信息 ===
节点类型: {previous_node_result.get('node_type', '未知')}
执行结果: {previous_node_result.get('result', '无结果')}
"""

        subtask_info = current_subtask.get('description', 'no description')

        error_summary = "无错误信息"
        if error_info:
            error_summary = f"{error_info.get('type', 'unknown')}: {error_info.get('message', 'no details')}"

        execution_summary = "无"
        if executed_actions:
            recent_actions = executed_actions[-3:]
            execution_summary = " -> ".join([action.get('action', 'unknown') for action in recent_actions])

        memory_error_solution = self._get_memory_error_solution(error_info, error_category)

        content = EXCEPTION_HANDLING_TEMPLATE.format(
            subtask_info=subtask_info,
            previous_info=previous_info,
            error_summary=error_summary,
            error_category=error_category,
            execution_summary=execution_summary,
            retry_count=retry_count,
            memory_error_solution=memory_error_solution
        )
        
        return Message(role="system", content=content)
    
    def _get_memory_solution(self, current_subtask: Dict, executed_actions: List) -> str:
        """从记忆中获取近似解决方案（预留接口）"""
        # TODO: 实现从记忆中搜索相似任务的解决方案
        return "=== 记忆中的解决方案 ===\n暂无相关解决方案"
    
    def _get_memory_error_solution(self, error_info: Dict, error_category: str) -> str:
        """从记忆中获取类似错误的解决方案（预留接口）"""
        # TODO: 实现从记忆中搜索相似错误的解决方案
        return "=== 记忆中的错误解决方案 ===\n暂无相关错误解决方案"

    async def _call_llm(self, messages: List[Message]) -> str:
        """调用LLM"""
        try:
            llm_response_raw = await self.llm.chat_completion(messages, stream=False)
            
            if isinstance(llm_response_raw, str):
                return llm_response_raw
            else:
                response_text = ""
                async for chunk in llm_response_raw:
                    response_text += chunk
                return response_text
                
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise e

    async def run_task(self, task_description: str):
        """
        执行给定的任务
        
        Args:
            task_description: 任务的文本描述
        """
        logger.info(f"开始执行任务: {task_description}")
        
        # 确保浏览器已启动
        await self.browser_manager.launch_browser()
        
        # 初始化状态 - 使用Pydantic模型
        agent_model = AgentStateModel(original_task=task_description)
        initial_state = AgentState(
            model=agent_model
        )
        
        # 添加初始消息到memory
        self.memory.add_message(Message(role="user", content=f"开始新任务: {task_description}"))
        
        try:
            # 运行状态图
            final_state = await self.graph.ainvoke(initial_state)
            
            # 处理最终结果
            final_model = final_state["model"]
            if final_model.control_flow_state.is_completed:
                logger.info("任务执行完成")
                self.memory.add_message(Message(role="assistant", content="任务执行完成"))
            elif final_model.control_flow_state.need_human_intervention:
                logger.info("任务需要人工干预")
                self.memory.add_message(Message(role="assistant", content="任务需要人工干预"))
            else:
                logger.info("任务执行结束（可能未完全完成）")
                self.memory.add_message(Message(role="assistant", content="任务执行结束"))
                
        except Exception as e:
            logger.error(f"任务执行过程中发生错误: {e}")
            self.memory.add_message(Message(role="assistant", content=f"任务执行失败: {str(e)}"))
        
        logger.info(f"任务 '{task_description}' 执行结束")