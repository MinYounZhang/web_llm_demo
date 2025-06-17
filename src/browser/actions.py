from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Awaitable, Optional, Union
import random
import functools
import numpy as np
from bs4 import NavigableString, BeautifulSoup
from patchright.async_api import Page, ElementHandle, Locator
from pydantic import BaseModel, Field, field_validator, ValidationError
from src.config import logger, config
from src.error_management import with_timeout_and_retry
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio

# Pydantic基础模型类
class ActionParams(BaseModel):
    """Action参数的基础模型"""
    model_config = {"extra": "forbid", "validate_assignment": True}

class ActionResult(BaseModel):
    """Action执行结果的标准模型"""
    success: bool = Field(..., description="动作是否执行成功")
    message: str = Field(..., description="执行结果描述")
    data: Optional[Any] = Field(None, description="具体的返回数据")
    error: Optional[str] = Field(None, description="错误信息（仅在失败时）")
    error_type: Optional[str] = Field(None, description="错误类型")

# 为各个Action定义参数模型
class NavigateParams(ActionParams):
    """导航动作参数"""
    url: str = Field(..., description="要导航到的URL")

class ClickParams(ActionParams):
    """点击动作参数"""
    element_id: Optional[str] = Field(None, description="元素ID（优先使用）")
    selector: Optional[str] = Field(None, description="CSS选择器（备选）")
    force: bool = Field(False, description="是否强制点击")
    use_js: bool = Field(False, description="是否使用JavaScript点击")
    
    def model_post_init(self, __context):
        """模型初始化后验证"""
        if not self.element_id and not self.selector:
            raise ValueError("必须提供element_id或selector中的一个")

class TypeParams(ActionParams):
    """输入文本动作参数"""
    element_id: Optional[str] = Field(None, description="元素ID（优先使用）")
    selector: Optional[str] = Field(None, description="CSS选择器（备选）")
    text: str = Field(..., description="要输入的文本")
    delay: int = Field(50, ge=0, le=1000, description="按键间延迟（毫秒）")
    
    def model_post_init(self, __context):
        """模型初始化后验证"""
        if not self.element_id and not self.selector:
            raise ValueError("必须提供element_id或selector中的一个")

class WaitParams(ActionParams):
    """等待动作参数 - 支持多种等待方式"""
    # 基础等待
    duration_ms: Optional[int] = Field(None, ge=0, description="等待时间（毫秒）")
    
    # 元素等待
    selector: Optional[str] = Field(None, description="等待元素出现的选择器")
    element_state: str = Field("visible", pattern="^(attached|detached|visible|hidden)$", description="元素状态")
    
    # 页面状态等待
    load_state: Optional[str] = Field(None, pattern="^(load|domcontentloaded|networkidle)$", description="页面加载状态")
    
    # 网络等待
    wait_for_response: Optional[str] = Field(None, description="等待特定响应的URL模式")
    wait_for_request: Optional[str] = Field(None, description="等待特定请求的URL模式")
    
    # 函数等待
    wait_for_function: Optional[str] = Field(None, description="等待JavaScript函数返回true")
    
    # 超时设置
    timeout_ms: int = Field(30000, ge=1000, le=300000, description="等待超时时间（毫秒）")
    
    def model_post_init(self, __context):
        """模型初始化后验证"""
        wait_conditions = [
            self.duration_ms is not None,
            self.selector is not None,
            self.load_state is not None,
            self.wait_for_response is not None,
            self.wait_for_request is not None,
            self.wait_for_function is not None
        ]
        
        if not any(wait_conditions):
            raise ValueError("必须提供至少一种等待条件：duration_ms, selector, load_state, wait_for_response, wait_for_request, wait_for_function")

class GetTextParams(ActionParams):
    """获取文本动作参数"""
    element_id: Optional[str] = Field(None, description="元素ID（优先使用）")
    selector: Optional[str] = Field(None, description="CSS选择器（备选）")
    
    def model_post_init(self, __context):
        """模型初始化后验证"""
        if not self.element_id and not self.selector:
            raise ValueError("必须提供element_id或selector中的一个")

class ScrollParams(ActionParams):
    """滚动动作参数"""
    direction: str = Field(..., pattern="^(up|down|left|right)$", description="滚动方向")
    distance: int = Field(500, ge=1, le=5000, description="滚动距离（像素）")

class GetAllTabsParams(ActionParams):
    """获取所有标签页参数（无参数）"""
    pass

class SwitchTabParams(ActionParams):
    """切换标签页参数"""
    tab_id: Optional[int] = Field(None, ge=0, description="标签页ID")
    tab_index: Optional[int] = Field(None, ge=0, description="标签页索引（备选参数名）")
    
    def model_post_init(self, __context):
        """模型初始化后验证"""
        if self.tab_id is None and self.tab_index is None:
            raise ValueError("必须提供tab_id或tab_index中的一个")

class NewTabParams(ActionParams):
    """新建标签页参数"""
    url: Optional[str] = Field(None, description="在新标签页中导航到的URL（可选）")

class CloseTabParams(ActionParams):
    """关闭标签页参数"""
    tab_id: Optional[int] = Field(None, ge=0, description="要关闭的标签页ID")
    tab_index: Optional[int] = Field(None, ge=0, description="要关闭的标签页索引（备选参数名）")
    
    def model_post_init(self, __context):
        """模型初始化后验证"""
        if self.tab_id is None and self.tab_index is None:
            raise ValueError("必须提供tab_id或tab_index中的一个")

class KeyboardInputParams(ActionParams):
    """键盘输入参数"""
    keys: Union[str, List[str]] = Field(..., description="按键列表，支持组合键")

class SaveToFileParams(ActionParams):
    """保存文件参数"""
    filename: str = Field(..., min_length=1, description="文件名")
    content: str = Field(..., description="要保存的内容")
    format: str = Field("txt", pattern="^(txt|html|json)$", description="文件格式")

class RefreshPageParams(ActionParams):
    """刷新页面参数（无参数）"""
    pass

class WebSearchParams(ActionParams):
    """网页搜索参数"""
    query: str = Field(..., min_length=1, description="搜索查询字符串")
    search_engine: str = Field("auto", description="搜索引擎")

class GetAllElementsParams(ActionParams):
    """获取页面元素参数"""
    method: str = Field("dom", pattern="^(dom|aom|xpath)$", description="遍历方法")
    enable_highlight: bool = Field(True, description="是否高亮元素")

# 浏览器导航参数
class BrowserBackParams(ActionParams):
    """浏览器后退参数（无参数）"""
    pass

class BrowserForwardParams(ActionParams):
    """浏览器前进参数（无参数）"""
    pass

# 保存到文件参数
class FinishParams(ActionParams):
    """任务完成参数（无参数）"""
    pass


def with_human_like_mouse_move(func: Callable[[Any, Page, Any], Awaitable[Any]]) -> Callable[[Any, Page, Any], Awaitable[Any]]:
    """装饰器：在执行动作前模拟人类鼠标移动。
    
    Args:
        func: 要装饰的异步函数
        
    Returns:
        装饰后的异步函数
    """
    @functools.wraps(func)
    async def wrapper(self: Action, page: Page, **kwargs: Any) -> Any:
        # 如果动作涉及特定元素，则移动到该元素
        element_id = kwargs.get("element_id")
        selector = kwargs.get("selector")
        
        if element_id and hasattr(page, '_agent_element_handles'):
            # 使用ElementHandle进行鼠标移动
            handle = page._agent_element_handles.get(element_id)
            if handle:
                try:
                    await handle.hover(timeout=5000)
                    # 在元素附近进行小范围随机移动
                    bounding_box = await handle.bounding_box()
                    if bounding_box:
                        x = bounding_box['x'] + bounding_box['width'] / 2 + random.uniform(-5, 5)
                        y = bounding_box['y'] + bounding_box['height'] / 2 + random.uniform(-5, 5)
                        await page.mouse.move(x, y, steps=random.randint(5, 10))
                except Exception as e:
                    logger.warning(f"类人鼠标移动到ElementHandle时出错: {e}")
        elif selector:
            try:
                element = page.locator(selector).first
                await element.hover(timeout=5000)
                # 在元素附近进行小范围随机移动
                bounding_box = await element.bounding_box()
                if bounding_box:
                    x = bounding_box['x'] + bounding_box['width'] / 2 + random.uniform(-5, 5)
                    y = bounding_box['y'] + bounding_box['height'] / 2 + random.uniform(-5, 5)
                    await page.mouse.move(x, y, steps=random.randint(5, 10))
            except Exception as e:
                logger.warning(f"类人鼠标移动到元素时出错: {e}")
        else:
            # 随机移动到屏幕上的某个点
            try:
                viewport_size = page.viewport_size
                if viewport_size:
                    x = random.uniform(0, viewport_size['width'])
                    y = random.uniform(0, viewport_size['height'])
                    await page.mouse.move(x, y, steps=random.randint(5, 15))
            except Exception as e:
                logger.warning(f"类人随机鼠标移动时出错: {e}")

        await page.wait_for_timeout(random.randint(100, 300))  # 移动后的短暂暂停
        return await func(self, page, **kwargs)

    return wrapper

class Action(ABC):
    """浏览器操作的抽象基类。"""
    def __init__(self, name: str, description: str, params_model: BaseModel = None):
        self.name = name
        self.description = description
        self.params_model = params_model

    @abstractmethod
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        """执行动作。

        Args:
            page: Playwright 页面对象。
            **kwargs: 动作所需的参数。

        Returns:
            ActionResult: 标准格式的动作执行结果
        """
        pass

    def validate_params(self, **kwargs: Any) -> ActionParams:
        """验证并返回参数"""
        if self.params_model is None:
            return ActionParams()
        
        try:
            return self.params_model(**kwargs)
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                field = error.get('loc', ['unknown'])[0]
                msg = error.get('msg', 'validation error')
                error_details.append(f"{field}: {msg}")
            
            raise ValueError(f"参数验证失败: {'; '.join(error_details)}")

    def get_params_schema(self) -> Dict[str, Any]:
        """获取参数的JSON Schema"""
        if self.params_model is None:
            return {}
        return self.params_model.model_json_schema()

    def to_dict(self) -> Dict[str, Any]:
        """返回动作的字典表示，用于 LLM 理解。"""
        schema = self.get_params_schema()
        properties = schema.get('properties', {})
        
        # 转换为简化的参数描述
        args_desc = {}
        for field_name, field_info in properties.items():
            args_desc[field_name] = field_info.get('description', field_name)
        
        return {
            "name": self.name,
            "args": args_desc,
            "description": self.description 
        }



class NavigateAction(Action):
    """导航到指定 URL 的动作。"""
    def __init__(self):
        super().__init__(
            name="navigate_to_url",
            description="Navigate to a specified URL and wait for page to load completely. Automatically waits for network idle to handle slow-loading websites.",
            params_model=NavigateParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        try:
            logger.info(f"导航到 URL: {params.url}")
            await page.goto(params.url, timeout=config.browser_config.timeout, wait_until="domcontentloaded")
            logger.info(f"已导航到: {page.url}")
            return ActionResult(
                success=True,
                message=f"已成功导航到: {page.url}",
                data={"final_url": page.url}
            )
        except Exception as e:
            error_msg = f"导航到 {params.url} 失败: {str(e)}"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e)
            )

class ClickAction(Action):
    """点击指定元素，支持多种定位方式和点击策略。"""
    def __init__(self):
        super().__init__(
            name="click_element",
            description="Click on an element",
            params_model=ClickParams
        )

    @with_human_like_mouse_move
    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        element_id = params.element_id
        selector = params.selector
        force_click = params.force
        use_js = params.use_js
        
        # 存储原始element_id用于日志
        original_element_id = element_id
        
        # 首先验证元素是否在页面元素列表中
        if element_id:
            element_exists = await self._verify_element_in_page(page, element_id)
            if not element_exists:
                logger.warning(f"元素 {element_id} 不在当前页面元素列表中，可能需要刷新页面元素")
                # 尝试刷新页面元素
                try:
                    await page.evaluate("window.location.reload()")
                    await page.wait_for_load_state("domcontentloaded")
                    await asyncio.sleep(1)  # 等待页面稳定
                except Exception as e:
                    logger.warning(f"刷新页面失败: {e}")
        
        # 尝试多种定位策略
        located_element = None
        
        # 使用增强的点击策略
        try:
            result = await self._enhanced_click_strategies(page, element_id, selector, force_click, use_js)
            
            # 如果点击成功，等待页面加载完成
            if result.success:
                try:
                    logger.info("点击成功，等待页面加载完成...")
                    # 等待页面加载完成，防止后续动作获取到空白页面
                    await page.wait_for_load_state("domcontentloaded", timeout=10000)
                    # 额外等待网络空闲，确保动态内容加载完成
                    await page.wait_for_load_state("networkidle", timeout=15000)
                    # 给页面一些时间渲染内容
                    await asyncio.sleep(1)
                    logger.info("页面加载完成")
                except Exception as e:
                    logger.warning(f"等待页面加载时出现警告: {e}")
                    # 即使等待失败，也不认为点击失败，只是给个警告
                    result.message += f" (页面加载等待警告: {str(e)})"
            
            return result
            
        except Exception as e:
            # 如果增强策略也失败，使用原有的回退逻辑作为最终保险
            logger.warning(f"增强点击策略失败，尝试最后的回退方案: {e}")
            
            # 最后的回退策略：智能重新定位
            if element_id:
                try:
                    success = await self._try_smart_fallback_click(page, element_id, force_click, use_js)
                    if success:
                        # 回退策略成功后也等待页面加载
                        try:
                            await page.wait_for_load_state("domcontentloaded", timeout=10000)
                            await page.wait_for_load_state("networkidle", timeout=15000)
                            await asyncio.sleep(1)
                        except Exception:
                            pass  # 回退策略的等待失败不影响结果
                        
                        return ActionResult(
                            success=True,
                            message=f"最终回退策略成功: {element_id}"
                        )
                except Exception as fallback_e:
                    logger.error(f"最终回退策略也失败: {fallback_e}")
            
            # 所有策略都失败
            error_msg = f"所有点击策略（包括增强策略和回退策略）都失败: {original_element_id or selector}"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="click_all_failed"
            )

    async def _try_element_handle_click(self, handle: Any, element_id: str, force_click: bool, use_js: bool) -> bool:
        """尝试使用ElementHandle进行点击，包含有效性检查"""
        try:
            # 检查ElementHandle是否仍然有效
            try:
                await handle.is_visible()
            except Exception as e:
                logger.warning(f"ElementHandle已失效: {e}")
                return False
            
            # 滚动到元素位置
            try:
                await handle.scroll_into_view_if_needed()
                await handle.page.wait_for_timeout(200)
            except Exception as e:
                logger.debug(f"滚动到元素失败: {e}")
            
            # 尝试点击
            if use_js:
                await handle.evaluate("el => el.click()")
                logger.info(f"ElementHandle JavaScript点击成功: {element_id}")
            else:
                click_options = {"timeout": config.browser_config.timeout}
                if force_click:
                    click_options["force"] = True
                await handle.click(**click_options)
                logger.info(f"ElementHandle常规点击成功: {element_id}")
            
            return True
            
        except Exception as e:
            logger.warning(f"ElementHandle点击失败: {e}")
            return False

    async def _try_locator_click(self, locator: Any, selector: str, force_click: bool, use_js: bool) -> bool:
        """尝试使用Locator进行点击，包含多种点击策略"""
        try:
            # 检查元素是否存在和可见
            if await locator.count() == 0:
                logger.debug(f"选择器未找到元素: {selector}")
                return False
            
            # 滚动到元素位置
            try:
                await locator.scroll_into_view_if_needed()
                await locator.page.wait_for_timeout(200)
            except Exception as e:
                logger.debug(f"滚动到元素失败: {e}")
            
            # 等待元素可见
            try:
                await locator.wait_for(state="visible", timeout=2000)
            except Exception as e:
                logger.debug(f"等待元素可见超时: {e}")
            
            # 尝试不同的点击方式
            click_strategies = []
            
            if use_js:
                click_strategies.append(("JavaScript点击", lambda: locator.evaluate("el => el.click()")))
            else:
                if force_click:
                    click_strategies.append(("强制点击", lambda: locator.click(force=True, timeout=config.browser_config.timeout)))
                else:
                    click_strategies.append(("常规点击", lambda: locator.click(timeout=config.browser_config.timeout)))
            
            # 备选策略
            if not use_js:
                click_strategies.append(("JavaScript备选点击", lambda: locator.evaluate("el => el.click()")))
            if not force_click:
                click_strategies.append(("强制备选点击", lambda: locator.click(force=True, timeout=config.browser_config.timeout)))
            
            # 高级策略
            click_strategies.append(("模拟事件点击", lambda: locator.evaluate("""
                el => {
                    const rect = el.getBoundingClientRect();
                    const x = rect.left + rect.width / 2;
                    const y = rect.top + rect.height / 2;
                    
                    ['mousedown', 'mouseup', 'click'].forEach(eventType => {
                        const event = new MouseEvent(eventType, {
                            view: window,
                            bubbles: true,
                            cancelable: true,
                            clientX: x,
                            clientY: y
                        });
                        el.dispatchEvent(event);
                    });
                }
            """)))
            
            # 逐一尝试策略
            for strategy_name, strategy_func in click_strategies:
                try:
                    await strategy_func()
                    logger.info(f"{strategy_name}成功: {selector}")
                    return True
                except Exception as e:
                    logger.debug(f"{strategy_name}失败: {e}")
                    continue
            
            logger.warning(f"所有点击策略都失败: {selector}")
            return False
            
        except Exception as e:
            logger.warning(f"Locator点击过程出错: {e}")
            return False

    async def _try_smart_fallback_click(self, page: Page, element_id: str, force_click: bool, use_js: bool) -> bool:
        """智能回退策略：尝试根据element_id重新定位元素"""
        try:
            logger.info(f"执行智能回退策略，重新定位元素: {element_id}")
            
            # 尝试从element_id中提取信息并重新定位
            fallback_selectors = []
            
            # 根据element_id前缀判断元素类型
            if element_id.startswith("element_"):
                # 默认方法的元素，尝试常见的交互元素选择器
                fallback_selectors.extend([
                    "button:visible",
                    "input:visible", 
                    "a[href]:visible",
                    "[role='button']:visible",
                    "[onclick]:visible"
                ])
            elif element_id.startswith("aom_elem_"):
                # AOM方法的元素
                fallback_selectors.extend([
                    "[role='button']:visible",
                    "[role='link']:visible", 
                    "button:visible",
                    "a[href]:visible"
                ])
            elif element_id.startswith("xpath_elem_"):
                # XPath方法的元素
                fallback_selectors.extend([
                    "button:visible",
                    "input:visible",
                    "a[href]:visible"
                ])
            elif element_id.startswith("fallback_element_"):
                # 回退方法的元素
                fallback_selectors.extend([
                    "button:visible",
                    "input:visible",
                    "a[href]:visible",
                    "[role='button']:visible"
                ])
            
            # 尝试每个回退选择器
            for selector in fallback_selectors:
                try:
                    elements = page.locator(selector)
                    count = await elements.count()
                    
                    if count > 0:
                        # 提取element_id中的索引（如果有）
                        element_index = self._extract_element_index(element_id)
                        
                        # 选择对应索引的元素，如果索引无效则选择第一个
                        target_index = min(element_index, count - 1) if element_index >= 0 else 0
                        target_element = elements.nth(target_index)
                        
                        logger.info(f"回退策略找到元素: {selector}, 索引: {target_index}")
                        success = await self._try_locator_click(target_element, f"{selector}[{target_index}]", force_click, use_js)
                        if success:
                            return True
                        
                except Exception as e:
                    logger.debug(f"回退选择器失败: {selector}, 错误: {e}")
                    continue
            
            logger.warning(f"智能回退策略未能定位到元素: {element_id}")
            return False
            
        except Exception as e:
            logger.error(f"智能回退策略执行出错: {e}")
            return False

    def _extract_element_index(self, element_id: str) -> int:
        """从element_id中提取索引数字"""
        try:
            # 提取ID中的数字部分
            import re
            numbers = re.findall(r'\d+', element_id)
            if numbers:
                return int(numbers[-1]) - 1  # 转换为0基索引
            return 0
        except Exception:
            return 0

    async def _verify_element_in_page(self, page: Page, element_id: str) -> bool:
        """验证元素是否在当前页面元素列表中"""
        try:
            # 检查是否有缓存的页面元素信息
            if hasattr(page, '_page_elements_cache'):
                elements = page._page_elements_cache.get('elements', [])
                return any(elem.get('id') == element_id for elem in elements)
            
            # 如果没有缓存，尝试直接检查DOM
            exists = await page.evaluate(f"""() => {{
                return document.querySelector('[data-agent-id="{element_id}"]') !== null;
            }}""")
            return exists
        except Exception as e:
            logger.debug(f"验证元素存在性失败: {e}")
            return False

    async def _enhanced_click_strategies(self, page: Page, element_id: str, selector: str, force_click: bool, use_js: bool) -> ActionResult:
        """增强的点击策略，包含多种方法和智能回退"""
        strategies = [
            ("ElementHandle直接点击", self._strategy_element_handle),
            ("选择器定位点击", self._strategy_selector_click),
            ("坐标点击", self._strategy_coordinate_click),
            ("事件模拟点击", self._strategy_event_simulation),
            ("JavaScript强制点击", self._strategy_js_force_click),
            ("滚动后重试", self._strategy_scroll_and_retry),
            ("等待后重试", self._strategy_wait_and_retry)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"尝试策略: {strategy_name}")
                result = await strategy_func(page, element_id, selector, force_click, use_js)
                if result.success:
                    logger.info(f"策略 {strategy_name} 成功")
                    return result
                else:
                    logger.debug(f"策略 {strategy_name} 失败: {result.message}")
            except Exception as e:
                logger.debug(f"策略 {strategy_name} 异常: {e}")
                continue
        
        # 所有策略都失败
        return ActionResult(
            success=False,
            message=f"所有点击策略都失败: {element_id or selector}",
            error="All click strategies failed",
            error_type="click_all_strategies_failed"
        )

    async def _strategy_element_handle(self, page: Page, element_id: str, selector: str, force_click: bool, use_js: bool) -> ActionResult:
        """策略1: ElementHandle直接点击"""
        if not element_id or not hasattr(page, '_agent_element_handles'):
            raise Exception("ElementHandle不可用")
        
        handle = page._agent_element_handles.get(element_id)
        if not handle:
            raise Exception("ElementHandle不存在")
        
        # 检查ElementHandle有效性
        try:
            await handle.is_visible()
        except Exception:
            raise Exception("ElementHandle已失效")
        
        # 滚动到元素
        await handle.scroll_into_view_if_needed()
        await page.wait_for_timeout(300)
        
        # 执行点击
        if use_js:
            await handle.evaluate("el => el.click()")
        else:
            click_options = {"timeout": config.browser_config.timeout}
            if force_click:
                click_options["force"] = True
            await handle.click(**click_options)
        
        return ActionResult(success=True, message=f"ElementHandle点击成功: {element_id}")

    async def _strategy_selector_click(self, page: Page, element_id: str, selector: str, force_click: bool, use_js: bool) -> ActionResult:
        """策略2: 选择器定位点击"""
        selectors_to_try = []
        
        # 如果有selector参数，优先使用它
        if selector:
            # 对于DOM生成的CSS选择器，先清理再使用
            cleaned_selector = self._clean_css_selector(selector)
            selectors_to_try.append(cleaned_selector)
        
        # 如果有element_id，尝试不同的data属性格式
        if element_id:
            # 直接使用element_id作为data-agent-id
            selectors_to_try.append(f'[data-agent-id="{element_id}"]')
            
            # 如果element_id看起来像uniqueId格式，也尝试原始格式
            if "_" in element_id:
                selectors_to_try.extend([
                    f'[data-aom-id="{element_id}"]',
                    f'[data-xpath-id="{element_id}"]'
                ])
        
        if not selectors_to_try:
            raise Exception("没有可用的选择器")
        
        logger.info(f"尝试选择器列表: {selectors_to_try}")
        
        for sel in selectors_to_try:
            try:
                # 对于某些特殊选择器，先做验证
                if not self._is_valid_css_selector(sel):
                    logger.debug(f"无效的CSS选择器，跳过: {sel}")
                    continue
                
                # 检查元素是否存在
                element_count = await page.locator(sel).count()
                if element_count == 0:
                    logger.debug(f"选择器未找到元素: {sel}")
                    continue
                
                logger.info(f"选择器找到 {element_count} 个元素: {sel}")
                element = page.locator(sel).first
                
                # 等待元素可见和可点击
                try:
                    await element.wait_for(state="visible", timeout=3000)
                except Exception as e:
                    logger.debug(f"等待元素可见超时: {e}")
                
                # 滚动到元素位置
                try:
                    await element.scroll_into_view_if_needed()
                    await page.wait_for_timeout(500)
                except Exception as e:
                    logger.debug(f"滚动到元素失败: {e}")
                
                # 执行点击
                if use_js:
                    await element.evaluate("el => el.click()")
                    logger.info(f"JavaScript点击成功: {sel}")
                else:
                    click_options = {"timeout": config.browser_config.timeout}
                    if force_click:
                        click_options["force"] = True
                    await element.click(**click_options)
                    logger.info(f"常规点击成功: {sel}")
                
                return ActionResult(success=True, message=f"选择器点击成功: {sel}")
                
            except Exception as e:
                logger.debug(f"选择器 {sel} 点击失败: {e}")
                continue
        
        raise Exception(f"所有选择器都失败: {selectors_to_try}")
    
    def _clean_css_selector(self, selector: str) -> str:
        """清理CSS选择器，移除可能导致问题的伪选择器"""
        import re
        
        # 移除可能有问题的伪选择器
        problematic_pseudo_selectors = [
            r':first-child',
            r':last-child',
            r':nth-child\([^)]*\)',
            r':nth-last-child\([^)]*\)',
            r':first-of-type',
            r':last-of-type',
            r':nth-of-type\([^)]*\)',
            r':nth-last-of-type\([^)]*\)',
            r':text\([^)]*\)',
            r':contains\([^)]*\)'
        ]
        
        cleaned_selector = selector
        for pattern in problematic_pseudo_selectors:
            cleaned_selector = re.sub(pattern, '', cleaned_selector)
        
        # 清理多余的空格和连续的选择器组合符
        cleaned_selector = re.sub(r'\s+', ' ', cleaned_selector)
        cleaned_selector = re.sub(r'\s*>\s*', ' > ', cleaned_selector)
        cleaned_selector = re.sub(r'\s*\+\s*', ' + ', cleaned_selector)
        cleaned_selector = re.sub(r'\s*~\s*', ' ~ ', cleaned_selector)
        
        # 移除末尾可能的多余字符
        cleaned_selector = cleaned_selector.strip()
        
        # 如果清理后选择器为空或只包含空格，返回原选择器
        if not cleaned_selector.strip():
            logger.warning(f"选择器清理后为空，使用原选择器: {selector}")
            return selector
            
        if cleaned_selector != selector:
            logger.info(f"选择器已清理: '{selector}' -> '{cleaned_selector}'")
            
        return cleaned_selector

    async def _strategy_coordinate_click(self, page: Page, element_id: str, selector: str, force_click: bool, use_js: bool) -> ActionResult:
        """策略3: 坐标点击（基于缓存的元素位置）"""
        if not hasattr(page, '_page_elements_cache') or not element_id:
            raise Exception("没有元素位置缓存或元素ID")
        
        elements = page._page_elements_cache.get('elements', [])
        target_element = None
        
        for elem in elements:
            if elem.get('id') == element_id:
                target_element = elem
                break
        
        if not target_element:
            raise Exception("元素不在缓存中")
        
        pos = target_element.get('pos', [])
        if len(pos) < 4:
            raise Exception("元素位置信息不完整")
        
        x, y, w, h = pos
        click_x = x + w // 2
        click_y = y + h // 2
        
        # 滚动到目标位置
        await page.evaluate(f"window.scrollTo({click_x - 400}, {click_y - 300})")
        await page.wait_for_timeout(300)
        
        # 坐标点击
        await page.mouse.click(click_x, click_y)
        
        return ActionResult(success=True, message=f"坐标点击成功: ({click_x}, {click_y})")

    async def _strategy_event_simulation(self, page: Page, element_id: str, selector: str, force_click: bool, use_js: bool) -> ActionResult:
        """策略4: 事件模拟点击"""
        target_selector = None
        
        if element_id:
            target_selector = f'[data-agent-id="{element_id}"]'
        elif selector:
            target_selector = selector
        else:
            raise Exception("没有目标选择器")
        
        # 使用JavaScript模拟完整的鼠标事件序列
        click_result = await page.evaluate(f"""(selector) => {{
            const element = document.querySelector(selector);
            if (!element) return false;
            
            // 滚动到元素
            element.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            
            // 获取元素中心点
            const rect = element.getBoundingClientRect();
            const x = rect.left + rect.width / 2;
            const y = rect.top + rect.height / 2;
            
            // 模拟完整的鼠标事件序列
            const events = ['mousedown', 'mouseup', 'click'];
            for (const eventType of events) {{
                const event = new MouseEvent(eventType, {{
                    view: window,
                    bubbles: true,
                    cancelable: true,
                    clientX: x,
                    clientY: y,
                    button: 0
                }});
                element.dispatchEvent(event);
            }}
            
            // 额外触发focus和blur事件
            if (element.focus) element.focus();
            
            return true;
        }}""", target_selector)
        
        if click_result:
            return ActionResult(success=True, message=f"事件模拟点击成功: {target_selector}")
        else:
            raise Exception("事件模拟失败")

    async def _strategy_js_force_click(self, page: Page, element_id: str, selector: str, force_click: bool, use_js: bool) -> ActionResult:
        """策略5: JavaScript强制点击"""
        target_selector = None
        
        if element_id:
            target_selector = f'[data-agent-id="{element_id}"]'
        elif selector:
            target_selector = selector
        else:
            raise Exception("没有目标选择器")
        
        # 强制JavaScript点击，包含多种触发方式
        click_result = await page.evaluate(f"""(selector) => {{
            const element = document.querySelector(selector);
            if (!element) return false;
            
            // 滚动到元素
            element.scrollIntoView({{ behavior: 'instant', block: 'center' }});
            
            // 移除可能的阻止因素
            element.style.pointerEvents = 'auto';
            element.style.visibility = 'visible';
            element.style.display = 'block';
            
            // 多种点击方式
            try {{
                // 方式1: 直接调用click
                element.click();
                
                // 方式2: 如果是链接，尝试导航
                if (element.href) {{
                    window.location.href = element.href;
                }}
                
                // 方式3: 如果是表单元素，尝试提交
                if (element.form && element.type === 'submit') {{
                    element.form.submit();
                }}
                
                // 方式4: 触发自定义事件
                element.dispatchEvent(new Event('click', {{ bubbles: true }}));
                
                return true;
            }} catch (e) {{
                console.error('强制点击失败:', e);
                return false;
            }}
        }}""", target_selector)
        
        if click_result:
            return ActionResult(success=True, message=f"JavaScript强制点击成功: {target_selector}")
        else:
            raise Exception("JavaScript强制点击失败")

    async def _strategy_scroll_and_retry(self, page: Page, element_id: str, selector: str, force_click: bool, use_js: bool) -> ActionResult:
        """策略6: 滚动后重试"""
        # 先尝试滚动到页面不同位置
        scroll_positions = [
            "window.scrollTo(0, 0)",  # 顶部
            "window.scrollTo(0, document.body.scrollHeight / 2)",  # 中间
            "window.scrollTo(0, document.body.scrollHeight)",  # 底部
        ]
        
        for scroll_cmd in scroll_positions:
            try:
                await page.evaluate(scroll_cmd)
                await page.wait_for_timeout(500)
                
                # 尝试基本的选择器点击
                result = await self._strategy_selector_click(page, element_id, selector, force_click, use_js)
                return result
            except Exception as e:
                logger.debug(f"滚动位置 {scroll_cmd} 后点击失败: {e}")
                continue
        
        raise Exception("滚动后重试失败")

    async def _strategy_wait_and_retry(self, page: Page, element_id: str, selector: str, force_click: bool, use_js: bool) -> ActionResult:
        """策略7: 等待后重试"""
        wait_times = [1000, 2000, 3000]  # 1秒、2秒、3秒
        
        for wait_time in wait_times:
            try:
                logger.info(f"等待 {wait_time}ms 后重试")
                await page.wait_for_timeout(wait_time)
                
                # 等待页面稳定
                await page.wait_for_load_state("domcontentloaded")
                
                # 尝试选择器点击
                result = await self._strategy_selector_click(page, element_id, selector, force_click, use_js)
                return result
            except Exception as e:
                logger.debug(f"等待 {wait_time}ms 后点击失败: {e}")
                continue
        
        raise Exception("等待后重试失败")

class TypeAction(Action):
    """在指定元素中输入文本（优先使用ElementHandle）。"""
    def __init__(self):
        super().__init__(
            name="type_text",
            description="Type text into an element",
            params_model=TypeParams
        )

    @with_human_like_mouse_move
    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        element_id = params.element_id
        selector = params.selector
        text_to_type = params.text
        delay = params.delay

        handle = None
        if element_id and hasattr(page, '_agent_element_handles'):
            # 优先使用ElementHandle
            handle = page._agent_element_handles.get(element_id)
            if handle:
                logger.info(f"在元素 (ElementHandle) '{element_id}' 中输入文本: '{text_to_type}'，延迟: {delay}ms")
                try:
                    await handle.type(text_to_type, delay=delay, timeout=config.browser_config.timeout)
                    logger.info(f"已在元素 '{element_id}' 中输入文本。")
                    return ActionResult(
                        success=True,
                        message=f"已在元素 '{element_id}' 中输入文本。"
                    )
                except Exception as e:
                    logger.warning(f"使用ElementHandle输入失败，尝试使用选择器: {e}")
                    selector = f'[data-agent-id="{element_id}"]'

        if selector:
            logger.info(f"在元素 (选择器) '{selector}' 中输入文本: '{text_to_type}'，延迟: {delay}ms")
            try:
                element = page.locator(selector).first
                await element.type(text_to_type, delay=delay, timeout=config.browser_config.timeout)
                logger.info(f"已在元素 '{selector}' 中输入文本。")
                return ActionResult(
                    success=True,
                    message=f"已在元素 '{selector}' 中输入文本。"
                )
            except Exception as e:
                error_msg = f"在元素 '{selector}' 中输入文本失败: {str(e)}"
                logger.error(error_msg)
                return ActionResult(
                    success=False,
                    message=error_msg,
                    error=str(e),
                    error_type="type_text_failed"
                )
        else:
            # 这种情况不应该发生，因为Pydantic已经验证了参数
            error_msg = "参数验证失败：未提供element_id或selector"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error="No element locator provided",
                error_type="parameter_error"
            )

class WaitAction(Action):
    """等待指定时间、元素出现或页面状态变化。"""
    def __init__(self):
        super().__init__(
            name="wait",
            description="Wait for various conditions: duration, element state, page load state, network events, or JavaScript functions",
            params_model=WaitParams
        )
    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        try:
            # 1. 基础时间等待
            if params.duration_ms is not None:
                logger.info(f"等待 {params.duration_ms} 毫秒")
                await page.wait_for_timeout(params.duration_ms)
                return ActionResult(
                    success=True,
                    message=f"已完成等待 {params.duration_ms} 毫秒"
                )
            
            # 2. 元素状态等待
            elif params.selector is not None:
                logger.info(f"等待元素 '{params.selector}' 状态变为 '{params.element_state}'")
                await page.wait_for_selector(
                    params.selector, 
                    state=params.element_state,
                    timeout=params.timeout_ms
                )
                return ActionResult(
                    success=True,
                    message=f"元素 '{params.selector}' 已达到状态 '{params.element_state}'"
                )
            
            # 3. 页面加载状态等待
            elif params.load_state is not None:
                logger.info(f"等待页面加载状态: {params.load_state}")
                await page.wait_for_load_state(params.load_state, timeout=params.timeout_ms)
                return ActionResult(
                    success=True,
                    message=f"页面已达到加载状态: {params.load_state}"
                )
            
            # 4. 网络响应等待
            elif params.wait_for_response is not None:
                logger.info(f"等待响应: {params.wait_for_response}")
                async with page.expect_response(params.wait_for_response, timeout=params.timeout_ms) as response_info:
                    response = await response_info.value
                return ActionResult(
                    success=True,
                    message=f"已收到响应: {response.url}",
                    data={"url": response.url, "status": response.status}
                )
            
            # 5. 网络请求等待
            elif params.wait_for_request is not None:
                logger.info(f"等待请求: {params.wait_for_request}")
                async with page.expect_request(params.wait_for_request, timeout=params.timeout_ms) as request_info:
                    request = await request_info.value
                return ActionResult(
                    success=True,
                    message=f"已发送请求: {request.url}",
                    data={"url": request.url, "method": request.method}
                )
            
            # 6. JavaScript函数等待
            elif params.wait_for_function is not None:
                logger.info(f"等待JavaScript函数: {params.wait_for_function}")
                await page.wait_for_function(params.wait_for_function, timeout=params.timeout_ms)
                return ActionResult(
                    success=True,
                    message=f"JavaScript函数已返回true: {params.wait_for_function}"
                )
            
            else:
                # 这种情况不应该发生，因为Pydantic已经验证了参数
                error_msg = "参数验证失败：未提供有效的等待条件"
                logger.error(error_msg)
                return ActionResult(
                    success=False,
                    message=error_msg,
                    error="No valid wait condition provided",
                    error_type="parameter_error"
                )
                
        except Exception as e:
            error_msg = f"等待操作失败: {str(e)}"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="wait_timeout" if "timeout" in str(e).lower() else "wait_failed"
            )

class GetTextAction(Action):
    """获取指定元素的文本内容（优先使用ElementHandle）。"""
    def __init__(self):
        super().__init__(
            name="get_element_text",
            description="Get the text content of an element",
            params_model=GetTextParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        element_id = params.element_id
        selector = params.selector
        
        if element_id and hasattr(page, '_agent_element_handles'):
            # 优先使用ElementHandle
            handle = page._agent_element_handles.get(element_id)
            if handle:
                logger.info(f"获取元素 (ElementHandle) '{element_id}' 的文本内容。")
                try:
                    text_content = await handle.text_content(timeout=config.browser_config.timeout)
                    logger.info(f"元素 '{element_id}' 的文本内容为: '{text_content}'")
                    return ActionResult(
                        success=True,
                        message=f"元素 '{element_id}' 的文本内容为: '{text_content}'",
                        data=text_content or ""
                    )
                except Exception as e:
                    logger.warning(f"使用ElementHandle获取文本失败，尝试使用选择器: {e}")
                    selector = f'[data-agent-id="{element_id}"]'

        if selector:
            # 清理和验证选择器
            cleaned_selector = self._clean_css_selector(selector)
            
            try:
                logger.info(f"获取元素 (选择器) '{cleaned_selector}' 的文本内容。")
                
                # 使用多种策略尝试获取文本
                text_content = await self._get_text_with_strategies(page, cleaned_selector)
                
                logger.info(f"元素 '{cleaned_selector}' 的文本内容为: '{text_content}'")
                return ActionResult(
                    success=True,
                    message=f"元素 '{cleaned_selector}' 的文本内容为: '{text_content}'",
                    data=text_content or ""
                )
            except Exception as e:
                error_msg = f"获取元素 '{cleaned_selector}' 的文本内容失败: {str(e)}"
                logger.error(error_msg)
                return ActionResult(
                    success=False,
                    message=error_msg,
                    error=str(e),
                    error_type="get_text_failed"
                )
        else:
            # 这种情况不应该发生，因为Pydantic已经验证了参数
            error_msg = "参数验证失败：未提供element_id或selector"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error="No element locator provided",
                error_type="parameter_error"
            )

    def _clean_css_selector(self, selector: str) -> str:
        """清理CSS选择器，移除可能导致问题的伪选择器"""
        import re
        
        # 移除可能有问题的伪选择器
        problematic_pseudo_selectors = [
            r':first-child',
            r':last-child',
            r':nth-child\([^)]*\)',
            r':nth-last-child\([^)]*\)',
            r':first-of-type',
            r':last-of-type',
            r':nth-of-type\([^)]*\)',
            r':nth-last-of-type\([^)]*\)',
            r':text\([^)]*\)',
            r':contains\([^)]*\)'
        ]
        
        cleaned_selector = selector
        for pattern in problematic_pseudo_selectors:
            cleaned_selector = re.sub(pattern, '', cleaned_selector)
        
        # 清理多余的空格和连续的选择器组合符
        cleaned_selector = re.sub(r'\s+', ' ', cleaned_selector)
        cleaned_selector = re.sub(r'\s*>\s*', ' > ', cleaned_selector)
        cleaned_selector = re.sub(r'\s*\+\s*', ' + ', cleaned_selector)
        cleaned_selector = re.sub(r'\s*~\s*', ' ~ ', cleaned_selector)
        
        # 移除末尾可能的多余字符
        cleaned_selector = cleaned_selector.strip()
        
        # 如果清理后选择器为空或只包含空格，返回原选择器
        if not cleaned_selector.strip():
            logger.warning(f"选择器清理后为空，使用原选择器: {selector}")
            return selector
            
        if cleaned_selector != selector:
            logger.info(f"选择器已清理: '{selector}' -> '{cleaned_selector}'")
            
        return cleaned_selector

    async def _get_text_with_strategies(self, page: Page, selector: str) -> str:
        """使用多种策略获取元素文本内容"""
        strategies = [
            ("常规文本获取", self._get_text_regular),
            ("JavaScript文本获取", self._get_text_javascript),
            ("内部文本获取", self._get_text_inner_text),
            ("所有文本获取", self._get_text_all_text)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.debug(f"尝试策略: {strategy_name}")
                text_content = await strategy_func(page, selector)
                if text_content and text_content.strip():
                    logger.debug(f"策略 {strategy_name} 成功获取文本")
                    return text_content
                else:
                    logger.debug(f"策略 {strategy_name} 获取到空文本")
            except Exception as e:
                logger.debug(f"策略 {strategy_name} 失败: {e}")
                continue
        
        # 所有策略都失败，抛出异常
        raise Exception("所有文本获取策略都失败")

    async def _get_text_regular(self, page: Page, selector: str) -> str:
        """常规文本获取方法"""
        element = page.locator(selector).first
        return await element.text_content(timeout=config.browser_config.timeout)

    async def _get_text_javascript(self, page: Page, selector: str) -> str:
        """使用JavaScript获取文本"""
        return await page.evaluate(f"""
            (selector) => {{
                const element = document.querySelector(selector);
                if (!element) return null;
                return element.textContent || element.innerText || '';
            }}
        """, selector)

    async def _get_text_inner_text(self, page: Page, selector: str) -> str:
        """获取innerText"""
        element = page.locator(selector).first
        return await element.inner_text(timeout=config.browser_config.timeout)

    async def _get_text_all_text(self, page: Page, selector: str) -> str:
        """获取所有文本内容"""
        return await page.evaluate(f"""
            (selector) => {{
                const elements = document.querySelectorAll(selector);
                if (elements.length === 0) return null;
                return Array.from(elements).map(el => el.textContent || el.innerText || '').join(' ');
            }}
        """, selector)


# 可以根据需要添加更多 Action 类

class ScrollAction(Action):
    """页面滚动动作。"""
    def __init__(self):
        super().__init__(
            name="scroll",
            description="Scroll the page in specified direction and distance",
            params_model=ScrollParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        direction = params.direction
        distance = params.distance
        
        logger.info(f"滚动页面: 方向={direction}, 距离={distance}px")
        
        # 根据方向计算滚动参数
        delta_x = 0
        delta_y = 0
        
        if direction == "down":
            delta_y = distance
        elif direction == "up":
            delta_y = -distance
        elif direction == "right":
            delta_x = distance
        elif direction == "left":
            delta_x = -distance
        
        try:
            await page.mouse.wheel(delta_x, delta_y)
            
            # 等待滚动完成
            await page.wait_for_timeout(500)
            logger.info(f"页面滚动完成: {direction} {distance}px")
            return ActionResult(
                success=True,
                message=f"页面滚动完成: {direction} {distance}px"
            )
        except Exception as e:
            error_msg = f"页面滚动失败: {direction} {distance}px - {str(e)}"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="scroll_failed"
            )

class BrowserBackAction(Action):
    """浏览器后退动作。"""
    def __init__(self):
        super().__init__(
            name="browser_back",
            description="Navigate back in browser history",
            params_model=BrowserBackParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        logger.info("浏览器后退")
        try:
            await page.go_back(timeout=config.browser_config.timeout)
            logger.info(f"已后退到: {page.url}")
            return ActionResult(
                success=True,
                message=f"已后退到: {page.url}",
                data={"url": page.url}
            )
        except Exception as e:
            error_msg = f"浏览器后退失败: {e}"
            logger.warning(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="navigation_failed"
            )

class BrowserForwardAction(Action):
    """浏览器前进动作。"""
    def __init__(self):
        super().__init__(
            name="browser_forward",
            description="Navigate forward in browser history",
            params_model=BrowserForwardParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        logger.info("浏览器前进")
        try:
            await page.go_forward(timeout=config.browser_config.timeout)
            logger.info(f"已前进到: {page.url}")
            return ActionResult(
                success=True,
                message=f"已前进到: {page.url}",
                data={"url": page.url}
            )
        except Exception as e:
            error_msg = f"浏览器前进失败: {e}"
            logger.warning(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="navigation_failed"
            )

class GetAllTabsAction(Action):
    """获取所有标签页信息的动作。"""
    def __init__(self):
        super().__init__(
            name="get_all_tabs",
            description="Get information about all open tabs. This action is typically executed automatically at the beginning of task nodes to gather current browser state.",
            params_model=GetAllTabsParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        logger.info("获取所有打开的标签页信息。")
        
        context = page.context
        if not context:
            error_msg = "无法获取浏览器上下文"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error_type="context_error"
            )
        
        pages = context.pages
        tabs_info = []
        
        for i, tab_page in enumerate(pages):
            try:
                tab_info = {
                    "id": i,
                    "title": await tab_page.title(),
                    "url": tab_page.url,
                    "is_current": tab_page == page
                }
                tabs_info.append(tab_info)
            except Exception as e:
                logger.warning(f"获取标签页 {i} 信息时出错: {e}")
                tabs_info.append({
                    "id": i,
                    "title": "无法获取标题",
                    "url": "unknown",
                    "is_current": False
                })
        
        logger.info(f"获取到 {len(tabs_info)} 个标签页信息")
        return ActionResult(
            success=True,
            message=f"获取到 {len(tabs_info)} 个标签页信息",
            data=tabs_info
        )

class SwitchTabAction(Action):
    """切换到指定标签页的动作。"""
    def __init__(self):
        super().__init__(
            name="switch_to_tab",
            description="Switch to a specific tab by ID or index",
            params_model=SwitchTabParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        # 兼容两种参数名称
        target_index = params.tab_id if params.tab_id is not None else params.tab_index
        
        context = page.context
        if not context:
            error_msg = "无法获取浏览器上下文"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error_type="context_error"
            )
        
        pages = context.pages
        
        if target_index < 0 or target_index >= len(pages):
            error_msg = f"标签页索引 {target_index} 超出范围。当前有 {len(pages)} 个标签页。"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error_type="index_out_of_range"
            )
        
        target_page = pages[target_index]
        
        try:
            await target_page.bring_to_front()
            tab_title = await target_page.title()
            logger.info(f"已切换到标签页 {target_index}: {tab_title} ({target_page.url})")
            return ActionResult(
                success=True,
                message=f"已切换到标签页 {target_index}: {tab_title}",
                data={"tab_id": target_index, "title": tab_title, "url": target_page.url}
            )
        except Exception as e:
            error_msg = f"切换到标签页 {target_index} 失败: {e}"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="switch_tab_failed"
            )

class NewTabAction(Action):
    """新建标签页的动作。"""
    def __init__(self):
        super().__init__(
            name="new_tab",
            description="Create a new tab with optional URL navigation",
            params_model=NewTabParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        url = params.url
        
        context = page.context
        if not context:
            error_msg = "无法获取浏览器上下文"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error_type="context_error"
            )
        
        logger.info(f"创建新标签页{f', 导航到: {url}' if url else ''}")
        
        try:
            new_page = await context.new_page()
            
            if url:
                await new_page.goto(url, timeout=config.browser_config.timeout)
                logger.info(f"新标签页已创建并导航到: {url}")
                return ActionResult(
                    success=True,
                    message=f"新标签页已创建并导航到: {url}",
                    data={"url": url, "title": await new_page.title()}
                )
            else:
                logger.info("新标签页已创建")
                return ActionResult(
                    success=True,
                    message="新标签页已创建"
                )
            
        except Exception as e:
            error_msg = f"创建新标签页失败: {e}"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="new_tab_failed"
            )

class CloseTabAction(Action):
    """关闭标签页的动作。"""
    def __init__(self):
        super().__init__(
            name="close_tab",
            description="Close a specific tab by ID or index",
            params_model=CloseTabParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        # 兼容两种参数名称
        target_index = params.tab_id if params.tab_id is not None else params.tab_index
        
        context = page.context
        if not context:
            error_msg = "无法获取浏览器上下文"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error_type="context_error"
            )
        
        pages = context.pages
        
        if target_index < 0 or target_index >= len(pages):
            error_msg = f"标签页索引 {target_index} 超出范围。当前有 {len(pages)} 个标签页。"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error_type="index_out_of_range"
            )
        
        if len(pages) <= 1:
            error_msg = "无法关闭最后一个标签页"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error_type="last_tab_error"
            )
        
        target_page = pages[target_index]
        is_current_page = target_page == page
        
        try:
            # 获取标题（在关闭前）
            tab_title = "未知标题"
            try:
                tab_title = await target_page.title()
            except:
                pass
                
            await target_page.close()
            logger.info(f"已关闭标签页 {target_index}: {tab_title}")
            
            # 如果关闭的是当前页面，需要切换到另一个页面
            if is_current_page:
                remaining_pages = context.pages
                if remaining_pages:
                    new_current_page = remaining_pages[0]  # 切换到第一个可用页面
                    await new_current_page.bring_to_front()
                    new_title = await new_current_page.title()
                    logger.info(f"已切换到页面: {new_title} ({new_current_page.url})")
                    return ActionResult(
                        success=True,
                        message=f"已关闭标签页 {target_index}: {tab_title}，并切换到: {new_title}",
                        data={"closed_tab": target_index, "new_current_tab": new_title}
                    )
                else:
                    error_msg = "关闭标签页后没有可用的页面"
                    logger.error(error_msg)
                    return ActionResult(
                        success=False,
                        message=error_msg,
                        error_type="no_remaining_tabs"
                    )
            else:
                return ActionResult(
                    success=True,
                    message=f"已关闭标签页 {target_index}: {tab_title}",
                    data={"closed_tab": target_index}
                )
                
        except Exception as e:
            error_msg = f"关闭标签页 {target_index} 失败: {e}"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="close_tab_failed"
            )

class KeyboardInputAction(Action):
    """键盘输入动作，支持按键组合和特殊键。"""
    def __init__(self):
        super().__init__(
            name="keyboard_input",
            description="Send keyboard input with support for key combinations and special keys",
            params_model=KeyboardInputParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        keys = params.keys
        
        if not isinstance(keys, list):
            keys = [keys]  # 确保是列表格式
        
        logger.info(f"执行键盘输入: {keys}")
        
        try:
            executed_keys = []
            for key in keys:
                if isinstance(key, str):
                    # 处理组合键
                    if '+' in key:
                        # 分离修饰键和主键
                        parts = key.split('+')
                        modifiers = parts[:-1]
                        main_key = parts[-1]
                        
                        # 按下修饰键
                        for modifier in modifiers:
                            await page.keyboard.down(modifier)
                        
                        # 按下主键
                        await page.keyboard.press(main_key)
                        
                        # 释放修饰键
                        for modifier in reversed(modifiers):
                            await page.keyboard.up(modifier)
                        
                        logger.info(f"已按下组合键: {key}")
                        executed_keys.append(key)
                    else:
                        # 单个按键
                        await page.keyboard.press(key)
                        logger.info(f"已按下键: {key}")
                        executed_keys.append(key)
                    
                    # 按键间短暂延迟
                    await page.wait_for_timeout(100)
                else:
                    logger.warning(f"忽略无效的键值: {key}")
                    
            return ActionResult(
                success=True,
                message=f"键盘输入完成: {executed_keys}",
                data={"executed_keys": executed_keys}
            )
                    
        except Exception as e:
            error_msg = f"键盘输入失败: {e}"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="keyboard_input_failed"
            )

class SaveToFileAction(Action):
    """保存页面内容或数据到文件的动作。"""
    def __init__(self):
        super().__init__(
            name="save_to_file",
            description="Save page content or data to file with specified format",
            params_model=SaveToFileParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        filename = params.filename
        content = params.content
        file_format = params.format
        
        # 确保filename有正确的扩展名
        if not filename.endswith(f'.{file_format}'):
            filename = f"{filename}.{file_format}"
        
        logger.info(f"保存内容到文件: {filename} (格式: {file_format})")
        
        try:
            # 创建保存目录（如果不存在）
            import os
            save_dir = "saved_files"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            file_path = os.path.join(save_dir, filename)
            
            # 根据格式处理内容
            if file_format == "json":
                import json
                if isinstance(content, str):
                    # 如果content是字符串，尝试解析为JSON再保存
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        # 如果解析失败，就当作普通字符串处理
                        pass
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                # txt, html等文本格式
                with open(file_path, 'w', encoding='utf-8') as f:
                    if isinstance(content, (dict, list)):
                        import json
                        content = json.dumps(content, ensure_ascii=False, indent=2)
                    f.write(str(content))
            
            logger.info(f"内容已保存到文件: {file_path}")
            return ActionResult(
                success=True,
                message=f"内容已保存到文件: {file_path}",
                data={"file_path": file_path, "format": file_format}
            )
            
        except Exception as e:
            error_msg = f"保存文件失败: {e}"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="save_file_failed"
            )

class RefreshPageAction(Action):
    """刷新当前页面的动作。"""
    def __init__(self):
        super().__init__(
            name="refresh_page",
            description="Refresh the current page and wait for load completion",
            params_model=RefreshPageParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        logger.info("刷新当前页面")
        
        try:
            current_url = page.url
            await page.reload(timeout=config.browser_config.timeout, wait_until="networkidle")
            await page.wait_for_load_state("domcontentloaded")
            
            new_url = page.url
            logger.info(f"页面刷新完成: {new_url}")
            
            return ActionResult(
                success=True,
                message=f"页面刷新完成: {new_url}",
                data={"old_url": current_url, "new_url": new_url}
            )
            
        except Exception as e:
            error_msg = f"页面刷新失败: {str(e)}"
            logger.error(error_msg)
            return ActionResult(
                success=False,
                message=error_msg,
                error=str(e),
                error_type="page_refresh_error"
            )

class WebSearchAction(Action):
    """网页搜索动作，默认使用Google，如果不可访问则使用Bing。"""
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Perform web search using Google (fallback to Bing if Google is not accessible)",
            params_model=WebSearchParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        # 验证参数
        params = self.validate_params(**kwargs)
        
        query = params.query
        search_engine = params.search_engine
        
        logger.info(f"执行网页搜索: '{query}' (搜索引擎: {search_engine})")
        
        # 构建搜索URL
        search_urls = {
            "google": f"https://www.google.com/search?q={query}",
            "bing": f"https://www.bing.com/search?q={query}"
        }
        
        if search_engine == "auto":
            # 自动选择：先尝试Google，失败则使用Bing
            engines_to_try = ["google", "bing"]
        elif search_engine in search_urls:
            engines_to_try = [search_engine]
        else:
            logger.error(f"不支持的搜索引擎: {search_engine}")
            return {
                "success": False,
                "message": f"不支持的搜索引擎: {search_engine}",
                "error": f"Unsupported search engine: {search_engine}",
                "error_type": "invalid_parameter"
            }
        
        last_error = None
        
        for engine in engines_to_try:
            try:
                search_url = search_urls[engine]
                logger.info(f"尝试使用 {engine.upper()} 搜索: {search_url}")
                
                # 导航到搜索页面
                await page.goto(search_url, timeout=config.browser_config.timeout, wait_until="networkidle")
                await page.wait_for_load_state("domcontentloaded")
                
                # 检查是否成功加载搜索结果
                current_url = page.url
                page_title = await page.title()
                
                # 验证是否成功到达搜索页面
                if engine == "google":
                    if "google.com" in current_url.lower() and "search" in current_url.lower():
                        logger.info(f"Google搜索成功: {page_title}")
                        return ActionResult(
                            success=True,
                            message=f"Google搜索成功: {page_title}",
                            data={
                                "search_engine": "google",
                                "query": query,
                                "url": current_url,
                                "title": page_title
                            }
                        )
                elif engine == "bing":
                    if "bing.com" in current_url.lower() and "search" in current_url.lower():
                        logger.info(f"Bing搜索成功: {page_title}")
                        return ActionResult(
                            success=True,
                            message=f"Bing搜索成功: {page_title}",
                            data={
                                "search_engine": "bing", 
                                "query": query,
                                "url": current_url,
                                "title": page_title
                            }
                        )
                
                # 如果到达这里，说明没有成功到达预期的搜索页面
                error_msg = f"{engine.upper()}搜索可能失败，当前URL: {current_url}"
                logger.warning(error_msg)
                last_error = error_msg
                
            except Exception as e:
                error_msg = f"{engine.upper()}搜索失败: {str(e)}"
                logger.warning(error_msg)
                last_error = error_msg
                
                # 如果是最后一个搜索引擎，继续到下一个
                continue
        
        # 所有搜索引擎都失败
        final_error = f"所有搜索引擎都无法访问。最后错误: {last_error}"
        logger.error(final_error)
        return ActionResult(
            success=False,
            message=final_error,
            error=last_error,
            error_type="network_error"
        )


class FinishAction(Action):
    """任务完成动作，用于表示当前子任务或整个任务已经完成，无需执行任何实际操作。"""
    def __init__(self):
        super().__init__(
            name="finish",
            description="Indicate that the current subtask or entire task has been completed. This action performs no actual operations but signals task completion to the system.",
            params_model=FinishParams
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> ActionResult:
        """执行任务完成动作 - 实际上什么都不做，只是返回成功状态"""
        # 验证参数
        params = self.validate_params(**kwargs)
        
        logger.info("任务完成动作被触发 - 表示当前任务已完成")
        
        return ActionResult(
            success=True,
            message="任务完成信号已发出",
            data={"action_type": "task_completion", "timestamp": "now"}
        )

