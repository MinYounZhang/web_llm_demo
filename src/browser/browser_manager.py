import asyncio
import random
import os
import json
from typing import Any, Dict, List, Optional, Type, Tuple

from patchright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from src.config import config, logger
from src.browser.actions import Action, GetAllTabsAction, SwitchTabAction, NavigateAction, ActionResult
from pydantic import ValidationError
from src.browser.cookie_manager import CookieManager
from src.error_management import (
    error_manager, 
    HumanInterventionError, 
    ActionTimeoutError,
    detect_human_intervention_needed,
    get_error_type,
    ErrorType
)


class BrowserManager:
    """管理 Playwright 浏览器实例、上下文、页面以及执行浏览器操作。"""

    def __init__(self):
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._actions: Dict[str, Action] = {}
        # 缓存tab信息，避免重复获取
        self._cached_tabs: List[Dict[str, Any]] = []
        self._tabs_cache_valid = False
        # 缓存页面元素信息，避免重复获取
        self._cached_page_elements: Dict[str, Any] = {}
        self._page_elements_cache_valid = False
        self._page_elements_cache_time = 0
        self._page_elements_cache_method = ""
        self._page_elements_cache_url = ""
        # 记录执行动作前的页面数量，用于检测新窗口
        self._previous_page_count = 0
        # Cookie管理器实例
        self._cookie_manager = CookieManager()
        # 添加元素句柄映射，用于维护element_id到实际元素的对应关系
        self._element_handles: Dict[str, Any] = {}
        self._element_handles_counter = 0
        self._register_default_actions()
        logger.info("BrowserManager 初始化完成，已注册默认 actions。")

    def _invalidate_tabs_cache(self):
        """使tab缓存失效"""
        self._tabs_cache_valid = False
        self._cached_tabs = []

    def _invalidate_page_elements_cache(self):
        """使页面元素缓存失效"""
        self._page_elements_cache_valid = False
        self._cached_page_elements = {}
        self._page_elements_cache_time = 0
        self._page_elements_cache_method = ""
        self._page_elements_cache_url = ""
        logger.debug("页面元素缓存已失效")

    def _is_tab_related_action(self, action_name: str) -> bool:
        """判断是否为可能影响tab的动作"""
        tab_related_actions = [
            "get_all_tabs",      # 获取所有标签页
            "switch_to_tab",     # 切换标签页
            "new_tab",           # 新建标签页
            "close_tab",         # 关闭标签页
            "navigate_to_url",   # 导航可能会打开新标签页
            "click_element"      # 点击链接可能会打开新标签页
        ]
        return action_name in tab_related_actions

    def _is_page_elements_changing_action(self, action_name: str) -> bool:
        """判断是否为可能改变页面元素的动作"""
        element_changing_actions = [
            "navigate_to_url",    # 导航会完全改变页面
            "click_element",      # 点击可能触发页面变化、弹窗、动态内容加载
            "type_text",          # 输入可能触发动态内容变化
            "scroll",             # 滚动可能触发懒加载
            "keyboard_input",     # 键盘输入可能触发快捷键或表单变化
            "browser_back",       # 浏览器后退会改变页面
            "browser_forward",    # 浏览器前进会改变页面
            "switch_to_tab",      # 切换标签页会改变当前页面
            "new_tab",            # 新建标签页
            "close_tab"           # 关闭标签页可能影响当前页面
        ]
        return action_name in element_changing_actions

    def _should_use_page_elements_cache(self, method: str, current_url: str, cache_timeout: int = 60) -> bool:
        """判断是否应该使用页面元素缓存"""
        import time
        
        # 如果缓存无效，不使用缓存
        if not self._page_elements_cache_valid:
            return False
        
        # 如果缓存为空，不使用缓存
        if not self._cached_page_elements:
            return False
        
        # 如果URL发生变化，不使用缓存
        if self._page_elements_cache_url != current_url:
            logger.debug(f"URL已变化，缓存失效: {self._page_elements_cache_url} -> {current_url}")
            return False
        
        # 如果方法不同，不使用缓存
        if self._page_elements_cache_method != method:
            logger.debug(f"获取方法已变化，缓存失效: {self._page_elements_cache_method} -> {method}")
            return False
        
        # 检查时间间隔（默认60秒）
        current_time = time.time()
        if current_time - self._page_elements_cache_time > cache_timeout:
            logger.debug(f"缓存超时，已过去 {current_time - self._page_elements_cache_time:.1f} 秒")
            return False
        
        return True

    def _register_default_actions(self):
        """注册项目定义的默认 Action 实例。"""
        from src.browser import actions # 延迟导入，避免循环依赖
        default_action_classes = [
            actions.NavigateAction, actions.ClickAction, actions.TypeAction, actions.WaitAction,actions.GetTextAction, 
            actions.ScrollAction, actions.BrowserBackAction, 
            actions.BrowserForwardAction,
            actions.SwitchTabAction, actions.NewTabAction, 
            actions.CloseTabAction, actions.KeyboardInputAction, actions.SaveToFileAction,
            actions.RefreshPageAction, actions.WebSearchAction
        ]
        for action_cls in default_action_classes:
            try:
                instance = action_cls()
                self.add_action(instance)
            except Exception as e:
                logger.warning(f"无法注册 Action {action_cls.__name__}: {e}")

    def add_action(self, action: Action):
        """添加一个 Action 到管理器中。"""
        if action.name in self._actions:
            logger.warning(f"名为 '{action.name}' 的 Action 已存在，将被覆盖。")
        self._actions[action.name] = action
        logger.debug(f"Action '{action.name}' 已添加。")

    def get_action(self, name: str) -> Action | None:
        """根据名称获取 Action。"""
        return self._actions.get(name)

    def get_all_actions_description(self) -> List[Dict[str, str]]:
        """获取所有已注册 Action 的名称和描述，用于 LLM 参考。"""
        return [action.to_dict() for action in self._actions.values()]

    def get_action_params_schema(self, action_name: str) -> Dict[str, Any]:
        """
        获取指定动作的参数Schema
        
        Args:
            action_name: 动作名称
            
        Returns:
            参数的JSON Schema
        """
        if action_name not in self._actions:
            raise ValueError(f"未找到名为 '{action_name}' 的 Action")
        
        action = self._actions[action_name]
        return action.get_params_schema()

    async def get_page_state(self, force_refresh_tabs: bool = False) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        获取当前页面的基本状态：URL, Title, 以及标签页信息。
        
        Args:
            force_refresh_tabs: 是否强制刷新tab信息
        """
        try:
            current_page = await self.get_current_page()
            url = current_page.url
            title = await current_page.title()
            
            # 只在需要时获取tab信息
            if force_refresh_tabs or not self._tabs_cache_valid:
                try:
                    tabs = await self.get_all_tabs()
                    self._cached_tabs = tabs
                    self._tabs_cache_valid = True
                    logger.info(f"刷新tab信息: TabsCount={len(tabs)}")
                except Exception as e:
                    logger.warning(f"获取tab信息失败，使用缓存: {e}")
                    tabs = self._cached_tabs
            else:
                tabs = self._cached_tabs
                logger.debug(f"使用缓存的tab信息: TabsCount={len(tabs)}")
            
            logger.info(f"当前页面状态: URL='{url[:50]}...', Title='{title}', TabsCount={len(tabs)}")
            return url, title, tabs
            
        except Exception as e:
            logger.error(f"获取页面状态失败: {e}")
            # 尝试启动浏览器（如果尚未启动）
            try:
                logger.info("尝试启动浏览器以获取页面状态...")
                await self.launch_browser()
                current_page = await self.get_current_page()
                url = current_page.url
                title = await current_page.title()
                # 强制获取tab信息
                tabs = await self.get_all_tabs()
                self._cached_tabs = tabs
                self._tabs_cache_valid = True
                return url, title, tabs
            except Exception as launch_e:
                logger.error(f"再次尝试获取页面状态失败（启动浏览器后）: {launch_e}")
                return "unknown_url", "unknown_title", []

    async def get_page_elements_with_fallback(self, method: str = "dom", force_refresh: bool = False, enable_highlight: bool = False) -> Any:
        """
        获取页面元素信息，支持多种方法和智能缓存
        
        Args:
            method: 元素获取方法 ('dom', 'aom', 'xpath')
            force_refresh: 是否强制刷新，忽略缓存
            enable_highlight: 是否启用元素高亮
        """
        try:
            current_page = await self.get_current_page()
            current_url = current_page.url
            
            # 检查缓存
            if not force_refresh and self._should_use_page_elements_cache(method, current_url):
                logger.info(f"使用缓存的页面元素信息 - 方法:{method}, URL:{current_url[:60]}...")
                return self._cached_page_elements
            
            logger.info(f"获取页面元素信息，使用方法: {method}")
            
            # 直接调用内部方法
            result = await self._get_all_elements_info(current_page, method=method, enable_highlight=enable_highlight)
            
            # 缓存结果
            import time
            self._cached_page_elements = result
            self._page_elements_cache_valid = True
            self._page_elements_cache_time = time.time()
            self._page_elements_cache_method = method
            self._page_elements_cache_url = current_url
            
            logger.info(f"页面元素信息已缓存 - 方法:{method}, 元素数量:{result.get('summary', {}).get('total', 0)}")
            return result
            
        except Exception as e:
            logger.error(f"获取页面元素信息失败（方法: {method}）: {e}")
            # 如果指定方法失败，尝试使用DOM方法
            if method != "dom":
                try:
                    logger.info("尝试使用DOM方法重新获取页面元素...")
                    current_page = await self.get_current_page()
                    result = await self._get_all_elements_info(current_page, method="dom", enable_highlight=enable_highlight)
                    
                    # 缓存DOM方法的结果
                    import time
                    self._cached_page_elements = result
                    self._page_elements_cache_valid = True
                    self._page_elements_cache_time = time.time()
                    self._page_elements_cache_method = "dom"
                    self._page_elements_cache_url = current_page.url
                    
                    return result
                except Exception:
                    logger.error("DOM方法也失败")
            
            # 返回空结果
            return {
                "elements": [],
                "summary": {"total": 0, "interactive": 0, "text": 0},
                "method": method,
                "error": str(e)
            }

    async def _detect_and_switch_to_new_page_after_action(self, action_name: str) -> None:
        """检测并切换到新页面"""
        if not self._context or action_name not in ["click_element"]:
            if self._context:
                self._previous_page_count = len(self._context.pages)
            return
            
        await asyncio.sleep(1.5)
        current_page_count = len(self._context.pages)
        
        if current_page_count > self._previous_page_count:
            new_pages = self._context.pages[self._previous_page_count:]
            if new_pages:
                target_page = new_pages[-1]
                
                # 简单验证页面有效性
                await asyncio.sleep(1.0)
                if (target_page.url and target_page.url not in ["about:blank", ""] and 
                    not target_page.url.startswith(("chrome-", "moz-")) and "error" not in target_page.url.lower()):
                    
                    # 直接切换到新页面
                    await target_page.bring_to_front()
                    self._page = target_page
                    self._invalidate_tabs_cache()
                    logger.info(f"切换到新页面: {target_page.url}")
        
        self._previous_page_count = len(self._context.pages)

    async def get_current_page(self) -> Page:
        """获取当前活动的页面。如果不存在则尝试创建。"""
        if self._page and not self._page.is_closed():
            return self._page
        elif self._context:
            logger.info("当前页面不存在或已关闭，正在创建新页面...")
            return await self.new_page()
        else:
            logger.info("浏览器未启动，正在尝试启动并创建新页面...")
            return await self.launch_browser()
        
    async def get_all_tabs(self) -> List[Dict[str, Any]]:
        """便捷方法，使用 GetAllTabsAction 获取所有标签页信息。"""
        page = await self.get_current_page() # 需要一个有效的 page 对象来访问 context
        action = self.get_action("get_all_tabs")
        if action:
            return await self.execute_action(action_name="get_all_tabs", page_override=page)
        logger.warning("'get_all_tabs' action 未注册。")
        return []

    async def save_cookies(self):
        """保存cookies的公共接口。"""
        await self._cookie_manager.save_cookies(self._context)

    async def clear_cookies(self):
        """清除所有cookies的公共接口。"""
        await self._cookie_manager.clear_all_cookies(self._context)

    async def close_browser(self):
        """关闭浏览器和 Playwright 实例。"""
        # 在关闭前保存cookies
        if config.browser_config.auto_save_cookies:
            await self._cookie_manager.save_cookies(self._context)
        
        if self._browser:
            await self._browser.close()
            logger.info("浏览器已关闭。")
        if self._playwright:
            await self._playwright.stop()
            logger.info("Playwright 实例已停止。")
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None

    async def launch_browser(self, headless: bool | None = None, user_agent: str | None = None, viewport_size: tuple[int, int] | None = None, **kwargs) -> Page:
        """
        启动浏览器，创建上下文和页面。

        Args:
            headless: 是否无头模式，None表示使用配置文件设置。
            user_agent: 用户代理字符串，None表示使用配置文件设置。
            viewport_size: 视窗大小 (width, height)，None表示使用默认大小 1366x768。
            **kwargs: 传递给 browser.new_context() 的额外参数。

        Returns:
            创建的 Page 实例。

        Raises:
            RuntimeError: 如果启动浏览器失败。
        """
        logger.info(f"正在启动浏览器... (headless: {headless}, user_agent: {user_agent[:50] + '...' if user_agent and len(user_agent) > 50 else user_agent})")

        try:
            # 启动Playwright
            self._playwright = await async_playwright().start()
            
            # 配置启动选项
            launch_options = {
                "headless": headless if headless is not None else config.browser_config.headless,
                "args": [
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-infobars",
                    "--disable-extensions",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-ipc-flooding-protection"
                ]
            }
            
            # 启动Chromium浏览器
            self._browser = await self._playwright.chromium.launch(**launch_options)
            
            # 设置默认User-Agent（如果未提供）
            default_user_agent = user_agent or config.browser_config.user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            
            # 设置视窗大小（如果未提供则使用默认值）
            viewport_width, viewport_height = viewport_size or (1366, 768)
            
            # 配置上下文选项 - 模拟真实浏览器环境
            context_options = {
                # 基本设置 - 可配置的视窗大小
                "viewport": {"width": viewport_width, "height": viewport_height},
                "user_agent": default_user_agent,
                
                # 语言和地区设置
                "locale": "zh-CN",
                "timezone_id": "Asia/Shanghai",
                
                # 媒体设置
                "color_scheme": "light",
                "reduced_motion": "no-preference",
                
                # 权限设置
                "permissions": ["geolocation", "notifications"],
                
                # 设备信息
                "device_scale_factor": 1.0,
                "is_mobile": False,
                "has_touch": False,
                
                # 额外的HTTP头部
                "extra_http_headers": {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Cache-Control": "max-age=0"
                },
                
                # JavaScript启用
                "java_script_enabled": True,
                
                **kwargs
            }
            
            # 如果启用了Cookie管理，尝试加载已保存的状态（只加载重要cookie）
            if self._cookie_manager.is_cookie_management_enabled():
                storage_state_path = await self._cookie_manager.load_cookies()
                if storage_state_path:
                    context_options["storage_state"] = storage_state_path
                    logger.info(f"使用保存的重要Cookie状态: {storage_state_path}")
                else:
                    logger.info("没有找到有效的重要Cookie，使用全新浏览器状态")
            
            # 创建浏览器上下文
            self._context = await self._browser.new_context(**context_options)
            
            # 设置超时时间
            self._context.set_default_navigation_timeout(config.browser_config.navigation_timeout)
            self._context.set_default_timeout(config.browser_config.timeout)
            
            # 创建新页面
            self._page = await self._context.new_page()
            
            # 应用反检测机制
            await self._apply_stealth_mechanisms(self._page)
            
            # 更新页面计数
            self._previous_page_count = len(self._context.pages)
            
            logger.info(f"浏览器启动成功。页面URL: {self._page.url}")
            logger.info(f"浏览器配置: User-Agent: {default_user_agent}")
            logger.info(f"浏览器配置: 视窗: {viewport_width}x{viewport_height}, 语言: zh-CN, 时区: Asia/Shanghai")
            return self._page

        except Exception as e:
            logger.error(f"启动浏览器失败: {e}")
            await self.close_browser()
            raise RuntimeError(f"无法启动浏览器: {e}")

    async def new_page(self) -> Page:
        """在当前上下文中创建一个新页面，并应用反自动化措施。"""
        if not self._context:
            logger.error("浏览器上下文未初始化。请先调用 launch_browser。")
            raise RuntimeError("浏览器上下文未初始化。请先调用 launch_browser。")
        page = await self._context.new_page()
        await self._apply_stealth_mechanisms(page)
        self._page = page # 将最新创建的页面设为当前活动页面
        
        # 更新页面计数
        if self._context:
            self._previous_page_count = len(self._context.pages)
        
        logger.info(f"新页面已创建: {page.url}")
        return page

    async def _apply_stealth_mechanisms(self, page: Page):
        """应用基本的反机器人检测规避机制。"""
        # 1. 伪造 WebGL 信息
        try:
            await page.add_init_script("""
                if (navigator.webdriver === false) {
                    // Do nothing if webdriver is already false
                } else if (navigator.webdriver === undefined) {
                    // Define webdriver as false if it's undefined
                    Object.defineProperty(navigator, 'webdriver', { get: () => false });
                } else {
                    // If webdriver is true, try to redefine it as false
                    try {
                        Object.defineProperty(navigator, 'webdriver', { get: () => false });
                    } catch (e) {
                        console.warn('Could not redefine navigator.webdriver: ', e);
                    }
                }

                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );

                // Overriding plugins, languages, and mimeTypes
                if (navigator.plugins) {
                    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] }); // Example
                }
                if (navigator.languages) {
                    Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh'] });
                }
                 if (navigator.mimeTypes) {
                    Object.defineProperty(navigator, 'mimeTypes', { get: () => { length: 0 } });
                }
            """)
            logger.debug("已应用反检测脚本 (webdriver, permissions, plugins, languages, mimeTypes)。")
        except Exception as e:
            logger.warning(f"应用反检测脚本时出错: {e}")

    async def execute_action(self, action_name: str, page_override: Optional[Page] = None, **kwargs: Any) -> Any:
        """
        执行指定的 Action，包含完整的错误管理机制。

        Args:
            action_name: 要执行的 Action 的名称。
            page_override: 如果提供，则在该页面上执行操作，否则使用当前的 self._page。
            **kwargs: 传递给 Action 的 execute 方法的参数。

        Returns:
            Action 执行的结果。

        Raises:
            ValueError: 如果 Action 未找到或者页面未初始化。
            HumanInterventionError: 如果需要人工干预。
        """
        # 获取执行前的Cookie状态（用于智能保存）
        cookies_before = await self._cookie_manager.get_current_cookies(self._context)
        
        if action_name not in self._actions:
            raise ValueError(f"未找到名为 '{action_name}' 的 Action。")

        action_to_execute = self._actions[action_name]

        # 确定要使用的页面
        target_page = page_override or await self.get_current_page()
        if target_page is None:
            raise ValueError("页面未初始化，无法执行 Action。")

        logger.info(f"在页面 <{await target_page.title()}> ({target_page.url}) 上执行 Action: '{action_name}'，参数: {kwargs}")

        # 执行前检测是否需要人工干预
        needs_intervention, intervention_reason = await detect_human_intervention_needed(target_page)
        if needs_intervention:
            logger.warning(f"执行前检测到需要人工干预: {intervention_reason}")
            
            # 等待人工干预
            intervention_resolved = await error_manager.wait_for_human_intervention(
                target_page, intervention_reason
            )
            
            if not intervention_resolved:
                raise HumanInterventionError(
                    f"无法解决人工干预问题: {intervention_reason}",
                    "请手动解决后重新尝试"
                )
        
        # 使用重试机制执行Action
        max_retries = config.agent_config.max_action_retries
        retry_delay = config.agent_config.action_retry_delay_ms / 1000
        
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 因为包括第一次尝试
            try:
                if attempt > 0:
                    logger.info(f"Action '{action_name}' 第 {attempt + 1} 次尝试...")
                    await asyncio.sleep(retry_delay * (attempt ** 2))  # 指数退避
                
                # 计算超时时间
                base_timeout = config.browser_config.timeout / 1000
                timeout_duration = base_timeout * config.agent_config.action_timeout_multiplier
                
                # 执行Action，带超时控制
                action_result = await asyncio.wait_for(
                    action_to_execute.execute(target_page, **kwargs),
                    timeout=timeout_duration
                )
                
                # 处理ActionResult返回值
                if isinstance(action_result, ActionResult):
                    # 新的Pydantic模型格式
                    final_result = action_result.dict()
                    
                    if not action_result.success:
                        # 动作执行失败，但不抛出异常，让重试机制处理
                        logger.warning(f"Action '{action_name}' 执行失败: {action_result.error}")
                        raise RuntimeError(f"Action failed: {action_result.error}")
                    
                elif isinstance(action_result, dict) and "success" in action_result:
                    # 兼容旧的字典格式
                    if not action_result.get("success", False):
                        # 动作执行失败，但不抛出异常，让重试机制处理
                        error_msg = action_result.get("error", "Unknown error")
                        logger.warning(f"Action '{action_name}' 执行失败: {error_msg}")
                        raise RuntimeError(f"Action failed: {error_msg}")
                    
                    final_result = action_result
                    
                elif hasattr(action_result, 'url'):
                    # 兼容旧格式：直接返回页面对象的情况
                    self._page = action_result
                    final_result = {
                        "success": True,
                        "message": f"Action completed: {await action_result.title()} ({action_result.url})",
                        "data": None
                    }
                else:
                    # 兼容其他旧格式
                    final_result = action_result
                
                # 对于特殊的tab操作，需要处理页面切换
                if isinstance(action_to_execute, SwitchTabAction):
                    # SwitchTabAction返回页面对象需要特殊处理
                    if final_result.get("success", False):
                        # 获取当前活动的页面
                        context = target_page.context
                        if context and context.pages:
                            # 切换成功，获取当前活动页面
                            self._page = context.pages[0]  # 临时方案

                # 执行成功后的短暂延迟
                await target_page.wait_for_timeout(random.randint(200, 800))
                
                # 执行后检测新页面（仅对点击动作）
                if action_name in ["click_element"]:
                    await self._detect_and_switch_to_new_page_after_action(action_name)
                else:
                    # 对于其他动作，简单更新页面计数
                    if self._context:
                        self._previous_page_count = len(self._context.pages)
                
                # 如果是tab相关动作，使缓存失效
                if self._is_tab_related_action(action_name):
                    self._invalidate_tabs_cache()
                    logger.debug(f"动作 '{action_name}' 导致tab缓存失效")
                
                # 如果是会改变页面元素的动作，使页面元素缓存失效
                if self._is_page_elements_changing_action(action_name):
                    self._invalidate_page_elements_cache()
                    logger.debug(f"动作 '{action_name}' 导致页面元素缓存失效")
                
                # 执行后智能保存Cookie
                await self._cookie_manager.smart_save_cookies_after_action(self._context, action_name, cookies_before)
                
                logger.info(f"Action '{action_name}' 执行完成。结果: {str(final_result)[:100]}...")
                return final_result
                
            except asyncio.TimeoutError:
                last_exception = ActionTimeoutError(action_name, timeout_duration)
                logger.warning(f"Action '{action_name}' 第 {attempt + 1} 次执行超时 ({timeout_duration}s)")
                
                # 处理超时情况
                should_retry, timeout_reason = await error_manager.handle_action_timeout(
                    action_name, target_page, timeout_duration
                )
                
                if not should_retry:
                    if "人工干预" in timeout_reason:
                        # 等待人工干预
                        intervention_resolved = await error_manager.wait_for_human_intervention(
                            target_page, timeout_reason
                        )
                        if intervention_resolved:
                            continue  # 重试
                        else:
                            raise HumanInterventionError(
                                f"Action '{action_name}' 超时且需要人工干预: {timeout_reason}",
                                "请手动解决问题后重试"
                            )
                    else:
                        # 不适合重试，直接抛出异常
                        raise last_exception
                
                # 如果这是最后一次尝试，抛出异常
                if attempt == max_retries:
                    raise last_exception
                    
            except (PlaywrightTimeoutError, ConnectionError) as e:
                last_exception = e
                error_type = get_error_type(e)
                logger.warning(f"Action '{action_name}' 第 {attempt + 1} 次执行失败: {e}, 错误类型: {error_type.value}")
                
                # 检测是否需要人工干预
                if error_type == ErrorType.HUMAN_INTERVENTION_NEEDED:
                    intervention_resolved = await error_manager.wait_for_human_intervention(
                        target_page, str(e)
                    )
                    if not intervention_resolved:
                        raise HumanInterventionError(
                            f"Action '{action_name}' 需要人工干预: {str(e)}",
                            "请手动处理后重试"
                        )
                    continue  # 重试
                
                # 如果这是最后一次尝试，抛出异常
                if attempt == max_retries:
                    raise last_exception
                    
            except HumanInterventionError:
                # 直接重新抛出人工干预错误，不进行重试
                raise
                
            except Exception as e:
                last_exception = e
                error_type = get_error_type(e)
                logger.error(f"Action '{action_name}' 第 {attempt + 1} 次执行时发生未预期错误: {e}, 错误类型: {error_type.value}")
                
                # 对于未知错误，检测是否需要人工干预
                needs_intervention, intervention_reason = await detect_human_intervention_needed(target_page)
                if needs_intervention:
                    intervention_resolved = await error_manager.wait_for_human_intervention(
                        target_page, f"执行错误后检测到问题: {intervention_reason}"
                    )
                    if not intervention_resolved:
                        raise HumanInterventionError(
                            f"Action '{action_name}' 执行错误且需要人工干预: {intervention_reason}",
                            "请手动解决问题后重试"
                        )
                    continue  # 重试
                
                # 如果这是最后一次尝试，抛出原始异常
                if attempt == max_retries:
                    raise last_exception
        
        # 理论上不应该到达这里，但为了安全起见
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Action '{action_name}' 执行失败，原因未知")

    async def clear_cookies_for_domain(self, domain: str):
        """清除特定域名的cookie。"""
        await self._cookie_manager.clear_cookies_for_domain(self._context, domain)
    
    async def _get_all_elements_info(self, page: Page, method: str = "dom", enable_highlight: bool = False) -> Dict[str, Any]:
        """
        获取页面元素信息的核心方法，改进版本包含拓扑结构和明确的元素类型
        
        Args:
            page: 页面对象
            method: 获取方法 ('dom', 'aom', 'xpath')
            enable_highlight: 是否启用高亮
        """
        logger.info(f"获取页面元素 - 方法:{method}, 高亮:{enable_highlight}")
        
        # 确保页面加载完成
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            await page.wait_for_load_state("networkidle", timeout=15000)
            await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"页面加载检查警告: {e}")
        
        # 清除之前的高亮
        await self._clear_element_highlights(page)
        
        elements = []
        actual_method = method
        
        # 根据方法获取元素，支持自动回退
        if method == "aom":
            for retry in range(2):
                try:
                    elements = await self._aom_element_traversal(page)
                    break
                except Exception as e:
                    if retry == 0:
                        logger.warning(f"AOM方法第{retry+1}次失败，重试: {e}")
                        await asyncio.sleep(0.5)
                    else:
                        logger.warning(f"AOM方法失败，回退到DOM: {e}")
                        break
            
            if not elements:
                elements = await self._dom_element_traversal(page)
                actual_method = "dom"
        
        elif method == "xpath":
            try:
                elements = await self._xpath_element_traversal(page)
            except Exception as e:
                logger.warning(f"XPath方法失败，回退到DOM: {e}")
                elements = await self._dom_element_traversal(page)
                actual_method = "dom"
        else:
            elements = await self._dom_element_traversal(page)
        
        if not elements:
            return await self._fallback_element_extraction(page, enable_highlight)
        
        # 改进的元素处理：添加拓扑结构和明确类型
        processed_elements = self._process_and_enhance_elements(elements)
        
        # 创建Element Handle映射
        element_handles = await self._create_enhanced_element_handles(page, processed_elements)
        
        # 添加高亮
        if enable_highlight:
            await self._add_element_highlights(page, list(element_handles.keys()))
        
        # 构建增强的结果
        result = self._build_enhanced_result(processed_elements, actual_method, len(element_handles))
        result["_element_handles"] = element_handles
        
        # 缓存到页面对象
        if not hasattr(page, '_page_elements_cache'):
            page._page_elements_cache = {}
        page._page_elements_cache.update({
            'elements': processed_elements,
            'method': actual_method,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        return result

    async def _dom_element_traversal(self, page: Page) -> List[Dict[str, Any]]:
        """增强的DOM遍历，包含拓扑结构信息"""
        return await page.evaluate("""() => {
            const elements = [];
            let elementId = 1;
            
            // 生成唯一CSS选择器的函数
            function generateSelector(element) {
                // 优先使用ID
                if (element.id) {
                    return `#${element.id}`;
                }
                
                // 构建路径选择器
                const path = [];
                let current = element;
                
                while (current && current !== document.body && current !== document.documentElement) {
                    let selector = current.tagName.toLowerCase();
                    
                    // 添加类名
                    if (current.className && typeof current.className === 'string') {
                        const classes = current.className.trim().split(/\\s+/).filter(c => c);
                        if (classes.length > 0) {
                            selector += '.' + classes.slice(0, 2).join('.');
                        }
                    }
                    
                    // 添加属性
                    if (current.getAttribute('role')) {
                        selector += `[role="${current.getAttribute('role')}"]`;
                    }
                    
                    // 计算同类元素的位置
                    const parent = current.parentElement;
                    if (parent) {
                        const siblings = Array.from(parent.children).filter(sibling => 
                            sibling.tagName === current.tagName
                        );
                        if (siblings.length > 1) {
                            const index = siblings.indexOf(current) + 1;
                            selector += `:nth-of-type(${index})`;
                        }
                    }
                    
                    path.unshift(selector);
                    current = current.parentElement;
                }
                
                return path.slice(-3).join(' > '); // 最多3层深度
            }
            
            // 分层选择器：按重要性和类型分组
            const selectorGroups = {
                interactive: [
                    'button, input, textarea, select, a[href]',
                    '[role="button"], [role="link"], [role="textbox"], [role="combobox"]',
                    '[onclick], [tabindex]:not([tabindex="-1"]), .btn, .click, .link'
                ],
                semantic: [
                    'h1, h2, h3, h4, h5, h6',
                    'nav, main, section, article, aside, header, footer',
                    'label, strong, em, mark'
                ],
                content: [
                    'p, div[class*="text"], span[class*="content"]',
                    'li, td, th, caption'
                ]
            };
            
            const processedElements = new Set();
            const elementMap = new Map(); // 用于构建层次关系
            
            // 按组处理元素
            for (const [groupType, selectors] of Object.entries(selectorGroups)) {
                for (const selector of selectors) {
                    const nodeList = document.querySelectorAll(selector);
                    for (const el of nodeList) {
                        if (processedElements.has(el)) continue;
                        
                        const rect = el.getBoundingClientRect();
                        if (rect.width <= 0 || rect.height <= 0) continue;
                        
                        // 生成CSS选择器
                        let cssSelector = '';
                        if (el.id) {
                            cssSelector = `#${el.id}`;
                        } else {
                            const tag = el.tagName.toLowerCase();
                            const classList = el.className ? `.${el.className.trim().split(/\s+/).slice(0, 2).join('.')}` : '';
                            const role = el.getAttribute('role') ? `[role="${el.getAttribute('role')}"]` : '';
                            
                            // 计算同类型元素的索引
                            const siblings = el.parentElement ? Array.from(el.parentElement.children).filter(s => s.tagName === el.tagName) : [el];
                            const index = siblings.indexOf(el) + 1;
                            const nthType = siblings.length > 1 ? `:nth-of-type(${index})` : '';
                            
                            cssSelector = `${tag}${classList}${role}${nthType}`;
                        }
                        
                        const elementInfo = {
                            id: `dom_elem_${elementId++}`,
                            tag: el.tagName.toLowerCase(),
                            text: (el.textContent || '').trim(),
                            rect: {
                                x: Math.round(rect.left),
                                y: Math.round(rect.top),
                                w: Math.round(rect.width),
                                h: Math.round(rect.height)
                            },
                            attrs: {},
                            cssSelector: cssSelector,
                            elementType: groupType, // 明确的元素类型
                            interactive: groupType === 'interactive',
                            semantic: ['interactive', 'semantic'].includes(groupType),
                            // 拓扑信息
                            depth: 0,
                            parentId: null,
                            childIds: [],
                            siblingIndex: 0
                        };
                        
                        // 提取关键属性
                        ['id', 'class', 'role', 'type', 'href', 'placeholder', 'aria-label', 'title'].forEach(attr => {
                            const value = el.getAttribute(attr);
                            if (value) elementInfo.attrs[attr] = value;
                        });
                        
                        // 计算DOM深度
                        let depth = 0;
                        let parent = el.parentElement;
                        while (parent && parent !== document.body) {
                            depth++;
                            parent = parent.parentElement;
                        }
                        elementInfo.depth = depth;
                        
                        // 兄弟节点索引
                        if (el.parentElement) {
                            const siblings = Array.from(el.parentElement.children);
                            elementInfo.siblingIndex = siblings.indexOf(el);
                        }
                        
                        elements.push(elementInfo);
                        processedElements.add(el);
                        elementMap.set(el, elementInfo);
                    }
                }
            }
            
            // 构建父子关系
            for (const elementInfo of elements) {
                const el = Array.from(processedElements).find(e => 
                    e.tagName.toLowerCase() === elementInfo.tag && 
                    Math.abs(e.getBoundingClientRect().left - elementInfo.rect.x) < 5
                );
                
                if (el && el.parentElement && elementMap.has(el.parentElement)) {
                    const parentInfo = elementMap.get(el.parentElement);
                    elementInfo.parentId = parentInfo.id;
                    parentInfo.childIds.push(elementInfo.id);
                }
            }
            
            console.log('DOM遍历完成，找到元素数量:', elements.length);
            return elements;
        }""")

    async def _aom_element_traversal(self, page: Page) -> List[Dict[str, Any]]:
        """增强的AOM遍历，包含无障碍性信息"""
        try:
            client = await page.context.new_cdp_session(page)
            await client.send('Accessibility.enable')
            await client.send('DOM.enable')
            ax_tree = await client.send('Accessibility.getFullAXTree')
            
            elements = []
            element_id = 1
            nodes = ax_tree.get('nodes', [])
            
            for node in nodes:
                role = node.get('role', {}).get('value', '')
                if not role:
                    continue
                
                name = node.get('name', {}).get('value', '')
                description = node.get('description', {}).get('value', '')
                bbox = node.get('boundingBox', {})
                
                if not bbox:
                    bbox = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
                
                # 组合文本内容
                text_parts = [name, description] if description != name else [name]
                combined_text = ' '.join(filter(None, text_parts))
                
                # 确定元素类型
                element_type = self._classify_aom_element_type(role)
                
                element_info = {
                    'id': f'aom_elem_{element_id}',
                    'tag': role.lower(),
                    'text': combined_text,
                    'rect': {
                        'x': int(bbox.get('x', 0)),
                        'y': int(bbox.get('y', 0)),
                        'w': int(bbox.get('width', 0)),
                        'h': int(bbox.get('height', 0))
                    },
                    'attrs': {'role': role},
                    'elementType': element_type,
                    'interactive': self._is_interactive_aom_role(role),
                    'semantic': True,
                    'backend_node_id': node.get('backendDOMNodeId'),
                    'source_method': 'aom',
                    # 拓扑信息（简化版）
                    'depth': 0,
                    'parentId': None,
                    'childIds': [],
                    'siblingIndex': 0
                }
                
                # 添加更多属性
                for prop_name in ['value', 'checked', 'disabled', 'expanded', 'selected']:
                    prop_value = node.get(prop_name)
                    if prop_value is not None:
                        element_info['attrs'][prop_name] = str(prop_value.get('value', '')) if isinstance(prop_value, dict) else str(prop_value)
                
                elements.append(element_info)
                element_id += 1
            
            await client.detach()
            return elements
            
        except Exception as e:
            logger.warning(f"AOM遍历失败: {e}")
            raise

    async def _xpath_element_traversal(self, page: Page) -> List[Dict[str, Any]]:
        """增强的XPath遍历"""
        return await page.evaluate("""() => {
            const elements = [];
            let elementId = 1;
            
            const xpathGroups = {
                interactive: [
                    '//button | //input | //textarea | //select | //a[@href]',
                    '//*[@role="button"] | //*[@role="link"] | //*[@role="textbox"]'
                ],
                semantic: [
                    '//h1 | //h2 | //h3 | //h4 | //h5 | //h6',
                    '//nav | //main | //section | //article'
                ],
                content: [
                    '//*[@onclick] | //*[contains(@class, "btn")] | //*[contains(@class, "click")]'
                ]
            };
            
            const processedElements = new Set();
            
            for (const [groupType, xpaths] of Object.entries(xpathGroups)) {
                for (const xpath of xpaths) {
                    try {
                        const result = document.evaluate(xpath, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
                        
                        for (let i = 0; i < result.snapshotLength; i++) {
                            const el = result.snapshotItem(i);
                            if (processedElements.has(el)) continue;
                            
                            const rect = el.getBoundingClientRect();
                            if (rect.width <= 0 || rect.height <= 0) continue;
                            
                            // 生成CSS选择器
                            let cssSelector = '';
                            if (el.id) {
                                cssSelector = `#${el.id}`;
                            } else {
                                const tag = el.tagName.toLowerCase();
                                const classList = el.className ? `.${el.className.trim().split(/\\s+/).slice(0, 2).join('.')}` : '';
                                const role = el.getAttribute('role') ? `[role="${el.getAttribute('role')}"]` : '';
                                
                                // 计算同类型元素的索引
                                const siblings = el.parentElement ? Array.from(el.parentElement.children).filter(s => s.tagName === el.tagName) : [el];
                                const index = siblings.indexOf(el) + 1;
                                const nthType = siblings.length > 1 ? `:nth-of-type(${index})` : '';
                                
                                cssSelector = `${tag}${classList}${role}${nthType}`;
                            }
                            
                            const elementInfo = {
                                id: `xpath_elem_${elementId++}`,
                                tag: el.tagName.toLowerCase(),
                                text: (el.textContent || '').trim(),
                                rect: {
                                    x: Math.round(rect.left),
                                    y: Math.round(rect.top),
                                    w: Math.round(rect.width),
                                    h: Math.round(rect.height)
                                },
                                attrs: {},
                                cssSelector: cssSelector,
                                elementType: groupType,
                                interactive: groupType === 'interactive',
                                semantic: ['interactive', 'semantic'].includes(groupType),
                                depth: 0,
                                parentId: null,
                                childIds: [],
                                siblingIndex: 0
                            };
                            
                            // 提取属性
                            ['id', 'class', 'type', 'role'].forEach(attr => {
                                const value = el.getAttribute(attr);
                                if (value) elementInfo.attrs[attr] = value;
                            });
                            
                            elements.push(elementInfo);
                            processedElements.add(el);
                        }
                    } catch (e) {
                        console.warn('XPath查询失败:', xpath, e);
                    }
                }
            }
            
            return elements;
        }""")

    def _classify_aom_element_type(self, role: str) -> str:
        """根据AOM角色分类元素类型"""
        interactive_roles = {'button', 'link', 'textbox', 'combobox', 'checkbox', 'radio', 'menuitem', 'tab'}
        semantic_roles = {'heading', 'navigation', 'main', 'region', 'article', 'section'}
        
        if role.lower() in interactive_roles:
            return 'interactive'
        elif role.lower() in semantic_roles:
            return 'semantic'
        else:
            return 'content'

    def _is_interactive_aom_role(self, role: str) -> bool:
        """判断AOM角色是否为交互性角色"""
        interactive_roles = {
            'button', 'link', 'textbox', 'combobox', 'checkbox', 'radio', 
            'menuitem', 'tab', 'tabpanel', 'listbox', 'option', 'slider',
            'spinbutton', 'searchbox', 'switch', 'tree', 'treeitem'
        }
        return role.lower() in interactive_roles

    def _process_and_enhance_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理和增强元素信息"""
        enhanced_elements = []
        
        for element in elements:
            # 生成唯一ID
            unique_id = element.get('id', f"elem_{len(enhanced_elements) + 1}")
            
            # 计算可点击性评分
            clickability = self._calculate_clickability_score(element)
            
            # 计算文本重要性评分
            text_importance = self._calculate_text_importance(element)
            
            # 获取屏幕象限位置
            rect = element.get('rect', {})
            quadrant = self._get_screen_quadrant(rect)
            
            # 确保有selector字段 - 优先使用cssSelector字段
            selector = element.get('cssSelector') or element.get('selector')
            if not selector:
                # 如果没有selector，基于元素属性生成一个
                tag = element.get('tag', 'div')
                attrs = element.get('attrs', {})
                
                if attrs.get('id'):
                    selector = f"#{attrs['id']}"
                elif attrs.get('class'):
                    # 清理class名称
                    classes = attrs['class'].replace(' ', '.').replace(':', '\\:')
                    selector = f"{tag}.{classes}"
                elif attrs.get('role'):
                    selector = f'{tag}[role="{attrs["role"]}"]'
                else:
                    # 最后回退到data-agent-id
                    selector = f'[data-agent-id="{unique_id}"]'
            
            # 增强元素信息
            enhanced_element = {
                **element,
                'uniqueId': unique_id,
                'selector': selector,  # 确保selector字段存在
                'clickability': clickability,
                'textImportance': text_importance,
                'position': {
                    'quadrant': quadrant,
                    'center_x': rect.get('x', 0) + rect.get('w', 0) // 2,
                    'center_y': rect.get('y', 0) + rect.get('h', 0) // 2
                }
            }
            
            enhanced_elements.append(enhanced_element)
        
        # 按综合评分排序
        enhanced_elements.sort(key=lambda x: x['clickability'] + x['textImportance'], reverse=True)
        
        return enhanced_elements

    def _calculate_clickability_score(self, element: Dict[str, Any]) -> float:
        """计算元素的可点击性评分"""
        score = 0.0
        
        # 基础交互性
        if element.get('interactive', False):
            score += 5.0
        
        # 标签类型评分
        tag = element.get('tag', '')
        tag_scores = {'button': 5.0, 'a': 4.0, 'input': 4.0, 'select': 3.0}
        score += tag_scores.get(tag, 0.0)
        
        # 属性评分
        attrs = element.get('attrs', {})
        if attrs.get('onclick') or 'click' in attrs.get('class', '').lower():
            score += 3.0
        if 'btn' in attrs.get('class', '').lower():
            score += 2.0
        
        # 大小评分
        rect = element.get('rect', {})
        area = rect.get('w', 0) * rect.get('h', 0)
        if area > 1000:
            score += 1.0
        
        return min(score, 10.0)  # 最高10分

    def _calculate_text_importance(self, element: Dict[str, Any]) -> float:
        """计算文本重要性评分"""
        text = element.get('text', '').strip()
        if not text:
            return 0.0
        
        score = 0.0
        
        # 长度评分
        if len(text) > 50:
            score += 3.0
        elif len(text) > 10:
            score += 2.0
        elif len(text) > 0:
            score += 1.0
        
        # 语义关键词评分
        important_keywords = ['点击', '查看', '更多', '详情', '阅读', '继续', '下一步', '确认', '提交', '搜索']
        for keyword in important_keywords:
            if keyword in text:
                score += 2.0
                break
        
        # 标题元素额外加分
        if element.get('tag', '') in ['h1', 'h2', 'h3']:
            score += 2.0
        
        return min(score, 10.0)

    def _get_screen_quadrant(self, rect: Dict[str, Any]) -> str:
        """获取元素在屏幕中的象限"""
        x = rect.get('x', 0) + rect.get('w', 0) // 2
        y = rect.get('y', 0) + rect.get('h', 0) // 2
        
        # 假设1366x768的视窗
        if x < 683:
            if y < 384:
                return 'top-left'
            else:
                return 'bottom-left'
        else:
            if y < 384:
                return 'top-right'
            else:
                return 'bottom-right'

    async def _create_enhanced_element_handles(self, page: Page, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建增强的元素句柄映射"""
        element_handles = {}
        
        # 批量设置data-agent-id属性
        element_mapping_results = await page.evaluate("""(elements) => {
            const results = [];
            elements.forEach(elementInfo => {
                const rect = elementInfo.rect;
                if (rect.w > 0 && rect.h > 0) {
                    const x = rect.x + rect.w / 2;
                    const y = rect.y + rect.h / 2;
                    
                    const element = document.elementFromPoint(x, y);
                    if (element) {
                        // 使用新的ID格式
                        const agentId = elementInfo.uniqueId;
                        element.setAttribute('data-agent-id', agentId);
                        element.setAttribute('data-element-type', elementInfo.elementType);
                        element.setAttribute('data-clickability', elementInfo.clickability.toFixed(1));
                        
                        results.push({
                            uniqueId: agentId,
                            found: true,
                            tagName: element.tagName.toLowerCase(),
                            selector: `[data-agent-id="${agentId}"]`
                        });
                    } else {
                        results.push({
                            uniqueId: elementInfo.uniqueId,
                            found: false
                        });
                    }
                }
            });
            return results;
        }""", elements)
        
        # 获取ElementHandle并更新元素信息
        for i, element_info in enumerate(elements):
            unique_id = element_info['uniqueId']
            mapping_result = element_mapping_results[i] if i < len(element_mapping_results) else None
            
            if mapping_result and mapping_result.get('found'):
                try:
                    selector = f'[data-agent-id="${unique_id}"]'
                    handle = await page.query_selector(selector)
                    if handle:
                        element_handles[unique_id] = handle
                        # 更新元素信息中的selector - 优先使用cssSelector
                        element_info['selector'] = element_info.get('cssSelector') or selector
                        element_info['data_agent_id'] = unique_id
                        logger.debug(f"成功创建ElementHandle: {unique_id}")
                    else:
                        # 优先使用DOM生成的cssSelector
                        css_selector = element_info.get('cssSelector')
                        if css_selector:
                            element_info['selector'] = css_selector
                        else:
                            fallback_selector = self._generate_fallback_selector(element_info)
                            element_info['selector'] = fallback_selector
                            element_info['fallback_selector'] = fallback_selector
                except Exception as e:
                    logger.debug(f"创建ElementHandle失败: {unique_id}, {e}")
                    # 优先使用DOM生成的cssSelector
                    css_selector = element_info.get('cssSelector')
                    if css_selector:
                        element_info['selector'] = css_selector
                    else:
                        fallback_selector = self._generate_fallback_selector(element_info)
                        element_info['selector'] = fallback_selector
                        element_info['fallback_selector'] = fallback_selector
            else:
                # 无法通过坐标定位，优先使用DOM生成的cssSelector
                css_selector = element_info.get('cssSelector')
                if css_selector:
                    element_info['selector'] = css_selector
                else:
                    fallback_selector = self._generate_fallback_selector(element_info)
                    element_info['selector'] = fallback_selector
                    element_info['fallback_selector'] = fallback_selector
                    logger.warning(f"无法通过坐标定位元素，使用备用selector: {unique_id} -> {fallback_selector}")
        
        # 存储到页面对象
        if not hasattr(page, '_agent_element_handles'):
            page._agent_element_handles = {}
        page._agent_element_handles.update(element_handles)
        
        logger.info(f"成功创建 {len(element_handles)} 个ElementHandle，共处理 {len(elements)} 个元素")
        return element_handles

    def _generate_fallback_selector(self, element_info: Dict[str, Any]) -> str:
        """为无法通过坐标定位的元素生成备用CSS选择器"""
        tag = element_info.get('tag', 'div')
        attrs = element_info.get('attrs', {})
        text = element_info.get('text', '').strip()
        
        # 优先使用ID属性
        if attrs.get('id'):
            return f"#{attrs['id']}"
        
        # 使用class属性
        if attrs.get('class'):
            classes = attrs['class'].replace(' ', '.')
            return f"{tag}.{classes}"
        
        # 使用role属性
        if attrs.get('role'):
            return f'{tag}[role="{attrs["role"]}"]'
        
        # 使用href属性（对于链接）
        if tag == 'a' and attrs.get('href'):
            return f'a[href="{attrs["href"]}"]'
        
        # 使用type属性（对于input）
        if tag == 'input' and attrs.get('type'):
            return f'input[type="{attrs["type"]}"]'
        
        # 使用文本内容（如果文本不太长且具有唯一性）
        if text and len(text) < 50 and len(text) > 2:
            # 简化文本，去除特殊字符
            clean_text = text.replace('"', '\\"').replace('\n', ' ').strip()
            if clean_text:
                # 使用属性选择器而不是:text()伪选择器
                return f'{tag}[title*="{clean_text[:20]}"]'
        
        # 最后回退到标签名加nth-of-type
        return f'{tag}:first-of-type'

    async def _add_element_highlights(self, page: Page, element_ids: List[str]) -> None:
        """添加元素高亮"""
        try:
            highlighted_count = await page.evaluate("""(elementIds) => {
                // 创建高亮样式
                let style = document.getElementById('agent-highlight-style');
                if (!style) {
                    style = document.createElement('style');
                    style.id = 'agent-highlight-style';
                    style.textContent = `
                        .agent-highlight {
                            outline: 3px solid #ff4444 !important;
                            outline-offset: 2px !important;
                            background: rgba(255, 68, 68, 0.15) !important;
                            position: relative !important;
                            z-index: 999998 !important;
                            box-shadow: 0 0 5px rgba(255, 68, 68, 0.5) !important;
                        }
                        .agent-highlight::before {
                            content: attr(data-agent-id) " [" attr(data-clickability) "]" !important;
                            position: absolute !important;
                            top: -25px !important;
                            left: 0 !important;
                            background: #ff4444 !important;
                            color: white !important;
                            font-size: 11px !important;
                            font-weight: bold !important;
                            padding: 3px 6px !important;
                            border-radius: 4px !important;
                            font-family: 'Courier New', monospace !important;
                            pointer-events: none !important;
                            z-index: 999999 !important;
                            white-space: nowrap !important;
                            border: 1px solid #cc3333 !important;
                        }
                    `;
                    document.head.appendChild(style);
                }
                
                let count = 0;
                console.log('开始高亮元素，数量:', elementIds.length);
                
                elementIds.forEach(elementId => {
                    const element = document.querySelector(`[data-agent-id="${elementId}"]`);
                    if (element) {
                        element.classList.add('agent-highlight');
                        count++;
                        console.log('高亮元素成功:', elementId, element.tagName);
                    } else {
                        console.log('未找到元素:', elementId);
                    }
                });
                
                console.log('高亮完成，成功数量:', count);
                return count;
            }""", element_ids)
            
            logger.info(f"高亮完成: {highlighted_count}/{len(element_ids)} 个元素")
            
        except Exception as e:
            logger.warning(f"高亮失败: {e}")

    async def _clear_element_highlights(self, page: Page) -> None:
        """清除元素高亮"""
        try:
            await page.evaluate("""() => {
                document.querySelectorAll('.agent-highlight').forEach(el => {
                    el.classList.remove('agent-highlight');
                });
                
                const style = document.getElementById('agent-highlight-style');
                if (style) style.remove();
                
                document.querySelectorAll('[data-agent-id]').forEach(el => {
                    el.removeAttribute('data-agent-id');
                    el.removeAttribute('data-element-type');
                    el.removeAttribute('data-clickability');
                });
            }""")
        except Exception as e:
            logger.debug(f"清除高亮失败: {e}")

    def _build_enhanced_result(self, elements: List[Dict[str, Any]], method: str, handle_count: int) -> Dict[str, Any]:
        """构建增强的结果"""
        # 按类型分组
        grouped_elements = {}
        for element in elements:
            element_type = element.get('elementType', 'unknown')
            if element_type not in grouped_elements:
                grouped_elements[element_type] = []
            grouped_elements[element_type].append(element)
        
        # 构建拓扑结构
        topology = self._build_element_topology(elements)
        
        # 生成markdown
        markdown = self._generate_enhanced_markdown(elements, grouped_elements, method)
        
        return {
            'elements': elements,
            'grouped': grouped_elements,
            'topology': topology,
            'summary': {
                'total': len(elements),
                'interactive': len([e for e in elements if e.get('interactive')]),
                'semantic': len([e for e in elements if e.get('semantic')]),
                'by_type': {k: len(v) for k, v in grouped_elements.items()},
                'method': method
            },
            'markdown': markdown,
            'elementHandlesCount': handle_count
        }

    def _build_element_topology(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建元素拓扑结构"""
        # 按深度分层
        depth_groups = {}
        for element in elements:
            depth = element.get('depth', 0)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(element['uniqueId'])
        
        # 按象限分组
        quadrant_groups = {}
        for element in elements:
            quadrant = element.get('position', {}).get('quadrant', 'unknown')
            if quadrant not in quadrant_groups:
                quadrant_groups[quadrant] = []
            quadrant_groups[quadrant].append(element['uniqueId'])
        
        return {
            'depth_layers': depth_groups,
            'screen_quadrants': quadrant_groups,
            'parent_child_relations': [(e.get('parentId'), e['uniqueId']) for e in elements if e.get('parentId')]
        }

    def _generate_enhanced_markdown(self, elements: List[Dict[str, Any]], grouped_elements: Dict[str, List], method: str) -> str:
        """生成增强的markdown描述"""
        lines = [f"# 页面元素分析 ({method.upper()})", ""]
        
        # 按类型显示
        for element_type, type_elements in grouped_elements.items():
            lines.append(f"## {element_type.title()} 元素 ({len(type_elements)}个)")
            lines.append("")
            
            # 按可点击性评分排序
            sorted_elements = sorted(type_elements, key=lambda x: x.get('clickability', 0), reverse=True)
            
            for element in sorted_elements[:10]:  # 只显示前10个
                unique_id = element['uniqueId']
                tag = element['tag']
                text = element.get('text', '')[:50] + "..." if len(element.get('text', '')) > 50 else element.get('text', '')
                clickability = element.get('clickability', 0)
                text_importance = element.get('textImportance', 0)
                selector = element.get('selector') or element.get('cssSelector') or f'[data-agent-id="{unique_id}"]'
                
                line_parts = [f"- **{unique_id}** `{tag}`"]
                
                if element.get('interactive'):
                    line_parts.append("🔗")
                if element.get('semantic'):
                    line_parts.append("📝")
                
                if text:
                    line_parts.append(f'"{text}"')
                
                line_parts.append(f"(点击性:{clickability:.1f}, 文本:{text_importance:.1f})")
                
                # 添加selector信息
                if selector and selector != 'N/A':
                    line_parts.append(f"selector:`{selector}`")
                
                lines.append(" ".join(line_parts))
            
            lines.append("")
        
        return "\n".join(lines)

    async def _fallback_element_extraction(self, page: Page, enable_highlight: bool) -> Dict[str, Any]:
        """简单回退方案"""
        try:
            elements = await page.evaluate("""() => {
                const elements = [];
                const selectors = ['button', 'input', 'a[href]'];
                let id = 1;
                
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            elements.push({
                                uniqueId: `fallback_${selector}_${id}`,
                                id: `fallback_elem_${id++}`,
                                tag: el.tagName.toLowerCase(),
                                text: (el.textContent || '').trim(),
                                rect: {
                                    x: Math.round(rect.left),
                                    y: Math.round(rect.top),
                                    w: Math.round(rect.width),
                                    h: Math.round(rect.height)
                                },
                                elementType: 'interactive',
                                interactive: true,
                                clickability: 5.0
                            });
                        }
                    });
                });
                
                return elements;
            }""")
            
            if enable_highlight:
                await self._add_element_highlights(page, [e['uniqueId'] for e in elements])
            
            return {
                'elements': elements,
                'summary': {'total': len(elements), 'interactive': len(elements), 'method': 'fallback'},
                'markdown': f"# 页面元素 (FALLBACK)\n\n" + "\n".join([f"- **{e['uniqueId']}** `{e['tag']}` 🔗 {e.get('text', '')} (点击性:{e['clickability']})" for e in elements]),
                'elementHandlesCount': 0
            }
            
        except Exception as e:
            logger.error(f"回退方案失败: {e}")
            return {
                'elements': [],
                'summary': {'total': 0, 'interactive': 0, 'method': 'failed'},
                'markdown': "# 页面元素获取失败",
                'elementHandlesCount': 0
            }


