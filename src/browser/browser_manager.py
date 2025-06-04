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
from src.browser.actions import Action, GetAllTabsAction, SwitchTabAction, NavigateAction
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
        # 记录执行动作前的页面数量，用于检测新窗口
        self._previous_page_count = 0
        # Cookie管理器实例
        self._cookie_manager = CookieManager()
        self._register_default_actions()
        logger.info("BrowserManager 初始化完成，已注册默认 actions。")

    def _invalidate_tabs_cache(self):
        """使tab缓存失效"""
        self._tabs_cache_valid = False
        self._cached_tabs = []

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

    def _register_default_actions(self):
        """注册项目定义的默认 Action 实例。"""
        from src.browser import actions # 延迟导入，避免循环依赖
        default_action_classes = [
            actions.NavigateAction, actions.ClickAction, actions.TypeAction, actions.WaitAction,actions.GetTextAction, 
            # actions.GetAllElementsAction, 
            actions.GetAllElementsActionBs4, 
            actions.ScrollAction, actions.BrowserBackAction, actions.BrowserForwardAction, actions.GetAllTabsAction,
            actions.SwitchTabAction, actions.NewTabAction, actions.CloseTabAction, 
            actions.KeyboardInputAction, actions.SaveToFileAction
        ]
        for action_cls in default_action_classes:
            instance = action_cls()
            self.add_action(instance)

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
            
            logger.info(f"当前页面状态: URL='{url}', Title='{title}', TabsCount={len(tabs)}")
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

    async def get_page_elements_with_fallback(self) -> Any:
        """
        获取页面元素信息，简化版本直接调用action
        """
        try:
            logger.info("获取页面元素信息...")
            return await self.execute_action("get_all_elements_info_bs4") # get_all_elements_info
        except Exception as e:
            logger.error(f"获取页面元素信息失败: {e}")
            # 返回空结果，让调用方处理
            return {
                "flatElements": [],
                "hierarchicalStructure": [],
                "summary": {
                    "totalElements": 0,
                    "interactiveElements": 0,
                    "textElements": 0,
                    "pageTitle": "未知页面",
                    "pageUrl": "unknown"
                },
                "elementHandles": {},
                "elementCount": 0,
                "error": str(e)
            }

    async def _detect_and_switch_to_new_page_after_action(self, action_name: str) -> None:
        """
        动作执行后检测是否有新页面打开，如果有则切换到新页面
        使用更加保守的策略，只在明确的点击操作后检测新页面
        """
        if not self._context:
            return
            
        # 只有在点击操作后才检测新页面（去掉navigate_to_url，因为它在当前tab中导航）
        page_creating_actions = ["click_element"]
        if action_name not in page_creating_actions:
            if self._context:
                self._previous_page_count = len(self._context.pages)
            return
            
        # 适当等待新页面加载
        await asyncio.sleep(1.5)
        
        current_page_count = len(self._context.pages)
        
        # 如果页面数量增加，说明有新页面打开
        if current_page_count > self._previous_page_count:
            logger.info(f"检测到新页面打开：动作 '{action_name}' 后页面数量从 {self._previous_page_count} 增加到 {current_page_count}")
            
            # 找到最新的页面（最后一个）
            new_pages = self._context.pages[self._previous_page_count:]
            
            if new_pages:
                target_page = new_pages[-1]  # 选择最新的页面
                
                try:
                    # 等待新页面开始加载
                    await asyncio.sleep(1.0)
                    
                    # 检查页面是否开始加载内容
                    await target_page.wait_for_load_state("domcontentloaded", timeout=6000)
                    
                    # 检查页面URL是否有效且有意义
                    if (target_page.url and 
                        target_page.url not in ["about:blank", ""] and 
                        not target_page.url.startswith("chrome-extension://") and
                        not target_page.url.startswith("chrome://") and
                        not target_page.url.startswith("moz-extension://") and
                        "error" not in target_page.url.lower()):
                        
                        # 使用SwitchTabAction来切换到新页面
                        try:
                            # 找到新页面的tab_id
                            new_page_index = None
                            for i, page in enumerate(self._context.pages):
                                if page == target_page:
                                    new_page_index = i
                                    break
                            
                            if new_page_index is not None:
                                # 使用SwitchTabAction进行切换
                                switch_action = self.get_action("switch_to_tab")
                                if switch_action:
                                    switched_page = await switch_action.execute(self._page, tab_id=new_page_index)
                                    self._page = switched_page
                                    self._invalidate_tabs_cache()
                                    logger.info(f"使用SwitchTabAction切换到新页面: {await target_page.title()} ({target_page.url})")
                                else:
                                    # 回退到直接切换
                                    await target_page.bring_to_front()
                                    self._page = target_page
                                    self._invalidate_tabs_cache()
                                    logger.info(f"直接切换到新页面: {await target_page.title()} ({target_page.url})")
                            
                        except Exception as switch_e:
                            logger.warning(f"使用SwitchTabAction切换失败，使用直接切换: {switch_e}")
                            # 回退到直接切换
                            await target_page.bring_to_front()
                            self._page = target_page
                            self._invalidate_tabs_cache()
                            logger.info(f"直接切换到新页面: {await target_page.title()} ({target_page.url})")
                    else:
                        logger.info(f"新页面URL无效或无意义，不进行切换: {target_page.url}")
                        
                except Exception as e:
                    logger.warning(f"检查新页面时出错: {e}")
        
        # 更新页面计数
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

    async def _clear_element_highlights(self, page: Page) -> None:
        """清除页面上的元素高亮显示"""
        try:
            await page.evaluate("""() => {
                // 移除所有高亮框
                const highlights = document.querySelectorAll('[data-agent-highlight]');
                highlights.forEach(highlight => highlight.remove());
            }""")
            
            logger.debug("已清除页面元素高亮显示")
            
        except Exception as e:
            logger.debug(f"清除元素高亮时出错（可能页面未加载完成）: {e}")

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
                if isinstance(action_to_execute, SwitchTabAction):
                    result = await asyncio.wait_for(
                        action_to_execute.execute(target_page, **kwargs),
                        timeout=timeout_duration
                    )
                    self._page = result  # 更新当前活动页面
                    final_result = f"Switched to tab: {await result.title()} ({result.url})"
                elif isinstance(action_to_execute, GetAllTabsAction):
                    final_result = await asyncio.wait_for(
                        action_to_execute.execute(target_page, **kwargs),
                        timeout=timeout_duration
                    )
                elif action_name in ["new_tab", "close_tab"]:
                    # 处理新建或关闭tab的动作，这些动作会返回页面对象
                    result = await asyncio.wait_for(
                        action_to_execute.execute(target_page, **kwargs),
                        timeout=timeout_duration
                    )
                    if hasattr(result, 'url'):  # 检查返回的是页面对象
                        self._page = result  # 更新当前活动页面
                        final_result = f"{action_name} completed: {await result.title()} ({result.url})"
                    else:
                        final_result = result
                else:
                    final_result = await asyncio.wait_for(
                        action_to_execute.execute(target_page, **kwargs),
                        timeout=timeout_duration
                    )

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