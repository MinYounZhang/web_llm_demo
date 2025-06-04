from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Awaitable
import random
import functools
from bs4 import NavigableString, BeautifulSoup
from patchright.async_api import Page, ElementHandle, Locator
from src.config import logger, config
from src.error_management import with_timeout_and_retry
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, page: Page, **kwargs: Any) -> Any:
        """执行动作。

        Args:
            page: Playwright 页面对象。
            **kwargs: 动作所需的参数。

        Returns:
            动作执行的结果，例如获取的文本、元素等。
        """
        pass

    def to_dict(self) -> Dict[str, str]:
        """返回动作的字典表示，用于 LLM 理解。"""
        return {
            "name": self.name,
            "description": self.description
        }

class NavigateAction(Action):
    """导航到指定 URL 的动作。"""
    def __init__(self):
        super().__init__(
            name="navigate_to_url",
            description="Navigate to a specified URL. Requires 'url' argument."
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> None:
        url = kwargs.get("url")
        if not url:
            logger.error(f"'{self.name}' 动作需要 'url' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'url' 参数。")
        logger.info(f"导航到 URL: {url}")
        await page.goto(url, timeout=config.browser_config.timeout)
        logger.info(f"已导航到: {page.url}")

class ClickAction(Action):
    """点击指定元素（优先使用ElementHandle）。"""
    def __init__(self):
        super().__init__(
            name="click_element",
            description="Click on an element. Requires either 'element_id' (preferred) or 'selector' argument. element_id refers to elements from get_all_elements_info."
        )

    @with_human_like_mouse_move
    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> None:
        element_id = kwargs.get("element_id")
        selector = kwargs.get("selector")
        
        if element_id and hasattr(page, '_agent_element_handles'):
            # 优先使用ElementHandle
            handle = page._agent_element_handles.get(element_id)
            if handle:
                logger.info(f"点击元素 (ElementHandle): {element_id}")
                try:
                    await handle.click(timeout=config.browser_config.timeout)
                    logger.info(f"已点击元素: {element_id}")
                    return
                except Exception as e:
                    logger.warning(f"使用ElementHandle点击失败，尝试使用选择器: {e}")
                    # 回退到选择器方式
                    selector = f'[data-agent-id="{element_id}"]'
        
        if selector:
            logger.info(f"点击元素 (选择器): {selector}")
            element = page.locator(selector).first
            if not await element.is_visible():
                logger.warning(f"元素 '{selector}' 不可见，尝试强制点击。")
            await element.click(timeout=config.browser_config.timeout)
            logger.info(f"已点击元素: {selector}")
        else:
            logger.error(f"'{self.name}' 动作需要 'element_id' 或 'selector' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'element_id' 或 'selector' 参数。")

class TypeAction(Action):
    """在指定元素中输入文本（优先使用ElementHandle）。"""
    def __init__(self):
        super().__init__(
            name="type_text",
            description="Type text into an element. Requires either 'element_id' (preferred) or 'selector', and 'text' arguments. Optional 'delay' in milliseconds between key presses."
        )

    @with_human_like_mouse_move
    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> None:
        element_id = kwargs.get("element_id")
        selector = kwargs.get("selector")
        text_to_type = kwargs.get("text")
        delay = kwargs.get("delay", 50)

        if text_to_type is None:
            logger.error(f"'{self.name}' 动作需要 'text' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'text' 参数。")

        handle = None
        if element_id and hasattr(page, '_agent_element_handles'):
            # 优先使用ElementHandle
            handle = page._agent_element_handles.get(element_id)
            if handle:
                logger.info(f"在元素 (ElementHandle) '{element_id}' 中输入文本: '{text_to_type}'，延迟: {delay}ms")
                try:
                    await handle.type(text_to_type, delay=delay, timeout=config.browser_config.timeout)
                    logger.info(f"已在元素 '{element_id}' 中输入文本。")
                    return
                except Exception as e:
                    logger.warning(f"使用ElementHandle输入失败，尝试使用选择器: {e}")
                    selector = f'[data-agent-id="{element_id}"]'

        if selector:
            logger.info(f"在元素 (选择器) '{selector}' 中输入文本: '{text_to_type}'，延迟: {delay}ms")
            element = page.locator(selector).first
            await element.type(text_to_type, delay=delay, timeout=config.browser_config.timeout)
            logger.info(f"已在元素 '{selector}' 中输入文本。")
        else:
            logger.error(f"'{self.name}' 动作需要 'element_id' 或 'selector' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'element_id' 或 'selector' 参数。")

class WaitAction(Action):
    """等待指定时间或元素出现。"""
    def __init__(self):
        super().__init__(
            name="wait",
            description="Wait for a specified duration or for an element to appear. Requires either 'duration_ms' (milliseconds) or 'selector' argument."
        )
    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> None:
        duration_ms = kwargs.get("duration_ms")
        selector = kwargs.get("selector")

        if duration_ms is not None:
            logger.info(f"等待 {duration_ms} 毫秒。")
            await page.wait_for_timeout(duration_ms)
            logger.info(f"已完成等待 {duration_ms} 毫秒。")
        elif selector is not None:
            logger.info(f"等待元素 '{selector}' 出现。")
            await page.wait_for_selector(selector, timeout=config.browser_config.timeout)
            logger.info(f"元素 '{selector}' 已出现。")
        else:
            logger.error(f"'{self.name}' 动作需要 'duration_ms' 或 'selector' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'duration_ms' 或 'selector' 参数。")

class GetTextAction(Action):
    """获取指定元素的文本内容（优先使用ElementHandle）。"""
    def __init__(self):
        super().__init__(
            name="get_element_text",
            description="Get the text content of an element. Requires either 'element_id' (preferred) or 'selector' argument. Returns the text content as a string."
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> str:
        element_id = kwargs.get("element_id")
        selector = kwargs.get("selector")
        
        if element_id and hasattr(page, '_agent_element_handles'):
            # 优先使用ElementHandle
            handle = page._agent_element_handles.get(element_id)
            if handle:
                logger.info(f"获取元素 (ElementHandle) '{element_id}' 的文本内容。")
                try:
                    text_content = await handle.text_content(timeout=config.browser_config.timeout)
                    logger.info(f"元素 '{element_id}' 的文本内容为: '{text_content}'")
                    return text_content or ""
                except Exception as e:
                    logger.warning(f"使用ElementHandle获取文本失败，尝试使用选择器: {e}")
                    selector = f'[data-agent-id="{element_id}"]'

        if selector:
            logger.info(f"获取元素 (选择器) '{selector}' 的文本内容。")
            element = page.locator(selector).first
            text_content = await element.text_content(timeout=config.browser_config.timeout)
            logger.info(f"元素 '{selector}' 的文本内容为: '{text_content}'")
            return text_content or ""
        else:
            logger.error(f"'{self.name}' 动作需要 'element_id' 或 'selector' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'element_id' 或 'selector' 参数。")

class GetAllElementsAction(Action):
    """获取页面上所有对LLM有用的元素信息。"""
    def __init__(self):
        super().__init__(
            name="get_all_elements_info",
            description="Get information for all useful elements on the page (interactive elements and meaningful text). Optional 'selector' argument to filter elements. Returns element information with handles."
        )

    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=1, min=1, max=5),
    #     retry=retry_if_exception_type((Exception,)),
    #     reraise=True
    # )
    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> Dict[str, Any]:
        selector = kwargs.get("selector")
        enable_highlight = kwargs.get("enable_highlight", True)
        
        logger.info(f"获取页面有用元素，自定义选择器: {selector}, 启用高亮: {enable_highlight}")

        try:
            # 先清除之前的高亮
            await self._clear_highlights(page)

            # 获取页面元素信息
            page_structure = await self._get_page_elements(page, selector)
            
            # 创建ElementHandle映射
            element_handles = await self._create_element_handles(page, page_structure.get("flatElements", []))
            
            # 添加高亮功能
            if enable_highlight and element_handles:
                await self._highlight_elements(page, list(element_handles.keys()))

            result = {
                **page_structure,
                "elementHandles": element_handles,
                "elementCount": len(element_handles)
            }

            logger.info(f"获取到 {len(element_handles)} 个有用元素，交互元素: {result['summary']['interactiveElements']}，文本元素: {result['summary']['textElements']}")
            return result
            
        except Exception as e:
            logger.warning(f"获取页面元素失败，尝试回退方案: {e}")
            return await self._fallback_get_elements(page, enable_highlight)

    async def _get_page_elements(self, page: Page, selector: str = None) -> Dict[str, Any]:
        """获取页面元素的主要逻辑"""
        return await page.evaluate("""(args) => {
            const customSelector = args.customSelector;
            const attrsList = args.attrsList;
            
            // 生成唯一ID
            let elementIdCounter = 0;
            const elementIdMap = new WeakMap();
            
            function getElementId(el) {
                if (!elementIdMap.has(el)) {
                    elementIdMap.set(el, `element_${elementIdCounter++}`);
                    el.setAttribute('data-agent-id', elementIdMap.get(el));
                }
                return elementIdMap.get(el);
            }
            
            // 元素可见性检测
            function isVisible(el) {
                if (!el || !document.contains(el)) return false;
                try {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0 && 
                           style.visibility !== 'hidden' && style.display !== 'none' && style.opacity !== '0';
                } catch (e) { return false; }
            }
            
            // 交互元素判断
            function isInteractiveElement(el) {
                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role');
                const classList = Array.from(el.classList);
                
                // 明确的交互元素
                if (['button', 'input', 'textarea', 'select', 'a', 'label', 'option'].includes(tag)) return true;
                
                // 交互角色
                const interactiveRoles = ['button', 'link', 'textbox', 'searchbox', 'combobox', 'checkbox', 'radio', 'menuitem', 'tab', 'switch', 'slider'];
                if (role && interactiveRoles.includes(role)) return true;
                
                // 可点击属性
                if (el.onclick || el.hasAttribute('onclick') || el.tabIndex >= 0 || el.contentEditable === 'true') return true;
                
                // 交互类名
                const interactivePatterns = ['btn', 'button', 'clickable', 'click', 'search-btn', 'nav-btn', 'menu-btn', 'submit', 'cancel', 'confirm', 'toggle', 'tab', 'dropdown'];
                if (classList.some(cls => interactivePatterns.some(pattern => cls.toLowerCase().includes(pattern.toLowerCase())))) return true;
                
                // SVG图标按钮或pointer cursor
                if (['div', 'span'].includes(tag)) {
                    const hasSvgIcon = el.querySelector('svg');
                    const hasIconClass = classList.some(cls => ['icon', 'ico', 'fa-', 'material-icons'].some(iconPattern => cls.toLowerCase().includes(iconPattern)));
                    if (hasSvgIcon || hasIconClass) return true;
                    
                    try {
                        const style = window.getComputedStyle(el);
                        if (style.cursor === 'pointer') return true;
                    } catch (e) {}
                }
                
                return false;
            }
            
            // 有用文本元素判断
            function isUsefulTextElement(el) {
                const tag = el.tagName.toLowerCase();
                const text = (el.textContent || '').trim();
                
                if (!text || text.length === 0 || text.length > 150) return false;
                
                const textTags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'strong', 'em', 'label'];
                if (textTags.includes(tag)) return true;
                
                const importantClasses = ['title', 'heading', 'label', 'name', 'text', 'message'];
                return importantClasses.some(cls => Array.from(el.classList).some(elCls => elCls.toLowerCase().includes(cls)));
            }
            
            // 获取元素信息
            function getElementInfo(el) {
                const rect = el.getBoundingClientRect();
                const tag = el.tagName.toLowerCase();
                const text = (el.textContent || "").trim().slice(0, 100);
                const directText = Array.from(el.childNodes)
                    .filter(node => node.nodeType === Node.TEXT_NODE)
                    .map(node => node.textContent.trim())
                    .join(' ').slice(0, 50);
                
                const attrs = {};
                for (const attr of attrsList) {
                    const value = el.getAttribute(attr) || el[attr];
                    if (value && value.toString().trim()) {
                        attrs[attr] = value.toString().trim().slice(0, 100);
                    }
                }
                
                return {
                    id: getElementId(el),
                    tag,
                    text: directText || text,
                    attrs,
                    position: {
                        x: Math.round(rect.left),
                        y: Math.round(rect.top),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    },
                    isVisible: isVisible(el),
                    isInteractive: isInteractiveElement(el),
                    isUsefulText: isUsefulTextElement(el),
                    inViewport: (rect.top >= 0 && rect.left >= 0 && 
                                rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && 
                                rect.right <= (window.innerWidth || document.documentElement.clientWidth))
                };
            }
            
            // 收集有用元素
            function collectUsefulElements() {
                const allElements = [];
                const processedElements = new Set();
                
                let targetElements = [];
                
                if (customSelector) {
                    try {
                        targetElements = Array.from(document.querySelectorAll(customSelector));
                    } catch (e) {
                        console.warn('自定义选择器无效:', e);
                        targetElements = [];
                    }
                } else {
                    // 使用默认选择器
                    const selectors = [
                        'button, input, textarea, select, a, label, option',
                        '[role="button"], [role="link"], [role="textbox"], [role="searchbox"], [role="combobox"], [role="checkbox"], [role="radio"], [role="menuitem"]',
                        '[onclick], [tabindex], [contenteditable="true"]',
                        '.btn, .button, .clickable',
                        'h1, h2, h3, h4, h5, h6, p, span, strong, em, label'
                    ];
                    
                    for (const selector of selectors) {
                        try {
                            targetElements.push(...Array.from(document.querySelectorAll(selector)));
                        } catch (e) {
                            console.warn(`选择器查询失败: ${selector}`, e);
                        }
                    }
                }
                
                // 过滤和处理元素
                for (const el of targetElements) {
                    if (processedElements.has(el)) continue;
                    
                    try {
                        if (isVisible(el)) {
                            const elementInfo = getElementInfo(el);
                            if (elementInfo.isInteractive || elementInfo.isUsefulText) {
                                allElements.push(elementInfo);
                                processedElements.add(el);
                            }
                        }
                    } catch (e) {
                        console.warn('处理元素时出错:', e);
                    }
                }
                
                // 按位置排序
                allElements.sort((a, b) => {
                    if (Math.abs(a.position.y - b.position.y) > 10) {
                        return a.position.y - b.position.y;
                    }
                    return a.position.x - b.position.x;
                });
                
                return allElements;
            }
            
            // 执行元素收集
            const flatElements = collectUsefulElements();
            
            return {
                flatElements,
                hierarchicalStructure: [], // 简化，不再构建复杂的层级结构
                summary: {
                    totalElements: flatElements.length,
                    interactiveElements: flatElements.filter(el => el.isInteractive).length,
                    textElements: flatElements.filter(el => el.isUsefulText).length,
                    pageTitle: document.title,
                    pageUrl: window.location.href
                }
            };
        }""", {
            "customSelector": selector,
            "attrsList": ["id", "name", "class", "placeholder", "aria-label", "role", "href", "value", "type", "title", "alt"]
        })

    async def _create_element_handles(self, page: Page, flat_elements: List[Dict]) -> Dict[str, ElementHandle]:
        """创建ElementHandle映射"""
        element_handles = {}
        
        for element_info in flat_elements:
            element_id = element_info["id"]
            try:
                handle = await page.query_selector(f'[data-agent-id="{element_id}"]')
                if handle:
                    element_handles[element_id] = handle
                    element_info["llm_description"] = self._generate_element_description(element_info)
                    element_info["selector"] = f'[data-agent-id="{element_id}"]'
            except Exception as e:
                logger.warning(f"无法获取元素 {element_id} 的ElementHandle: {e}")

        # 存储ElementHandle映射到页面对象中
        if not hasattr(page, '_agent_element_handles'):
            page._agent_element_handles = {}
        page._agent_element_handles.update(element_handles)
        
        return element_handles

    async def _fallback_get_elements(self, page: Page, enable_highlight: bool = True) -> Dict[str, Any]:
        """回退方案：获取基本元素信息"""
        try:
            logger.info("使用回退方案获取页面元素...")
            
            # 等待页面基本加载
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
            
            # 获取基本元素
            basic_elements = await page.query_selector_all("button, input, a, [role='button']")
            fallback_elements = []
            element_handles = {}
            
            for i, element in enumerate(basic_elements):
                try:
                    element_id = f"fallback_element_{i}"
                    await element.evaluate(f"el => el.setAttribute('data-agent-id', '{element_id}')")
                    
                    tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                    text_content = (await element.text_content() or "").strip()[:50]
                    is_visible = await element.is_visible()
                    
                    if is_visible:
                        element_info = {
                            "id": element_id,
                            "tag": tag_name,
                            "text": text_content,
                            "attrs": {},
                            "position": {"x": 0, "y": 0, "width": 0, "height": 0},
                            "isVisible": True,
                            "isInteractive": True,
                            "isUsefulText": bool(text_content),
                            "inViewport": True,
                            "llm_description": f"<{tag_name}> text: '{text_content}'",
                            "selector": f'[data-agent-id="{element_id}"]'
                        }
                        
                        fallback_elements.append(element_info)
                        element_handles[element_id] = element
                        
                except Exception as e:
                    logger.debug(f"处理回退元素时出错: {e}")
                    continue
            
            # 存储ElementHandle映射
            if not hasattr(page, '_agent_element_handles'):
                page._agent_element_handles = {}
            page._agent_element_handles.update(element_handles)
            
            # 添加高亮
            if enable_highlight and element_handles:
                await self._highlight_elements(page, list(element_handles.keys()))
            
            result = {
                "flatElements": fallback_elements,
                "hierarchicalStructure": [],
                "summary": {
                    "totalElements": len(fallback_elements),
                    "interactiveElements": len([el for el in fallback_elements if el.get("isInteractive")]),
                    "textElements": len([el for el in fallback_elements if el.get("isUsefulText")]),
                    "pageTitle": await page.title(),
                    "pageUrl": page.url
                },
                "elementHandles": element_handles,
                "elementCount": len(element_handles),
                "isFallback": True
            }
            
            logger.info(f"回退方案获取到 {len(fallback_elements)} 个基本元素")
            return result
            
        except Exception as e:
            logger.error(f"回退方案也失败: {e}")
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
                "isFallback": True,
                "fallbackFailed": True
            }

    def _generate_element_description(self, element_info: Dict[str, Any]) -> str:
        """为LLM生成简洁的元素描述"""
        tag = element_info.get("tag", "unknown")
        text = element_info.get("text", "")
        attrs = element_info.get("attrs", {})
        is_interactive = element_info.get("isInteractive", False)
        
        desc_parts = [f"<{tag}>"]
        
        if is_interactive:
            desc_parts.append("interactive")
        
        if text and len(text.strip()) > 0:
            desc_parts.append(f"text:'{text.strip()}'")
        
        # 添加重要属性
        important_attrs = ["id", "name", "placeholder", "aria-label", "type", "role"]
        for attr_name in important_attrs:
            if attr_name in attrs and attrs[attr_name]:
                desc_parts.append(f"{attr_name}:'{attrs[attr_name]}'")
        
        return ", ".join(desc_parts)

    async def _highlight_elements(self, page: Page, element_ids: List[str]) -> None:
        """为元素添加简洁的高亮显示"""
        try:
            await page.evaluate("""(elementIds) => {
                let styleSheet = document.getElementById('agent-highlight-styles');
                if (!styleSheet) {
                    styleSheet = document.createElement('style');
                    styleSheet.id = 'agent-highlight-styles';
                    document.head.appendChild(styleSheet);
                    styleSheet.textContent = `
                        .agent-highlight {
                            position: absolute !important;
                            border: 2px solid #ff4444 !important;
                            background: rgba(255, 68, 68, 0.1) !important;
                            pointer-events: none !important;
                            z-index: 999999 !important;
                            box-sizing: border-box !important;
                        }
                        .agent-highlight-label {
                            position: absolute !important;
                            background: #ff4444 !important;
                            color: white !important;
                            padding: 2px 6px !important;
                            font-size: 11px !important;
                            font-family: monospace !important;
                            font-weight: bold !important;
                            border-radius: 3px !important;
                            top: -20px !important;
                            left: 0 !important;
                            pointer-events: none !important;
                            z-index: 1000000 !important;
                        }
                    `;
                }

                const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
                const scrollY = window.pageYOffset || document.documentElement.scrollTop;

                elementIds.forEach(elementId => {
                    const element = document.querySelector(`[data-agent-id="${elementId}"]`);
                    if (element) {
                        const rect = element.getBoundingClientRect();
                        
                        const highlightBox = document.createElement('div');
                        highlightBox.className = 'agent-highlight';
                        highlightBox.setAttribute('data-agent-highlight', elementId);
                        highlightBox.style.left = (rect.left + scrollX) + 'px';
                        highlightBox.style.top = (rect.top + scrollY) + 'px';
                        highlightBox.style.width = rect.width + 'px';
                        highlightBox.style.height = rect.height + 'px';
                        
                        const label = document.createElement('div');
                        label.className = 'agent-highlight-label';
                        label.textContent = elementId;
                        highlightBox.appendChild(label);
                        
                        document.body.appendChild(highlightBox);
                    }
                });
            }""", element_ids)
            
            logger.info(f"已为 {len(element_ids)} 个元素添加高亮显示")
            
        except Exception as e:
            logger.warning(f"添加元素高亮时出错: {e}")

    async def _clear_highlights(self, page: Page) -> None:
        """清除所有高亮显示和相关样式"""
        try:
            highlight_count = await page.evaluate("""() => {
                const highlights = document.querySelectorAll('[data-agent-highlight]');
                const count = highlights.length;
                highlights.forEach(highlight => highlight.remove());
                return count;
            }""")
            
            if highlight_count > 0:
                logger.debug(f"已清除 {highlight_count} 个元素高亮显示")
            
        except Exception as e:
            logger.warning(f"清除元素高亮时出错: {e}")

class ScrollAction(Action):
    """滚动页面。"""
    def __init__(self):
        super().__init__(
            name="scroll_page",
            description="Scroll the page. Requires 'direction' ('up' or 'down') and optional 'amount' (pixels or 'full'). Defaults to scrolling down by one viewport height."
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> None:
        direction = kwargs.get("direction", "down")
        amount = kwargs.get("amount") # 'full' or pixels

        logger.info(f"滚动页面，方向: {direction}, 数量: {amount}")

        if amount == "full":
            if direction == "down":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                logger.info("已滚动到页面底部。")
            else: # up
                await page.evaluate("window.scrollTo(0, 0)")
                logger.info("已滚动到页面顶部。")
        else:
            scroll_amount_px = amount if isinstance(amount, int) else await page.evaluate("window.innerHeight")
            if direction == "down":
                await page.evaluate(f"window.scrollBy(0, {scroll_amount_px})")
            else: # up
                await page.evaluate(f"window.scrollBy(0, -{scroll_amount_px})")
            logger.info(f"页面已向 {direction} 滚动 {scroll_amount_px} 像素。")

class BrowserBackAction(Action):
    """浏览器后退。"""
    def __init__(self):
        super().__init__(name="browser_back", description="Navigate to the previous page in history.")

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> None:
        logger.info("执行浏览器后退操作。")
        await page.go_back(timeout=config.browser_config.timeout)
        logger.info(f"已后退到: {page.url}")

class BrowserForwardAction(Action):
    """浏览器前进。"""
    def __init__(self):
        super().__init__(name="browser_forward", description="Navigate to the next page in history.")

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> None:
        logger.info("执行浏览器前进操作。")
        await page.go_forward(timeout=config.browser_config.timeout)
        logger.info(f"已前进到: {page.url}")

class GetAllTabsAction(Action):
    """获取所有打开的标签页信息。"""
    def __init__(self):
        super().__init__(
            name="get_all_tabs",
            description="Get information (title, URL) for all open tabs in the current browser context. Returns a list of dictionaries, each with 'id', 'title', and 'url'."
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> List[Dict[str, Any]]:
        logger.info("获取所有打开的标签页信息。")
        context = page.context
        tabs_info = []
        for i, tab_page in enumerate(context.pages):
            tabs_info.append({
                "id": i, # 使用索引作为临时ID
                "title": await tab_page.title(),
                "url": tab_page.url,
                "is_current": tab_page == page
            })
        logger.info(f"获取到 {len(tabs_info)} 个标签页信息: {tabs_info}")
        return tabs_info

class SwitchTabAction(Action):
    """切换到指定的标签页。"""
    def __init__(self):
        super().__init__(
            name="switch_to_tab",
            description="Switch to a specific tab. Requires 'tab_id' (index from get_all_tabs) or 'url' or 'title' of the tab to switch to."
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> Page:
        tab_id = kwargs.get("tab_id") # Index-based ID from GetAllTabsAction
        url_to_switch = kwargs.get("url")
        title_to_switch = kwargs.get("title")
        
        context = page.context
        target_page: Page | None = None

        if tab_id is not None:
            logger.info(f"尝试切换到标签页 ID (索引): {tab_id}")
            if 0 <= tab_id < len(context.pages):
                target_page = context.pages[tab_id]
            else:
                logger.error(f"无效的标签页 ID: {tab_id}。标签页数量: {len(context.pages)}")
                raise ValueError(f"无效的标签页 ID: {tab_id}")
        elif url_to_switch:
            logger.info(f"尝试切换到 URL 包含 '{url_to_switch}' 的标签页。")
            for p in context.pages:
                if url_to_switch.lower() in p.url.lower():
                    target_page = p
                    break
        elif title_to_switch:
            logger.info(f"尝试切换到标题包含 '{title_to_switch}' 的标签页。")
            for p in context.pages:
                if title_to_switch.lower() in (await p.title()).lower():
                    target_page = p
                    break
        else:
            logger.error(f"'{self.name}' 动作需要 'tab_id' 或 'url' 或 'title' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'tab_id' 或 'url' 或 'title' 参数。")

        if target_page:
            await target_page.bring_to_front()
            logger.info(f"已切换到标签页: {await target_page.title()} ({target_page.url})")
            return target_page # 返回新的当前页面对象
        else:
            logger.error(f"未找到匹配的标签页进行切换。参数: tab_id={tab_id}, url={url_to_switch}, title={title_to_switch}")
            raise ValueError("未找到匹配的标签页进行切换。")

class NewTabAction(Action):
    """新建标签页。"""
    def __init__(self):
        super().__init__(
            name="new_tab",
            description="Create a new tab in the current browser context. Optional 'url' parameter to navigate to a specific URL immediately after creating the tab."
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> Page:
        url = kwargs.get("url")
        context = page.context
        
        logger.info("创建新标签页...")
        new_page = await context.new_page()
        
        if url:
            logger.info(f"新标签页导航到 URL: {url}")
            await new_page.goto(url, timeout=config.browser_config.timeout)
        
        # 切换到新标签页
        await new_page.bring_to_front()
        
        logger.info(f"已创建并切换到新标签页: {await new_page.title()} ({new_page.url})")
        return new_page  # 返回新页面对象

class CloseTabAction(Action):
    """关闭指定标签页。"""
    def __init__(self):
        super().__init__(
            name="close_tab",
            description="Close a specific tab. Requires 'tab_id' (index from get_all_tabs) or 'url' or 'title' of the tab to close. If closing the current tab, will automatically switch to another available tab."
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> Page:
        tab_id = kwargs.get("tab_id")
        url_to_close = kwargs.get("url") 
        title_to_close = kwargs.get("title")
        
        context = page.context
        target_page: Page | None = None

        if tab_id is not None:
            logger.info(f"尝试关闭标签页 ID (索引): {tab_id}")
            if 0 <= tab_id < len(context.pages):
                target_page = context.pages[tab_id]
            else:
                logger.error(f"无效的标签页 ID: {tab_id}。标签页数量: {len(context.pages)}")
                raise ValueError(f"无效的标签页 ID: {tab_id}")
        elif url_to_close:
            logger.info(f"尝试关闭 URL 包含 '{url_to_close}' 的标签页。")
            for p in context.pages:
                if url_to_close.lower() in p.url.lower():
                    target_page = p
                    break
        elif title_to_close:
            logger.info(f"尝试关闭标题包含 '{title_to_close}' 的标签页。")
            for p in context.pages:
                if title_to_close.lower() in (await p.title()).lower():
                    target_page = p
                    break
        else:
            logger.error(f"'{self.name}' 动作需要 'tab_id' 或 'url' 或 'title' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'tab_id' 或 'url' 或 'title' 参数。")

        if not target_page:
            logger.error(f"未找到匹配的标签页进行关闭。参数: tab_id={tab_id}, url={url_to_close}, title={title_to_close}")
            raise ValueError("未找到匹配的标签页进行关闭。")

        # 检查是否只剩一个标签页
        if len(context.pages) <= 1:
            logger.error("无法关闭最后一个标签页")
            raise ValueError("无法关闭最后一个标签页")

        # 检查是否要关闭当前页面
        is_current_page = target_page == page
        
        # 如果要关闭的是当前页面，先切换到其他页面
        new_current_page = page
        if is_current_page:
            # 找到另一个页面作为新的当前页面
            for p in context.pages:
                if p != target_page:
                    new_current_page = p
                    await p.bring_to_front()
                    logger.info(f"切换到其他标签页: {await p.title()} ({p.url})")
                    break

        # 关闭目标页面
        await target_page.close()
        logger.info(f"已关闭标签页: {await target_page.title() if not target_page.is_closed() else '已关闭的页面'}")
        
        return new_current_page  # 返回新的当前页面对象

class KeyboardInputAction(Action):
    """键盘输入动作，支持特殊键和组合键。"""
    def __init__(self):
        super().__init__(
            name="keyboard_input",
            description="Press keyboard keys or key combinations. Requires 'keys' (string or list of keys). Supports special keys like 'Enter', 'Tab', 'Escape', 'Control+A', etc."
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> None:
        keys = kwargs.get("keys")
        if not keys:
            logger.error(f"'{self.name}' 动作需要 'keys' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'keys' 参数。")
        
        # 如果是字符串，转换为列表
        if isinstance(keys, str):
            keys = [keys]
        
        logger.info(f"执行键盘输入: {keys}")
        
        for key in keys:
            try:
                # 处理组合键（如 Control+A）
                if '+' in key:
                    # 分解组合键
                    key_parts = key.split('+')
                    # 按下所有修饰键
                    for i, part in enumerate(key_parts[:-1]):
                        await page.keyboard.down(part)
                    # 按下最后一个键
                    await page.keyboard.press(key_parts[-1])
                    # 释放所有修饰键（反向顺序）
                    for i in range(len(key_parts) - 2, -1, -1):
                        await page.keyboard.up(key_parts[i])
                else:
                    # 单个键
                    await page.keyboard.press(key)
                
                logger.info(f"已按下键: {key}")
                
                # 按键之间的短暂延迟
                await page.wait_for_timeout(50)
                
            except Exception as e:
                logger.error(f"按键 '{key}' 失败: {e}")
                raise

class SaveToFileAction(Action):
    """保存内容到文件（txt或md格式）。"""
    def __init__(self):
        super().__init__(
            name="save_to_file",
            description="Save content to a file. Requires 'content' (text to save), 'filename' (with .txt or .md extension), and optional 'directory' (default: current directory)."
        )

    @with_timeout_and_retry
    async def execute(self, page: Page, **kwargs: Any) -> str:
        content = kwargs.get("content")
        filename = kwargs.get("filename")
        directory = kwargs.get("directory", ".")
        
        if not content:
            logger.error(f"'{self.name}' 动作需要 'content' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'content' 参数。")
        
        if not filename:
            logger.error(f"'{self.name}' 动作需要 'filename' 参数。")
            raise ValueError(f"'{self.name}' 动作需要 'filename' 参数。")
        
        # 验证文件扩展名
        if not filename.endswith(('.txt', '.md')):
            logger.error(f"文件名必须以 .txt 或 .md 结尾，当前: {filename}")
            raise ValueError(f"文件名必须以 .txt 或 .md 结尾")
        
        # 构建完整路径
        import os
        full_path = os.path.join(directory, filename)
        
        try:
            # 确保目录存在
            os.makedirs(directory, exist_ok=True)
            
            # 写入文件
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"内容已保存到文件: {full_path}")
            return full_path
            
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            raise

class GetAllElementsActionBs4(Action):
    """使用BeautifulSoup实现的高效元素提取动作，保留拓扑结构，专注提取关键信息和可交互元素"""
    
    def __init__(self):
        super().__init__(
            name="get_all_elements_info_bs4",
            description="Get information for all useful elements on the page using BeautifulSoup for better performance and topology preservation. Returns hierarchical element structure with handles."
        )
    
    @with_timeout_and_retry  
    async def execute(self, page: Page, **kwargs: Any) -> Dict[str, Any]:
        selector = kwargs.get("selector")
        enable_highlight = kwargs.get("enable_highlight", True)
        preserve_hierarchy = kwargs.get("preserve_hierarchy", True)
        
        logger.info(f"使用BeautifulSoup获取页面元素，保留层级: {preserve_hierarchy}, 启用高亮: {enable_highlight}")
        
        try:
            # 先清除之前的高亮
            await self._clear_highlights(page)
            
            # 获取页面HTML内容
            html_content = await page.content()
            
            # 使用BeautifulSoup解析
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取有用元素并保留拓扑结构
            elements_data = self._extract_useful_elements(soup, preserve_hierarchy)
            
            # 创建ElementHandle映射（基于属性匹配）
            element_handles = await self._create_element_handles_bs4(page, elements_data['flatElements'])
            
            # 添加高亮功能
            if enable_highlight and element_handles:
                await self._highlight_elements(page, list(element_handles.keys()))
            
            result = {
                **elements_data,
                "elementHandles": element_handles,
                "elementCount": len(element_handles),
                "method": "BeautifulSoup"
            }
            
            logger.info(f"BeautifulSoup提取到 {len(element_handles)} 个有用元素")
            return result
            
        except Exception as e:
            logger.warning(f"BeautifulSoup元素提取失败，回退到原方案: {e}")
            # 回退到原始方案
            original_action = GetAllElementsAction()
            return await original_action.execute(page, **kwargs)
    
    def _extract_useful_elements(self, soup: BeautifulSoup, preserve_hierarchy: bool = True) -> Dict[str, Any]:
        """提取有用元素，保留拓扑结构"""
        
        # 定义重要的交互元素
        interactive_tags = {
            'button', 'input', 'textarea', 'select', 'a', 'label', 'option',
            'summary', 'details', 'audio', 'video'  
        }
        
        # 定义重要的文本元素
        text_tags = {
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'strong', 'em',
            'title', 'caption', 'legend', 'figcaption'
        }
        
        # 定义布局/装饰元素（通常忽略）
        decorative_tags = {
            'script', 'style', 'noscript', 'meta', 'link', 'base', 'head',
            'svg', 'path', 'g', 'circle', 'rect', 'line', 'polygon'
        }
        
        flat_elements = []
        hierarchical_structure = [] if preserve_hierarchy else None
        element_counter = 0
        
        def is_visible_element(element):
            """判断元素是否可能可见（基于HTML属性）"""
            if not hasattr(element, 'name') or not element.name:
                return False
                
            # 检查display和visibility样式
            style = element.get('style', '')
            if any(hidden in style.lower() for hidden in ['display:none', 'display: none', 'visibility:hidden', 'visibility: hidden']):
                return False
                
            # 检查hidden属性
            if element.get('hidden') is not None:
                return False
                
            # 检查aria-hidden
            if element.get('aria-hidden') == 'true':
                return False
                
            return True
        
        def is_interactive_element(element):
            """判断是否为交互元素"""
            if not hasattr(element, 'name'):
                return False
                
            tag_name = element.name.lower()
            
            # 明确的交互标签
            if tag_name in interactive_tags:
                return True
                
            # 检查role属性
            role = element.get('role', '').lower()
            interactive_roles = {
                'button', 'link', 'textbox', 'searchbox', 'combobox', 
                'checkbox', 'radio', 'menuitem', 'tab', 'switch', 'slider'
            }
            if role in interactive_roles:
                return True
                
            # 检查事件属性
            event_attrs = ['onclick', 'onchange', 'onsubmit', 'onkeydown', 'onkeyup']
            if any(element.get(attr) for attr in event_attrs):
                return True
                
            # 检查tabindex
            if element.get('tabindex') is not None:
                return True
                
            # 检查contenteditable
            if element.get('contenteditable') == 'true':
                return True
                
            # 检查类名中的交互模式
            class_names = element.get('class', [])
            if isinstance(class_names, str):
                class_names = class_names.split()
                
            interactive_patterns = [
                'btn', 'button', 'clickable', 'click', 'link', 'nav-item',
                'menu-item', 'dropdown', 'toggle', 'search-btn', 'submit'
            ]
            
            for class_name in class_names:
                if any(pattern in class_name.lower() for pattern in interactive_patterns):
                    return True
                    
            return False
        
        def is_useful_text_element(element):
            """判断是否为有用的文本元素"""
            if not hasattr(element, 'name'):
                return False
                
            tag_name = element.name.lower()
            
            # 明确的文本标签
            if tag_name in text_tags:
                text = element.get_text(strip=True)
                # 过滤空文本和过长文本
                return 0 < len(text) <= 200
                
            # 检查是否有重要的类名
            class_names = element.get('class', [])
            if isinstance(class_names, str):
                class_names = class_names.split()
                
            important_patterns = [
                'title', 'heading', 'label', 'name', 'text', 'message',
                'description', 'content', 'info', 'note'
            ]
            
            for class_name in class_names:
                if any(pattern in class_name.lower() for pattern in important_patterns):
                    text = element.get_text(strip=True)
                    return 0 < len(text) <= 200
                    
            return False
        
        def should_ignore_element(element):
            """判断是否应该忽略元素"""
            if not hasattr(element, 'name'):
                return True
                
            tag_name = element.name.lower()
            
            # 忽略装饰性元素
            if tag_name in decorative_tags:
                return True
                
            # 忽略不可见元素
            if not is_visible_element(element):
                return True
                
            return False
        
        def get_element_importance_score(element):
            """计算元素重要性评分"""
            score = 0
            
            if is_interactive_element(element):
                score += 10
                
            if is_useful_text_element(element):
                text_length = len(element.get_text(strip=True))
                if text_length < 50:
                    score += 5  # 短文本更重要
                else:
                    score += 2  # 长文本次要
                    
            # 根据标签重要性加分
            tag_name = element.name.lower()
            tag_importance = {
                'h1': 8, 'h2': 7, 'h3': 6, 'h4': 5, 'h5': 4, 'h6': 3,
                'button': 8, 'input': 7, 'textarea': 6, 'select': 6,
                'a': 5, 'label': 4, 'p': 3, 'span': 2, 'div': 1
            }
            score += tag_importance.get(tag_name, 0)
            
            return score
        
        def extract_element_info(element):
            """提取单个元素的信息"""
            nonlocal element_counter
            element_counter += 1
            
            tag_name = element.name.lower()
            element_id = f"bs4_element_{element_counter}"
            
            # 获取文本内容
            text = element.get_text(strip=True)
            # 获取直接文本内容（不包括子元素）
            direct_text_parts = []
            for child in element.children:
                if isinstance(child, NavigableString):
                    text_part = str(child).strip()
                    if text_part:
                        direct_text_parts.append(text_part)
            direct_text = ' '.join(direct_text_parts)
            
            # 提取关键属性
            attrs = {}
            important_attrs = [
                'id', 'name', 'class', 'placeholder', 'aria-label', 'role',
                'href', 'value', 'type', 'title', 'alt', 'src', 'data-*'
            ]
            
            for attr_name in important_attrs:
                if attr_name == 'data-*':
                    # 提取所有data-属性
                    for attr, value in element.attrs.items():
                        if attr.startswith('data-'):
                            attrs[attr] = str(value)[:100] if value else ''
                else:
                    value = element.get(attr_name)
                    if value:
                        if isinstance(value, list):
                            attrs[attr_name] = ' '.join(value)[:100]
                        else:
                            attrs[attr_name] = str(value)[:100]
            
            # 生成用于匹配的选择器
            selectors = []
            if attrs.get('id'):
                selectors.append(f"#{attrs['id']}")
            if attrs.get('class'):
                classes = attrs['class'].split()[:3]  # 限制类名数量
                if classes:
                    selectors.append(f".{'.'.join(classes)}")
            if not selectors:
                selectors.append(tag_name)
            
            return {
                'id': element_id,
                'tag': tag_name,
                'text': (direct_text or text)[:100],  # 限制文本长度
                'fullText': text[:200],  # 完整文本（稍长）
                'attrs': attrs,
                'isInteractive': is_interactive_element(element),
                'isUsefulText': is_useful_text_element(element),
                'importanceScore': get_element_importance_score(element),
                'selectors': selectors,  # 用于后续匹配ElementHandle
                'llm_description': self._generate_element_description_bs4({
                    'tag': tag_name,
                    'text': (direct_text or text)[:50],
                    'attrs': attrs,
                    'isInteractive': is_interactive_element(element),
                    'isUsefulText': is_useful_text_element(element)
                }),
                'xpath': None,  # 可能的XPath，暂时留空
                'children_count': len([child for child in element.children if hasattr(child, 'name')])
            }
        
        def extract_recursive(element, parent_info=None, depth=0):
            """递归提取元素信息"""
            if depth > 10:  # 限制递归深度
                return None
                
            if should_ignore_element(element):
                return None
            
            element_info = extract_element_info(element)
            
            # 只保留重要的元素
            if element_info['importanceScore'] < 2:
                return None
                
            # 处理子元素
            children = []
            if hasattr(element, 'children'):
                for child in element.children:
                    if hasattr(child, 'name'):  # 跳过文本节点
                        child_info = extract_recursive(child, element_info, depth + 1)  
                        if child_info:
                            children.append(child_info)
            
            element_info['children'] = children
            flat_elements.append(element_info)
            
            return element_info
        
        # 从body开始提取（如果没有body则从html开始）
        root_element = soup.find('body') or soup.find('html') or soup
        
        if preserve_hierarchy:
            hierarchical_structure = extract_recursive(root_element)
        else:
            # 扁平化提取
            for element in root_element.find_all():
                if not should_ignore_element(element):
                    element_info = extract_element_info(element)
                    if element_info['importanceScore'] >= 2:
                        flat_elements.append(element_info)
        
        # 按重要性和位置排序
        flat_elements.sort(key=lambda x: (-x['importanceScore'], x['tag']))
        
        # 生成摘要
        summary = {
            'totalElements': len(flat_elements),
            'interactiveElements': len([el for el in flat_elements if el['isInteractive']]),
            'textElements': len([el for el in flat_elements if el['isUsefulText']]),
            'pageTitle': soup.title.string if soup.title else "无标题",
            'pageUrl': 'bs4_extracted',
            'averageImportanceScore': sum(el['importanceScore'] for el in flat_elements) / len(flat_elements) if flat_elements else 0
        }
        
        return {
            'flatElements': flat_elements,
            'hierarchicalStructure': [hierarchical_structure] if hierarchical_structure else [],
            'summary': summary
        }
    
    def _generate_element_description_bs4(self, element_info: Dict[str, Any]) -> str:
        """为LLM生成元素描述（BeautifulSoup版本）"""
        tag = element_info.get('tag', 'unknown')
        text = element_info.get('text', '')
        attrs = element_info.get('attrs', {})
        is_interactive = element_info.get('isInteractive', False)
        is_useful_text = element_info.get('isUsefulText', False)
        
        desc_parts = [f"<{tag}>"]
        
        if is_interactive:
            desc_parts.append("interactive")
        if is_useful_text:
            desc_parts.append("text-content")
            
        if text and len(text.strip()) > 0:
            desc_parts.append(f"text:'{text.strip()}'")
        
        # 添加关键属性
        important_attrs = ["id", "name", "placeholder", "aria-label", "type", "role", "href"]
        for attr_name in important_attrs:
            if attr_name in attrs and attrs[attr_name]:
                desc_parts.append(f"{attr_name}:'{attrs[attr_name]}'")
        
        return ", ".join(desc_parts)
    
    async def _create_element_handles_bs4(self, page: Page, flat_elements: List[Dict]) -> Dict[str, ElementHandle]:
        """为BeautifulSoup提取的元素创建ElementHandle映射"""
        element_handles = {}
        
        for element_info in flat_elements:
            element_id = element_info["id"]
            selectors = element_info.get("selectors", [])
            
            # 尝试多个选择器来匹配元素
            handle = None
            for selector in selectors:
                try:
                    # 尝试匹配唯一元素
                    elements = await page.query_selector_all(selector)
                    if len(elements) == 1:
                        handle = elements[0]
                        break
                    elif len(elements) > 1:
                        # 多个匹配时，尝试通过文本内容进一步过滤
                        expected_text = element_info.get('text', '').strip()
                        if expected_text:
                            for elem in elements:
                                try:
                                    elem_text = (await elem.text_content() or '').strip()
                                    if expected_text in elem_text or elem_text in expected_text:
                                        handle = elem
                                        break
                                except:
                                    continue
                        if handle:
                            break
                except Exception as e:
                    logger.debug(f"选择器 '{selector}' 匹配失败: {e}")
                    continue
            
            if handle:
                try:
                    # 为元素添加标识属性
                    await handle.evaluate(f"el => el.setAttribute('data-agent-id', '{element_id}')")
                    element_handles[element_id] = handle
                    element_info["selector"] = f'[data-agent-id="{element_id}"]'
                except Exception as e:
                    logger.debug(f"无法为元素 {element_id} 设置标识: {e}")
                    # 即使无法设置标识，也保留ElementHandle
                    element_handles[element_id] = handle
            else:
                logger.debug(f"无法为元素 {element_id} 找到匹配的ElementHandle (选择器: {selectors})")
        
        # 存储ElementHandle映射到页面对象中
        if not hasattr(page, '_agent_element_handles'):
            page._agent_element_handles = {}
        page._agent_element_handles.update(element_handles)
        
        return element_handles
    
    async def _clear_highlights(self, page: Page) -> None:
        """清除所有高亮显示（复用原有方法）"""
        try:
            highlight_count = await page.evaluate("""() => {
                const highlights = document.querySelectorAll('[data-agent-highlight]');
                const count = highlights.length;
                highlights.forEach(highlight => highlight.remove());
                return count;
            }""")
            
            if highlight_count > 0:
                logger.debug(f"已清除 {highlight_count} 个元素高亮显示")
                
        except Exception as e:
            logger.warning(f"清除元素高亮时出错: {e}")
    
    async def _highlight_elements(self, page: Page, element_ids: List[str]) -> None:
        """为元素添加高亮显示（复用原有方法）"""
        try:
            await page.evaluate("""(elementIds) => {
                let styleSheet = document.getElementById('agent-highlight-styles');
                if (!styleSheet) {
                    styleSheet = document.createElement('style');
                    styleSheet.id = 'agent-highlight-styles';
                    document.head.appendChild(styleSheet);
                    styleSheet.textContent = `
                        .agent-highlight {
                            position: absolute !important;
                            border: 2px solid #00ff00 !important;
                            background: rgba(0, 255, 0, 0.1) !important;
                            pointer-events: none !important;
                            z-index: 999999 !important;
                            box-sizing: border-box !important;
                        }
                        .agent-highlight-label {
                            position: absolute !important;
                            background: #00ff00 !important;
                            color: black !important;
                            padding: 2px 6px !important;
                            font-size: 11px !important;
                            font-family: monospace !important;
                            font-weight: bold !important;
                            border-radius: 3px !important;
                            top: -20px !important;
                            left: 0 !important;
                            pointer-events: none !important;
                            z-index: 1000000 !important;
                        }
                    `;
                }

                const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
                const scrollY = window.pageYOffset || document.documentElement.scrollTop;

                elementIds.forEach(elementId => {
                    const element = document.querySelector(`[data-agent-id="${elementId}"]`);
                    if (element) {
                        const rect = element.getBoundingClientRect();
                        
                        const highlightBox = document.createElement('div');
                        highlightBox.className = 'agent-highlight';
                        highlightBox.setAttribute('data-agent-highlight', elementId);
                        highlightBox.style.left = (rect.left + scrollX) + 'px';
                        highlightBox.style.top = (rect.top + scrollY) + 'px';
                        highlightBox.style.width = rect.width + 'px';
                        highlightBox.style.height = rect.height + 'px';
                        
                        const label = document.createElement('div');
                        label.className = 'agent-highlight-label';
                        label.textContent = elementId;
                        highlightBox.appendChild(label);
                        
                        document.body.appendChild(highlightBox);
                    }
                });
            }""", element_ids)
            
            logger.info(f"已为 {len(element_ids)} 个元素添加绿色高亮显示 (BeautifulSoup)")
            
        except Exception as e:
            logger.warning(f"添加元素高亮时出错: {e}")

# 可以根据需要添加更多 Action 类

if __name__ == '__main__':
    # 简单的测试
    navigate = NavigateAction()
    print(navigate.to_dict())

    click = ClickAction()
    print(click.to_dict())

    type_text = TypeAction()
    print(type_text.to_dict()) 