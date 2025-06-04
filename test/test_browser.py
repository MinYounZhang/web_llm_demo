import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from src.browser.browser_manager import BrowserManager
from src.browser.actions import (
    NavigateAction, ClickAction, TypeAction, WaitAction,
    GetTextAction, GetAllElementsAction, ScrollAction,
    BrowserBackAction, BrowserForwardAction, GetAllTabsAction,
    SwitchTabAction
)
from src.config import config

# Mock playwright 模块
patchright_mock = MagicMock()
patch('patchright.async_api', patchright_mock).start()

@pytest.fixture
def mock_config():
    mock_config = Mock()
    mock_config.browser_config = Mock()
    mock_config.browser_config.headless = False
    mock_config.browser_config.timeout = 60000
    mock_config.browser_config.user_agent = "Mozilla/5.0"
    mock_config.browser_config.navigation_timeout = 60000
    return mock_config

@pytest.fixture
def browser_manager():
    manager = BrowserManager()
    return manager

@pytest.mark.asyncio
async def test_browser_manager_initialization(browser_manager):
    """测试浏览器管理器初始化"""
    with patch('patchright.async_api.async_playwright') as mock_playwright:
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        
        mock_playwright.return_value.start.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        
        await browser_manager.launch_browser()
        
        assert browser_manager._browser == mock_browser
        assert browser_manager._context == mock_context
        assert browser_manager._page == mock_page
        
        mock_playwright.return_value.start.return_value.chromium.launch.assert_called_once()
        mock_browser.new_context.assert_called_once()
        mock_context.new_page.assert_called_once()

@pytest.mark.asyncio
async def test_browser_manager_close(browser_manager):
    """测试浏览器管理器关闭"""
    mock_browser = AsyncMock()
    mock_playwright = AsyncMock()
    browser_manager._browser = mock_browser
    browser_manager._playwright = mock_playwright
    
    await browser_manager.close_browser()
    
    mock_browser.close.assert_awaited_once()
    mock_playwright.stop.assert_awaited_once()
    assert browser_manager._browser is None
    assert browser_manager._context is None
    assert browser_manager._page is None
    assert browser_manager._playwright is None

@pytest.mark.asyncio
async def test_navigate_action():
    """测试导航动作"""
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    action = NavigateAction()
    
    await action.execute(mock_page, url="https://example.com")
    
    mock_page.goto.assert_awaited_with("https://example.com", timeout=config.timeout)
    
    # 测试错误情况
    with pytest.raises(ValueError):
        await action.execute(mock_page)  # 没有提供 URL

@pytest.mark.asyncio
async def test_click_action():
    """测试点击动作"""
    mock_page = AsyncMock()
    mock_element = AsyncMock()
    mock_locator = AsyncMock()
    mock_locator.first = mock_element
    mock_page.locator.return_value = mock_locator
    mock_element.is_visible = AsyncMock(return_value=True)
    mock_element.click = AsyncMock()
    mock_element.hover = AsyncMock()
    mock_element.bounding_box = AsyncMock(return_value={'x': 100, 'y': 100, 'width': 50, 'height': 20})
    mock_page.mouse = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    
    action = ClickAction()
    
    await action.execute(mock_page, selector="#test-button")
    
    mock_page.locator.assert_called_with("#test-button")
    mock_element.click.assert_awaited_once()
    
    # 测试元素不可见的情况
    mock_element.is_visible = AsyncMock(return_value=False)
    await action.execute(mock_page, selector="#test-button")
    mock_element.click.assert_awaited()

@pytest.mark.asyncio
async def test_type_action():
    """测试输入文本动作"""
    mock_page = AsyncMock()
    mock_element = AsyncMock()
    mock_locator = AsyncMock()
    mock_locator.first = mock_element
    mock_page.locator.return_value = mock_locator
    mock_element.type = AsyncMock()
    mock_element.hover = AsyncMock()
    mock_element.bounding_box = AsyncMock(return_value={'x': 100, 'y': 100, 'width': 50, 'height': 20})
    mock_page.mouse = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    
    action = TypeAction()
    
    await action.execute(mock_page, selector="#input", text="test text", delay=10)
    
    mock_page.locator.assert_called_with("#input")
    mock_element.type.assert_awaited_with("test text", delay=10, timeout=config.timeout)
    
    # 测试错误情况
    with pytest.raises(ValueError):
        await action.execute(mock_page, selector="#input")  # 没有提供文本

@pytest.mark.asyncio
async def test_wait_action():
    """测试等待动作"""
    mock_page = AsyncMock()
    action = WaitAction()
    
    # 测试等待时间
    await action.execute(mock_page, duration_ms=1000)
    mock_page.wait_for_timeout.assert_awaited_with(1000)
    
    # 测试等待元素
    await action.execute(mock_page, selector="#element")
    mock_page.wait_for_selector.assert_awaited_with("#element", timeout=config.timeout)
    
    # 测试错误情况
    with pytest.raises(ValueError):
        await action.execute(mock_page)  # 没有提供任何参数

@pytest.mark.asyncio
async def test_get_text_action():
    """测试获取文本动作"""
    mock_page = AsyncMock()
    mock_element = AsyncMock()
    mock_locator = AsyncMock()
    mock_locator.first = mock_element
    mock_page.locator.return_value = mock_locator
    mock_element.text_content = AsyncMock(return_value="test content")
    
    action = GetTextAction()
    
    result = await action.execute(mock_page, selector="#text")
    
    assert result == "test content"
    mock_page.locator.assert_called_with("#text")
    mock_element.text_content.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_all_elements_action():
    """测试获取所有元素动作"""
    mock_page = AsyncMock()
    mock_elements = [AsyncMock(), AsyncMock()]
    mock_page.query_selector_all = AsyncMock(return_value=mock_elements)
    
    # 模拟元素属性
    for element in mock_elements:
        element.is_visible = AsyncMock(return_value=True)
        element.evaluate = AsyncMock(return_value="button")
        element.text_content = AsyncMock(return_value="Button Text")
        element.get_attribute = AsyncMock(return_value="test-class")
    
    action = GetAllElementsAction()
    
    result = await action.execute(mock_page)
    
    assert len(result) == 2
    assert all("tag" in item for item in result)
    assert all("text" in item for item in result)
    assert all("attributes" in item for item in result)

@pytest.mark.asyncio
async def test_scroll_action():
    """测试滚动动作"""
    mock_page = AsyncMock()
    action = ScrollAction()
    
    # 测试完整滚动
    await action.execute(mock_page, direction="down", amount="full")
    mock_page.evaluate.assert_awaited_once()
    
    # 测试指定像素滚动
    mock_page.evaluate.reset_mock()
    await action.execute(mock_page, direction="up", amount=100)
    mock_page.evaluate.assert_awaited_once()

@pytest.mark.asyncio
async def test_browser_navigation_actions():
    """测试浏览器导航动作（前进/后退）"""
    mock_page = AsyncMock()
    mock_page.go_back = AsyncMock()
    mock_page.go_forward = AsyncMock()
    
    # 测试后退
    back_action = BrowserBackAction()
    await back_action.execute(mock_page)
    mock_page.go_back.assert_awaited_once()
    
    # 测试前进
    forward_action = BrowserForwardAction()
    await forward_action.execute(mock_page)
    mock_page.go_forward.assert_awaited_once()

@pytest.mark.asyncio
async def test_tab_management_actions():
    """测试标签页管理动作"""
    mock_page = AsyncMock()
    mock_context = AsyncMock()
    mock_page.context = mock_context
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.url = "https://example.com"
    
    # 测试获取所有标签页
    mock_pages = [mock_page, mock_page]
    mock_context.pages = mock_pages
    get_tabs_action = GetAllTabsAction()
    tabs = await get_tabs_action.execute(mock_page)
    assert len(tabs) == 2
    
    # 测试切换标签页
    switch_tab_action = SwitchTabAction()
    mock_context.pages = [mock_page]
    with pytest.raises(ValueError):
        await switch_tab_action.execute(mock_page, tab_index=1)  # 索引越界

@pytest.mark.asyncio
async def test_browser_manager_new_page(browser_manager):
    """测试创建新页面"""
    mock_page = AsyncMock()
    mock_context = AsyncMock()
    browser_manager._context = mock_context
    mock_context.new_page = AsyncMock(return_value=mock_page)
    
    with patch.object(browser_manager, '_apply_stealth_mechanisms', new_callable=AsyncMock):
        new_page = await browser_manager.new_page()
        assert new_page == mock_page
        mock_context.new_page.assert_awaited_once()

@pytest.mark.asyncio
async def test_browser_manager_get_current_page(browser_manager):
    """测试获取当前页面"""
    mock_page = AsyncMock()
    mock_page.is_closed = AsyncMock(return_value=False)
    browser_manager._page = mock_page
    
    current_page = await browser_manager.get_current_page()
    assert current_page == mock_page
    
    # 测试页面已关闭的情况
    mock_page.is_closed = AsyncMock(return_value=True)
    mock_new_page = AsyncMock()
    with patch.object(browser_manager, 'new_page', new_callable=AsyncMock, return_value=mock_new_page):
        result = await browser_manager.get_current_page()
        browser_manager.new_page.assert_awaited_once()

@pytest.mark.asyncio
async def test_browser_manager_execute_action(browser_manager):
    """测试执行动作"""
    mock_page = AsyncMock()
    mock_page.is_closed = AsyncMock(return_value=False)
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.goto = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    browser_manager._page = mock_page
    
    # 注册一个测试动作
    test_action = NavigateAction()
    browser_manager.add_action(test_action)
    
    # 测试正常执行
    await browser_manager.execute_action("navigate_to_url", url="https://example.com")
    mock_page.goto.assert_awaited_with("https://example.com", timeout=config.timeout)
    
    # 测试未注册的动作
    with pytest.raises(ValueError):
        await browser_manager.execute_action("non_existent_action")
    
    # 测试页面已关闭的情况
    mock_page.is_closed = AsyncMock(return_value=True)
    with pytest.raises(ValueError):
        await browser_manager.execute_action("navigate_to_url", url="https://example.com")

@pytest.mark.asyncio
async def test_browser_manager_stealth_mechanisms(browser_manager):
    """测试反自动化检测机制"""
    mock_page = AsyncMock()
    mock_page.add_init_script = AsyncMock()
    
    await browser_manager._apply_stealth_mechanisms(mock_page)
    mock_page.add_init_script.assert_awaited_once() 