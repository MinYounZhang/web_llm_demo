"""
错误管理模块
提供重试机制、超时处理和人工干预检测功能。
"""
import logging
import asyncio
import time
from typing import Any, Callable, Optional, Tuple, Union
from functools import wraps
from enum import Enum

from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)
from patchright.async_api import TimeoutError as PlaywrightTimeoutError, Page

from src.config import config, logger


class ErrorType(Enum):
    """错误类型枚举"""
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    ELEMENT_NOT_FOUND = "element_not_found"
    HUMAN_INTERVENTION_NEEDED = "human_intervention_needed"
    LLM_CALL_FAILED = "llm_call_failed"
    BROWSER_CRASHED = "browser_crashed"
    UNKNOWN = "unknown"


class HumanInterventionError(Exception):
    """需要人工干预的错误"""
    def __init__(self, message: str, suggested_action: str = None):
        super().__init__(message)
        self.suggested_action = suggested_action


class ActionTimeoutError(Exception):
    """Action执行超时错误"""
    def __init__(self, action_name: str, timeout_duration: float):
        super().__init__(f"Action '{action_name}' 在 {timeout_duration}s 后超时")
        self.action_name = action_name
        self.timeout_duration = timeout_duration


def get_error_type(exception: Exception) -> ErrorType:
    """根据异常类型判断错误类型"""
    if isinstance(exception, (PlaywrightTimeoutError, ActionTimeoutError, asyncio.TimeoutError)):
        return ErrorType.TIMEOUT
    elif isinstance(exception, HumanInterventionError):
        return ErrorType.HUMAN_INTERVENTION_NEEDED
    elif "connection" in str(exception).lower() or "network" in str(exception).lower():
        return ErrorType.CONNECTION
    elif "element" in str(exception).lower() and "not found" in str(exception).lower():
        return ErrorType.ELEMENT_NOT_FOUND
    elif "browser" in str(exception).lower() and "crash" in str(exception).lower():
        return ErrorType.BROWSER_CRASHED
    else:
        return ErrorType.UNKNOWN


async def detect_human_intervention_needed(page: Page) -> Tuple[bool, str]:
    """
    检测是否需要人工干预（更加宽松的版本）
    主要针对验证码、明确的登录要求等复杂场景
    
    Returns:
        Tuple[bool, str]: (是否需要干预, 检测到的问题描述)
    """
    try:
        # 1. 检测验证码（更精确的检测）
        captcha_indicators = [
            # reCAPTCHA
            "iframe[src*='recaptcha']",
            ".g-recaptcha",
            "#recaptcha",
            
            # 通用验证码元素
            "[class*='captcha'][style*='display: block'], [class*='captcha']:not([style*='display: none'])",
            "[id*='captcha'][style*='display: block'], [id*='captcha']:not([style*='display: none'])",
            
            # 图片验证码
            "img[src*='captcha']",
            "img[alt*='验证码'], img[alt*='captcha']",
            
            # 滑动验证码
            "[class*='slider'][class*='captcha']",
            "[class*='slide'][class*='verify']"
        ]
        
        for selector in captcha_indicators:
            try:
                elements = await page.locator(selector).all()
                for element in elements:
                    if await element.is_visible():
                        return True, f"检测到验证码元素: {selector}"
            except Exception:
                continue

        # 2. 检测明确的登录弹窗或页面
        login_indicators = [
            # 登录弹窗
            ".login-modal:visible, .login-popup:visible, .login-dialog:visible",
            "[class*='login'][class*='modal']:visible",
            "[class*='login'][class*='popup']:visible",
            
            # 登录表单（必须同时包含用户名和密码字段）
            "form:has(input[type='password']):has(input[name*='user'], input[name*='email'], input[name*='login'])",
            
            # 明确的登录页面标识
            "body:has(h1:text-matches('登录|Login|Sign In', 'i')):has(input[type='password'])",
            "body:has(.login-title, .login-header):has(input[type='password'])"
        ]
        
        for selector in login_indicators:
            try:
                if await page.locator(selector).count() > 0:
                    return True, f"检测到登录界面: {selector}"
            except Exception:
                continue

        # 3. 检测明确的权限错误页面（非常严格）
        permission_error_indicators = [
            # 403/401错误页面
            "body:has(h1:text-matches('403|401|Forbidden|Unauthorized', 'i'))",
            
            # 明确的权限错误信息
            "body:has(:text-matches('您没有权限访问|Access Denied|Permission Denied|Unauthorized Access', 'i'))",
            
            # 需要登录的明确提示（必须是页面主要内容）
            "main:has(:text-matches('请先登录|Please log in|Login required', 'i'))",
            ".content:has(:text-matches('请先登录|Please log in|Login required', 'i'))"
        ]
        
        for selector in permission_error_indicators:
            try:
                if await page.locator(selector).count() > 0:
                    return True, f"检测到权限错误: {selector}"
            except Exception:
                continue

        # 4. 检测二次验证（2FA）
        twofa_indicators = [
            "input[placeholder*='验证码'], input[placeholder*='verification'], input[placeholder*='code']",
            ".two-factor, .2fa, .verification-code",
            ":text-matches('请输入验证码|Enter verification code|Two-factor', 'i')"
        ]
        
        for selector in twofa_indicators:
            try:
                elements = await page.locator(selector).all()
                for element in elements:
                    if await element.is_visible():
                        # 确保这是一个验证码输入场景，而不是普通的搜索框等
                        parent_text = await element.locator("..").text_content()
                        if any(keyword in parent_text.lower() for keyword in ['验证', 'verification', 'code', '2fa', 'two-factor']):
                            return True, f"检测到二次验证: {selector}"
            except Exception:
                continue

        return False, "未检测到需要人工干预的情况"
    
    except Exception as e:
        logger.warning(f"检测人工干预时出错: {e}")
        # 检测出错时不认为需要人工干预，让系统继续运行
        return False, f"检测异常但继续执行: {str(e)}"


class ErrorManager:
    """错误管理器"""
    
    def __init__(self):
        self.intervention_timeout = config.agent_config.human_intervention_timeout_s
    
    def create_action_retry_decorator(self, action_name: str):
        """为Action创建重试装饰器"""
        
        def decorator(func):
            @retry(
                stop=stop_after_attempt(config.agent_config.max_action_retries),
                wait=wait_exponential(
                    multiplier=1, 
                    min=config.agent_config.action_retry_delay_ms / 1000,
                    max=10
                ),
                retry=retry_if_exception_type((
                    PlaywrightTimeoutError,
                    ActionTimeoutError,
                    ConnectionError,
                    asyncio.TimeoutError
                )),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True
            )
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_type = get_error_type(e)
                    logger.error(f"Action '{action_name}' 执行失败: {e}, 错误类型: {error_type.value}")
                    
                    # 如果是需要人工干预的错误，不重试
                    if error_type == ErrorType.HUMAN_INTERVENTION_NEEDED:
                        raise HumanInterventionError(
                            f"Action '{action_name}' 需要人工干预: {str(e)}",
                            f"请手动处理后重试"
                        )
                    
                    raise
            
            return wrapper
        return decorator
    
    def create_llm_retry_decorator(self):
        """为LLM调用创建重试装饰器"""
        
        @retry(
            stop=stop_after_attempt(config.agent_config.max_llm_retries),
            wait=wait_exponential(
                multiplier=1,
                min=config.agent_config.llm_retry_delay_ms / 1000,
                max=30
            ),
            retry=retry_if_exception_type((
                ConnectionError,
                TimeoutError,
                Exception  # 对于LLM调用，我们对大部分异常都进行重试
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"LLM调用失败: {e}")
                    raise
            return wrapper
        
        return decorator
    
    async def handle_action_timeout(self, action_name: str, page: Page, timeout_duration: float) -> Tuple[bool, str]:
        """
        处理Action超时（宽松版本）
        
        Returns:
            Tuple[bool, str]: (是否应该重试, 处理建议)
        """
        logger.warning(f"Action '{action_name}' 超时 ({timeout_duration}s)")
        
        # 只检测明确需要人工干预的情况（如验证码、权限提示）
        needs_intervention, reason = await detect_human_intervention_needed(page)
        if needs_intervention:
            logger.warning(f"检测到需要人工干预: {reason}")
            return False, f"需要人工干预: {reason}"
        
        # 默认情况下都允许重试
        logger.info("超时但未检测到阻断性问题，建议重试")
        return True, "操作超时但可以继续重试"
    
    async def wait_for_human_intervention(self, page: Page, issue_description: str) -> bool:
        """
        等待人工干预（更加宽松的处理）
        
        Args:
            page: 当前页面
            issue_description: 问题描述
            
        Returns:
            bool: 是否成功解决问题
        """
        logger.info(f"等待人工干预解决问题: {issue_description}")
        logger.info(f"请在 {self.intervention_timeout} 秒内手动解决问题，然后程序将自动继续...")
        
        start_time = time.time()
        initial_url = page.url
        
        while time.time() - start_time < self.intervention_timeout:
            await asyncio.sleep(5)  # 每5秒检查一次
            
            try:
                # 检查页面是否发生了变化（可能表示用户进行了操作）
                current_url = page.url
                if current_url != initial_url:
                    logger.info(f"检测到页面变化: {initial_url} -> {current_url}")
                    
                    # 再次检测是否还需要人工干预
                    needs_intervention, _ = await detect_human_intervention_needed(page)
                    if not needs_intervention:
                        logger.info("人工干预问题已解决")
                        return True
                
                # 检查是否还存在原始问题
                needs_intervention, current_reason = await detect_human_intervention_needed(page)
                if not needs_intervention:
                    logger.info("问题已自动解决或被用户解决")
                    return True
                    
            except Exception as e:
                logger.warning(f"检查人工干预结果时出错: {e}")
                # 检查出错时也认为可以继续执行
                break
        
        logger.warning(f"人工干预超时 ({self.intervention_timeout}s)，将继续自动执行")
        # 超时后返回True，让系统继续执行而不是中断
        return True


# 创建全局错误管理器实例
error_manager = ErrorManager()


def with_timeout_and_retry(func=None, *, timeout_multiplier: float = None):
    """
    为Action的execute方法添加超时和重试机制的装饰器
    
    Args:
        func: 被装饰的函数
        timeout_multiplier: 超时时间倍数，默认使用配置值
    """
    if timeout_multiplier is None:
        timeout_multiplier = config.agent_config.action_timeout_multiplier
    
    def decorator(execute_func):
        @wraps(execute_func)
        async def wrapper(self, page, **kwargs):
            # 从Action实例获取action_name
            action_name = getattr(self, 'name', 'unknown_action')
            
            # 应用重试装饰器
            retry_decorator = error_manager.create_action_retry_decorator(action_name)
            retried_func = retry_decorator(execute_func)
            
            # 计算超时时间
            base_timeout = config.browser_config.timeout / 1000  # 转换为秒
            timeout_duration = base_timeout * timeout_multiplier
            
            try:
                # 使用asyncio.wait_for添加超时
                result = await asyncio.wait_for(
                    retried_func(self, page, **kwargs),
                    timeout=timeout_duration
                )
                return result
                
            except asyncio.TimeoutError:
                # 抛出自定义的超时错误
                raise ActionTimeoutError(action_name, timeout_duration)
                
        return wrapper
    
    # 支持不带参数和带参数两种使用方式
    if func is None:
        return decorator
    else:
        return decorator(func) 