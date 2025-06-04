import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from patchright.async_api import BrowserContext

from src.config import config, logger


class CookieManager:
    """专门管理浏览器Cookie的类，提供保存、加载、过滤、智能检测等功能"""
    
    def __init__(self):
        self._cookie_path = config.browser_config.cookie_save_path
        self._enable_management = config.browser_config.enable_cookie_management
        self._auto_save = config.browser_config.auto_save_cookies
        self._cookie_domains = config.browser_config.cookie_domains
        self._enable_smart_save = config.browser_config.enable_smart_cookie_save
        self._smart_threshold = config.browser_config.smart_cookie_save_threshold
        
        # 确保cookie目录存在
        self._ensure_cookie_directory()
        
        logger.info("CookieManager 初始化完成")
    
    def _ensure_cookie_directory(self):
        """确保cookie目录存在"""
        cookie_dir = os.path.dirname(self._cookie_path)
        if cookie_dir and not os.path.exists(cookie_dir):
            os.makedirs(cookie_dir, exist_ok=True)
            logger.info(f"创建cookie目录: {cookie_dir}")
    
    def _is_important_cookie(self, cookie: Dict) -> bool:
        """判断是否为重要cookie（登录相关）"""
        cookie_name = cookie.get('name', '').lower()
        cookie_domain = cookie.get('domain', '').lower()
        
        # 重要的cookie名称模式（只保存登录和认证相关）
        important_patterns = [
            # 核心认证相关
            'auth', 'token', 'jwt', 'session', 'login', 'user', 'account',
            # 会话相关
            'sess', 'sid', 'phpsessid', 'jsessionid', 'aspsessionid',
            # 记住登录状态
            'remember', 'keep_logged', 'stay_signed', 'persistent_login',
            # 常见的认证cookie
            'access_token', 'refresh_token', 'csrf_token', 'xsrf_token',
            # 社交登录
            'oauth', 'saml', 'sso', 'openid',
            # 特定重要平台的认证
            'github_user', 'google_auth', 'facebook_login', 'microsoft_auth',
            'baidu_login', 'wechat_auth', 'alipay_auth', 'taobao_login'
        ]
        
        # 检查cookie名称是否包含重要模式
        for pattern in important_patterns:
            if pattern in cookie_name:
                return True
        
        # 排除一些明显不重要的cookie
        unimportant_patterns = [
            # 追踪和分析
            '_ga', '_gid', '_gtm', '_fbp', '_gcl', 'utm_', 'track', 'analytics',
            # 广告相关
            'ads', 'doubleclick', 'adsense', 'advertising',
            # 偏好设置（非认证相关）
            'theme', 'language', 'locale', 'timezone', 'font_size',
            # 临时数据
            'temp_', 'tmp_', 'cache_', 'visit_', 'page_view'
        ]
        
        for pattern in unimportant_patterns:
            if pattern in cookie_name:
                return False
        
        # 如果cookie值看起来像是会话ID或令牌（长度较长且包含随机字符）
        cookie_value = cookie.get('value', '')
        if len(cookie_value) >= 16 and any(c.isalnum() for c in cookie_value):
            # 进一步检查是否可能是重要的会话数据
            if any(keyword in cookie_name for keyword in ['id', 'key', 'hash', 'code']):
                return True
        
        return False
    
    async def load_cookies(self) -> Optional[str]:
        """加载保存的cookie状态文件。
        
        Returns:
            如果存在有效的cookie文件，返回文件路径；否则返回None。
        """
        if not self._enable_management:
            logger.debug("Cookie管理已禁用，跳过加载。")
            return None
            
        if os.path.exists(self._cookie_path):
            try:
                # 验证cookie文件是否有效
                with open(self._cookie_path, 'r', encoding='utf-8') as f:
                    cookie_data = json.load(f)
                    if isinstance(cookie_data, dict) and 'cookies' in cookie_data:
                        # 检查cookie是否过期或有问题
                        cookies = cookie_data.get('cookies', [])
                        valid_cookies = await self._validate_and_filter_cookies(cookies)
                        
                        if valid_cookies:
                            # 更新cookie数据，只保留有效的重要cookie
                            cookie_data['cookies'] = valid_cookies
                            # 临时保存清理后的cookie
                            temp_cookie_path = self._cookie_path + '.cleaned'
                            with open(temp_cookie_path, 'w', encoding='utf-8') as temp_f:
                                json.dump(cookie_data, temp_f, indent=2, ensure_ascii=False)
                            
                            logger.info(f"找到有效的cookie文件: {self._cookie_path}，有效重要cookie数量: {len(valid_cookies)}")
                            return temp_cookie_path
                        else:
                            logger.info(f"Cookie文件中没有有效的重要cookie，跳过加载: {self._cookie_path}")
                            return None
                    else:
                        logger.warning(f"Cookie文件格式无效: {self._cookie_path}")
                        return None
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"读取cookie文件失败: {e}")
                return None
        else:
            logger.info(f"Cookie文件不存在: {self._cookie_path}")
            return None
    
    async def _validate_and_filter_cookies(self, cookies: List[Dict]) -> List[Dict]:
        """验证cookies的有效性并过滤出重要的cookie"""
        valid_important_cookies = []
        current_time = asyncio.get_event_loop().time()
        
        for cookie in cookies:
            # 首先检查cookie是否过期
            if 'expires' in cookie and cookie['expires'] > 0:
                if cookie['expires'] < current_time:
                    logger.debug(f"跳过过期cookie: {cookie.get('name', 'unknown')}")
                    continue
            
            # 检查cookie域名是否合理
            domain = cookie.get('domain', '')
            if domain and not domain.startswith('.') and '.' not in domain and domain != 'localhost':
                logger.debug(f"跳过可疑cookie域名: {domain}")
                continue
            
            # 只保留重要的cookie（登录相关）
            if self._is_important_cookie(cookie):
                valid_important_cookies.append(cookie)
                logger.debug(f"保留重要cookie: {cookie.get('name')} (域名: {cookie.get('domain')})")
            else:
                logger.debug(f"跳过非重要cookie: {cookie.get('name')}")
        
        logger.info(f"从 {len(cookies)} 个cookie中筛选出 {len(valid_important_cookies)} 个重要cookie")
        return valid_important_cookies
    
    async def save_cookies(self, context: BrowserContext):
        """保存当前浏览器状态（包括cookies）到文件，只保存重要cookie。"""
        if not self._enable_management or not context:
            logger.debug("Cookie管理已禁用或浏览器上下文不存在，跳过保存。")
            return
            
        try:
            # 获取当前浏览器状态
            storage_state = await context.storage_state()
            
            # 过滤出重要的cookie
            all_cookies = storage_state.get('cookies', [])
            important_cookies = []
            
            for cookie in all_cookies:
                if self._is_important_cookie(cookie):
                    important_cookies.append(cookie)
            
            # 如果配置了特定域名，则进一步过滤
            if self._cookie_domains is not None:
                filtered_cookies = await self.filter_cookies_by_domain(important_cookies, self._cookie_domains)
                storage_state['cookies'] = filtered_cookies
                logger.info(f"已过滤cookies，保留域名: {self._cookie_domains}，重要cookie数量: {len(filtered_cookies)}")
            else:
                storage_state['cookies'] = important_cookies
                logger.info(f"保存重要cookie数量: {len(important_cookies)}")
            
            # 保存到文件
            with open(self._cookie_path, 'w', encoding='utf-8') as f:
                json.dump(storage_state, f, indent=2, ensure_ascii=False)
            
            cookie_count = len(storage_state.get('cookies', []))
            logger.info(f"已保存 {cookie_count} 个重要cookies到: {self._cookie_path}")
            
        except Exception as e:
            logger.error(f"保存cookies失败: {e}")
    
    async def filter_cookies_by_domain(self, cookies: List[Dict], domains: List[str]) -> List[Dict]:
        """根据域名列表过滤cookies。
        
        Args:
            cookies: cookie列表
            domains: 允许的域名列表
            
        Returns:
            过滤后的cookie列表
        """
        if not domains:
            return cookies
            
        filtered_cookies = []
        for cookie in cookies:
            cookie_domain = cookie.get('domain', '')
            if any(domain in cookie_domain for domain in domains):
                filtered_cookies.append(cookie)
        
        logger.debug(f"从 {len(cookies)} 个cookies中过滤出 {len(filtered_cookies)} 个")
        return filtered_cookies
    
    async def is_important_cookie_change(self, old_cookies: List[Dict], new_cookies: List[Dict]) -> bool:
        """
        检测是否有重要的Cookie变化（认证、会话相关）
        """
        # 只检查重要cookie的变化
        old_important = [c for c in old_cookies if self._is_important_cookie(c)]
        new_important = [c for c in new_cookies if self._is_important_cookie(c)]
        
        # 创建cookie字典以便比较
        old_cookie_dict = {f"{c.get('name', '')}_{c.get('domain', '')}": c for c in old_important}
        new_cookie_dict = {f"{c.get('name', '')}_{c.get('domain', '')}": c for c in new_important}
        
        # 检查新增的重要cookie
        for key, new_cookie in new_cookie_dict.items():
            if key not in old_cookie_dict:
                logger.info(f"检测到新增重要Cookie: {new_cookie.get('name')} (域名: {new_cookie.get('domain')})")
                return True
        
        # 检查重要cookie的值变化
        for key, new_cookie in new_cookie_dict.items():
            if key in old_cookie_dict:
                old_cookie = old_cookie_dict[key]
                if old_cookie.get('value') != new_cookie.get('value'):
                    logger.info(f"检测到重要Cookie值变化: {new_cookie.get('name')} (域名: {new_cookie.get('domain')})")
                    return True
        
        # 检查删除的重要cookie
        for key in old_cookie_dict:
            if key not in new_cookie_dict:
                old_cookie = old_cookie_dict[key]
                logger.info(f"检测到重要Cookie被删除: {old_cookie.get('name')} (域名: {old_cookie.get('domain')})")
                return True
        
        return False
    
    async def get_current_cookies(self, context: BrowserContext) -> List[Dict]:
        """获取当前浏览器的所有cookies"""
        if not context:
            return []
        
        try:
            return await context.cookies()
        except Exception as e:
            logger.warning(f"获取当前cookies失败: {e}")
            return []
    
    async def smart_save_cookies_after_action(self, context: BrowserContext, action_name: str, cookies_before: List[Dict]) -> None:
        """
        在动作执行后智能保存Cookie，只关注重要cookie的变化
        """
        if not self._enable_management or not self._enable_smart_save:
            return
        
        # 对于认证相关动作，更积极地保存cookie
        auth_related_actions = [
            "click_element",      # 可能点击登录按钮
            "type_text",          # 可能输入用户名密码
            "keyboard_input",     # 可能按回车提交表单
        ]
        
        # 对于导航相关动作，较少保存cookie（避免保存过多跳转cookie）
        navigation_actions = [
            "navigate_to_url"     # 普通导航，除非有重要cookie变化否则不保存
        ]
        
        cookies_after = await self.get_current_cookies(context)
        
        # 检查是否有重要cookie变化
        has_important_change = await self.is_important_cookie_change(cookies_before, cookies_after)
        
        # 计算重要cookie数量变化
        important_before = [c for c in cookies_before if self._is_important_cookie(c)]
        important_after = [c for c in cookies_after if self._is_important_cookie(c)]
        important_count_change = len(important_after) - len(important_before)
        
        # 决定是否保存cookie
        should_save = False
        
        if has_important_change:
            should_save = True
            logger.info(f"检测到重要Cookie变化，将保存")
        elif action_name in auth_related_actions and important_count_change > 0:
            should_save = True
            logger.info(f"认证相关动作 '{action_name}' 后重要Cookie增加，将保存")
        elif action_name in navigation_actions:
            # 导航动作只在有重要变化时才保存
            should_save = False
            logger.debug(f"导航动作 '{action_name}' 后无重要Cookie变化，跳过保存")
        elif abs(important_count_change) >= max(1, self._smart_threshold // 2):
            # 重要cookie数量显著变化时保存
            should_save = True
            logger.info(f"重要Cookie数量显著变化 ({important_count_change})，将保存")
        
        if should_save:
            logger.info(f"动作 '{action_name}' 后保存Cookie (重要变化: {has_important_change}, 重要cookie数量变化: {important_count_change})")
            await self.save_cookies(context)
        else:
            logger.debug(f"动作 '{action_name}' 后跳过Cookie保存 (重要cookie数量变化: {important_count_change})")
    
    async def clear_cookies_for_domain(self, context: BrowserContext, domain: str):
        """清除特定域名的cookie。"""
        if not self._enable_management:
            logger.debug("Cookie管理已禁用，跳过清除特定域名cookie。")
            return
            
        if not context:
            logger.warning("浏览器上下文不存在，无法清除cookie。")
            return
            
        try:
            # 获取当前所有cookie
            all_cookies = await context.cookies()
            
            # 过滤出要删除的cookie
            cookies_to_clear = []
            for cookie in all_cookies:
                cookie_domain = cookie.get('domain', '')
                if domain in cookie_domain or cookie_domain in domain:
                    cookies_to_clear.append({
                        'name': cookie['name'],
                        'domain': cookie['domain'],
                        'path': cookie.get('path', '/')
                    })
            
            # 清除cookie
            for cookie in cookies_to_clear:
                await context.clear_cookies(
                    name=cookie['name'],
                    domain=cookie['domain'],
                    path=cookie['path']
                )
            
            logger.info(f"已清除域名 '{domain}' 的 {len(cookies_to_clear)} 个cookie")
            
        except Exception as e:
            logger.error(f"清除域名cookie失败: {e}")
    
    async def clear_all_cookies(self, context: BrowserContext):
        """清除所有cookies"""
        if not self._enable_management:
            logger.debug("Cookie管理已禁用，跳过清除所有cookie。")
            return
            
        if not context:
            logger.warning("浏览器上下文不存在，无法清除cookie。")
            return
            
        try:
            await context.clear_cookies()
            logger.info("已清除所有cookies")
        except Exception as e:
            logger.error(f"清除所有cookies失败: {e}")
    
    def is_cookie_management_enabled(self) -> bool:
        """检查cookie管理是否启用"""
        return self._enable_management
    
    def get_cookie_save_path(self) -> str:
        """获取cookie保存路径"""
        return self._cookie_path 