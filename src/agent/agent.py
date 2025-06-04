import asyncio
import json
import logging
from typing import Any, Dict, List, Tuple

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

class Agent:
    """
    Agent 类负责根据任务、浏览器状态和 LLM 的决策来执行操作。
    """

    def __init__(self, memory: Memory | None = None, browser_manager: BrowserManager | None = None, llm: BaseLLM | None = None):
        """
        初始化 Agent。

        Args:
            memory: Memory 实例。如果为 None，则创建一个新的。
            browser_manager: BrowserManager 实例。如果为 None，则创建一个新的。
            llm: BaseLLM 实例。如果为 None，则通过 LLMFactory 创建一个。
        """
        self.memory = memory or Memory()
        self.browser_manager = browser_manager or BrowserManager()
        self.llm = llm or LLMFactory.create_llm()
        # 迭代轮次跟踪 - 设置更大的默认值
        self.max_iterations = 50  # 设置为50步，覆盖config中的默认值
        self.current_iteration = 0
        logger.info(f"Agent 初始化完成。LLM Provider: {self.llm.get_config().get('provider')}, 最大迭代次数: {self.max_iterations}")

    def _build_llm_prompt(self, task_description: str, current_url: str, current_title: str, available_tabs: List[Dict], page_elements: Any, previous_actions: List[Dict]) -> List[Message]:
        """
        构建发送给 LLM 的提示信息。
        """
        system_prompt = (
            f"你是一个智能Web Agent，你的任务是：{task_description}。\n"
            f"当前执行状态：第 {self.current_iteration + 1} 轮 / 总共 {self.max_iterations} 轮\n\n"
            f"=== 当前浏览器状态 ===\n"
            f"URL: {current_url}\n"
            f"标题: {current_title}\n"
            f"打开的标签页: {json.dumps(available_tabs, ensure_ascii=False, indent=2)}\n\n"
            f"=== 可用的浏览器操作 ===\n"
            f"{json.dumps(self.browser_manager.get_all_actions_description(), ensure_ascii=False, indent=2)}\n\n"
        )
        
        # 添加前面动作的信息（如果有）
        if previous_actions:
            action_summary = "=== 上一轮执行的动作和结果 ===\n"
            for pa in previous_actions:
                action_summary += f"- 动作: {pa.get('action')}\n"
                action_summary += f"  参数: {pa.get('args')}\n"
                action_summary += f"  结果: {str(pa.get('result', '无结果'))[:200]}\n\n"
            system_prompt += action_summary
        
        # 处理页面元素信息
        system_prompt += "=== 当前页面元素信息 ===\n"
        if page_elements:
            if isinstance(page_elements, dict):
                # 新的层级结构格式
                flat_elements = page_elements.get("flatElements", [])
                hierarchical_structure = page_elements.get("hierarchicalStructure")
                summary = page_elements.get("summary", {})
                
                system_prompt += f"页面摘要: {json.dumps(summary, ensure_ascii=False)}\n"
                system_prompt += f"可交互元素数量: {len(flat_elements)}\n\n"
                
                # 显示元素列表（限制数量避免过长）
                system_prompt += "主要元素列表:\n"
                for i, el in enumerate(flat_elements[:100]):  # 最多显示100个元素
                    desc = el.get("llm_description", f"Element {el.get('id')}")
                    element_id = el.get("id")
                    system_prompt += f"  - {element_id}: {desc}\n"
                
                if len(flat_elements) > 100:
                    system_prompt += f"  ... 还有 {len(flat_elements) - 50} 个元素未显示\n"
                
                # 如果有层级结构，简要展示
                if hierarchical_structure:
                    system_prompt += "\n元素层级结构已获取（可用于理解页面布局）\n"
                
            elif isinstance(page_elements, list):
                # 兼容旧格式
                system_prompt += f"元素数量: {len(page_elements)}\n"
                for i, el in enumerate(page_elements[:30]):
                    desc = el.get("llm_description", f"Element {i}")
                    system_prompt += f"  - {desc}\n"
        else:
            system_prompt += "注意：当前无法获取页面元素信息，可能页面正在加载或存在技术问题。\n"
        
        # 添加分析和决策指令
        system_prompt += (
            f"\n=== 任务执行要求 ===\n"
            f"请按以下步骤进行分析和决策：\n\n"
            f"1. **阅读当前页面元素**：仔细分析上述页面元素信息，理解页面结构和可用的交互元素。\n\n"
            f"2. **判断前一个动作是否完成**：\n"
            f"   - 如果这是第一轮，标记为 null\n"
            f"   - 否则，根据动作结果和当前页面状态判断是否成功完成\n"
            f"   - 考虑页面是否正确响、URL是否正确\n\n"
            f"3. **判断当前处于任务的第几个步骤**：\n"
            f"   - 分析整体任务目标\n"
            f"   - 评估已完成的进度\n"
            f"   - 明确当前所处的阶段（如：初始阶段、搜索阶段、结果处理阶段等）\n\n"
            f"4. **规划后续动作序列**：\n"
            f"   - 基于当前状态和任务目标，设计接下来需要执行的动作\n"
            f"   - 每个动作都应该有明确的目的\n"
            f"   - 优先使用 element_id 来指定元素（如 'element_0'）\n\n"
            f"=== 输出格式要求 ===\n"
            f"必须返回严格的JSON格式，包含以下字段：\n"
            f"{{\n"
            f"  \"previous_action_success\": true/false/null,  // 前面动作是否成功\n"
            f"  \"task_stage_analysis\": \"当前处于任务的什么阶段，已完成什么，还需要做什么\",\n"
            f"  \"current_step_number\": 1,  // 当前是任务的第几步（整数）\n"
            f"  \"actions\": [  // 要执行的动作序列\n"
            f"    {{\n"
            f"      \"action\": \"action_name\",\n"
            f"      \"args\": {{\"arg1\": \"value1\"}},\n"
            f"      \"purpose\": \"这个动作的目的\"\n"
            f"    }}\n"
            f"  ],\n"
            f"  \"reasoning\": \"详细的推理过程，解释为什么选择这些动作\"\n"
            f"}}\n\n"
            f"注意事项：\n"
            f"- 如果任务已完成，actions 返回空列表 []\n"
            f"- 使用 element_id（如 'element_0'）而不是复杂的选择器\n"
            f"- 每个动作都要有明确的 purpose 说明\n"
            f"- 如果需要用户帮助，使用 ask_user 动作\n"
        )
        
        messages = [Message(role="system", content=system_prompt)]
        return messages

    async def _parse_llm_response(self, response_text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        解析 LLM 返回的响应，现在返回完整的分析结果和操作序列。
        
        Returns:
            Tuple[Dict, List]: (完整的LLM响应字典, 操作序列列表)
        """
        try:
            # 尝试去除Markdown代码块标记
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            elif response_text.strip().startswith("```"):
                 response_text = response_text.strip()[3:-3].strip()

            response_dict = json.loads(response_text)
            if not isinstance(response_dict, dict):
                logger.warning(f"LLM 返回的不是一个字典，而是 {type(response_dict)}: {response_text}")
                return {}, []
            
            # 提取必要字段
            previous_action_success = response_dict.get("previous_action_success", None)
            task_stage_analysis = response_dict.get("task_stage_analysis", "未提供阶段分析")
            current_step_number = response_dict.get("current_step_number", 0)
            actions = response_dict.get("actions", [])
            reasoning = response_dict.get("reasoning", "未提供推理过程")
            
            # 验证actions格式
            if not isinstance(actions, list):
                logger.warning(f"actions字段不是列表格式: {actions}")
                actions = []
            
            valid_actions = []
            for act in actions:
                if isinstance(act, dict) and "action" in act and "args" in act and isinstance(act["args"], dict):
                    # 保留purpose字段（如果有）
                    valid_action = {
                        "action": act["action"],
                        "args": act["args"]
                    }
                    if "purpose" in act:
                        valid_action["purpose"] = act["purpose"]
                    valid_actions.append(valid_action)
                else:
                    logger.warning(f"LLM 返回了无效的动作格式: {act}")
            
            # 构建完整的响应信息
            full_response = {
                "previous_action_success": previous_action_success,
                "task_stage_analysis": task_stage_analysis,
                "current_step_number": current_step_number,
                "actions": valid_actions,
                "reasoning": reasoning,
                "raw_response": response_dict
            }
            
            logger.info(f"LLM分析结果:")
            logger.info(f"  - 前面动作成功: {previous_action_success}")
            logger.info(f"  - 当前步骤: {current_step_number}")
            logger.info(f"  - 阶段分析: {task_stage_analysis}")
            logger.info(f"  - 推理过程: {reasoning}")
            
            return full_response, valid_actions
            
        except json.JSONDecodeError as e:
            logger.error(f"解析 LLM 响应失败: {e}. 响应内容: {response_text}")
            return {}, []
        except Exception as e:
            logger.error(f"解析LLM响应时发生未知错误: {e}. 响应内容: {response_text}")
            return {}, []

    async def _call_llm_with_retry(self, messages: List[Message]) -> str:
        """
        带重试机制的LLM调用
        
        Args:
            messages: 发送给LLM的消息列表
            
        Returns:
            LLM的响应文本
        """
        max_retries = config.agent_config.max_llm_retries
        retry_delay = config.agent_config.llm_retry_delay_ms / 1000
        
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 因为包括第一次尝试
            try:
                if attempt > 0:
                    logger.info(f"LLM调用第 {attempt + 1} 次尝试...")
                    await asyncio.sleep(retry_delay * attempt)  # 线性退避
                
                # 调用LLM
                llm_response_raw = await self.llm.chat_completion(messages, stream=False)
                
                if isinstance(llm_response_raw, str):
                    return llm_response_raw
                else:
                    # 处理异步生成器的情况（虽然stream=False，但为了安全起见）
                    response_text = ""
                    async for chunk in llm_response_raw:
                        response_text += chunk
                    return response_text
                    
            except Exception as e:
                last_exception = e
                error_type = get_error_type(e)
                logger.warning(f"LLM调用第 {attempt + 1} 次失败: {e}, 错误类型: {error_type.value}")
                
                # 如果这是最后一次尝试，抛出异常
                if attempt == max_retries:
                    logger.error(f"LLM调用经过 {max_retries + 1} 次尝试后仍然失败")
                    raise last_exception
        
        # 理论上不应该到达这里
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("LLM调用失败，原因未知")

    async def _analyze_failure_with_llm(self, error_description: str, page_state: Dict[str, Any]) -> str:
        """
        使用LLM分析失败原因并提供建议
        
        Args:
            error_description: 错误描述
            page_state: 当前页面状态
            
        Returns:
            LLM的分析结果和建议
        """
        if not config.agent_config.enable_fallback_analysis:
            return "未启用失败分析功能"
        
        try:
            analysis_prompt = [
                Message(role="system", content=(
                    "你是一个专业的Web自动化错误分析专家。请根据提供的错误信息和页面状态，"
                    "分析可能的原因并提供具体的解决建议。请用中文回答，格式简洁明确。"
                )),
                Message(role="user", content=f"""
                执行Web自动化任务时遇到了问题：
                
                错误描述：{error_description}
                
                页面状态：
                - URL: {page_state.get('url', '未知')}
                - 标题: {page_state.get('title', '未知')}
                - 可见元素数量: {page_state.get('element_count', '未知')}
                
                请分析可能的原因并提供解决建议。
                """)
            ]
            
            analysis_result = await self._call_llm_with_retry(analysis_prompt)
            logger.info(f"LLM分析结果: {analysis_result}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"LLM分析失败: {e}")
            return f"分析过程中出错: {str(e)}"

    async def run_task(self, task_description: str):
        """
        执行给定的任务。

        Args:
            task_description: 任务的文本描述。
        """
        logger.info(f"开始执行任务: {task_description}")
        # 只存储与LLM交互的信息到Memory
        self.memory.add_message(Message(role="user", content=f"开始新任务: {task_description}"))

        await self.browser_manager.launch_browser() # 确保浏览器已启动

        previous_executed_actions: List[Dict] = [] # 用于传递给LLM的上一步结果

        for i in range(self.max_iterations):
            self.current_iteration = i
            logger.info(f"Agent 迭代 {i + 1}/{self.max_iterations}")

            # 1. 获取页面状态（第一次迭代强制刷新tab信息）
            force_refresh = (i == 0) or any(self.browser_manager._is_tab_related_action(act.get("action", "")) for act in previous_executed_actions)
            if force_refresh:
                logger.info(f"第{i+1}轮：需要刷新tab信息（首次迭代或执行了tab相关动作）")
            else:
                logger.info(f"第{i+1}轮：使用缓存的tab信息（前面没有执行tab相关动作）")
            
            current_url, current_title, available_tabs = await self.browser_manager.get_page_state(force_refresh_tabs=force_refresh)
            
            # 2. 获取页面元素信息（带回退机制）
            page_elements = await self.browser_manager.get_page_elements_with_fallback()

            # 3. 构建 Prompt 并请求 LLM
            prompt_messages = self._build_llm_prompt(task_description, current_url, current_title, available_tabs, page_elements, previous_executed_actions)
            
            # 记录构建的prompt到memory (主要是system和最后的用户请求)
            self.memory.add_message(prompt_messages[0]) # System prompt
            if len(prompt_messages) > 1:
                self.memory.add_message(prompt_messages[-1]) # User request part

            llm_response_text = ""
            try:
                llm_response_text = await self._call_llm_with_retry(prompt_messages)
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                continue # 进入下一次迭代
            
            # 4. 解析 LLM 响应得到完整分析和动作序列
            llm_analysis, actions_to_perform = await self._parse_llm_response(llm_response_text)
            
            # 存储LLM的响应到Memory：原始信息为content，解析的action序列为meta_content
            self.memory.add_message(Message(
                role="assistant", 
                content=llm_response_text,
                meta_content={
                    "parsed_analysis": llm_analysis,
                    "actions_to_perform": actions_to_perform
                }
            ))
            
            logger.info(f"LLM 原始响应: {llm_response_text}")
            logger.info(f"LLM 解析后的动作: {actions_to_perform}")

            if not actions_to_perform:
                logger.info("LLM 未返回有效动作或任务可能已完成。")
                break # 结束任务循环
            
            previous_executed_actions = [] # 清空，记录本次迭代执行的动作
            
            # 5. 执行动作
            for action_detail in actions_to_perform:
                action_name = action_detail.get("action")
                action_args = action_detail.get("args", {})

                if action_name == "ask_user":
                    question = action_args.get("question", "我需要您的帮助才能继续。")
                    logger.info(f"Agent 请求用户输入: {question}")
                    print(f"AGENT ASKS: {question}") 
                    return # 结束整个run_task

                        
                if self.browser_manager.get_action(action_name):
                    try:
                        result = await self.browser_manager.execute_action(action_name, **action_args)
                        logger.info(f"动作 '{action_name}' 执行成功，结果: {str(result)[:100]}...")
                        action_detail["result"] = result if isinstance(result, (str, int, float, bool, list, dict)) else str(result)
                        
                    except HumanInterventionError as e:
                        logger.warning(f"动作 '{action_name}' 需要人工干预: {e}")
                        action_detail["result"] = f"需要人工干预: {str(e)}"
                        
                        # 获取当前页面状态用于分析
                        current_page = await self.browser_manager.get_current_page()
                        page_state = {
                            "url": current_page.url,
                            "title": await current_page.title(),
                            "element_count": len(await current_page.query_selector_all("*"))
                        }
                        
                        # 使用LLM分析问题
                        analysis = await self._analyze_failure_with_llm(
                            f"动作 '{action_name}' 需要人工干预: {str(e)}", 
                            page_state
                        )
                        logger.info(f"LLM分析结果: {analysis}")
                        
                    except ActionTimeoutError as e:
                        logger.error(f"动作 '{action_name}' 执行超时: {e}")
                        action_detail["result"] = f"执行超时: {str(e)}"
                        
                        # 获取当前页面状态
                        current_page = await self.browser_manager.get_current_page()
                        page_state = {
                            "url": current_page.url,
                            "title": await current_page.title(),
                            "element_count": len(await current_page.query_selector_all("*"))
                        }
                        
                        # 使用LLM分析超时原因
                        analysis = await self._analyze_failure_with_llm(
                            f"动作 '{action_name}' 执行超时: {str(e)}", 
                            page_state
                        )
                        logger.info(f"超时分析结果: {analysis}")
                        
                    except Exception as e:
                        error_type = get_error_type(e)
                        logger.error(f"执行动作 '{action_name}' (参数: {action_args}) 失败: {e}, 错误类型: {error_type.value}")
                        action_detail["result"] = f"错误: {e}"
                        
                        # 对于其他类型的错误，也可以使用LLM进行分析
                        if error_type != ErrorType.UNKNOWN:
                            try:
                                current_page = await self.browser_manager.get_current_page()
                                page_state = {
                                    "url": current_page.url,
                                    "title": await current_page.title(),
                                    "element_count": len(await current_page.query_selector_all("*"))
                                }
                                
                                analysis = await self._analyze_failure_with_llm(
                                    f"动作 '{action_name}' 失败: {str(e)}, 错误类型: {error_type.value}", 
                                    page_state
                                )
                                logger.info(f"错误分析结果: {analysis}")
                            except Exception as analysis_error:
                                logger.warning(f"分析错误时出现问题: {analysis_error}")
                        
                else:
                    logger.warning(f"LLM 请求执行未知动作: '{action_name}'")
                    action_detail["result"] = f"错误: 未知动作 '{action_name}'"
                
                previous_executed_actions.append(action_detail) # 记录已执行的动作及其结果
            
            await asyncio.sleep(config.agent_config.action_delay_ms / 1000) # 每次迭代后等待

        logger.info(f"任务 '{task_description}' 执行结束。")
        self.memory.add_message(Message(role="user", content=f"任务结束: {task_description}"))