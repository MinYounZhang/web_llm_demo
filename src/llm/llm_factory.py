from src.llm.base_llm import BaseLLM
from src.llm.gemini_llm import GeminiLLM
from src.llm.deepseek_llm import DeepSeekLLM
from src.config import config, logger

class LLMFactory:
    """LLM 工厂类，用于创建不同类型的 LLM 实例。"""

    @staticmethod
    def create_llm(
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None, # 仅 DeepSeek 需要
        model_name: str | None = None
    ) -> BaseLLM:
        """
        根据提供的配置创建并返回一个 LLM 实例。

        Args:
            provider: LLM 提供商 (e.g., "gemini", "deepseek")。如果为 None，则从全局配置读取。
            api_key: 对应 LLM 提供商的 API 密钥。如果为 None，则由具体 LLM 类从全局配置读取。
            base_url: DeepSeek API 的基础 URL。如果为 None，则由 DeepSeekLLM 从全局配置读取。
            model_name: 要使用的模型名称。如果为 None，则使用各 LLM 类中定义的默认模型。

        Returns:
            一个 BaseLLM 的子类实例。

        Raises:
            ValueError: 如果配置的 LLM 提供商不受支持。
        """
        provider = provider or config.llm_config.provider
        logger.info(f"LLMFactory 正在创建 LLM 实例，提供商: {provider}")

        if provider == "gemini":
            # GeminiLLM 的 __init__ 会自行从 config 读取 api_key (如果调用时不提供)
            # model_name 如果不提供，GeminiLLM 内部也有默认值
            return GeminiLLM(api_key=api_key, model_name=model_name or "gemini-pro")
        elif provider == "deepseek":
            # DeepSeekLLM 的 __init__ 会自行从 config 读取 api_key 和 base_url (如果调用时不提供)
            # model_name 如果不提供，DeepSeekLLM 内部也有默认值
            return DeepSeekLLM(api_key=api_key, base_url=base_url, model_name=model_name or "deepseek-chat")
        else:
            logger.error(f"不支持的 LLM 提供商: {provider}")
            raise ValueError(f"不支持的 LLM 提供商: {provider}")

# 默认的 LLM 实例，可以直接导入使用
# llm_instance = LLMFactory.create_llm()

if __name__ == '__main__':
    # 测试 Gemini (需要配置 GEMINI_API_KEY 环境变量)
    if config.llm_config.gemini_api_key:
        print("\n--- 测试创建 Gemini LLM ---")
        try:
            gemini_llm = LLMFactory.create_llm(provider="gemini")
            print(f"Gemini LLM 创建成功: {gemini_llm.get_config()}")
        except Exception as e:
            print(f"创建 Gemini LLM 失败: {e}")
    else:
        print("\n未配置 GEMINI_API_KEY，跳过 Gemini LLM 创建测试。")

    # 测试 DeepSeek (需要配置 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL 环境变量)
    if config.llm_config.deepseek_api_key:
        print("\n--- 测试创建 DeepSeek LLM ---")
        try:
            deepseek_llm = LLMFactory.create_llm(provider="deepseek")
            print(f"DeepSeek LLM 创建成功: {deepseek_llm.get_config()}")
        except Exception as e:
            print(f"创建 DeepSeek LLM 失败: {e}")
    else:
        print("\n未配置 DEEPSEEK_API_KEY，跳过 DeepSeek LLM 创建测试。")

    # 测试不支持的 provider
    print("\n--- 测试不支持的 LLM Provider ---")
    try:
        LLMFactory.create_llm(provider="unsupported_provider")
    except ValueError as e:
        print(f"捕获到预期错误: {e}")

    # 测试从配置中读取 provider (假设 .env 中 LLM_PROVIDER=gemini 或 deepseek)
    print(f"\n--- 测试从默认配置 ({config.llm_config.provider}) 创建 LLM ---")
    if (config.llm_config.provider == "gemini" and config.llm_config.gemini_api_key) or \
       (config.llm_config.provider == "deepseek" and config.llm_config.deepseek_api_key):
        try:
            default_llm = LLMFactory.create_llm()
            print(f"默认 LLM ({config.llm_config.provider}) 创建成功: {default_llm.get_config()}")
        except Exception as e:
            print(f"创建默认 LLM ({config.llm_config.provider}) 失败: {e}")
    else:
        print(f"默认 LLM ({config.llm_config.provider}) 的 API Key 未配置，跳过测试。") 