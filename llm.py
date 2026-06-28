"""DeepSeek 云端 LLM 客户端。
默认 deepseek-v4-flash。ChatDeepSeek 会自动从环境变量 DEEPSEEK_API_KEY 读取密钥。
"""
from langchain_deepseek import ChatDeepSeek

from config import DEEPSEEK_API_KEY, LLM_MODEL


def get_llm(model: str = LLM_MODEL, temperature: float = 0.0) -> ChatDeepSeek:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("请在 .env 里设置 DEEPSEEK_API_KEY")
    return ChatDeepSeek(model=model, temperature=temperature, max_retries=2)
