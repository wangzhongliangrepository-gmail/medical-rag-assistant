"""DeepSeek 云端 LLM 客户端。

相比本地 Qwen3 简单很多：
- 不用 SSH 隧道连 Ollama，云端直连。
- 思考内容走独立的 reasoning_content 字段，.content 是干净的——不会像 Qwen3-via-Ollama
  那样把 <think> 漏进正文，所以不再需要 strip_think。
- 模型更强，工具调用稳定，不必为小模型不可靠的 function calling 做规避。

默认 deepseek-v4-flash。ChatDeepSeek 会自动从环境变量 DEEPSEEK_API_KEY 读取密钥。
"""
from langchain_deepseek import ChatDeepSeek

from config import DEEPSEEK_API_KEY, LLM_MODEL


def get_llm(temperature: float = 0.0) -> ChatDeepSeek:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("请在 .env 里设置 DEEPSEEK_API_KEY")
    return ChatDeepSeek(model=LLM_MODEL, temperature=temperature, max_retries=2)
