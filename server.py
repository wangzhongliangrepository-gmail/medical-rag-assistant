"""医疗知识助手 FastAPI 服务（P6 + P5 记忆）。

启动：
  uvicorn server:app --host 0.0.0.0 --port 8000
接口：
  GET  /         网页前端（聊天式）
  GET  /health   健康检查
  POST /chat     {question, use_external, session_id, user_id}
                 → {answer, standalone_question, sub_questions, evidence, ...}

记忆：session_id → 短期会话（checkpointer 的 thread_id）；user_id → 长期用户记忆（Store namespace）。
"""
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from med_graph import get_med_graph

app = FastAPI(
    title="医疗知识助手",
    description="混合检索 + 知识融合 + 记忆的医疗 RAG（仅供学习演示，非医疗建议）",
)
STATIC = Path(__file__).parent / "static"
GRAPH = get_med_graph()  # 启动时编译一次，复用（checkpointer/store 随之常驻）


class ChatRequest(BaseModel):
    question: str
    use_external: bool = False              # 联网搜索开关（前端按钮）
    session_id: str = ""                    # 会话 id → 短期记忆 thread_id（空则单次）
    user_id: str = "anonymous"             # 用户 id → 长期记忆 namespace
    model: str = "flash"                    # 模型档位：flash（快）/ pro（强）


class EvidenceItem(BaseModel):
    text: str
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    standalone_question: str               # 指代消解后的自包含问题（前端可展示「理解为」）
    sub_questions: list[str]
    evidence: list[EvidenceItem]
    n_internal: int
    n_external: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    thread_id = req.session_id or str(uuid.uuid4())  # 无 session 则一次性 thread
    model_tier = "pro" if req.model == "pro" else "flash"
    config = {"configurable": {"thread_id": thread_id, "user_id": req.user_id, "model_tier": model_tier}}
    result = GRAPH.invoke(
        {"question": req.question, "use_external": req.use_external},
        config=config,
    )
    return ChatResponse(
        answer=result["answer"],
        standalone_question=result.get("standalone_question", req.question),
        sub_questions=result.get("plan_questions", []),
        evidence=[
            EvidenceItem(
                text=e["text"][:300],
                source=e["source"],
                score=round(float(e.get("score", 0.0)), 3),
            )
            for e in result.get("evidence", [])
        ],
        n_internal=len(result.get("internal_evidence", [])),
        n_external=len(result.get("external_evidence", [])),
    )


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
