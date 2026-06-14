"""医疗知识助手 FastAPI 服务（P6）。

启动：
  uvicorn server:app --host 0.0.0.0 --port 8000
接口：
  GET  /         网页前端
  GET  /health   健康检查
  POST /chat     {question, use_external} → {answer, sub_questions, evidence, ...}
"""
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from med_graph import get_med_graph

app = FastAPI(
    title="医疗知识助手",
    description="混合检索 + 知识融合的医疗 RAG（仅供学习演示，非医疗建议）",
)
STATIC = Path(__file__).parent / "static"
GRAPH = get_med_graph()  # 启动时编译一次，复用


class ChatRequest(BaseModel):
    question: str
    use_external: bool = False   # 联网搜索开关（前端按钮）


class EvidenceItem(BaseModel):
    text: str
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sub_questions: list[str]
    evidence: list[EvidenceItem]
    n_internal: int
    n_external: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = GRAPH.invoke({"question": req.question, "use_external": req.use_external})
    return ChatResponse(
        answer=result["answer"],
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
