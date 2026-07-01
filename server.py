"""医疗知识助手 FastAPI 服务（P6 + P5 记忆）。

启动：
  uvicorn server:app --host 0.0.0.0 --port 8000
接口：
  GET  /             网页前端（聊天式）
  GET  /health       健康检查
  POST /chat         {question, use_external, session_id, user_id, model}
                     → {answer, standalone_question, sub_questions, evidence, ...}（一次性返回）
  POST /chat/stream  同入参 → SSE 流式：边跑边推进度（含反思补检索）+ 最终答案 + 证据

记忆：session_id → 短期会话（checkpointer 的 thread_id）；user_id → 长期用户记忆（Store namespace）。

工程化（相对裸 RAG 的三处加固）：
  ① 统一异常处理：任何未捕获异常 → 500 + trace_id（不把堆栈甩给前端），并落日志。
  ② 请求超时：/chat 把阻塞的图调用丢线程池 + asyncio 超时，超了回 504。
  ③ 流式响应：/chat/stream 用 SSE 把节点级进度实时推给前端，体验从"转圈"变"有反馈"。
"""
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool   # 把阻塞函数丢到线程池跑，不卡事件循环
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from med_graph import get_med_graph

# ── 日志：结构化输出，异常和超时都留痕（生产可换 JSON formatter + 接入采集） ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("medrag")

# 单请求最长处理时间（秒）。联网检索 + 多轮反思可能慢，给个上限兜底。可用环境变量覆盖。
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "120"))

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
    revisions: int                         # 反思轮数（>1 表示触发了补检索重答）


# ──────────────────────────────────────────────────────────────────────────
# ① 统一异常处理：兜底所有没被业务代码捕获的异常
#    作用：前端永远拿到结构化错误体（含 trace_id 便于查日志），而不是一堆 Python 堆栈。
# ──────────────────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    trace_id = uuid.uuid4().hex[:12]
    # logger.exception 把完整 traceback 写进服务端日志，但不返回给客户端
    logger.exception("[%s] 未处理异常 %s %s", trace_id, request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "服务内部错误，请稍后重试", "trace_id": trace_id},
    )


# ── 小工具：把请求转成图的 config / 输入 / 响应（/chat 和 /chat/stream 共用，避免重复） ──
def _build_config(req: ChatRequest) -> dict:
    thread_id = req.session_id or str(uuid.uuid4())  # 无 session 则一次性 thread
    model_tier = "pro" if req.model == "pro" else "flash"
    return {"configurable": {"thread_id": thread_id, "user_id": req.user_id, "model_tier": model_tier}}


def _graph_inputs(req: ChatRequest) -> dict:
    return {"question": req.question, "use_external": req.use_external}


def _run_graph(req: ChatRequest) -> dict:
    """同步阻塞地跑完整张图（被丢进线程池执行，见 /chat）。"""
    return GRAPH.invoke(_graph_inputs(req), config=_build_config(req))


def _pack_evidence(evidence: list[dict]) -> list[dict]:
    return [
        {
            "text": e["text"][:300],                        # 截断，别把整段塞给前端
            "source": e["source"],
            "score": round(float(e.get("score", 0.0)), 3),  # 规整分数
        }
        for e in evidence
    ]


def _to_response(result: dict, req: ChatRequest) -> ChatResponse:
    return ChatResponse(
        answer=result["answer"],
        standalone_question=result.get("standalone_question", req.question),
        sub_questions=result.get("plan_questions", []),
        evidence=[EvidenceItem(**e) for e in _pack_evidence(result.get("evidence", []))],
        n_internal=len(result.get("internal_evidence", [])),
        n_external=len(result.get("external_evidence", [])),
        revisions=result.get("revisions", 0),
    )



@app.get("/health")
def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────
# ② 请求超时：图调用是同步阻塞的重活，这里
#    run_in_threadpool(_run_graph)  → 把阻塞调用丢线程池，不卡 async 事件循环
#    asyncio.wait_for(..., timeout) → 超过 REQUEST_TIMEOUT_S 就抛 TimeoutError → 回 504
#    ⚠️ 诚实边界：Python 无法强杀线程，超时只是让"请求"提前返回，图仍会在后台跑完。
# ──────────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(_run_graph, req),
            timeout=REQUEST_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        trace_id = uuid.uuid4().hex[:12]
        logger.warning("[%s] 请求超时（> %.0fs）", trace_id, REQUEST_TIMEOUT_S)
        return JSONResponse(
            status_code=504,
            content={
                "error": f"处理超时（>{REQUEST_TIMEOUT_S:.0f}s），可关闭联网检索或稍后重试",
                "trace_id": trace_id,
            },
        )
    return _to_response(result, req)


# ──────────────────────────────────────────────────────────────────────────
# ③ 流式响应（SSE）：边跑边把进度推给前端，而不是等整张图跑完才返回
#    GRAPH.stream(stream_mode="updates") 每个节点跑完吐一个 {节点名: state增量}
#    用 text/event-stream（SSE）协议逐条 yield；StreamingResponse 接收同步生成器，
#    框架自动在线程池里迭代它，不卡事件循环。
# ──────────────────────────────────────────────────────────────────────────
# 节点名 → 给用户看的中文进度文案
_STAGE_LABEL = {
    "contextualize": "理解问题…",
    "recall_memory": "回忆你的健康背景…",
    "plan": "拆解问题…",
    "retrieve_internal": "检索内部教材…",
    "retrieve_external": "联网检索…",
    "fuse": "融合重排证据…",
    "answer": "作答…",
    "reflect": "自检证据是否充分…",
    "augment_retrieve": "证据不足，补充检索…",
    "extract_memory": "记录健康事实…",
}


def _sse(event: str, data: dict) -> str:
    """拼一条 SSE 消息：event: 类型 + data: JSON，以空行结尾。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    config = _build_config(req)
    inputs = _graph_inputs(req)

    def event_stream():
        try:
            final: dict = {}  # 累积各节点的 state 增量，收尾时一次性给出证据等结构化信息
            for update in GRAPH.stream(inputs, config=config, stream_mode="updates"):
                for node, delta in update.items():
                    # 1) 进度事件：当前在跑哪个节点
                    yield _sse("progress", {"stage": node, "label": _STAGE_LABEL.get(node, node)})
                    # 2) answer 节点产出答案 → 单独推一条（反思重答时会再来一条，前端覆盖即可）
                    if node == "answer" and isinstance(delta, dict) and delta.get("answer"):
                        yield _sse("answer", {"text": delta["answer"]})
                    if isinstance(delta, dict):
                        final.update(delta)
            # 3) 收尾事件：证据 / 子问题 / 反思轮数等结构化元数据
            yield _sse("done", {
                "standalone_question": final.get("standalone_question", req.question),
                "sub_questions": final.get("plan_questions", []),
                "n_internal": len(final.get("internal_evidence", [])),
                "n_external": len(final.get("external_evidence", [])),
                "revisions": final.get("revisions", 0),
                "evidence": _pack_evidence(final.get("evidence", [])),
            })
        except Exception:
            trace_id = uuid.uuid4().hex[:12]
            logger.exception("[%s] 流式处理异常", trace_id)
            yield _sse("error", {"error": "服务内部错误，请稍后重试", "trace_id": trace_id})

    # SSE 必备响应头：text/event-stream + 关缓冲（防反向代理攒包）
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")