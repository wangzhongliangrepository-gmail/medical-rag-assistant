# 学习串讲：FastAPI 服务（逐行精读 server.py）

> 配合 `server.py`、`static/index.html`、`Dockerfile`（`CMD` 那行）、`docs/LEARN_DOCKER.md` 阅读。
> 全项目的 FastAPI 就 `server.py` 一个文件——**面试问「你的服务怎么写的」，把这一个文件讲透就够了**。
> 它从最初的「裸 RAG 接口」加固过三处工程化：**统一异常处理 + 请求超时 + SSE 流式**（第八章逐行讲），现在约 227 行。
> 建议读完亲手 `uvicorn server:app --reload` 跑起来，打开 `http://localhost:8000/docs` 点几下。

## 0. 先建立大局

- **FastAPI 到底是什么** —— 一句话：一个**用 Python 类型注解写 Web API**的框架。你声明「请求长这样、响应长这样」，
  它替你做三件事：① 自动**校验**进来的 JSON（不合法直接 422，代码里拿到的一定是干净数据）；
  ② 自动**序列化**返回值成 JSON；③ 自动**生成交互式文档**（`/docs`）。核心卖点是「**类型即契约**」。

- **三个角色别混**（面试高频）：
  - **FastAPI**：框架，定义路由、校验、文档。`app = FastAPI()` 是一个 **ASGI 应用对象**，它**自己不会跑**。
  - **uvicorn**：ASGI **服务器**，真正监听端口、收 TCP、把 HTTP 请求喂给 `app`。`uvicorn server:app` 里 `server` 是模块、`app` 是对象。
  - **Pydantic**：数据校验库。FastAPI 用它把「请求体 JSON ↔ Python 对象」来回转换并校验。`BaseModel` 就是它的。

  ```
  浏览器/前端  --HTTP-->  uvicorn(ASGI服务器)  -->  FastAPI(app, 路由+校验)  -->  你的函数 chat()
       ^                                                                              |
       └──────────────── JSON 响应  <── Pydantic 序列化 ChatResponse <────────────────┘
  ```

- **本项目这个服务在干嘛**：把命令行版的 RAG 图（`med_graph`）包成一个 HTTP 接口，前端网页用 `fetch` 调它。
  四个路由：`GET /`（发网页）、`GET /health`（探活）、`POST /chat`（一次性问诊）、`POST /chat/stream`（**SSE 流式问诊**，前端现在用这个）。

- **为什么选 FastAPI 而不是 Flask**：① 原生 async + ASGI，适合这种「请求里要等 LLM/检索」的 I/O 密集场景；
  ② Pydantic 自动校验省掉一堆手写 `if not request.json.get(...)`；③ 自动出 `/docs`，自测和交接都省事。

---

## 一、server.py 逐行

```python
import _bootstrap  # noqa: F401  ① 必须最先导入：放行 OpenMP 重复加载

import uuid
from pathlib import Path

from fastapi import FastAPI                 # ② 框架本体
from fastapi.responses import FileResponse  # ③ 直接回一个文件（发 HTML）
from pydantic import BaseModel              # ④ 请求/响应模型的基类

from med_graph import get_med_graph         # ⑤ 拿到编译好的 RAG 图

app = FastAPI(                              # ⑥ 创建 ASGI 应用对象
    title="医疗知识助手",
    description="混合检索 + 知识融合 + 记忆的医疗 RAG（仅供学习演示，非医疗建议）",
)
STATIC = Path(__file__).parent / "static"  # ⑦ static 目录的绝对路径
GRAPH = get_med_graph()                    # ⑧ 启动时编译一次，全局复用


class ChatRequest(BaseModel):              # ⑨ 请求体模型（入参契约）
    question: str
    use_external: bool = False
    session_id: str = ""
    user_id: str = "anonymous"
    model: str = "flash"


class EvidenceItem(BaseModel):             # ⑩ 嵌套模型：一条证据
    text: str
    source: str
    score: float


class ChatResponse(BaseModel):             # ⑪ 响应体模型（出参契约）
    answer: str
    standalone_question: str
    sub_questions: list[str]
    evidence: list[EvidenceItem]
    n_internal: int
    n_external: int
    revisions: int


@app.get("/health")                        # ⑫ 路由装饰器：GET /health
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)   # ⑬ POST /chat，声明响应模型
async def chat(req: ChatRequest):          # ⑭ 参数是模型 → 自动解析+校验；async 是为了加超时
    result = await asyncio.wait_for(       # ⑮ 阻塞的图调用丢线程池 + 超时兜底（第八章 8.2）
        run_in_threadpool(_run_graph, req),
        timeout=REQUEST_TIMEOUT_S,
    )
    return _to_response(result, req)       # ⑯ 组装成响应模型返回（裁剪/截断/规整在小工具里）


@app.get("/")                              # ⑰ GET / → 发网页前端
def index():
    return FileResponse(STATIC / "index.html")

# 另有：@app.exception_handler(Exception) 统一异常 + POST /chat/stream 流式 → 详见第八章
```

> 注：上面是核心骨架。真实 `server.py` 还抽了 `_build_config`/`_run_graph`/`_to_response`/`_pack_evidence`
> 几个小工具给 `/chat` 和 `/chat/stream` 共用，并加了**统一异常处理、请求超时、SSE 流式**三处工程化——**全部在第八章逐行讲**。

逐点讲：

- **① `import _bootstrap`**：放在**最顶上**，不是 FastAPI 的事，是本项目的硬约束——它放行 OpenMP 重复加载，
  必须在 torch/onnx 相关被 import 之前执行（`med_graph` 会间接拉起这些）。`# noqa: F401` 是告诉 linter「这个 import 没显式用到，别报未使用」。

- **⑥ `app = FastAPI(...)`**：创建应用对象。`title`/`description` 不只是装饰——它们会**显示在自动生成的 `/docs` 页面**顶部。
  这个 `app` 就是 `uvicorn server:app` 里那个 `app`，也是 Dockerfile `CMD` 里 `server:app` 指的东西。

- **⑦ `STATIC = Path(__file__).parent / "static"`**：用 `__file__` 算出**绝对路径**，而不是写 `"static"` 这种相对路径。
  关键原因：相对路径取决于「你在哪个目录启动进程」，容器里 / 本地工作目录一变就找不到文件；`__file__` 永远相对**这个 .py 自己**，稳。

- **⑧ `GRAPH = get_med_graph()`（重点）**：在**模块顶层**编译一次图，存全局变量，所有请求共用。
  - 为什么不放进 `chat()` 里？图编译 + 挂 checkpointer/store 是重活，每个请求都重建会很慢，而且**记忆会丢**（每次新建 store = 内存里的长期记忆清空）。放全局 = 进程生命周期内常驻，记忆才能跨请求累积。
  - 这其实是一种「**启动时初始化、全局单例**」模式。更正式的写法是用 FastAPI 的 `lifespan` 事件（见第七章），但模块级变量对单文件服务足够。

- **⑨ `class ChatRequest(BaseModel)`（核心：请求契约）**：继承 `BaseModel`，每个字段带类型注解。
  - 带默认值的（`use_external: bool = False`）= **可选**，前端不传就用默认；不带默认值的（`question: str`）= **必填**，不传直接 422。
  - 这就是 FastAPI「**声明即校验**」：你不用写一行 `if "question" not in body`，进到函数里 `req.question` 一定是 str。

- **⑩ `EvidenceItem` + ⑪ `ChatResponse` 里 `evidence: list[EvidenceItem]`（嵌套模型）**：模型可以**套模型**。
  `list[EvidenceItem]` 表示「证据是一个列表，每项必须有 text/source/score」。FastAPI 会**递归校验和序列化**整棵结构。

- **⑫ `@app.get("/health")`（路由装饰器）**：把下面的函数注册成「GET /health」的处理器。
  健康检查是惯例——给 Docker/K8s/负载均衡用来探活（对应 `LEARN_DOCKER` 第七章的 healthcheck）。返回 dict，FastAPI 自动转 JSON。

- **⑬ `@app.post("/chat", response_model=ChatResponse)`**：
  - `post` 因为要带请求体（问诊内容）。`get` 不该带 body。
  - **`response_model=ChatResponse` 做两件事**：① **过滤**——只有模型里声明的字段会出现在响应里（哪怕你 return 的对象多带了字段，也会被剔掉，防止意外泄露内部数据）；② 让 `/docs` 知道响应长啥样。

- **⑭ `def chat(req: ChatRequest)`（最关键的一行魔法）**：参数类型标成 `ChatRequest`，FastAPI 就知道
  「**从请求体读 JSON → 用 ChatRequest 校验 → 给我一个 `req` 对象**」。你完全不碰 `request.json()`，校验失败它自动回 422。
  - 对比：如果参数是简单类型（`q: str`）且没说来源，FastAPI 默认当它是 **查询参数**（`?q=...`）。是 `BaseModel` 才认作请求体。

- **⑮ `await asyncio.wait_for(run_in_threadpool(_run_graph, req), timeout=...)`**：真正干活的 `GRAPH.invoke()`（喂问题 + `config` 里的 `thread_id`/`user_id`/`model_tier`）是**同步阻塞**重活（要等 LLM + 检索）。所以这里两层包裹：`run_in_threadpool` 把它丢线程池（不卡事件循环）、`asyncio.wait_for` 加超时（超了回 504）。**为什么 `chat()` 是 `async def` 见 2.4，超时细节见第八章 8.2。**

- **⑯ `_to_response(...)` 组装 `ChatResponse`**：内部有几个**防御性处理**——`result.get("answer", ...)` 用 `.get` 带默认值防 KeyError；`e["text"][:300]` 截断证据文本（别把整段塞给前端）；`round(float(...), 3)` 规整分数。这些是「**出口处把数据修干净**」的好习惯。

- **⑰ `FileResponse(STATIC / "index.html")`**：直接把 HTML 文件作为响应体发回去。`GET /` 返回网页，浏览器拿到就渲染聊天界面。

---

## 二、核心机制（面试要能展开讲）

### 2.1 Pydantic 校验 —— 「类型即契约」

- 进来的 JSON 会被**逐字段校验 + 类型转换**。`use_external` 声明 `bool`，前端传 `"true"` 字符串它会试着转；
  传不出来（比如 `question` 缺失或类型错）→ FastAPI **自动返回 `422 Unprocessable Entity`**，带一份说明哪个字段错的 JSON，你一行校验代码都不用写。
- **价值面试一句话**：「我不用在业务函数里手写参数校验，Pydantic 在入口就挡掉脏数据，函数体里拿到的对象一定合法、有类型提示，IDE 还能补全。」

### 2.2 `response_model` —— 出口的过滤器 + 文档

- 两个作用：**过滤多余字段**（只放行模型声明的，防泄露 / 防意外）+ **生成响应文档**。
- 本项目用它把图返回的一大坨 state**裁剪成前端真正要的 7 个字段**，内部中间状态不外泄。

### 2.3 自动交互文档 `/docs` 和 `/redoc` —— FastAPI 的杀手锏

- 启动后访问 `http://localhost:8000/docs`：Swagger UI，**所有路由、请求/响应模型、必填项一目了然，还能直接在网页上点 "Try it out" 发请求**。
- 它是**从你的类型注解和 Pydantic 模型自动生成**的——你写好模型，文档免费。底层是 OpenAPI（`/openapi.json`）规范。
- 面试加分：「我交接 / 自测从不写 Postman 集合，直接用 `/docs`。」

### 2.4 `def` vs `async def` —— 阻塞重活到底怎么放（高频深水区）

- FastAPI 两种都支持，**核心铁律：绝不能在事件循环里直接跑同步阻塞代码**，否则所有请求一起卡死。规则：
  - `async def` + `await` 调**异步**库（异步 HTTP / DB）→ 在事件循环里跑，单线程高并发。
  - 普通 `def`（同步函数）→ FastAPI **自动把它丢线程池**里跑，**不阻塞事件循环**。
- 本项目的关键调用 `GRAPH.invoke()` 是**同步阻塞**重活（LangGraph 同步调 LLM/检索）。两种合法写法：
  - **写法 A（最初）**：`def chat()`，让框架自动把整个函数丢线程池。简单、够用。
  - **写法 B（现在）**：`async def chat()`，里面用 `await asyncio.wait_for(run_in_threadpool(_run_graph), timeout)`。
    手动把阻塞活丢线程池（`run_in_threadpool`），换来一个能力：**用 `wait_for` 给它套超时**（同步 `def` 里没法干净地加超时）。
- **为什么本项目从 A 换成 B**：为了**请求超时**（见第八章 8.2）。原理没变——阻塞活始终在线程池、没卡事件循环；
  只是从「靠框架隐式代劳」升级成「显式丢线程池 + 加超时控制」。
- **反模式（必背）**：`async def` 里**直接**调阻塞函数（不丢线程池）→ 卡死整个事件循环，所有并发请求一起挂。
  记住一句：**同步阻塞活，要么用 `def` 让框架丢线程池，要么在 `async def` 里显式 `run_in_threadpool`——就是不能在 `async` 里裸调。**

### 2.5 路由 / 装饰器 / 启动

- `@app.get` / `@app.post` 把函数绑定到「HTTP 方法 + 路径」。方法选择：读用 `GET`、带 body 的提交用 `POST`。
- 进程怎么起来的：`uvicorn server:app` → uvicorn import `server.py`、拿到 `app` 对象、开始监听端口、把请求路由给对应函数。
  开发时加 `--reload`（改代码自动重启）；容器里是 `--host 0.0.0.0 --port 8000`（见 `LEARN_DOCKER` 第一章 ⑪）。

---

## 三、一次 `/chat` 的请求全链路（能讲清这条 = 真懂）

```
1. 前端 static/index.html 第 322 行：
   fetch("/chat/stream", { method:"POST", headers:{"Content-Type":"application/json"},
                           body: JSON.stringify({question, use_external, session_id, user_id, model}) })
        │
        ▼
2. uvicorn 收到 TCP/HTTP 请求，按 ASGI 协议交给 app
        │
        ▼
3. FastAPI 按路径+方法匹配到处理器（/chat 走 chat()，/chat/stream 走 chat_stream()）
        │
        ▼
4. 读请求体 JSON → 用 ChatRequest 校验/转换 → 不合法直接 422 返回；合法则得到 req 对象
        │
        ▼
5. 执行：拼 config → 调 RAG 图
     · /chat        ：async def，await wait_for(run_in_threadpool(GRAPH.invoke), 超时) —— 一次性跑完，超时回 504
     · /chat/stream ：GRAPH.stream(stream_mode="updates")，每个节点跑完吐一个 state 增量
        │
        ▼
6. /chat        ：result(dict) → _to_response → ChatResponse（response_model 过滤多余字段）→ 序列化 JSON
   /chat/stream ：每个增量 → 拼成 SSE 事件（progress/answer/done）逐条 yield
        │
        ▼
7. uvicorn 写回 HTTP 响应（/chat 一次性 JSON；/chat/stream 是 text/event-stream 长连接，边算边发）
        │
        ▼
8. 前端：/chat 用 resp.json()；/chat/stream 用 sseEvents() 解析流 → 实时渲染进度 + 答案
   （renderAnswer 在第 349 行收尾渲染完整回答）
```

> 面试可压缩成一句：「前端 `fetch` 发 JSON → uvicorn 收包 → FastAPI 路由 → Pydantic 校验入参 →
> 调 RAG 图（同步阻塞活丢线程池、不卡事件循环；要进度就用 `GRAPH.stream` 逐节点产出）→
> 出参裁剪 / 拼 SSE 事件 → 回 JSON 或 event-stream → 前端一次性渲染或边流边渲染。」

---

## 四、核心概念速记

| 概念 | 一句话 |
|------|--------|
| FastAPI | 用类型注解写 API 的框架（校验+序列化+文档） |
| uvicorn | ASGI 服务器，真正监听端口、跑 app |
| ASGI | 异步网关接口（Flask 的 WSGI 的异步版） |
| Pydantic / BaseModel | 数据校验+序列化，请求/响应模型的基类 |
| `app = FastAPI()` | ASGI 应用对象，`uvicorn server:app` 里的 app |
| `@app.get/post` | 路由装饰器，绑定 HTTP 方法+路径 |
| 请求模型 (ChatRequest) | 入参契约，自动校验，缺字段→422 |
| `response_model` | 出参契约，过滤多余字段+生成文档 |
| 422 | 请求体校验失败自动返回的状态码 |
| `/docs` | 自动生成的交互式 API 文档（Swagger） |
| `def` vs `async def` | 同步函数走线程池；异步函数走事件循环 |
| `run_in_threadpool` | 在 async 里把阻塞活显式丢线程池（本项目配 `wait_for` 加超时） |
| `asyncio.wait_for` | 给阻塞调用套超时，超了抛 `TimeoutError` → 504 |
| `@app.exception_handler` | 统一异常兜底，返回脱敏错误体 + trace_id，堆栈只进日志 |
| FileResponse | 直接把文件作为响应（本项目发 HTML） |
| StreamingResponse | 流式响应，本项目用它发 SSE（`text/event-stream`） |
| SSE | 服务端推事件，单向流；POST 带 body 时需 fetch+手动解析（EventSource 仅 GET） |
| `GRAPH.stream(updates)` | LangGraph 逐节点产出 state 增量，流式进度的来源 |
| lifespan | 启动/关闭钩子（本项目用模块级全局替代） |

## 五、面试问答

1. **FastAPI、uvicorn、Pydantic 各是什么、怎么配合？** → 框架 / ASGI 服务器 / 校验库；uvicorn 收包喂给 FastAPI app，FastAPI 用 Pydantic 校验和序列化。
2. **请求参数怎么校验的？写了多少校验代码？** → 零。声明 `ChatRequest(BaseModel)`，FastAPI 自动校验，失败回 422。
3. **`response_model` 干嘛用？** → 过滤出参（只放行声明字段、防泄露）+ 生成响应文档。
4. **`/docs` 哪来的？** → 从类型注解和 Pydantic 模型自动生成的 OpenAPI/Swagger，免费。
5. **`chat()` 是 `async def` 还是 `def`？阻塞的 `GRAPH.invoke()` 怎么处理？** → 现在是 `async def`，里面 `await wait_for(run_in_threadpool(...), 超时)`：阻塞活显式丢线程池、不卡事件循环，同时拿到超时能力。铁律：同步阻塞活只能「`def` 让框架丢线程池」或「`async` 里显式 `run_in_threadpool`」，绝不能在 `async` 里裸调。
6. **图为什么在模块顶层编译一次（GRAPH 全局）？** → 编译重 + 记忆要常驻；每请求重建会慢且把内存里的长期记忆清空。
7. **`STATIC` 为什么用 `Path(__file__).parent`？** → 绝对路径不受启动目录影响，容器/本地都稳。
8. **必填和可选字段怎么区分？** → Pydantic 模型里有默认值=可选，无默认值=必填。
9. **健康检查 `/health` 干嘛的？** → 给 Docker/编排探活，配合 compose 的 healthcheck。
10. **GET 和 POST 怎么选？** → 读、无副作用、参数走 URL → GET；带请求体的提交 → POST。`/chat` 带问诊内容故用 POST。
11. **服务做了哪些工程化加固？** → ① 统一异常处理（`@app.exception_handler` → 500 + trace_id，堆栈只进日志）；② 请求超时（`run_in_threadpool` + `wait_for` → 504）；③ SSE 流式 `/chat/stream`（边跑边推进度 + 答案）。逐行见第八章。
12. **`/chat/stream` 怎么实现流式？为什么不用 WebSocket / EventSource？** → 后端 `GRAPH.stream(updates)` 逐节点产出 → `StreamingResponse` 发 SSE（`text/event-stream`）。单向推送用 SSE 比 WebSocket 轻；前端因为是 POST 带 body（EventSource 只支持 GET），用 `fetch` 读 `ReadableStream` 手动解析。
13. **还没做、生产该加什么？（主动说短板）** → 鉴权（API key/OAuth + `Depends`）、限流、CORS（前端分域时）、`lifespan` 管资源、token 级流式（见第七章）。

## 六、动手实验（5 分钟跑通，比读十遍强）

```bash
# 1. 启动（开发模式，改代码自动重启）
uvicorn server:app --reload --port 8000

# 2. 打开自动文档，点 "Try it out" 发请求
#    浏览器访问  http://localhost:8000/docs

# 3. 命令行探活
curl http://localhost:8000/health
#    → {"status":"ok"}

# 4. 命令行问诊（POST + JSON）
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question":"布洛芬的副作用？","use_external":false}'

# 5. 故意触发 422：少传必填的 question，看自动校验报错
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"use_external":false}'
#    → 422，body 里写明 question 字段缺失
```

> 重点体会两件事：① `/docs` 是你写好模型后**白送**的；② 第 5 步那个 422，**你没写一行校验代码**，是 Pydantic 干的。

## 七、还没用上的（短板展开，主动讲 = 边界感）

> 面试别假装全做了。能说「我知道有这些、本项目规模没上、上的话这么上」最加分。
> 注：**统一异常处理 / 请求超时 / SSE 流式**原本在这里当短板，现已实现 → 移到第八章逐行讲。下面是**仍没做**的。

- **CORS 中间件**：本项目前端和后端**同源**（都由这个 FastAPI 服务发出，`GET /` 发 HTML、`/chat*` 是同一域），
  所以不需要跨域配置。**如果**前端单独部署在别的域名，浏览器会因同源策略拦截，那时要加：
  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(CORSMiddleware, allow_origins=["https://前端域名"], allow_methods=["*"], allow_headers=["*"])
  ```

- **`lifespan` 启动/关闭钩子**：本项目用**模块级全局** `GRAPH = get_med_graph()` 在 import 时初始化。
  更正式的是 `lifespan` 上下文：启动时建资源、关闭时优雅释放（比如换成 SqliteSaver 时要关连接）。内存版没这需求。

- **依赖注入 `Depends`**：FastAPI 的招牌特性——把「鉴权 / 取数据库会话 / 取当前用户」抽成可复用依赖，注入到多个路由。
  本项目路由少、无鉴权，没必要。**但要会讲它解决什么**：复用 + 可测试（测试时替换依赖）。

- **鉴权 / 限流**：演示项目全无。生产要有 API key 或 OAuth（配 `Depends`）、限流防滥用（如 slowapi）。

- **token 级流式**：现在 `/chat/stream` 是**节点级**流（「检索中→作答中→出整段答案」），答案仍是一次成型推出。
  再进一步是**逐字 token** 推（`GRAPH.astream_events` 抓 `answer` 节点的 LLM token 流）。本项目没做，但能讲清「节点级 vs token 级」的差别就够。

> 一句收口：「我的服务刻意保持薄——一个文件几个路由，复杂度都压在 RAG 图里；但该有的工程化（异常兜底 / 超时 / 流式）我补上了。
> 更重的中间件（鉴权 / 限流 / CORS / lifespan / token 级流式）我清楚边界，按需要再加。」

---

## 八、工程化加固（已实现，逐行）

> 这三处是把「裸 RAG 接口」变成「像样的服务」的关键。面试讲「我不只是把模型包了个接口，还做了异常兜底 / 超时 / 流式」最显工程素养。

### 8.1 统一异常处理 —— 堆栈进日志，前端只看脱敏错误

```python
@app.exception_handler(Exception)                       # 兜底所有未被业务捕获的异常
async def unhandled_exception_handler(request: Request, exc: Exception):
    trace_id = uuid.uuid4().hex[:12]                    # 给这次错误发一个短 id
    logger.exception("[%s] 未处理异常 %s %s", trace_id, request.method, request.url.path)
    return JSONResponse(status_code=500,                # 返回结构化错误体，不含堆栈
        content={"error": "服务内部错误，请稍后重试", "trace_id": trace_id})
```
- **为什么**：默认情况未捕获异常会把 Python 堆栈/细节暴露给前端——既不专业也可能泄露内部信息。
- **怎么做**：注册一个 `Exception` 级别的处理器。`logger.exception` 把**完整 traceback 写进服务端日志**，
  返回给客户端的只有「脱敏文案 + `trace_id`」。线上排障：用户报错时报 `trace_id`，去日志一搜就定位。
- **面试点**：错误处理的核心是「**详细信息留给运维（日志），脱敏信息给用户（响应）**」，靠 `trace_id` 串起来。

### 8.2 请求超时 —— 阻塞活丢线程池 + `wait_for` 兜底

```python
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "120"))   # 可环境变量覆盖

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(_run_graph, req),         # 阻塞的 GRAPH.invoke 丢线程池
            timeout=REQUEST_TIMEOUT_S,                  # 超过就抛 TimeoutError
        )
    except asyncio.TimeoutError:
        trace_id = uuid.uuid4().hex[:12]
        logger.warning("[%s] 请求超时（> %.0fs）", trace_id, REQUEST_TIMEOUT_S)
        return JSONResponse(status_code=504, content={"error": "处理超时…", "trace_id": trace_id})
    return _to_response(result, req)
```
- **为什么 `chat()` 变 `async def`**：只有在 `async` 里才能用 `await asyncio.wait_for(...)` 加超时（见 2.4）。
- **两层包裹**：`run_in_threadpool` 保证阻塞活不卡事件循环；`wait_for` 给它设上限，超了回 `504 Gateway Timeout`。
- **⚠️ 诚实边界（面试加分）**：Python **无法强杀线程**，超时只是让「**请求**」提前返回 504，
  那个图调用仍会在后台线程里跑到完。真要能取消，得用支持中断的异步实现或子进程——这是已知权衡，不是 bug。

### 8.3 SSE 流式 `/chat/stream` —— 边跑边推，让反思回路「被看见」

```python
def _sse(event: str, data: dict) -> str:                # 拼一条 SSE 消息
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    config, inputs = _build_config(req), _graph_inputs(req)
    def event_stream():                                 # 同步生成器，框架自动在线程池迭代
        try:
            final = {}
            for update in GRAPH.stream(inputs, config=config, stream_mode="updates"):
                for node, delta in update.items():
                    yield _sse("progress", {"stage": node, "label": _STAGE_LABEL.get(node, node)})
                    if node == "answer" and delta.get("answer"):
                        yield _sse("answer", {"text": delta["answer"]})     # 反思重答会再来一条
                    if isinstance(delta, dict): final.update(delta)
            yield _sse("done", {"evidence": _pack_evidence(final.get("evidence", [])), ...})
        except Exception:
            yield _sse("error", {"error": "服务内部错误", "trace_id": uuid.uuid4().hex[:12]})
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```
- **数据从哪来**：`GRAPH.stream(stream_mode="updates")` 每个节点跑完吐一个 `{节点名: state增量}`，**不必改图或节点**。
- **三种事件**：`progress`（当前在跑哪个节点）、`answer`（answer 节点产出答案，反思重答时再推一条覆盖）、`done`（收尾的证据/反思轮数等元数据）；出错推 `error`。
- **为什么 `def`（不是 `async def`）**：`event_stream` 是**同步生成器**，`StreamingResponse` 接收它后框架**自动在线程池迭代**，照样不卡事件循环。
- **SSE 协议**：响应头 `text/event-stream`，每条消息 `event: 类型\ndata: JSON\n\n`（空行分隔）。`X-Accel-Buffering: no` 防反向代理攒包。
- **项目亮点**：节点级流让 P4 反思回路「证据不足 → 补检索 → 重答」**实时显示给用户**，而不是干等。

### 8.4 前端怎么消费 SSE（`static/index.html`）

`EventSource` 只支持 GET，而 `/chat/stream` 是 POST 带 body，所以用 `fetch` + 手动解析 `ReadableStream`：

```javascript
async function* sseEvents(resp) {                       // 把字节流切成一条条 SSE 事件
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });     // {stream:true} 正确拼回跨块的多字节中文
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {            // 按空行切出完整事件
      const raw = buf.slice(0, i); buf = buf.slice(i + 2);
      let ev = "message", data = "";
      raw.split("\n").forEach(line => {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      });
      if (data) yield { event: ev, data: JSON.parse(data) };
    }
  }
}
// 用法：for await (const {event, data} of sseEvents(resp)) { ...按 event 类型更新 UI... }
```
- **两个易踩坑**：① 网络会在**任意字节**切断，事件可能跨 chunk → 必须用 `buf` 缓冲、按 `\n\n` 切；
  ② 中文是多字节，`TextDecoder({stream:true})` 才能跨 chunk 正确拼回（这两点都实测过 1 字节/块都不崩）。
- **体验变化**：以前点发送干转圈十几秒；现在实时滚「检索内部教材… → 作答… →（证据不足）补充检索… → 作答…」，最后 `done` 用原 `renderAnswer` 渲染完整回答（含来源/反思提示）。

---

## 自测题

1. `uvicorn server:app` 里的 `server` 和 `app` 分别是什么？谁负责监听端口？

2. `def chat(req: ChatRequest)` 这行里，FastAPI 是怎么知道要「从请求体读 JSON 并校验」的？如果参数写成 `q: str` 会怎样？

3. 少传必填字段 `question`，会发生什么？这套校验你写了几行代码？

4. `response_model=ChatResponse` 有哪两个作用？

5. `chat()` 现在是 `async def`，为什么？里面阻塞的 `GRAPH.invoke()` 怎么放才不卡事件循环？「在 `async def` 里直接裸调阻塞函数」会怎样？

6. 为什么 `GRAPH = get_med_graph()` 放在模块顶层、而不是放进 `chat()` 函数里？（提示：性能 + 记忆）

7. `STATIC = Path(__file__).parent / "static"`，为什么不直接写 `"static"`？

8. `/docs` 页面是哪来的？你为它额外写了代码吗？

9. 前端 `fetch("/chat")` 到拿到答案，中间经过哪些环节？（按顺序说 uvicorn → FastAPI → Pydantic → chat → 图 → 序列化）

10. 这个服务没做 CORS，为什么现在不出问题？什么情况下必须加？

11. `/chat/stream` 怎么实现「边生成边显示」？用了 FastAPI 的什么 + 图的什么能力？三种 SSE 事件分别是啥？

12. 统一异常处理里，为什么完整堆栈进日志、却只把脱敏文案 + `trace_id` 返回给前端？`trace_id` 有什么用？

13. 请求超时用 `wait_for` + `run_in_threadpool` 实现，它能真的「掐断」那个跑超时的图调用吗？为什么？（提示：Python 线程）

14. 前端为什么不用 `EventSource` 而用 `fetch` 读流？SSE 解析里 `buf` 缓冲和 `TextDecoder({stream:true})` 各防的是什么坑？