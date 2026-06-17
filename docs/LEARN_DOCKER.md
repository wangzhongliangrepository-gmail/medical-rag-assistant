# 学习串讲：Docker 部署（逐行精读 Dockerfile + docker-compose）

> 配合 `Dockerfile`、`docker-compose.yml`、`.dockerignore`、`docs/P6_DEPLOY.md` 阅读。
> 全项目就这两个文件 + 十几条命令，是最好补的一块。建议读完亲手 `down`/`up` 跑一遍。

## 0. 先建立大局

- **镜像（image） vs 容器（container）**：镜像是"模板/安装包"（类），容器是镜像跑起来的"运行实例"（对象）。一个镜像可跑出多个容器。
- **两个文件分工**：
  - `Dockerfile`：**怎么造 app 这一个镜像**（装 Python、装依赖、拷代码、设启动命令）。
  - `docker-compose.yml`：**怎么把多个容器编排起来一起跑**（app + qdrant，配端口/网络/卷/环境变量）。
- **本项目部署全貌**：
  ```
  容器：app(FastAPI :8000) + qdrant(向量库 :6333)
  宿主机：Xinference(BGE, 要 GPU) ← app 经 host.docker.internal 连
  云端：DeepSeek / Tavily(外部 API) ← 密钥 .env 注入
  ```

---

## 一、Dockerfile 逐行

```dockerfile
# 医疗知识助手 FastAPI 服务镜像          # ① 注释，无作用
FROM python:3.11-slim                    # ② 基础镜像
WORKDIR /app                             # ③ 容器内工作目录
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*   # ④ 装编译依赖并清缓存
COPY requirements.txt .                  # ⑤ 先只拷依赖清单
RUN pip install --no-cache-dir -r requirements.txt   # ⑥ 装 Python 包
COPY fastembed_cache /tmp/fastembed_cache    # ⑦ 预置 BM25 模型（踩坑解决）
COPY . .                                 # ⑧ 拷全部代码
ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONUNBUFFERED=1                    # ⑨ 环境变量
EXPOSE 8000                              # ⑩ 声明端口
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]   # ⑪ 启动命令
```

- **② `FROM python:3.11-slim`**：从一个已装好 Python 3.11 的精简 Debian 镜像开始造。`slim` = 去掉文档/多余包的精简版，镜像更小。
- **③ `WORKDIR /app`**：之后的命令都在容器的 `/app` 目录里执行（相当于 `cd /app`）。
- **④ `RUN apt-get install gcc g++`**：装 C/C++ 编译器——`fastembed`/`onnxruntime` 装的时候可能要现场编译。`--no-install-recommends` 不装推荐的附带包、结尾 `rm -rf /var/lib/apt/lists/*` 删 apt 索引缓存，都是**为了镜像更小**。
- **⑤⑥⑧ 分三步拷贝（分层缓存，重点）**：为什么先拷 `requirements.txt` 装包、最后才 `COPY . .`？因为 Docker 镜像是**一层层叠的**，每条指令一层，**没变的层会复用缓存**。把"装依赖"放前面，只要 `requirements.txt` 没改，重新 build 时第 ⑥ 步（最慢的装包）直接走缓存。如果把 `COPY . .` 放前面，改一行代码就会让装包缓存失效、每次重装。
- **⑥ `--no-cache-dir`**：装完不保留 pip 下载缓存，减小镜像。
- **⑦ `COPY fastembed_cache /tmp/fastembed_cache`**：把 BM25 模型预先放进镜像。**这是真踩过的坑**——容器内联网下载 HuggingFace 失败，所以把宿主机缓存好的模型打进镜像，运行时零下载。
- **⑨ `ENV`**：设两个环境变量。`KMP_DUPLICATE_LIB_OK=TRUE` 放行 torch/onnx 的 OpenMP 重复加载（否则崩）；`PYTHONUNBUFFERED=1` 让日志实时输出不缓冲。
- **⑩ `EXPOSE 8000`**：**只是文档声明**"我用 8000"，本身不映射端口。真正对外要靠 compose 的 `ports`。
- **⑪ `CMD [...]`**：容器启动时跑的命令——用 uvicorn 起 FastAPI。`--host 0.0.0.0` 关键：容器内必须监听 `0.0.0.0`（所有网卡）外面才连得进，写 `127.0.0.1` 就只有容器自己能访问。

---

## 二、docker-compose.yml 逐段

```yaml
services:                                # 要跑的服务（每个 = 一类容器）
  qdrant:                                # 服务①：向量库
    image: qdrant/qdrant:latest          # 用现成官方镜像（不自己造）
    ports:
      - "6333:6333"                      # 端口映射 宿主机:容器
    volumes:
      - qdrant_storage:/qdrant/storage   # 命名卷：数据持久化
    restart: unless-stopped              # 异常退出自动重启

  app:                                   # 服务②：我们的应用
    build: .                             # 用当前目录的 Dockerfile 造镜像
    ports:
      - "8000:8000"
    env_file:
      - .env                             # 从 .env 注入密钥（DEEPSEEK/TAVILY...）
    environment:
      - QDRANT_URL=http://qdrant:6333    # 连容器内 qdrant（服务名当主机名）
      - XINFERENCE_URL=http://host.docker.internal:9997   # 连宿主机 Xinference
    extra_hosts:
      - "host.docker.internal:host-gateway"   # 让容器能访问宿主机
    volumes:
      - ./static:/app/static             # bind 挂载：改前端刷新即生效，不用重建
    depends_on:
      - qdrant                           # 先起 qdrant 再起 app
    restart: unless-stopped

volumes:
  qdrant_storage:                        # 声明上面用到的命名卷
```

逐点：
- **`services`**：每个服务对应一类容器。这里两个：`qdrant`、`app`。
- **`image:` vs `build:`**：qdrant 用现成镜像（`image`），app 用本地 Dockerfile 现造（`build: .`）。
- **`ports: "6333:6333"`**：`宿主机端口:容器端口`。把容器端口映射到宿主机，你浏览器才能访问 `localhost:8000`。
- **`qdrant 的 volumes: qdrant_storage:/qdrant/storage`（命名卷）**：把 qdrant 数据存到 Docker 管理的**命名卷**里，**容器删了重建数据还在**（你灌的 54759 向量靠它持久化）。
- **`app 的 volumes: ./static:/app/static`（bind 挂载）**：把宿主机 `static` 目录直接映射进容器，**改 `index.html` 刷新即生效、不用重建镜像**（开发热更新）。
  - 两种卷的区别：**命名卷**（`名字:路径`）= Docker 管的持久化数据；**bind 挂载**（`./本地路径:容器路径`）= 直接映射本地目录，常用于开发。
- **`env_file: .env`**：把 `.env` 里的密钥**运行时注入**容器，不烤进镜像（`.dockerignore` 排除了 `.env`）。
- **`environment:`**：直接写死的环境变量。`QDRANT_URL=http://qdrant:6333` 里的 **`qdrant` 是服务名**——compose 建了内部网络，**服务名自动当主机名**，app 用 `qdrant` 就能连到 qdrant 容器，不用管 IP。
- **`extra_hosts: host.docker.internal:host-gateway`**：让容器能用 `host.docker.internal` 这个域名访问**宿主机**（Xinference 在宿主机上）。Linux 上需要这行才生效。
- **`depends_on: qdrant`**：保证**启动顺序**（先起 qdrant）。⚠️ 但它只保证"容器起来了"，**不保证 qdrant 内部服务真就绪**——严格要配 healthcheck。
- **`restart: unless-stopped`**：容器异常退出自动重启（除非你手动停的）。
- **顶层 `volumes: qdrant_storage:`**：声明命名卷（上面 qdrant 用到了）。

---

## 三、核心概念速记

| 概念 | 一句话 |
|------|--------|
| 镜像 image | 模板/安装包（类）|
| 容器 container | 镜像的运行实例（对象）|
| Dockerfile | 怎么造一个镜像 |
| docker-compose | 怎么编排多个容器一起跑 |
| 端口映射 ports | 宿主机端口:容器端口，对外暴露 |
| 命名卷 named volume | Docker 管的持久化数据（重启不丢）|
| bind 挂载 | 映射本地目录进容器（开发热更新）|
| env_file / environment | 运行时注入环境变量 |
| 服务名网络 | compose 内服务名自动当主机名互连 |
| host.docker.internal | 容器访问宿主机的域名 |
| depends_on | 启动顺序（不保证就绪）|
| EXPOSE | 仅声明端口（不等于映射）|

## 四、常用命令

```bash
docker compose ps              # 看跑着哪些容器
docker compose up -d            # 后台拉起所有服务
docker compose up -d --build app   # 重建并起 app（改了 Dockerfile/Python 代码时）
docker compose down             # 停掉并删除容器（命名卷数据还在）
docker compose logs app          # 看 app 日志
docker compose logs -f app       # 实时跟踪日志
docker compose restart app       # 重启 app
docker compose exec app sh       # 进 app 容器里看
```

> 改 `static/`（前端）→ 刷新即可（bind 挂载）；改 Python 代码或 Dockerfile → 要 `up -d --build app`。

## 五、面试问答

1. 镜像 vs 容器？→ 模板 vs 运行实例。
2. 为什么先拷 requirements 再拷代码？→ 分层缓存，改代码不重装包。
3. EXPOSE 和 ports 区别？→ EXPOSE 仅文档声明，对外靠 ports 映射。
4. 镜像怎么瘦身？→ slim 基础镜像、`--no-cache-dir`、清 apt 缓存；进阶多阶段构建。
5. 容器之间怎么通信？→ compose 内部网络，服务名当主机名（app 连 `qdrant:6333`）。
6. 容器怎么连宿主机服务？→ `host.docker.internal`（连宿主机 Xinference）。
7. 命名卷 vs bind 挂载？→ 前者 Docker 管的持久化数据；后者映射本地目录（开发热更新）。
8. 密钥怎么管、为什么不进镜像？→ `.env` 运行时注入 + `.dockerignore` 排除。
9. depends_on 保证依赖就绪吗？→ 不，只保证启动顺序，要 healthcheck。
10. 怎么上云？→ 镜像推仓库（ACR/Docker Hub）→ 云主机 pull 运行；GPU 服务需 GPU 实例或换云端 API。
11. 踩过什么坑？→ ① langchain-community 版本约束错；② BM25 模型容器内无法联网下载，预置进镜像；③ 内存版记忆容器重启全丢。

**主动说短板（显边界感）**：多阶段构建、healthcheck、资源限制、推镜像上云——了解，本项目规模没用上。

## 自测题

1. 为什么 Dockerfile 把 `COPY . .` 放在 `pip install` 后面？
2. `ports: "8000:8000"` 两个 8000 分别是谁？
3. `QDRANT_URL=http://qdrant:6333` 里的 `qdrant` 是什么？怎么解析到的？
4. 命名卷（qdrant_storage）和 bind 挂载（./static）有什么区别，各解决什么问题？
5. 改了 `index.html` 要不要重建容器？改了 `server.py` 呢？为什么？
