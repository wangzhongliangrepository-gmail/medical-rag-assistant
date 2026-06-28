# 学习串讲：Docker 部署（逐行精读 Dockerfile + docker-compose）

> 配合 `Dockerfile`、`docker-compose.yml`、`.dockerignore`、`docs/P6_DEPLOY.md` 阅读。
> 全项目就这两个文件 + 十几条命令，是最好补的一块。建议读完亲手 `down`/`up` 跑一遍。

## 0. 先建立大局

- **Docker 到底解决什么问题** —— 一句话面试答法："在我机器上能跑"问题。本项目依赖一堆东西：
  Python 3.11、一堆 pip 包、特定环境变量（`KMP_DUPLICATE_LIB_OK`）、还要连 qdrant。换台机器、版本一错就崩。
  Docker 把**「运行这个应用需要的整个环境」**打包成一个不可变的东西，搬到任何装了 Docker 的机器上，跑出来都一样。

- **镜像（image） vs 容器（container）**：
  - 镜像 image　＝　类 / 安装包 / 模板　　← 不可变，造一次
  - 容器 container ＝　对象 / 跑起来的实例　← 可销毁可重建，一个镜像能跑多个
  - 面试陷阱："容器里改的文件，重启还在吗?" → **不在**（除非挂了卷）。容器是临时的，这正是为什么 qdrant 的数据要挂 volumes。

- **两个文件分工**：
  - `Dockerfile`：造**一个**镜像的配方（`docker build` 执行它 → 得到 app 镜像）。装 Python、装依赖、拷代码、设启动命令。
  - `docker-compose.yml`：**怎么把多个容器编排起来一起跑**（app + qdrant，配端口 / 网络 / 卷 / 环境变量 / 启动顺序）。
  - `docker run` / `compose up` 把镜像跑成容器。

- **本项目部署全貌** —— 四类东西，三种待遇，这是整个部署设计的骨架：
  1. **进容器**：app（自己写的）+ qdrant（官方镜像）—— compose 管。
  2. **留宿主机**：Xinference —— 因为要 GPU，Windows 下容器化 GPU 太麻烦。
  3. **在云端**：DeepSeek、Tavily —— 本来就是外部 API，容器走网络访问。

  ```
  容器：app(FastAPI :8000) + qdrant(向量库 :6333)
  宿主机：Xinference(BGE, 要 GPU) ← app 经 host.docker.internal 连
  云端：DeepSeek / Tavily(外部 API) ← 密钥 .env 注入
  ```

### 自检 Q&A（建立大局）

**Q1. 镜像和容器的关系，用一个比喻说清（并回答：容器里写的文件重启后还在吗?为什么这对 qdrant 很关键?）**

镜像是类 / 模板 / 安装包，容器是这个镜像跑起来的运行实例——就像「类和对象」的关系。一个镜像能跑出多个容器。
镜像用 `docker build` 由 Dockerfile 造出来，不可变；容器用 `run`/`up` 启动，可随时销毁重建。

容器里写的文件，重启后还在吗? → **默认不在**。容器的文件系统是临时的（可写层），容器销毁就一起没了。

为什么这对 qdrant 很关键：qdrant 容器存的是我灌进去的 5 万多条医疗向量，如果只靠容器自己的文件系统，
容器一重建数据就全丢、得重新灌库（慢且要重新跑嵌入）。所以我在 compose 里给它挂了命名卷
`qdrant_storage:/qdrant/storage`，把数据存到容器外、由 Docker 管理的持久化卷里——容器随便删，数据还在。
这就对应 `docker-compose.yml` 第 9-10 行（qdrant 用卷）和第 30-31 行（声明卷）。

**Q2. 项目里哪些东西进了容器、哪些没进?各举一个，并说出「没进容器」的那个为什么不进。**

进容器：app（我自己写的 FastAPI）、qdrant（官方向量库镜像）。没进容器：Xinference。
为什么不进：Xinference 跑的 BGE 模型需要 GPU，在 Windows + Docker Desktop 下把 GPU 透传进容器很麻烦、成本高。
所以我把它留在宿主机直接吃 GPU，容器里的 app 通过 `host.docker.internal:9997` 这个特殊地址回连宿主机来用它。

**Q3. DeepSeek 是云端 API，那 app 容器要调 DeepSeek 时，密钥是怎么进到容器里的?**

密钥（`DEEPSEEK_API_KEY`、`TAVILY_API_KEY`）我没有写进代码、也没烤进镜像，而是放在宿主机的 `.env` 文件里。
compose 里用 `env_file: - .env`，容器启动时把这些值作为环境变量注入进去，代码用 `os.getenv` 读。

镜像会被推到镜像仓库、可能被很多人拉取，密钥烤进镜像 = 密钥泄露。而且 `.env` 我加进了 `.dockerignore`（第 2 行），
保证 `COPY . .` 拷代码时绝不会把 `.env` 打进镜像。所以密钥只在运行时注入，镜像本身是干净的、可安全分发的。

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

- **② `FROM python:3.11-slim`**：从一个已装好 Python 3.11 的精简 Debian 镜像开始造。`slim` = 去掉文档 / 多余包的精简版，镜像更小。

- **③ `WORKDIR /app`**：之后的命令都在容器的 `/app` 目录里执行（相当于 `cd /app`）。

- **④ `RUN apt-get install gcc g++`**：装 C/C++ 编译器——`fastembed`/`onnxruntime` 装的时候可能要现场编译。`--no-install-recommends` 不装推荐的附带包、结尾 `rm -rf /var/lib/apt/lists/*` 删 apt 索引缓存，都是**为了镜像更小**。

- **⑤⑥⑧ 分三步拷贝（分层缓存，重点）**：为什么先拷 `requirements.txt` 装包、最后才 `COPY . .`？因为 Docker 镜像是**一层层叠的**，每条指令一层，**没变的层会复用缓存**。把"装依赖"放前面，只要 `requirements.txt` 没改，重新 build 时第 ⑥ 步（最慢的装包）直接走缓存。如果把 `COPY . .` 放前面，改一行代码就会让装包缓存失效、每次重装。

- **⑥ `--no-cache-dir`**：装完不保留 pip 下载缓存，减小镜像。

- **⑦ `COPY fastembed_cache /tmp/fastembed_cache`**：把 BM25 模型预先放进镜像。**这是真踩过的坑**——容器内联网下载 HuggingFace 失败，所以把宿主机缓存好的模型打进镜像，运行时零下载。（`/tmp/fastembed_cache` 是 fastembed 在 Linux 下的默认缓存目录。）

- **⑨ `ENV`**：设两个环境变量。`KMP_DUPLICATE_LIB_OK=TRUE` 放行 torch/onnx 的 OpenMP 重复加载（否则崩）；`PYTHONUNBUFFERED=1` 让日志实时输出不缓冲。

- **⑩ `EXPOSE 8000`**：**只是文档声明**"我用 8000"，本身不映射端口。真正对外要靠 compose 的 `ports`。

- **⑪ `CMD [...]`**：容器启动时跑的命令——用 uvicorn 起 FastAPI。`--host 0.0.0.0` 关键：容器内必须监听 `0.0.0.0`（所有网卡）外面才连得进，写 `127.0.0.1` 就只有容器自己能访问。

### 自检 Q&A（Dockerfile）

**Q1. 层缓存题（必考）：为什么 `COPY requirements.txt` 和 `pip install` 要写在 `COPY . .` 之前?如果反过来写会有什么后果?**

因为要利用 Docker 的层缓存。Dockerfile 每条指令是一层，某层一变、它之后所有层全部重建。代码改动远比依赖改动频繁，
所以我把「装依赖」和「拷代码」拆开：先 `COPY requirements.txt` 再 `pip install`，只要依赖清单没变，这两层就一直命中缓存。

反过来写（先 `COPY . .` 再 `pip install`）的后果：每改一行代码，`COPY` 层就失效，后面的 `pip install` 层跟着失效，
几百个包每次构建都要重装一遍，构建从几秒变几分钟。

**Q2. `EXPOSE 8000` 这一行，是不是写了它别人就能从浏览器访问你的服务了?为什么?**

不是。`EXPOSE` 只是个文档性声明，表明「这个容器打算监听 8000 端口」，它不真正开放 / 映射端口。
要从宿主机或浏览器访问，必须靠 compose 里的 `ports: "8000:8000"`（或 `docker run -p 8000:8000`）做端口映射，
把宿主机的端口转发到容器端口。
加分句：`EXPOSE` 更多是给读 Dockerfile 的人和工具看的提示；真正打通内外的是 `ports`/`-p`。两者分工要分清。

**Q3. `CMD` 里为什么 host 必须写 `0.0.0.0` 而不是 `127.0.0.1`?**

`127.0.0.1`（localhost）只监听容器自己内部的回环网卡，这样即使做了端口映射，外面的流量进到容器也没人接，
宿主机 / 浏览器照样连不上。`0.0.0.0` 表示监听容器的所有网卡，这样经 `ports` 映射进来的请求才能被 uvicorn 接到。

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

**compose 解决什么问题**：Dockerfile 只管造一个 app 镜像。但实际要跑的是 app + qdrant 两个容器，
还要配端口、网络、卷、环境变量、谁先启动……一条条 `docker run` 写命令又长又容易错。
`docker-compose.yml` 就是把「多个容器怎么一起跑」声明式地写进一个文件，一句 `docker compose up` 全起来。

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
- **`depends_on: qdrant`**：保证**启动顺序**（先起 qdrant）。⚠️ 但它只保证"容器起来了"，**不保证 qdrant 内部服务真就绪**——严格要配 healthcheck（见第七章）。
- **`restart: unless-stopped`**：容器异常退出自动重启（除非你手动停的）。
- **顶层 `volumes: qdrant_storage:`**：声明命名卷（上面 qdrant 用到了）。

### 自检 Q&A（compose）

**Q1. `image:` 和 `build:` 各在什么时候用?你的两个 service 分别用了哪个，为什么?**

`image` 用别人造好的现成镜像（从仓库拉）；`build` 用本地 Dockerfile 现造自己的镜像。
- qdrant 用 `image: qdrant/qdrant:latest` —— 它是现成的中间件，官方有镜像，直接拉就行，没必要自己造。
- app 用 `build: .` —— 这是我自己写的代码，得用当前目录的 Dockerfile 把代码和环境打包成镜像。

**Q2. app 容器连 qdrant 用的是 `http://qdrant:6333`。这个 `qdrant` 是怎么被解析成 qdrant 容器的?如果把 qdrant 那行 `ports: 6333:6333` 删了，app 还连得上吗?**

compose 会自动建一个默认网络，把所有 service 放进去。在这个网络里每个服务名就是一个 DNS 主机名，
`qdrant` 会被解析成 qdrant 容器的内部 IP。所以 app 容器里 `http://qdrant:6333` 走的是容器间内部网络。
`ports` 映射只是把端口暴露给宿主机外部用的，跟容器间通信无关。
**删掉 `ports` 后唯一影响是**：我在 Windows 上 `localhost:6333` 连不到 qdrant 了，但 app→qdrant 照常。
加分句：三种找服务的方式要分清——容器找容器用**服务名**，容器找宿主机用 **`host.docker.internal`**，宿主机找容器用 **`localhost:映射端口`**。

**Q3. `qdrant_storage:/qdrant/storage`（命名卷）和 `./static:/app/static`（绑定挂载）有什么本质区别?各自解决什么问题?**

命名卷我不关心数据存在宿主机哪，交给 Docker 管，目的是持久化（qdrant 那 5 万向量别丢）。
绑定挂载是我明确指定宿主机某个目录直接映射进容器，改宿主机文件容器里立刻生效——
我用它做前端热更新，改 `index.html` 刷新即生效，不用重建镜像。
加分句：绑定挂载耦合了宿主机路径、是开发调试神器，生产一般不用；命名卷可移植、是持久化数据的标准做法。

**Q4. `depends_on: qdrant` 能保证 app 启动时 qdrant 已经"能接请求"了吗?**

不能。`depends_on` 只保证 qdrant 容器先启动（进程起来了），不保证它已经初始化完、能接收请求。
qdrant 进程刚起还在加载时，app 可能就来连了，会失败。
要真正「等服务就绪」，得给 qdrant 配 healthcheck，再让 app 用 `depends_on: condition: service_healthy` 等它健康（见第七章）。
加分句：这是个经典坑——「我配了 `depends_on` 为什么 app 还是连不上数据库」，答案就是它只保证启动顺序不保证就绪，
生产要配健康检查或在应用层做重试。

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
| healthcheck | 探活，配合 `condition: service_healthy` 保就绪 |
| 多阶段构建 | build 阶段 + 运行阶段分离，镜像只留运行所需 |

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
10. 怎么上云？→ 镜像推仓库（ACR/Docker Hub/GHCR）→ 云主机 pull 运行；GPU 服务需 GPU 实例或换云端 API。
11. 踩过什么坑？→ ① langchain-community 版本约束错；② BM25 模型容器内无法联网下载，预置进镜像；③ 内存版记忆容器重启全丢。

**主动说短板（显边界感）**：多阶段构建、healthcheck、资源限制、推镜像上云——了解，本项目当时没全用上；
其中多阶段构建、healthcheck、`.dockerignore` 的细节见第七章。

## 六、镜像分发：源码上 GitHub + 镜像上 GHCR + 新环境一键拉取跑（今日实操闭环）

> 这一章是亲手跑过的完整链路：**改代码 → 推源码 → 造镜像 → 推镜像 → 在没有源码的新环境拉下来跑**。
> 面试讲"怎么把项目部署到另一台机器"，这一套每一环都做过、还踩过坑，是最硬的底气。

### 6.0 先分清：一个项目有两种「push」，落在两个地方

最容易混的点——**git push 和 docker push 完全是两回事**：

| 操作 | 推的是 | 推到哪 | 网页在哪看 |
|------|--------|--------|-----------|
| `git push` | **源码**（.py 等文本） | GitHub **仓库** | 仓库主页 `<> Code` 标签 |
| `docker push` | **镜像**（二进制大包） | GHCR **镜像仓库(Packages)** | 仓库右侧 `Packages` / 账号 `?tab=packages` |

> 踩坑实录：推完镜像后在「Code 文件列表」里找镜像 → 永远找不到，因为镜像在 Packages 不在 Code。

### 6.1 源码上 GitHub（git）

```bash
git branch -M main                       # 分支改名 main
git remote add origin <仓库URL>           # origin = 远程仓库别名
git push -u origin main                  # 首推，-u 记住关联，之后 git push 即可
```
- 推送前必查 `.env` 没进过历史：`git log --all --full-history -- .env`（应为空）。
- `.gitignore` 排除 `.env`/`medical_data`/`qdrant_db`/`.idea`；`.env.example` 留模板供他人 `cp` 后填值。

### 6.2 镜像上 GHCR（docker）

```bash
# ① 建 PAT：GitHub Settings → Tokens(classic) → 勾 write:packages + read:packages
# ② 登录 GHCR（Password 处粘 PAT，不是 GitHub 密码）
docker login ghcr.io -u <用户名>
# ③ 构建并打标签（镜像名必须全小写）
docker build -t ghcr.io/<用户名>/medical-rag:latest .
# ④ 推送
docker push ghcr.io/<用户名>/medical-rag:latest
```

### 6.3 新环境拉下来跑（无源码！）

新环境只要**一个文件夹 + 两个文本文件**（`docker-compose.yml` 指向镜像 + `.env`），**不需要源码、不需要 venv/pip**——依赖全烤在镜像里。

```yaml
# fresh-env/docker-compose.yml —— 只引用镜像，不 build
services:
  app:
    image: ghcr.io/<用户名>/medical-rag:latest
    ports: ["8000:8000"]
    env_file: [.env]
    extra_hosts: ["host.docker.internal:host-gateway"]
    restart: unless-stopped
```
```bash
docker login ghcr.io -u <用户名>    # 私有镜像必须先登录，否则 pull 报 unauthorized
docker compose pull                  # 从 GHCR 拉镜像
docker compose up -d                 # 纯靠镜像跑起来
```
- `.env` 里两个 URL 指回宿主机以复用现有服务：`QDRANT_URL=http://host.docker.internal:6333`、`XINFERENCE_URL=http://host.docker.internal:9997`。
- **想证明代码真在容器里**：`docker exec <容器> ls /app` → 看到所有 .py。

### 6.4 把包 link 到仓库

包默认只在「账号级」可见，仓库右侧栏会显示 `No packages published`。两种 link 法：
- **网页**：package 页底部 `Link this package to a repository` → 选仓库。link 后仓库右侧栏出现该包，且**访问权继承仓库**。
- **Dockerfile 自动**（推荐，对以后的构建生效）：
  ```dockerfile
  LABEL org.opencontainers.image.source=https://github.com/<用户名>/medical-rag-assistant
  ```

### 6.5 今日踩的四个坑（面试"讲个你解决的问题"现成弹药）

| 坑 | 现象 | 根因 | 解法 |
|----|------|------|------|
| **端口占用** | `port is already allocated` | 宿主机 8000 被另一容器占 | 改映射 `8001:8000` 或停占用者 |
| **镜像名 ≠ 位置** | Packages 空 / downloads=0，但容器照跑 | `build -t ghcr.io/...` 只是起名，**没 push**；`compose up` 见本机有同名镜像就直接用，没去仓库拉 | 真 `docker push` 才上仓库 |
| **tag 拼写** | `No such image: ...:late` | tag 漏字母 | 用全名 `:latest` 或按镜像 ID 删 |
| **私有镜像拉取** | 换机 pull 报 `unauthorized` | 包是 Private | 目标机先 `docker login ghcr.io`，或改 Public |

### 6.6 两条核心认知（拉开差距）

1. **镜像名 ≠ 镜像在哪**：`ghcr.io/.../medical-rag` 只是名字，决定"将来 push/pull 去哪"，本机镜像可以叫任何名。`docker compose up` **本机有同名镜像就直接用，不会去拉**——所以"跑起来了"不等于"从仓库拉的"。删本机镜像 + `up` 看到 `Pulled` 才是真拉。
2. **镜像可移植 ≠ 整套系统零配置可移植**：fresh-env 实测——光有镜像还得补 ① `.env` 密钥（故意不进镜像）② 够得到 Xinference（GPU 留宿主机）③ 够得到有数据的 qdrant（数据可重建）。能讲清这条边界，就跟"只会抄 compose 模板"的人分开了。

### 6.7 命令速记（分发相关）

```bash
git push -u origin main                                  # 源码上 GitHub
docker login ghcr.io -u <用户名>                          # 登录镜像仓库(用 PAT)
docker build -t ghcr.io/<用户名>/medical-rag:latest .     # 造镜像并命名
docker push ghcr.io/<用户名>/medical-rag:latest           # 镜像上 GHCR
docker images | grep medical-rag                         # 看本机有没有这镜像
docker rmi <镜像ID>                                       # 删本机镜像(先 compose down)
docker compose pull && docker compose up -d              # 新环境真拉真跑
docker exec <容器> ls /app                                # 证明代码在容器里
```

### 6.8 面试金句

> "部署到新环境，目标机器**不需要源码、不需要 Python 环境**——镜像把这些全自包含了，目标机只要 Docker + 一份 compose 配置 + 运行时密钥。改代码的正确姿势是**改源码→重建镜像→发新版本→目标机拉新镜像**，而不是登上去改文件，这换来可复现和一键回滚。但镜像可移植不等于系统零配置：GPU 服务、密钥、数据这三类我刻意留在镜像外，按文档手动补。"

---

## 七、进阶补全（短板展开：多阶段构建 / healthcheck / .dockerignore）

> 这三块前面多次「点到为止」，这里补成正文。**面试主动讲「我了解但本项目没上，原因是……」比假装上了更显边界感。**

### 7.1 `.dockerignore` 逐行（最被低估的一块）

```dockerignore
# 密钥（经 compose 的 env_file 运行时注入，不烤进镜像）
.env

# 大数据集（灌库在宿主机跑，app 镜像不需要）
medical_data/

# 本地落盘向量库（容器用 qdrant 服务器，不需要）
qdrant_db/

# Python 缓存 / 日志 / 杂项
__pycache__/
*.py[cod]
*.log
*_out.txt
*_err.txt
.git/
.idea/
docs/
```

逐行为什么排除：
- **`.env`**：**安全第一**。`COPY . .` 会把当前目录一切拷进镜像，不排除就会把密钥烤进可分发的镜像（见第 0 节 Q3）。
- **`medical_data/`**：原始教材语料是几百 MB 的大数据集，灌库（`ingest.py`）在**宿主机**跑、产物进 qdrant，app 镜像运行时根本不读它 → 排除，镜像不被撑大。
- **`qdrant_db/`**：本地落盘模式的向量库。容器里用的是 qdrant **服务器**（独立容器 + 命名卷），这份本地副本进镜像纯属浪费。
- **`__pycache__/`、`*.py[cod]`**：Python 字节码缓存，进镜像无意义且可能跨平台不兼容。
- **`*.log` / `*_out.txt` / `*_err.txt`**：本地跑出来的日志 / 调试输出，跟运行无关。
- **`.git/`**：整个版本历史，往往比代码本身大几倍，**绝不该进镜像**。
- **`.idea/`**：IDE 配置，无关。
- **`docs/`**：学习 / 设计文档（**包括这份 `LEARN_DOCKER.md` 本身**）——不参与运行，排除以保持镜像精简。

> 一句话价值：`.dockerignore` 一管**安全**（挡密钥）、二管**镜像大小 / 构建速度**（挡大文件，`COPY` 的上下文更小、层缓存更稳）。
> 它和 `.gitignore` 是两套，作用域不同：前者管「什么进镜像」，后者管「什么进 git 历史」，**两边都要排 `.env`**。

### 7.2 healthcheck —— 把 `depends_on` 从「启动顺序」升级成「就绪保证」

前面反复说 `depends_on` 只保证启动先后、不保证就绪。补法是给 qdrant 加探活，再让 app 等它「健康」：

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    restart: unless-stopped
    healthcheck:                                  # 探活：周期性敲健康端点
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s        # 每 10s 探一次
      timeout: 5s          # 单次探测超时
      retries: 5           # 连续失败 5 次才判 unhealthy
      start_period: 20s    # 启动宽限期（这段时间内失败不计数，给它加载时间）

  app:
    build: .
    # ……（其余同原文件）
    depends_on:
      qdrant:
        condition: service_healthy             # ★ 等 qdrant healthy 再起 app，而非只等它"启动"
```

要点：
- `test` 用容器**内部** `localhost:6333`（健康检查在 qdrant 容器里跑，不走映射端口）。
- `start_period` 是关键：服务冷启动慢，给一段宽限期内的失败不计入 `retries`，避免一启动就被判死。
- 没有 `curl` 的精简镜像可改用语言内置探测（如 qdrant 也提供 `/healthz`；某些镜像得换 `wget` 或自带的 health 命令）。
- **本项目当时没上**：单机开发、qdrant 起得快，靠应用层重试 + `restart` 就够；生产 / 多依赖场景才值得配。

### 7.3 多阶段构建（multi-stage build）—— 怎么给这个项目瘦身

思路：**把「编译装包」和「运行」拆成两个阶段**，最终镜像只保留运行所需，把编译器（gcc/g++）、构建中间产物全丢掉。

```dockerfile
# ---------- 阶段一：builder，负责装依赖（带编译器） ----------
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# 把包装到一个独立前缀，方便整目录拷到下一阶段
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- 阶段二：运行镜像，不含编译器 ----------
FROM python:3.11-slim
WORKDIR /app
# 只把"装好的包"从 builder 拷过来，gcc/g++ 和 apt 缓存都留在 builder、不进最终镜像
COPY --from=builder /install /usr/local
COPY fastembed_cache /tmp/fastembed_cache
COPY . .
ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

要点 / 取舍：
- 省的是 **`gcc g++` + apt 列表 + pip 构建中间物**，运行阶段一概不留 → 镜像更小、攻击面更小。
- `COPY --from=builder` 是多阶段的核心：跨阶段只搬运需要的产物。
- **本项目收益有限**的诚实判断：依赖里 torch/onnxruntime 这些**运行时本来就需要的大库**才是镜像体积主力，
  砍掉编译器省的是「零头」。所以当时用单阶段够了——**但能讲清「为什么收益有限」恰恰是加分**，而不是盲目套模板。
- 进一步瘦身的方向（了解即可）：换更小基础镜像（`-alpine` 但 C 扩展兼容性差，慎用）、装 CPU 版 torch、合并 `RUN` 减层。

---

## 自测题

1. 为什么 Dockerfile 把 `COPY . .` 放在 `pip install` 后面？

2. `ports: "8000:8000"` 两个 8000 分别是谁？

3. `QDRANT_URL=http://qdrant:6333` 里的 `qdrant` 是什么？怎么解析到的？

4. 命名卷（qdrant_storage）和 bind 挂载（./static）有什么区别，各解决什么问题？

5. 改了 `index.html` 要不要重建容器？改了 `server.py` 呢？为什么？

6. `.dockerignore` 同时服务于哪两个目的？为什么里面**一定**要有 `.env`？它和 `.gitignore` 是一回事吗？

   <details><summary>参考答案</summary>

   两个目的：① **安全**——挡住 `.env` 等敏感文件，不让 `COPY . .` 把密钥烤进可分发的镜像；
   ② **镜像大小 / 构建速度**——挡掉 `medical_data/`、`.git/`、`qdrant_db/` 等大文件，让构建上下文更小、层缓存更稳。
   `.env` 一定要排，否则密钥进镜像 = 推到仓库就泄露。
   它和 `.gitignore` 不是一回事：作用域不同——`.dockerignore` 管「什么进镜像」，`.gitignore` 管「什么进 git 历史」，**但两边都得排 `.env`**。
   </details>

7. `depends_on: qdrant` 不保证 qdrant 就绪，怎么补？写出关键两处配置。

   <details><summary>参考答案</summary>

   给 qdrant 加 `healthcheck`（`test` 敲容器内 `localhost:6333/healthz`，配 `interval`/`timeout`/`retries`/`start_period`），
   再把 app 的 `depends_on` 从列表式改成 `depends_on: { qdrant: { condition: service_healthy } }`，
   这样 app 等到 qdrant **healthy** 才启动，而不只是「容器起来了」。生产也可在应用层做连接重试兜底。
   </details>

8. 面试官说「讲讲你的部署架构」，别一上来背命令，先给框架：

   > 我没有把所有东西都塞进容器，而是**按依赖的性质分类对待**：
   > - 自己的无状态代码（app）和现成中间件（qdrant）→ **容器化**
   > - 有状态数据（qdrant 的向量）→ **挂卷持久化**
   > - 要 GPU 的重依赖（Xinference）→ **留宿主机回连**（`host.docker.internal`）
   > - 外部云 API（DeepSeek/Tavily）和密钥 → **运行时注入，绝不进镜像**

9. qdrant 为什么进容器，且用「服务器版」而不是本地落盘？

   本地落盘模式（嵌入式）在数据量大时没有真正的 HNSW 索引，我灌了 5 万多向量后会有性能告警、检索慢。
   qdrant 服务器版有完整的 HNSW 近邻索引，检索快且稳。而且服务器版是独立服务，app 可以无状态、可水平扩展，
   数据和计算解耦——这才是生产形态。本地落盘只适合开发期单机调试。

10. 多阶段构建能给本项目省多少？为什么说「收益有限」反而是个好回答？

    <details><summary>参考答案</summary>

    多阶段把编译器（gcc/g++）、apt 缓存、pip 构建中间物留在 builder 阶段、不进最终镜像。
    但本项目镜像体积大头是 torch/onnxruntime 这些**运行时必须的大库**，砍编译器只省零头。
    所以当时用单阶段够了。能说清「为什么收益有限」证明你**理解镜像体积的真正来源**，而不是盲目套多阶段模板——这正是边界感。
    </details>