# 掌柜智库

掌柜智库是一个面向本地知识库的 FastAPI + LangGraph 应用。它可以把 Markdown、PDF、DOCX、Obsidian Vault 和教育资料导入统一知识库，再通过向量检索、Graph RAG、可选 HyDE / Web 检索和大语言模型生成回答。

项目当前由一个一体化服务提供网页、查询、导入、会话管理、任务进度和知识删除接口。

> 当前 README 按工作区现有代码编写。项目仍包含面向个人开发环境的配置和路径，部署到其他机器或生产环境前请先阅读“当前限制与安全边界”。

## 主要能力

- **网页问答**：支持多轮会话、历史消息、流式答案和查询进度展示。
- **普通文档导入**：支持 `.md`、`.pdf`、`.docx`；Markdown 可同时上传它引用的图片。
- **Obsidian 导入**：支持选择多个 Markdown / 图片文件，也支持拖入整个 Vault，保留 Vault 内的相对路径。
- **本地目录导入**：服务端递归扫描目录，自动收集 Markdown 引用的图片，并为每篇文档建立独立任务。
- **教育资料导入**：课程资料和题目资料使用独立解析链，支持后续课程和题目查询。
- **混合检索**：普通向量检索与 Graph RAG 结果可通过 RRF 融合，再经过重排和答案生成。
- **任务追踪**：导入和查询都会返回 `task_id`，前端可以通过轮询或 SSE 读取 `queued`、`processing`、`completed`、`failed`、`cancelled` 等状态。
- **文档级删除**：可以按 `document_id`、来源类型和来源目录删除已导入知识。

## 工作链路

```text
网页 / API 客户端
        |
        v
FastAPI 一体化服务
掌柜智库/atguigu/web/api/app_service.py
        |
        +--> 查询：QueryGraphRunner
        |      意图与主体确认
        |        -> 向量检索
        |        -> Graph RAG
        |        -> 可选 HyDE / Web 检索
        |        -> RRF -> Rerank -> LLM 流式回答
        |
        +--> 导入：ImportGraphRunner
        |      格式转换 -> 图片处理 -> 文档切分
        |        -> 主体识别 -> BGE-M3 向量化
        |        -> Milvus -> Neo4j 图谱构建
        |
        +--> 教育资料：确定性解析链
        |
        +--> MongoDB：会话、历史和任务同步信息
        +--> MinIO：原始文件备份和图片对象
        +--> Milvus：知识切片向量，当前集合为 knowledge_chunks_graph
        +--> Neo4j：文档实体和关系图谱
```

普通文档和 Obsidian 文档会进入导入图；查询侧默认走向量检索和 Graph RAG，HyDE 与 Web 检索默认关闭，可以通过环境变量按需开启。

## 环境要求

- Windows 10/11 或兼容的 Python 运行环境。
- Python `>= 3.12`。
- [uv](https://docs.astral.sh/uv/)：用于创建环境和安装依赖。
- 可用的模型平台：当前支持 ModelScope 和 OpenRouter。
- Milvus：存储知识切片向量。
- MongoDB：保存会话历史、会话目录和导入同步信息。
- Neo4j：构建和查询 Graph RAG 图谱。
- MinIO：备份原始文档并提供图片对象访问。
- MinerU API：导入 PDF 时使用；只导入 Markdown / DOCX 时可以不配置 PDF 解析能力。
- 能运行 BGE-M3 的设备和模型文件；`DEVICE`、`USE_FP16` 按本机 CPU / GPU 情况设置。

模型推理、Embedding、PDF 解析和外部数据库都可能产生网络、显存或服务可用性要求。README 不包含任何真实密钥，所有密钥都应只写入本地 `.env`。

## 快速启动

### 1. 安装依赖

在仓库根目录执行：

```powershell
uv sync
```

项目的 `pyproject.toml` 为 PyTorch 配置了 CUDA `cu130` 软件源。如果当前机器没有匹配的 CUDA 环境，安装 PyTorch 前请根据机器驱动和部署目标调整依赖源；不要把不匹配的 CUDA wheel 直接用于生产环境。

### 2. 创建环境文件

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`。最少需要配置当前模型平台、模型密钥、Embedding 模型、Milvus、MongoDB、Neo4j 和 MinIO。`.env.example` 已列出常用模型、答案预算、Neo4j、MinIO 和 Embedding 字段；Milvus、MongoDB 等连接字段也必须补到 `.env` 中。

### 3. 启动服务

推荐使用统一入口：

```powershell
uv run uvicorn atguigu.web.api.app_service:app --host 127.0.0.1 --port 8000
```

也可以直接运行模块：

```powershell
uv run python -m atguigu.web.api.app_service
```

服务启动后访问：

- 网页：<http://127.0.0.1:8000/>
- FastAPI 交互文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

健康检查示例：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

预期返回：

```json
{
  "message": "API已连接"
}
```

### 4. 验证服务是否可用

启动后先检查 `/health`，再打开 `/docs` 确认 FastAPI 已注册路由。真正执行文档导入或问答前，应确认对应的模型、Embedding、Milvus、MongoDB、Neo4j 和 MinIO 服务已经可连接；只启动 FastAPI 并不代表整条 RAG 链路已经就绪。

## 配置说明

### 模型平台

使用 `MODEL_PROVIDER` 选择平台：

```dotenv
MODEL_PROVIDER=modelscope

MODELSCOPE_API_KEY=你的魔搭密钥
MODELSCOPE_BASE_URL=https://api-inference.modelscope.cn/v1
MODELSCOPE_LLM_MODEL=Qwen/Qwen3.8-27B
MODELSCOPE_VL_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

切换到 OpenRouter 时：

```dotenv
MODEL_PROVIDER=openrouter

OPENROUTER_API_KEY=你的OpenRouter密钥
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_LLM_MODEL=你的文本模型
OPENROUTER_VL_MODEL=你的视觉模型
```

`MODEL_PROVIDER` 只接受 `modelscope` 或 `openrouter`。修改 `.env` 后需要重启服务才会重新读取配置。

### 数据库和对象存储

根据当前 `atguigu/config/config.py`，相关配置字段如下：

| 服务 | 环境变量 | 用途 |
| --- | --- | --- |
| Milvus | `MILVUS_URL` | 知识切片和向量检索；主体识别集合名可由 `ITEM_COLLECTION` 覆盖 |
| MongoDB | `MONGO_URL`、`MONGO_DB_NAME` | 会话、历史消息、会话目录和同步信息 |
| Neo4j | `NEO4J_URI`、`NEO4J_USERNAME`、`NEO4J_PASSWORD`、`NEO4J_DATABASE` | Graph RAG 的实体和关系图 |
| MinIO | `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`、`MINIO_BUCKET_NAME`、`MINIO_IMG_DIR` | 原始文件备份、图片上传和图片访问 |
| Embedding | `EMBEDDING_MODEL`、`DEVICE`、`USE_FP16` | BGE-M3 向量化模型、运行设备和半精度开关 |
| PDF 解析 | `MINERU_API_KEY` | PDF 转 Markdown 和图片提取 |

当前正文向量默认写入 Milvus 集合 `knowledge_chunks_graph`。不要只修改旧集合名就认为 Graph RAG 已切换完成，集合字段和图谱数据必须与当前导入、查询代码保持一致。

### 查询可选能力

默认关闭会增加外部调用或耗时的分支：

```dotenv
# 默认 false
ENABLE_HYDE=false
ENABLE_WEB_SEARCH=false
```

开启 Web 检索还需要配置 MCP 服务及其凭证：

```dotenv
ENABLE_WEB_SEARCH=true
MCP_SERVER=你的MCP服务地址
OPENAI_API_KEY=你的MCP或相关服务密钥
```

HyDE 会额外调用一次语言模型生成假设性文档，Web 检索会引入外部网络和 MCP 服务；建议先在本地知识库链路稳定后再启用。

### 运行时目录和答案预算

可以用以下配置调整答案上下文和临时目录：

```dotenv
KNOWLEDGE_RUNTIME_TEMP_ROOT=C:\path\to\runtime-data
LLM_MAX_TOKENS=32768
LLM_MAX_CONTINUATIONS=1
LLM_CONTINUATION_CONTEXT_CHARS=24000
ANSWER_MAX_CONTEXT_CHARS=48000
ANSWER_MAX_HISTORY_CHARS=24000
LLM_CONTEXT_WINDOW_TOKENS=262144
LLM_PROMPT_SAFETY_TOKENS=4096
```

模型实际上下文窗口和平台限制优先于这些预算。配置过大可能导致请求被平台拒绝、显存不足或回答延迟升高。

## 普通使用者流程

### 网页问答

1. 启动服务并打开 <http://127.0.0.1:8000/>。
2. 创建或选择会话。
3. 在导入区域上传文档，等待导入任务完成。
4. 在对话区域提问，前端会先创建查询任务，再通过 SSE 显示检索进度和增量答案。
5. 如果需要停止当前查询，可以使用前端的取消操作；接口形式见下文。

### 上传普通文档

网页上传支持一个主文档和附属图片：

- 主文档：`.md`、`.pdf` 或 `.docx`；
- 附属图片：`.jpg`、`.jpeg`、`.png`、`.gif`、`.bmp` 或 `.webp`；
- Markdown 中引用的图片应与上传的图片文件名匹配。

上传 Markdown 时，如果没有同时提交引用图片，文档文本仍可能导入成功，但图片引用无法在后续回答中正常显示。

### 导入 Obsidian Vault

可以在网页中拖入整个 Vault 文件夹或选择多个文件。服务会保留 Vault 内的相对路径，只接受 Markdown 和常见图片文件，并为每篇 Markdown 创建一个导入任务。

### 导入本地目录

`/import_local` 由服务端读取路径，因此路径必须是运行 FastAPI 进程的那台机器上的目录，不是浏览器所在机器的路径。它适合本地部署或可信内网环境；公网服务不要直接开放这个接口。

### 教育资料

`/education/import` 接受课程 Markdown 和题目 Markdown，可以只上传其中一类。导入完成后可通过 `/education/courses` 和 `/education/questions` 查询结构化结果。

## 常用 API

所有接口默认使用 `http://127.0.0.1:8000`。正式集成时请以运行中的 `/docs` 为准。

### 创建查询任务

PowerShell：

```powershell
$body = @{
    query = '设备出现故障时应该如何排查？'
    session_id = 'demo-session'
} | ConvertTo-Json

$result = Invoke-RestMethod `
    -Method Post `
    -Uri 'http://127.0.0.1:8000/query' `
    -ContentType 'application/json' `
    -Body $body

$result
```

返回值包含 `task_id`、`query` 和 `session_id`。使用返回的 `task_id` 读取流式事件：

```powershell
curl.exe -N 'http://127.0.0.1:8000/stream/<task_id>'
```

SSE 事件可能包含进度、上下文信息、答案增量和最终结果。查询状态取消接口为：

```powershell
curl.exe -X POST 'http://127.0.0.1:8000/query/<task_id>/cancel'
```

### 上传文档

```powershell
curl.exe -X POST `
    -F 'files=@D:\docs\设备手册.pdf' `
    -F 'files=@D:\docs\images\warning.png' `
    'http://127.0.0.1:8000/upload'
```

返回值包含 `task_id`、来源信息和 `document_id`。轮询导入状态：

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/status/<task_id>'
```

### 预览本地目录导入

`dry_run=true` 只扫描目录并配对图片，不执行 MinIO 备份和导入管线：

```powershell
$body = @{ dir = 'D:\knowledge\manuals' } | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri 'http://127.0.0.1:8000/import_local?dry_run=true' `
    -ContentType 'application/json' `
    -Body $body
```

确认结果后再去掉 `dry_run=true`。该接口读取服务端文件系统，必须只对可信调用方开放。

### 会话和知识管理

常用路由：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/sessions` | 获取会话目录 |
| `POST` | `/sessions` | 创建会话 |
| `PATCH` | `/sessions/{session_id}` | 重命名会话 |
| `DELETE` | `/sessions/{session_id}` | 删除会话及历史 |
| `GET` | `/history/{session_id}` | 获取会话历史 |
| `DELETE` | `/history/{session_id}` | 清空会话历史 |
| `GET` | `/stream/{task_id}` | 读取查询任务的 SSE 流 |
| `POST` | `/upload` | 上传普通文档和附属图片 |
| `POST` | `/upload_obsidian` | 上传 Obsidian Vault 文件 |
| `POST` | `/import_local` | 扫描服务端本地目录并导入 |
| `POST` | `/education/import` | 上传课程或题目 Markdown |
| `GET` | `/status/{task_id}` | 获取普通导入任务状态 |
| `GET` | `/education/status/{task_id}` | 获取教育导入状态 |
| `GET` | `/education/courses` | 查询课程 |
| `GET` | `/education/questions` | 查询题目 |
| `DELETE` | `/knowledge/documents/{document_id}` | 删除一个文档的知识切片 |
| `DELETE` | `/knowledge/sources/{source_type}?source_id=...` | 删除一个来源的知识 |
| `DELETE` | `/knowledge/directories` | 按来源目录删除知识 |

## 当前限制与安全边界

当前版本适合本地或受控内网开发环境，不应在没有额外防护的情况下直接暴露到公网：

- 当前 FastAPI 路由没有统一认证和授权层；上传、删除、会话访问及任务接口需要由部署层限制访问范围。
- 当前 CORS 配置允许所有来源，生产部署应改为明确的前端来源列表。
- `/import_local` 可以读取服务端提交的目录路径；必须做目录白名单、用户权限隔离和路径审计。
- `.env` 中包含模型、数据库、对象存储密钥，不要提交到 Git，不要把密钥写进 README、前端或日志。
- 任务队列和 SSE 状态主要保存在当前进程内，多进程、多实例或服务重启后的任务恢复能力需要单独设计。
- Milvus、MongoDB、Neo4j 和 MinIO 是多个外部存储，导入过程中发生部分成功时，应结合任务状态和来源身份做核对与重试。
- `app_service.py` 当前仍包含个人开发机上的输出目录配置。迁移到其他机器前，请搜索并替换相关 `E:\AI大模型\...\output` 路径，或将其收敛到统一环境变量配置。
- 未配置模型、Embedding 或外部存储时，服务可能可以启动，但查询和导入链路不一定可用；请用 `/health` 之外的真实导入和查询流程做验收。

## 项目结构

```text
.
├── 掌柜智库/
│   ├── atguigu/
│   │   ├── config/          # 环境变量和提示词配置
│   │   ├── import_process/  # 文档导入、切分、向量化和图谱构建
│   │   ├── query_process/   # 查询路由、召回、融合、重排和答案生成
│   │   ├── edu_process/     # 教育资料解析与导入
│   │   ├── tool/            # Milvus、MongoDB、MinIO、Neo4j 和任务工具
│   │   └── web/             # FastAPI API 和静态前端
│   ├── docs/architecture/   # 全链路架构说明
│   └── tests/               # 当前工作区中的测试
├── .env.example             # 环境变量模板
├── pyproject.toml            # Python 项目和依赖定义
├── uv.lock                  # uv 解析出的依赖锁定文件
└── README.md
```

关键入口：

- 服务入口：[app_service.py](掌柜智库/atguigu/web/api/app_service.py)
- 配置入口：[config.py](掌柜智库/atguigu/config/config.py)
- 前端页面：[index.html](掌柜智库/atguigu/web/page/index.html)
- 全链路说明：[掌柜智库-全链路思维图.md](掌柜智库/docs/architecture/掌柜智库-全链路思维图.md)

## 开发检查

提交代码前至少执行：

```powershell
git diff --check
```

如果已经配置好依赖和外部服务，再执行健康检查、一次最小 Markdown 导入和一次查询，确认不仅是服务进程启动，而且导入、检索、答案生成和状态回传都能闭环。项目当前没有在 README 中虚构统一测试命令；具体测试文件和运行条件以工作区实际内容为准。

## License

当前仓库未提供明确的 License 文件。对外发布或复用前，请先补充版权和授权说明。
