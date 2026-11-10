# 掌柜智库项目 AI 修改总指南

> 这是一份面向项目使用者和后续维护者的总说明。
>
> 目标是用通俗的话讲清楚：掌柜智库现在有哪些能力、AI 修改过哪些地方、每处修改解决了什么问题、日常应该怎么使用，以及遇到问题时应该先检查哪里。
>
> 项目目录：`D:\PythonProject\Knowledge_Database\掌柜智库`

---

## 一、先看整体：掌柜智库现在是怎么工作的

掌柜智库现在不是单纯的“上传文件，然后问问题”，而是一条完整的知识库问答链路：

```text
上传资料
   ↓
识别文件类型（Markdown / PDF / DOCX）
   ↓
转换和清洗内容
   ↓
识别图片、上传图片、生成图片说明
   ↓
按标题和语义切分内容
   ↓
BGE-M3 生成向量
   ↓
写入 Milvus 向量库和 MinIO 图片存储
   ↓
用户提问
   ↓
识别问题类型和主体
   ↓
向量检索 + 关键词/多路融合 + 重排序
   ↓
结合当前会话记忆生成答案
   ↓
前端流式显示答案、图片、来源、处理步骤和耗时
```

可以把它理解成四个部分：

1. **资料进入系统**：负责把 Markdown、PDF、DOCX 和图片变成系统能理解的知识。
2. **知识库存储**：Milvus 保存文本向量和元数据，MinIO 保存图片，MongoDB 保存会话历史和长期摘要。
3. **问题检索**：先判断用户想问什么，再从合适的知识库里找内容。
4. **答案展示**：模型根据检索结果和会话记忆生成答案，前端负责流式显示、图片渲染和状态提示。

---

## 二、最重要的整体改进

| 原来的问题 | 现在的处理方式 | 你能看到的效果 |
| --- | --- | --- |
| 问“你好”也进入知识库检索，容易报错 | 先识别闲聊、课程、题目、文档和普通知识问题 | 闲聊直接回复，减少无意义检索 |
| DOCX 不能直接进入原有 Markdown 流程 | 增加 DOCX 转 Markdown 节点 | 可以直接导入 Word 资料 |
| 教育课程和题目容易被普通文本切碎 | 增加教育资料专用解析器 | 一门课程一个知识块，一道题的题干、选项、答案、解析保持完整 |
| 标题上下文丢失，检索只看到局部内容 | 使用树形标题和面包屑路径切分 | 搜索结果更容易知道内容属于哪一章、哪一节 |
| 图片已经存到 MinIO，但回答不显示 | 打通“识别 → 上传 → 回填 URL → 检索 → 前端渲染”整条链路 | 问到流程图、截图、设备图片时可以直接看到图片 |
| 多文件导入偶发卡住或崩溃 | 图片视觉调用限流，BGE-M3 共享调用加锁，导入任务保留 Future | 并发导入更稳定，原生崩溃风险降低 |
| 会话名称一直是“新会话” | 第一轮问题自动作为会话标题 | 左侧会话列表能区分不同会话 |
| 广泛问题回答突然结束 | 根据模型上下文窗口动态压缩检索内容和历史 | 尽量把上下文控制在模型能接受的范围内，达到输出上限时会明确提示 |
| 只看到“失败”，不知道为什么 | 后端保存错误摘要并透传前端 | 导入卡片里能直接看到具体报错 |
| 查询速度很慢 | web_search 开关化、HyDE 缓存、重排序失败降级 | 默认查询链路更短，网络搜索不再拖慢每次回答 |

<!-- ==================== AI修改 开始 ==================== -->
### 主体识别节点优化（2026-08-21）

本轮优化针对普通资料导入链路中的 `node_item_name_recognition`，保持“一份文档识别一个核心主体”的数据模型不变，重点提高长文档和模型波动时的稳定性。

涉及文件：

- `atguigu/import_process/nodes/node_item_name_recognition.py`
- `atguigu/config/prompt.py`
- `tests/test_item_name_recognition.py`

主要变化：

1. 主体识别输入不再只取固定的前、中、后 9 个正文片段，而是同时建立章节标题索引，再补充文档前部、四分位、中部和尾部的代表性正文证据。
2. 证据总长度限制为 12000 字符，避免长文档把主体识别模型的上下文撑得过大。
3. 提示词不再把所有资料都称为“商品名称与型号”，现在允许课程、题库、设备、软件、系统、项目、技术、标准、模型和算法等主体类型。
4. 模型输出会去掉“主体名称：”等前缀和包裹符号，但保留 `Python 3.12`、`Windows Server 2022` 等名称内部的有效空格。
5. 如果模型输出的名称无法在文件名、章节标题或正文证据中找到，程序回退到文件名，避免把模型自行编造的概括写进主体库。
6. 模型调用失败、返回空值或返回不确定词语时，使用文件名作为保守兜底，不让一次外部模型波动直接阻断文档导入。
7. 主体向量化和结果校验完成后才执行旧主体记录清理，并统一转义 Milvus 字符串过滤条件，减少导入失败时的旧数据丢失和特殊字符解析问题。

维护注意：主体名称和主体向量发生变化后，已经导入的旧 chunk 不会自动更新。修改此节点后，建议重新导入受影响的资料；如果需要保留旧数据，优先使用新的 Milvus 集合名进行验证。

测试文件中的主体识别测试不访问真实 LLM、Milvus 或 embedding 服务，只验证证据采样、名称规范化、错误回退和过滤表达式转义等稳定契约。

本轮验证结果：`tests/test_item_name_recognition.py` 共 4 项测试通过，主体识别相关文件语法检查通过。现有 `tests/test_education_features.py` 为 20 项通过、1 项失败；失败原因是原有的 `get_answer_scope_instruction` 尚未在答案节点提供，本轮没有修改答案生成逻辑。
<!-- ==================== AI修改 结束 ==================== -->

---

## 三、配置和模型：项目使用哪些“记忆”和“能力”

主要配置在：

- `atguigu/config/config.py`
- 项目根目录的 `.env`

### 1. 生成答案的模型

项目已经把答案生成、主体识别、HyDE 等调用集中到配置中。语言模型和视觉模型可以通过 `MODEL_PROVIDER` 在魔搭与 OpenRouter 之间切换；图片摘要使用视觉模型；向量检索使用本地 BGE-M3；重排序仍然是单独的 OpenRouter 链路，不参加本次平台切换。

这里要特别区分四类模型：

- **答案模型**：把检索结果整理成自然语言答案。
- **主体识别模型**：判断用户问的是哪个课程、题库、项目或文档。
- **向量模型 BGE-M3**：把问题和资料变成向量，用来找相似内容。
- **重排序模型**：对初步召回结果重新排序，提高前几条结果的准确性。

所以，BGE-M3 是本地模型，重排序是独立链路；答案模型、主体识别、HyDE、主体确认统一走当前 `MODEL_PROVIDER` 的语言模型配置，图片摘要走当前平台的视觉模型配置。

### 2.1 魔搭/OpenRouter 一键切换

`.env` 中只改这一项即可切换语言模型和视觉模型：

```dotenv
MODEL_PROVIDER=modelscope
```

或：

```dotenv
MODEL_PROVIDER=openrouter
```

魔搭模式读取：

```dotenv
MODELSCOPE_API_KEY=...
MODELSCOPE_BASE_URL=https://api-inference.modelscope.cn/v1
MODELSCOPE_LLM_MODEL=Qwen/Qwen3.8-27B
MODELSCOPE_VL_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

OpenRouter 模式读取：

```dotenv
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_LLM_MODEL=dots-studio/dots-3-note-preview:free
OPENROUTER_VL_MODEL=dots-studio/dots-3-note-preview:free
```

当前按用户要求，OpenRouter 的语言模型和视觉模型默认都使用 `dots-studio/dots-3-note-preview:free`。如果平台后续调整该模型的图片输入能力，只需要单独修改 `OPENROUTER_VL_MODEL`，语言模型不受影响。

切换后必须重启后端，因为模型配置和 LangChain 客户端在进程启动/首次调用时读取并缓存。重排序仍只读取 `OPENROUTER_API_KEY`，不受 `MODEL_PROVIDER` 影响。

<!-- ==================== AI修改 开始 ====================
     余额耗尽和临时限流必须分开处理，避免无效指数退避。
     ==================== AI修改 结束 ==================== -->
如果模型接口返回 `insufficient balance`、`insufficient_quota`、`quota exceeded`、`余额不足` 或 `额度不足`，这类错误不会进入指数退避，因为等待不会补充额度；只有 `Too Many Requests`、RPM、TPM 等临时频率限制才会等待后重试。

### 2. 上下文和输出长度配置

目前重点配置是：

```text
LLM_CONTEXT_WINDOW_TOKENS=262144
LLM_MAX_TOKENS=16384
LLM_MAX_CONTINUATIONS=1
LLM_CONTINUATION_CONTEXT_CHARS=24000
ANSWER_MAX_CONTEXT_CHARS=48000
ANSWER_MAX_HISTORY_CHARS=24000
ANSWER_CONTEXT_DISPLAY_LIMIT=80000
LLM_PROMPT_SAFETY_TOKENS=4096
QUERY_HISTORY_LIMIT=20
```

简单理解：

- `LLM_CONTEXT_WINDOW_TOKENS`：模型这一轮最多能接收多少上下文。
- `LLM_MAX_TOKENS`：模型最多生成多少答案内容。
- `LLM_PROMPT_SAFETY_TOKENS`：给系统提示词、问题格式和估算误差预留的安全空间。
- `QUERY_HISTORY_LIMIT`：最多从会话历史中取多少条记录参与记忆处理。

这不是把数字越大越好。数字过大可能造成请求超限、速度变慢或模型接口拒绝。现在的代码会在发送前按总预算分配“检索内容”和“历史记忆”，比原来分别硬截断更加安全。

### 3. 三个数据库/存储分别负责什么

目前看到三个 Milvus 集合是正常的，不是重复创建：

1. **普通 chunks 集合**：保存项目文档、设备手册、商品资料等普通知识块。
2. **教育 chunks 集合**：保存课程介绍和题库数据，字段更多，支持课程和题目筛选。
3. **items 集合**：保存主体名称，例如课程系列名、题库名、项目名，用于先识别“用户问的是谁”。

查询通常是：

```text
用户问题
  ↓
主体识别：找到可能的课程/题库/项目名
  ↓
判断 query_type
  ↓
按 content_type、主体和向量相似度筛选 chunks
  ↓
融合和重排序
  ↓
生成答案
```

教育资料中的“主体识别码”主要体现在 `item_name`、`code`、`source_name`、`content_type` 等元数据中。它不是让用户手动输入的密码，而是帮助系统区分不同课程、题库和资料来源的标识。

---

## 四、查询侧修改：为什么现在能处理不同类型的问题

相关文件主要在：

- `atguigu/query_process/state.py`
- `atguigu/query_process/main_graph.py`
- `atguigu/query_process/nodes/`
- `atguigu/config/prompt.py`
- `atguigu/web/api/query_service.py`

### 1. 增加问题类型识别

系统现在会先把问题分成几类：

- `chitchat`：你好、谢谢、你是谁等闲聊。
- `course`：询问课程介绍、课程内容、学习路线。
- `question`：寻找题目、答案和解析。
- `doc`：查找某个项目或资料中的具体内容。
- `knowledge`：普通知识库问答。

以前所有问题都直接走知识库检索，闲聊没有检索目标时就容易出现空结果或异常。现在闲聊会直接走聊天分支，不再浪费向量检索和重排序时间。

### 2. 主体识别更宽松、更稳

`node_item_name_confirm.py` 做了几项修复：

- 允许问题没有明确主体，不强迫模型编造一个主体。
- 课程和题目使用软匹配，相似度达到合理阈值就可以继续，不会因为名字稍有差异而阻断。
- 增加健壮 JSON 解析，可以处理模型返回的 Markdown 代码围栏和多余文字。
- 主体识别失败时降级成友好回复，不让“你好”这类问题把整条流程打崩。
- 修复历史回填，只回填当前消息，不会把最近十条历史错误地重复绑定到当前问题。
- 增加元对话判断，例如“我刚才问了什么”“你还记得上一轮吗”，这类问题直接使用会话历史回答，不再被设备手册等检索结果干扰。

### 3. 教育问题使用专门过滤

教育检索会根据问题类型增加条件：

- 课程问题优先查 `content_type == course_intro`。
- 题目问题优先查 `content_type == question`。
- 返回 `source_name`、`code`、`q_type`、`content_type` 等字段，答案可以知道资料来源和题目类型。
- 课程和题目不强行走 HyDE，因为教育资料往往有明确的结构化主体，直接检索更稳定。

这也解释了为什么旧教育集合如果没有对应字段，会报：

```text
field course_name not exist
```

这不是查询语句本身突然坏了，而是代码已经按新 schema 查询，Milvus 里仍然是旧结构。解决办法是换新的 chunks 集合名称，或删除旧集合后重新导入教育资料。

### 4. 多路检索和 RRF 融合

现在的检索不是只看一个向量结果，而是尽量组合多路结果：

- 普通向量检索。
- HyDE 生成的假设性查询检索，默认可以关闭。
- 可选的 web_search 检索。
- 后续重排序。

原来的 RRF 公式把余弦相似度和排名分数直接混在一起，量纲不一致，导致某一路分数压过其他路。现在改为标准的排名融合公式，并保留原始 `sim_score`，这样“多路投票”真正发挥作用。

空结果也可以正常处理：某一路没有结果时不会让整个 RRF 或 rerank 报错。

### 5. 查询速度优化

默认配置：

```text
ENABLE_WEB_SEARCH=false
ENABLE_HYDE=true
```

含义是：

- 默认不访问网络搜索，避免每次问答都等待 MCP 建连和网络重试。
- 默认保留 HyDE，帮助普通知识问题扩展查询表达。
- HyDE 的模型客户端使用缓存，避免每轮重新初始化。
- 重排序只处理融合后的前 20 条候选，不把大量长文本全部发送到云端。
- 重排序失败时保留原有召回顺序，答案仍然可以继续生成。

如果要查实时新闻、价格或外部资料，可以在 `.env` 中打开 web_search，然后重启服务。日常查项目资料时建议保持关闭。

<!-- ==================== AI修改 开始 ==================== -->
### 6. 如何实测 Graph RAG 有没有带来收益

新增离线脚本：`atguigu/tool/eval_graph_ab.py`。它不改变线上查询链路，而是对同一批问题做两条对照：

```text
baseline：向量 + 可选 HyDE → RRF → Rerank
graph_on：向量 + 可选 HyDE + Graph → RRF → Rerank
```

向量和 HyDE 每个问题只执行一次，再把同一份结果分别送入两次 RRF；这样 Graph 开关前后的差异可以归因于 Graph，而不是两次独立检索的波动。Web Search 不参加这份评测，两条分支的条件保持一致。

先从 `掌柜智库` 目录运行自动 smoke 评测：

```powershell
& '..\.venv\Scripts\python.exe' -m atguigu.tool.eval_graph_ab `
  --sample 20 `
  --out graph_ab_smoke_report.json
```

也可以使用样例评测集：

```powershell
& '..\.venv\Scripts\python.exe' -m atguigu.tool.eval_graph_ab `
  --file tests\graph_eval_fixture.json `
  --out graph_ab_report.json
```

样例里的 `expect_item_names` 只能验证主体名是否被召回，报告会把它标记为 `smoke`，不能据此证明召回率提升。正式评测时，把问题对应的真实值写进 JSON：

```json
{
  "query": "FastAPI 和 Pydantic 分别负责什么？",
  "expect_chunk_ids": ["Milvus 中真实的 chunk_id"]
}
```

或者用更粗的文档级标准：

```json
{
  "query": "FastAPI 和 Pydantic 分别负责什么？",
  "expect_document_ids": ["Milvus 中真实的 document_id"]
}
```

重点看报告中的这些字段：

- `graph_final_recall_gain`：Graph 开启后最终 Recall@K 的变化，正数才代表收益。
- `graph_unique_hit_queries`：Graph 单路找到了、但 baseline 最终结果没有命中的问题，说明 Graph 具备补充召回能力。
- `graph_final_gain_queries`：Graph 最终结果真正新增命中的问题。
- `regression_queries`：开启 Graph 后反而丢失 baseline 命中的问题，需要重点检查 RRF 权重或 Rerank。

如果想排除 Rerank 的影响，只比较 RRF 候选：

```powershell
& '..\.venv\Scripts\python.exe' -m atguigu.tool.eval_graph_ab `
  --file tests\graph_eval_fixture.json `
  --skip-rerank `
  --out graph_ab_rrf_only.json
```

如果想做纯向量与向量加 Graph 的对照，可以关闭 HyDE：

```powershell
& '..\.venv\Scripts\python.exe' -m atguigu.tool.eval_graph_ab `
  --file tests\graph_eval_fixture.json `
  --disable-hyde `
  --out graph_ab_no_hyde.json
```

这份脚本只新增离线测试能力，没有新增线上 `.env` 开关，也没有改变生产查询是否调用 Graph。
<!-- ==================== AI修改 结束 ==================== -->

---

## 五、导入侧修改：资料是怎样进入知识库的

相关文件主要在：

- `atguigu/import_process/state.py`
- `atguigu/import_process/main_graph.py`
- `atguigu/import_process/nodes/`
- `atguigu/web/api/app_service.py`

### 1. 支持 Markdown、PDF、DOCX

入口节点现在会根据后缀分流：

- Markdown：直接进入图片处理和切分。
- PDF：走原有 PDF 解析流程。
- DOCX：先通过 `python-docx` 转成 Markdown，再复用后面的统一流程。

DOCX 转换规则包括：

- Word 标题转换为 Markdown 标题。
- 普通段落转换为文本。
- 表格转换为 Markdown 管道表格。
- 转换后的文件会生成 `_converted.md`，便于检查和复用。

如果 DOCX 导入失败，先确认依赖已经安装：

```text
pip install python-docx
```

### 2. 普通资料的树形切分

`node_document_split.py` 已经从简单按长度切分改成树形切分：

```text
第一章 基础
  └─ 1.2 安装
       └─ 1.2.1 配置环境
```

切出来的内容会携带类似这样的面包屑：

```text
第一章 基础 > 1.2 安装 > 1.2.1 配置环境
```

这样模型检索到一个小片段时，仍然知道它属于哪一章。

同时做了这些处理：

- 合并同一父标题下过短的碎片。
- 增加 50 字重叠，避免切断处丢上下文。
- 代码块内部的 `#` 不会被误判成 Markdown 标题。
- 表格可以在行边界切分，不轻易把一个表格单元格从中间截断。
- 增加 1500 字的整块上限，防止文档源码中出现 `<table` 字符串时误判为整块表格。

最后一项很重要：某个文档把完整 Python 源码嵌在正文里，里面恰好有 `<table` 字面量，旧代码会把一万多字当成一个块送进 BGE-M3，容易显存溢出甚至导致 Windows 原生崩溃。现在会强制继续切分。

### 3. 向量内容更完整

BGE-M3 编码时不再只编码正文，而是组合：

```text
文档名 + 主体名称 + 当前 chunk 内容
```

这能让“某个项目里的安装方法”和“另一个项目里的安装方法”区分得更清楚。

### 4. Milvus 加载保护

Milvus 的 collection 必须先加载到内存才能搜索或删除。现在增加了 `ensure_collection_loaded()`：

- 查询前自动检查并加载集合。
- 同一进程已经确认加载过的集合使用缓存，减少重复请求。
- 兼容不同 pymilvus 版本返回的状态格式。
- 多线程导入时避免多个任务同时争抢同一张表的加载操作。

因此，遇到 `collection not loaded` 的概率降低了。但如果是旧 schema，比如缺少 `course_name` 或教育字段，仍然需要重建/重导，加载保护不能改变已有表结构。

---

## 六、教育资料功能：课程和题库为什么要单独处理

教育资料不是普通长文档。普通文档可以按标题和长度切，但题库必须保证题干、选项、答案、解析在一起，否则用户问一道题时可能只召回到题干，没有答案。

相关文件：

- `atguigu/edu_process/edu_parsers.py`
- `atguigu/edu_process/edu_import.py`
- `atguigu/edu_process/__init__.py`
- `atguigu/web/api/education_service.py`

### 1. 课程解析

课程资料按课程系列整理，一个课程系列形成一个完整知识块，里面保留系列名称、模块和课程信息。

### 2. 题目解析

题目资料按“一道题一个 chunk”整理，以下内容不会被拆开：

- 题干。
- 选项。
- 标准答案。
- 题目解析。

这样用户问“这道题答案是什么”时，模型拿到的是完整题目对象，而不是零散段落。

### 3. 教育资料导入命令

如果使用命令行导入：

```text
python -m atguigu.edu_process.edu_import \
  --course "课程介绍.md路径" \
  --question "题目资料.md路径"
```

先做预检查可以加：

```text
--dry-run
```

教育前端导入的目标是让用户可以上传课程资料和题库资料，并显示解析出的目录、题目数量和导入状态。课程和题库不是“必须两份同时存在才能上传”的普通文件限制；如果页面当前提示两份，是教育导入流程为了完整建立教育库而设计的校验。普通项目资料走普通导入入口即可。

### 4. 教育集合变更后的处理

教育字段属于 Milvus schema。代码加了字段后，旧集合不会自动长出新字段。推荐做法：

1. 在 `.env` 中给教育 chunks 使用一个新的集合名。
2. 重新导入课程和题库。
3. 确认查询能查到新集合后，再考虑清理旧集合。

不要只重启服务期待旧 collection 自动升级，Milvus collection schema 通常不会这样变更。

---

## 七、图片链路：为什么现在可以从知识库回答里看到图

相关文件：

- `atguigu/import_process/nodes/node_md_img.py`
- `atguigu/import_process/nodes/node_document_split.py`
- `atguigu/tool/image_url_tool.py`
- `atguigu/web/api/app_service.py`
- `atguigu/web/page/index.html`
- `atguigu/web/page/import.html`

图片显示必须经过四步，少一步都可能只显示文字：

```text
1. 找到 Markdown 或 HTML img 引用
2. 把图片放入本次任务目录并上传 MinIO
3. 把原引用替换成可访问的 MinIO URL，并写回 chunk
4. 答案生成时提取图片 URL，前端渲染成图片
```

以前的问题是：

- 只识别 `![](image.png)`，不识别 `<img src="image.png">`。
- 前端只上传 Markdown，没有把它引用的图片一起上传。
- 切分节点重新从磁盘读取原始 Markdown，覆盖了图片替换后的内容。
- 多个文档共用一个 MinIO 图片目录，后导入文档会误删先导入文档的图片。

现在已经修复：

- 同时识别 Markdown 图片和 HTML `<img>`。
- 支持手动选择 Markdown+图片，也支持选择整个文件夹。
- 每个任务有独立目录，每个文档有独立图片子目录。
- 切分优先使用图片处理节点已经替换过的 `md_content`。
- 旧图片不会被其他文档的重导任务误删。
- 答案中的图片 URL 会被前端安全解析和展示。

### 图片导入的推荐方式

如果文档和图片本来就在同一个项目文件夹，推荐使用前端的“本地文件夹导入”：

1. 输入项目资料所在文件夹路径。
2. 点击扫描导入。
3. 系统递归查找 `.md`、`.pdf`、`.docx` 和图片。
4. 根据文档实际引用的图片自动配对。

也可以直接选择多个文件。系统会把某个 Markdown 实际引用的图片放到同一次上传中，不会把共享图片重复上传很多遍。

### 当前图片限制

目前自动识别的是 Markdown 图片和 HTML `<img>`。Obsidian 常见的 Wiki 图片写法：

```text
![[图片名称.png]]
```

还需要额外适配，不能假设它会像普通 Markdown 图片一样自动配对。后续接入 Obsidian 时，建议专门增加 WikiLink 图片解析、附件目录映射和重复文件名处理。

---

## 八、答案生成、上下文和记忆：为什么广泛问题不应该突然“断掉”

相关文件：

- `atguigu/query_process/nodes/node_answer_output.py`
- `atguigu/config/prompt.py`
- `atguigu/tool/mongo_client_tool.py`
- `atguigu/web/page/index.html`

### 1. 答案输出变长

答案模型的单次最大输出现在设置为 `16384` token，正常回答只请求一次；只有 provider 明确因为输出上限硬截断时，才最多自动续写一次。

提示词明确要求：

- 详细、完整地回答。
- 分点或分步骤展开。
- 不省略关键细节、步骤、注意事项和适用场景。
- 题目类回答要保留题干、选项、答案和解析。

“最大输出”只是硬上限，不代表每次一定输出 16384 token。真正的答案长度由问题类型、提示词要求和模型自然结束共同决定。

如果用户确实需要更多内容，建议继续追问具体章节，而不是让一次回答无限扩展。这样可以减少重复解释，也能让答案更容易阅读。

### 2. 动态上下文预算

现在不是简单地固定截取 48000 字检索内容、24000 字历史，然后再叠加问题和提示词。代码会先估算这一轮总预算：

```text
模型上下文窗口
  - 最大答案输出空间
  - 系统提示词和安全预留
  = 本轮可用输入空间
```

然后再在检索内容和历史记忆之间分配空间：

- 优先保留排名靠前的检索结果。
- 优先保留最近的会话历史。
- 问题本身不会被截掉。
- 内容超出预算时先压缩/截断输入，不让超长 prompt 直接撞上接口限制。

如果模型流式返回的结束原因是 `length`、`max_tokens` 或 `token_limit`，前端会在已生成内容后提示：

```text
回答已达到本轮输出上限，以上为已完成内容。
```

这比突然停住更容易判断是“达到输出上限”，还是服务真正报错。

### 3. 上下文进度条现在和记忆挂钩

前端显示的不是一个没有意义的总进度，而是当前 prompt 的组成：

- 紫色：会话记忆。
- 蓝色：检索到的知识。
- 灰色：当前问题。
- 长期摘要激活时会显示“已压缩”。

它主要用于观察，不是模型真实剩余 token 的精确计量。中文 token 受模型分词方式影响，界面显示的是保守估算。

### 4. 短期记忆和长期记忆

MongoDB 中增加了会话摘要集合 `chat_summary`：

- 最近几轮保留原文，方便理解“它”“这个”“上一题”等指代。
- 更早的对话压缩成滚动摘要，避免历史无限增长。
- 摘要压缩在后台执行，不阻塞当前答案流式输出。
- 压缩过程加锁，避免快速连续提问时多个线程同时删除历史。
- 历史写入加固，即使 LLM 流式失败，也会写入兜底 assistant 记录，避免记忆链断裂。
- 对近期历史采用“最近 10 条原文全保留”的策略，减少过度过滤导致的断片。

这就是为什么现在会话记忆不是简单的“取几条 Mongo 记录”，而是“长期摘要 + 最近原文”的组合。

### 5. 仍然需要知道的边界

记忆不是永久、无限的数据库问答。它主要用于理解当前会话上下文：

- 长期摘要是压缩后的内容，不能保证保留每个原句。
- 跨会话不会自动共享全部历史。
- 如果旧会话是在历史写入修复前产生的，可能已经缺少 assistant 记录，建议新建会话测试。
- 更换模型、修改上下文配置后必须重启后端。

---

## 九、会话窗口和前端界面修改

主要文件：

- `atguigu/web/page/index.html`
- `atguigu/web/page/chat.html`
- `atguigu/web/page/import.html`

### 1. 前后端合并

新增 `atguigu/web/api/app_service.py`，把原来的查询服务和导入服务合并到一个 FastAPI 应用：

- 统一使用 8000 端口。
- 根路径直接打开前端页面。
- 查询、导入、状态、历史、图片接口走同一个服务。
- 原来的独立服务文件仍可保留，但不要和合并服务同时启动，否则会端口冲突或前端连错服务。

启动方式：

```text
cd /d D:\PythonProject\Knowledge_Database\掌柜智库
set PYTHONDONTWRITEBYTECODE=1
python -m atguigu.web.api.app_service
```

Windows PowerShell 使用：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m atguigu.web.api.app_service
```

服务入口内部也会关闭字节码落盘；启动命令设置环境变量是为了覆盖包入口最早期的导入阶段，确保 `.pyc` 和 `__pycache__` 不写入项目目录。

浏览器访问：

```text
http://127.0.0.1:8000/
```

### 2. 多会话

现在会话由服务端保存，前端左侧列表切换不同 `session_id`。点击加号应该创建新的会话，不应该覆盖旧会话。

第一轮发送问题后，系统会把第一条问题设置为会话标题，所以不再一直显示“新会话”。标题只在仍是默认名称时自动设置，已经手动命名的会话不会被覆盖。

如果页面仍然显示旧名称，优先检查：

1. 后端是否已经重启。
2. 浏览器是否加载了旧的 `index.html`，可以强制刷新。
3. 当前是否连到了另一份服务。
4. 浏览器开发者工具中 PATCH `/sessions/{session_id}` 是否返回成功。

### 3. 状态、时间线和耗时

查询状态现在位于助手名称下方，不会再把助手名称顶到时间线中间。状态会显示：

- 当前正在执行的步骤。
- 已完成步骤和耗时。
- 总耗时。
- 展开后的时间线。
- 失败时的真实错误摘要。

导入状态也增加了未知任务和长时间无响应提示，不会永远停在“主体识别”而没有说明。

### 4. Markdown 和安全渲染

答案支持标题、列表、加粗、代码块、链接和图片。渲染前先做 HTML 转义，再转换允许的 Markdown，降低模型输出 HTML 造成 XSS 的风险。

### 5. 导入页面美化

导入页支持：

- 多文件选择。
- DOCX 文件。
- 图片附件。
- 拖拽导入。
- 本地文件夹导入。
- 每个文件独立的状态卡片。
- 目录、图片数量和失败原因展示。

普通项目资料建议用文件夹导入；教育课程和题库建议使用教育资料入口或教育 CLI，以便写入正确的教育元数据。

---

## 十、并发导入、显存和“时好时坏”的问题

这部分是最近修复中很关键的一组。

### 1. 图片视觉模型限流

多个文档同时导入时，每个文档可能有几十张图片。如果每张图片都同时请求视觉模型，很容易出现 429。

现在采用三重保护：

- 文档任务可以并发，提高整体速度。
- 图片视觉调用使用全局信号量，最多两张图片同时请求。
- 429、TPM、Too Many 等限流错误使用退避重试。

这样不会为了限制图片请求而让所有文档完全排队。

### 2. BGE-M3 共享模型加锁

用户曾经出现两个文件同时导入时，程序退出：

```text
Process finished with exit code -1073741819 (0xC0000005)
```

这个错误属于 Windows 原生访问冲突，通常不会留下普通 Python 异常。排查发现多个导入线程同时初始化或调用同一个 BGE-M3/PyTorch 模型，存在底层线程安全风险。

现在：

- 模型初始化使用初始化锁，确保只初始化一次。
- 向量编码使用共享编码锁，避免多个线程同时操作同一个模型实例。
- 文档级任务仍然可以并发，只有共享 BGE-M3 调用串行化。

因此“同时传一个文件”和“同时传两个文件”都能成功并不矛盾。之前是并发竞态，触发时机和机器负载不同，所以表现为时好时坏；现在通过锁把这个不稳定因素去掉了。

### 3. 前端连接中断提示

如果后端真的发生原生崩溃，Python 无法返回 failed 状态。前端轮询连续失败一段时间后会显示“后端连接中断”，而不是让某个文档永远停留在旧节点。

这时应先看：

- 后端进程是否还在。
- 是否有 CUDA 显存不足。
- 是否是模型加载时的 native crash。
- 是否重复启动了多个后端进程。

---

## 十一、日志和导入状态修改

主要文件：

- `atguigu/tool/logger.py`
- `atguigu/tool/task_utils.py`
- `atguigu/import_process/base.py`
- `atguigu/web/api/app_service.py`
- `atguigu/web/page/index.html`

做了三类改进：

1. 把 `httpx`、`openai`、`urllib3`、`asyncio` 等第三方库的 DEBUG 噪音压低，避免真正的业务错误被大量请求日志淹没。
2. 工作流异常保存最后几行 traceback，并通过 `/status` 返回给前端。
3. 前端失败卡片自动展开日志，直接显示错误摘要。

另外，后台 Future 会被集合持有，任务完成后再移除，避免极端情况下任务刚提交就失去引用；状态查询使用运行中的事件循环，兼容性更好。

<!-- ==================== AI修改 开始 ====================
     限流等待状态必须由后端明确表达，不能由前端根据空字符串猜测。
     ==================== AI修改 结束 ==================== -->
### 11.1 导入限流等待状态

图片摘要调用触发模型限流时，后端任务不会失败，而是按下面的状态流转：

```text
queued -> processing -> waiting_response -> processing -> completed
                                      \-> waiting_response (再次限流)
                                      \-> failed (超过重试上限)
```

`/status/{task_id}` 在等待期间返回：

```json
{
  "status": "waiting_response",
  "status_detail": {
    "message": "触发模型限流，30秒后自动重试",
    "retry_count": 1,
    "retry_after": 30
  }
}
```

首页和独立导入页收到 `waiting_response` 后显示“等待响应中”，并展示重试次数与预计等待时间。退避结束、下一次模型请求开始前，后端会恢复 `processing`，前端随之恢复“处理中”。重试次数耗尽后才返回 `failed`，并沿用原有错误详情展示。

任务提交到线程池后会先写入 `queued`，避免线程池排队期间 `/status` 返回空字符串。空字符串不再作为正常的限流、排队或处理中状态；真正部署多进程时仍需使用共享任务状态存储，因为当前任务字典仍是单进程内存。

本次涉及：

- `atguigu/tool/task_utils.py`：增加 `queued`、`waiting_response` 和结构化 `status_detail`。
- `atguigu/import_process/nodes/node_md_img.py`：退避前写入等待状态，退避后恢复处理状态。
- `atguigu/web/api/app_service.py`：上传任务和本地批量导入任务登记 `queued`。
- `atguigu/web/page/index.html`、`atguigu/web/page/import.html`：展示等待、排队和恢复处理状态。
- `tests/test_task_status.py`：覆盖状态详情、两个页面展示和图片重试接入。

如果状态永远是“排队中”或“主体识别”，优先按这个顺序处理：

1. 杀掉旧后端进程。
2. 只启动一个合并服务。
3. 不要使用 `uvicorn --workers` 多进程模式。
4. 强制刷新浏览器页面。
5. 再导入一个小文件验证。

原因是任务状态目前保存在当前进程内存中，多进程之间不会共享这份字典。

---

## 十二、项目中保留的工具和评估能力

### 1. 召回率评估

`atguigu/tool/eval_recall.py` 与线上检索上限同步到 20，并支持 Recall@20 和动态 Hit 判断。这样评估结果和真实线上配置一致，不会出现“评估说很好，线上只查 10 条”的偏差。

### 2. 图片 URL 工具

`atguigu/tool/image_url_tool.py` 负责识别答案中的图片 URL，处理 MinIO 地址和可访问路径，供前端统一渲染。

### 3. 会话目录和 Mongo 工具

- `session_catalog.py`：管理会话目录和标题。
- `mongo_client_tool.py`：保存历史、长期摘要和压缩状态。
- `task_utils.py`：保存导入任务状态和错误信息。

### 4. 测试

测试文件主要在：

- `掌柜智库/tests/test_education_features.py`
- `掌柜智库/tests/test_image_url_tool.py`

最近一次验证结果：

```text
23 passed, 1 warning
```

另外已做 Python 编译检查、前端 JavaScript 语法检查和 diff 空白检查。这里的测试结果表示代码结构和关键逻辑通过，不代表外部模型账号、Milvus、MongoDB、MinIO 和网络服务在任何环境下都一定可用。

---

## 十三、日常使用推荐流程

### 第一次启动

1. 确认 Milvus、MongoDB、MinIO 服务已经启动。
2. 确认 `.env` 中模型地址、密钥、集合名称正确。
3. 进入项目目录启动合并后端。
4. 浏览器打开 `http://127.0.0.1:8000/`。

### 导入普通项目资料

推荐使用“本地文件夹导入”：

1. 把 Markdown、DOCX、PDF 和图片放在项目资料目录中。
2. 确认 Markdown 中图片引用路径真实存在。
3. 在前端输入目录路径并扫描导入。
4. 等待每个文件的状态变为完成。
5. 随便提问一个文件中的明确内容，确认能查到。
6. 再测试带图片的问题，例如“给我这个项目的流程图”。

### 导入教育资料

1. 使用课程资料和题库资料的专用入口，或执行教育 CLI。
2. 首次导入前确认教育 chunks 集合是新 schema。
3. 导入后检查课程数量、题目数量和主体名称数量。
4. 分别测试课程问题和题目问题。

### 查询测试顺序

建议不要一上来就问非常宽泛的问题，可以按这个顺序：

1. “这个项目是什么？”
2. “给我讲讲某个明确章节。”
3. “给我某个流程图。”
4. “总结这个项目的导入流程，并带上相关图片。”
5. “我刚才问了什么？”
6. 最后再问“详细且深入地讲讲 Java 语法”这类宽问题。

这样更容易判断是导入、检索、记忆还是输出预算的问题。

---

## 十四、遇到常见错误应该怎么看

### `collection not found`

说明查询使用的集合不存在。检查 `.env` 的集合名、Milvus 数据库和是否导入过对应资料。

### `field xxx not exist`

说明代码需要的新字段不在旧 collection schema 中。换新集合名并重新导入，不能只重启服务。

### `collection not loaded`

现在代码会自动加载。若仍出现，检查 Milvus 是否正常运行、集合是否真的存在，以及是否是多个进程同时操作导致的服务端异常。

### `429 free-models-per-day`

这是 OpenRouter 免费模型额度耗尽，不是重启能解决的问题。检查答案模型实际配置是否仍然指向 OpenRouter，而不是只看某个 embedding 或重排序配置。

### `401 Please bind your Alibaba Cloud account`

说明请求已经到了阿里云/魔搭相关接口，但账号未绑定或凭证不满足平台要求。检查对应模型的 API Key、账号绑定状态、Endpoint 和模型名。

### 导入停在“主体识别”

先看后端是否仍在运行。若日志显示两个任务同时进入 BGE-M3，检查是否运行的是包含 BGE-M3 锁的最新代码，并重启旧进程。

### 导入成功但图片不显示

重点检查：

1. 是否重新导入过图片修复后的资料。
2. Markdown/HTML 中的图片路径是否真实存在。
3. MinIO 是否能访问对应 URL。
4. chunk 中是否已经保存图片 URL。
5. 是否浏览器缓存了旧页面。

旧 chunks 中没有图片 URL，单纯重启不能补历史数据，需要重导。

### 回答突然结束或过长

先看回答末尾有没有“达到本轮输出上限”的提示：

- 有：是输出长度或模型限制，可以缩小问题范围，或调整 `.env` 后重启。
- 没有且页面一直转圈：检查 SSE、后端日志和模型接口。
- 只针对宽泛问题发生：优先检查动态上下文预算和检索结果数量。

现在答案不会因为本地字符计数器达到某个软上限而主动截断。若末尾出现“达到本轮输出上限”的提示，说明 provider 触发了硬输出限制；确认后端已重启，并检查 `LLM_MAX_TOKENS` 与模型平台实际允许的输出上限。

---

## 十五、重导入和升级时必须注意

以下修改会影响已经写入的向量数据，最好换新集合或清理后重新导入：

- Milvus schema 增加或修改字段。
- 文档切分规则变化。
- BGE-M3 编码文本格式变化。
- 图片 URL 回填逻辑变化。
- 教育解析器字段变化。

推荐采用版本化集合名，例如：

```text
chunks_v1 → chunks_v2 → chunks_v3
```

确认新集合查询正常后，再清理旧集合。这样比直接覆盖唯一集合更容易回滚。

下列修改通常只需要重启服务，不需要重导资料：

- 前端样式和状态显示。
- 会话标题逻辑。
- 查询路由逻辑。
- 上下文预算和历史注入逻辑。
- web_search、HyDE、rerank 开关。

---

## 十六、所有 AI 修改代码的标记规则

项目中的代码修改统一使用以下标记包裹：

```python
# ==================== AI修改 开始 ====================
# 这里说明这段代码为什么存在，以及它解决了什么问题。
...
# ==================== AI修改 结束 ====================
```

HTML/JavaScript 中使用对应的 HTML 注释：

```html
<!-- ==================== AI修改 开始 ==================== -->
<!-- 这里说明界面逻辑或样式的作用。 -->
...
<!-- ==================== AI修改 结束 ==================== -->
```

这些标记的作用是让你以后能快速找到 AI 改过的地方。项目原有代码和 AI 新增代码可能交错，所以不要只凭文件名判断，优先搜索“AI修改 开始”。

已有的详细技术记录仍然保留：

- [AI修改说明](AI修改说明.md)：逐项记录改动文件和技术原因。
- [AI改进报告](AI改进报告.md)：改进过程和问题分析。
- [AI召回率优化报告](AI召回率优化报告.md)：切分、召回和 RRF 优化细节。
- [AI前后端合并与界面重构报告](AI前后端合并与界面重构报告.md)：服务合并和页面重构细节。

本文件是总览和使用手册，其他文件是更详细的技术档案。

---

## 十七、后续建议：Obsidian 接入应该怎么做

Obsidian 建议单独做“来源适配层”，但不再单独创建一张向量表。所有正文切片统一写入 `knowledge_chunks`，通过 `source_type=obsidian`、`source_id=Vault标识` 和稳定的 `source_path` 区分来源。

推荐架构：

```text
Obsidian Vault
  ↓
扫描 Inbox / 指定目录
  ↓
解析 frontmatter、标签和文件路径
  ↓
按笔记标题和小标题切分
  ↓
写入统一 knowledge_chunks
```

这样做的好处：

- 不会随着 Vault 和知识来源增加而不断创建 Collection。
- 可以按 `source_type`、`source_id` 和 `source_path` 过滤。
- 后续新增网页、Git、数据库等来源时，核心查询和删除逻辑不需要重写。
- Obsidian 的 `[[笔记链接]]` 第一阶段保留在正文中，不自动扩展到答案上下文，保证回答优先使用直接证据。
- 图片引用、增量同步和链接关系属于后续可选能力，不作为精确问答的必需链路。

---

## 十八、统一知识切片表与批量删除

当前正文知识统一使用一张 Milvus Collection：

```text
knowledge_chunks
```

主体识别仍使用独立的 `item_collection`，因为它和正文向量检索是两种不同用途。普通资料、教育资料和未来 Obsidian 资料都写入 `knowledge_chunks`。

每个 chunk 必须携带：

```text
document_id
source_type
source_id
source_path
file_hash
```

同一篇原始文档的所有 chunk 共用同一个 `document_id`。更新或删除文档时，按 `document_id` 一次删除全部切片，不再按 `file_title` 查找。

当前实现提供三类批量删除接口：

```text
DELETE /knowledge/documents/{document_id}
DELETE /knowledge/sources/{source_type}?source_id=...
DELETE /knowledge/directories
```

本次重构不保留旧普通/教育 Collection 的运行时兼容逻辑。由于原始资料可以重新导入，首次切换时直接删除旧 chunks Collection，使用新 schema 重新导入即可，不需要一次性迁移脚本。删除的是 Milvus 中的向量数据，不是原始资料文件。

本轮实际改动：

- `atguigu/tool/knowledge_chunk_store.py`：新增统一 schema、按 `document_id` 替换文档和按来源/目录批量删除。
- `atguigu/import_process/nodes/node_import_milvus.py`：从约 300 行的兼容型节点收缩为薄适配层。
- `atguigu/edu_process/edu_import.py`：教育导入直接复用统一知识存储模块，不再切换教育专用 chunks 表。
- `atguigu/web/api/app_service.py`：增加文档、来源和目录删除接口。
- `tests/test_knowledge_chunk_store.py`、`tests/test_import_metadata.py`：覆盖统一字段、稳定文档 ID、替换写入和批量删除条件。

验证结果：本轮相关测试 `17 passed`；全量测试 `39 passed, 1 failed`。唯一失败是既有的 `get_answer_scope_instruction` 未在答案输出节点提供，位于 `tests/test_education_features.py`，与本次统一知识表重构无关。编译全包时还发现既有的 `atguigu/query_process/nodes2/node_rerank.py` 存在缩进错误，本轮未修改该历史文件。

---

## 十九、重复导入的稳定文档身份

<!-- ==================== AI修改 开始 ==================== -->
之前的 `replace_document()` 已经会按 `document_id` 删除整篇旧文档，但导入接口把带随机任务 ID 的临时路径传给了入口节点。例如同一个文件两次导入时，路径可能分别是：

```text
output/2026-08-22/task-a/Python.md
output/2026-08-22/task-b/Python.md
```

临时目录每次不同，导致 `document_id` 每次不同，旧 chunk 就不会被删除，最终表现为同一文件重复累积。

本次修复把临时路径和文档身份分开：

- 本地目录导入：`source_id = 原始目录的规范绝对路径`，`source_path = 文件相对原始目录的 POSIX 路径`。
- 浏览器上传：`source_id = browser-upload`，`source_path = 上传文件名`，并且保存文件前只保留文件名，避免路径片段进入保存位置。
- `run_graph()` 将这些来源字段传入导入图，`NodeEntry` 再据此生成稳定的 `document_id`。
- 同一来源目录下同一相对路径的文件再次导入，会命中同一个 `document_id`，Milvus 会先删除该文档的全部旧切片，再插入新切片。
- 不同目录中同名文件仍然是不同文档；浏览器上传时同名文件按同一文档替换，这是无原始目录场景下最稳定、最容易理解的规则。
- `file_hash` 仍保留为后续增量同步的元数据，但本次不把内容哈希加入 `document_id`，否则文件更新会产生旧版本孤儿数据。

本轮修改文件：

- `atguigu/tool/knowledge_chunk_store.py`：新增本地目录和浏览器上传的稳定来源身份构造函数。
- `atguigu/web/api/app_service.py`：普通上传和本地目录导入都把稳定来源元数据传入 `run_graph()`。
- `tests/test_import_metadata.py`：覆盖任务目录变化、文件名规范化和图状态元数据传递。

注意：这次代码修复不会自动清理已经存在的历史重复 chunk。完成代码更新后，如果要清理旧数据，应先按来源或目录调用批量删除接口，再重新导入：

```text
DELETE /knowledge/sources/document?source_id=local-upload
DELETE /knowledge/directories
```

第二个接口需要在请求体中传入 `source_path`，并可选传入 `source_type`。如果旧数据来自此前的随机临时路径，最稳妥的方式仍是按你确认过的旧集合/来源范围整体删除后重新导入，避免误删其他来源。

<!-- ==================== AI修改 结束 ==================== -->

## 二十、视觉模型配置字段兼容

<!-- ==================== AI修改 开始 ==================== -->
视觉模型节点使用的是 `ModelConfig.VL_BASE_URL`，而配置类此前只暴露了 `VL_MODEL_BASE_URL`，因此图片导入时会出现：

```text
AttributeError: type object 'ModelConfig' has no attribute 'VL_BASE_URL'
```

本次在 `ModelConfig` 中补充了标准字段 `VL_BASE_URL`，并让旧字段 `VL_MODEL_BASE_URL` 指向同一个值。这样当前视觉节点、旧节点和已有环境变量都可以继续工作，不需要修改 `.env` 中已有的地址配置。

修改后需要重启后端进程，因为 `ModelConfig` 在模块导入时读取一次环境配置。已增加回归测试，确保 `VL_BASE_URL` 和 `VL_MODEL_BASE_URL` 始终一致。

<!-- ==================== AI修改 结束 ==================== -->

## 二十一、细节级代码瘦身

<!-- ==================== AI修改 开始 ==================== -->
本轮按“保留节点职责，只减少重复代码”的原则整理正式代码，没有拆分图片清洗、答案输出或导入节点，也没有修改原有业务注释。

修改内容：

- atguigu/config/prompt.py：删除前面已经被后一个定义覆盖的重复 ANSWER_PROMPT，只保留最终版本。
- atguigu/config/config.py：新增语义统一的 MODEL_TEMPERATURE，内部文本和视觉调用都读取它；旧 .env 中的 VL_MODEL_TEMPERATURE 仍作为兜底，不要求用户立即改配置。
- atguigu/query_process/nodes/node_search_embedding.py、atguigu/web/api/education_service.py：正文查询统一使用 milvus_chunks_collection，去掉运行时按教育/普通资料选择不同 chunks 集合的重复分支。旧配置别名仍保留给未迁移旧节点，不创建第二张表。
- atguigu/tool/milvus_client_create.py：集中定义正文输出字段，并新增 build_item_name_expr()；检索节点和 HyDE 共用同一过滤逻辑，兼容字符串或列表主体，避免重复实现和逐字符过滤问题。
- atguigu/import_process/nodes/node_md_img.py：OpenRouter 的 free-models-per-day、openrouter_free_tier_daily 和 daily reset 错误不再进入指数退避；每日免费额度耗尽无法靠等待恢复，瞬时 RPM/TPM 限流仍正常退避。
- tests/test_model_provider_switch.py、tests/test_knowledge_chunk_store.py：补充每日额度和主体过滤边界测试。

本轮没有操作 Milvus 数据，也没有把测试缓存或临时文件写回项目目录。
<!-- ==================== AI修改 结束 ==================== -->

## 二十二、一句话总结

现在的掌柜智库已经从“能检索文本的 Demo”扩展成了一个包含普通资料、教育题库、图片知识、会话记忆、流式回答、导入状态和多路检索的完整项目。

最需要记住的四件事是：

1. **改了 Milvus schema 或切分规则，要重新导入。**
2. **图片修复后，旧资料必须重新导入，旧 chunk 不会自动补图片。**
3. **后端用单进程合并服务启动，不要同时启动旧服务或 `--workers` 多进程。**
4. **正文资料统一进入一张知识切片表，新增来源优先增加适配器和 `source_type`，不要新增 Collection。**
5. **限流时前端显示“等待响应中”，退避结束恢复“处理中”，不要把正常等待当成状态丢失。**
6. **代码中的 AI 修改区域都有标记，遇到问题先看对应模块的日志和状态卡片。**
