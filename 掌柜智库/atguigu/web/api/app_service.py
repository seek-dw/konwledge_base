# ==================== AI修改 开始 ====================
# 合并版后端服务: 将原 query_service.py(对话,8001) 与 import_service.py(导入,8000)
# 两个独立 FastAPI 应用合并为一个应用, 统一跑在 8000 端口。
#
# 合并要点:
# 1. 两个服务的路由没有任何冲突, 可以直接挂在同一个 app 上:
#    - 对话侧: /health, /history/{sid}, DELETE /history/{sid}, /query, /stream/{tid}
#    - 导入侧: /upload, /status/{tid}
# 2. 两个 main_graph 里都叫 MainGraphRunner, 直接 import 会互相覆盖,
#    所以分别起别名 ImportGraphRunner / QueryGraphRunner。
# 3. 顺便把前端页面目录(page/)挂载到本服务上:
#    - 访问 http://127.0.0.1:8000/       → 直接打开 index.html
#    - 访问 http://127.0.0.1:8000/page/  → 静态资源目录
#    以后启动一个进程, 前后端就都有了。
#
# 启动方式(二选一):
#    python -m atguigu.web.api.app_service
#    uvicorn atguigu.web.api.app_service:app --host 0.0.0.0 --port 8000
#
# 注意: 原来的 query_service.py / import_service.py 仍保留可独立运行,
#       但不要再和本服务同时启动(端口冲突/重复执行任务)。
# ==================== AI修改 结束 ====================
# ==================== AI修改 开始 ====================
# 合并服务运行时不在项目目录生成 Python 字节码缓存，避免运行一次就污染工作区。
# 启动命令仍建议同时设置 PYTHONDONTWRITEBYTECODE=1，以覆盖包入口最早期的导入阶段。
# ==================== AI修改 结束 ====================
import sys

sys.dont_write_bytecode = True

import asyncio
import json
import re
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, List

import uvicorn
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.params import Path as PathParam, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

# 两个 main_graph 同名类, 起别名避免互相覆盖
from atguigu.import_process.main_graph import MainGraphRunner as ImportGraphRunner
from atguigu.query_process.main_graph import MainGraphRunner as QueryGraphRunner
from atguigu.config.config import MinioConfig, ModelConfig, build_runtime_temp_path
from atguigu.tool.logger import logger
from atguigu.tool.minio_client_tool import create_minio_client
# ==================== AI修改 开始 ====================
# 导入任务通过 Mongo 记录跨 Milvus/Neo4j 的同步状态。
from atguigu.tool.mongo_client_tool import (
    clear_history,
    get_recent_history_list,
    mark_knowledge_sync,
)
# ==================== AI修改 结束 ====================
from atguigu.tool.session_catalog import (
    delete_session_catalog,
    list_session_catalog,
    rename_session_catalog,
    upsert_session_catalog,
)
from atguigu.web.api.education_service import (
    get_education_status,
    list_courses,
    list_questions,
    run_education_import,
)
from atguigu.web.api.knowledge_service import (
    delete_directory_chunks,
    delete_document_chunks,
    delete_source_chunks,
)
# ==================== AI修改 开始 ====================
from atguigu.tool.knowledge_chunk_store import (
    build_document_id,
    build_local_source_identity,
    build_upload_source_identity,
)
# ==================== AI修改 结束 ====================
from atguigu.tool.task_utils import (
    get_data, create_queue, put_data, get_task_info,
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
    # ==================== AI修改 开始 ====================
    # 任务进入线程池后先标记 queued，避免排队期间 /status 返回空字符串。
    TASK_STATUS_QUEUED,
    TASK_STATUS_CANCELLED,
    TaskCancelledError,
    cancel_task,
    is_task_cancelled,
    # ==================== AI修改 结束 ====================
    update_task_status, add_running_task, add_done_task,
    # ==================== AI修改 开始 ====================
    # 导入 set_task_error: 导入失败时把异常摘要存进 task_info,
    # /status 带着一起返回, 前端展示真实失败原因(不用翻控制台)
    # ==================== AI修改 结束 ====================
    set_task_error,
)

# ==================== AI修改 开始 ====================
# 浏览器可能把附件的相对路径作为 filename 提交；上传接口只需要文件名，
# 否则 images/xxx.png 会被保存成任务目录/images/images/xxx.png。
def _uploaded_basename(filename: str | None) -> str:
    return PurePosixPath(str(filename or "").replace("\\", "/")).name
# ==================== AI修改 结束 ====================

app = FastAPI(title="掌柜智库 · 一体化服务")


# ==================== AI修改 开始 ====================
# 统一知识表删除参数。删除操作按 document_id/source_type/source_path
# 批量执行，不要求用户逐个寻找 chunk。
class KnowledgeDirectoryDeleteParams(BaseModel):
    source_path: str = Field(..., min_length=1, description="来源目录前缀")
    source_type: str | None = Field(default=None, description="可选来源类型")
    source_id: str = Field(..., min_length=1, description="来源实例")


@app.delete("/knowledge/documents/{document_id}")
async def delete_knowledge_document(document_id: str):
    return delete_document_chunks(document_id)


@app.delete("/knowledge/sources/{source_type}")
async def delete_knowledge_source(source_type: str, source_id: str | None = None):
    if not str(source_id or "").strip():
        raise HTTPException(status_code=400, detail="删除来源必须提供 source_id")
    return delete_source_chunks(source_type, source_id)


@app.delete("/knowledge/directories")
async def delete_knowledge_directory(body: KnowledgeDirectoryDeleteParams):
    return delete_directory_chunks(body.source_path, body.source_type, body.source_id)
# ==================== AI修改 结束 ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== AI修改 开始 ====================
# 前端页面托管: 把 page/ 目录挂到 /page, 根路径直接返回集成页 index.html
_PAGE_DIR = Path(__file__).resolve().parent.parent / "page"
app.mount("/page", StaticFiles(directory=str(_PAGE_DIR)), name="page")


@app.get("/")
async def index():
    """根路径直接打开前端集成页, 免去手动找 html 文件"""
    return FileResponse(
        str(_PAGE_DIR / "index.html"),
        # 禁用缓存: 改前端样式/JS 后刷新即可看到, 不用硬刷新或清缓存
        headers={"Cache-Control": "no-store"}
    )
# ==================== AI修改 结束 ====================


# ============================================================
# 以下为原 query_service.py 的全部接口(对话侧)
# ============================================================

# api of 1
# 心脏跳动检测接口,也就是测试前后端是否已经成功连接的接口
# 在前端页面的显示是"API未连接" | "API已连接"
@app.get("/health")
async def check_heart():
    return {"message": "API已连接"}


# ==================== AI修改 开始 ====================
# 会话目录接口：历史消息仍使用原有 /history/{session_id}，
# 这里额外管理左侧“会话列表”的标题和最近使用时间。
class SessionTitleParams(BaseModel):
    title: Annotated[str, Field(default="", description="会话标题")]


@app.get("/sessions")
async def get_sessions():
    """返回左侧会话列表。Mongo 暂时不可用时返回空列表，不阻断聊天页加载。"""
    try:
        return {"items": list_session_catalog()}
    except Exception as exc:
        logger.warning(f"读取会话目录失败: {exc}")
        return {"items": []}


@app.post("/sessions")
async def create_session(body: Annotated[SessionTitleParams, Body(...)]):
    session_id = str(uuid.uuid4())
    try:
        record = upsert_session_catalog(session_id, body.title or "新会话")
    except Exception as exc:
        logger.warning(f"创建会话目录失败: {exc}")
        record = {"session_id": session_id, "title": body.title or "新会话", "updated_ts": 0}
    return record


@app.patch("/sessions/{session_id}")
async def rename_session(
        session_id: Annotated[str, PathParam(..., description="会话ID")],
        body: Annotated[SessionTitleParams, Body(...)]):
    return rename_session_catalog(session_id, body.title)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: Annotated[str, PathParam(..., description="会话ID")]):
    clear_history(session_id=session_id)
    delete_session_catalog(session_id)
    return {"message": "会话已删除"}
# ==================== AI修改 结束 ====================


# api of 2
# 前端页面显示历史记录的接口,前端请求携带了session_id,所以此处使用路径参数进行接收
@app.get("/history/{session_id}")
async def send_history(session_id: Annotated[str, PathParam(..., description="会话ID")]):
    # 直接通过定义好的方法拿出历史记录返回给前端
    # ==================== AI修改 开始 ====================
    # 刷新页面与查询节点统一读取同一个历史窗口，默认呈现最近20条消息。
    history_list = get_recent_history_list(
        session_id, limit=ModelConfig.QUERY_HISTORY_LIMIT
    )
    # ==================== AI修改 结束 ====================
    history_list = sorted(history_list, key=lambda x: x["ts"])
    # 历史记录里面的id是一个对象,这里要把id转为字符串
    history_list = [{
        "_id": str(history["_id"]),
        "session_id": history.get("session_id"),
        "role": history.get("role"),
        "text": history.get("text"),
         "rewritten_query": history.get("rewritten_query"),
         "item_names": history.get("item_names"),
         # ==================== AI修改 开始 ====================
         # MongoDB 已保存 assistant 消息的 image_urls；历史接口也必须返回它，
         # 否则刷新页面后前端没有图片候选地址，只能显示文字答案。
         # 旧历史没有该字段时返回空列表，保证前端始终拿到稳定的数据类型。
         # ==================== AI修改 结束 ====================
         "image_urls": history.get("image_urls") or [],
         "ts": history.get("ts")
    }
        for history in history_list]
    # 因为先写的前端,后写的后端,所以这里返回的键一定要和前端拿值的键一样,"items"
    return {"items": history_list}


# api of 3
# 通过前端的请求可以看出前端的判断是只要请求的ok存在,就会执行前端的删除历史记录的函数
@app.delete("/history/{session_id}")
async def delete_history(session_id: Annotated[str, PathParam(..., description="会话ID")]):
    clear_history(session_id=session_id)
    return {"message": "历史记录删除成功"}


# api of 4
# 前端的输入框接口对接，也就是用户输入问题的接口
# 通过前端测试发现请求固定路径为query,携带参数为请求体参数 {query,session_id}
class BodyParams(BaseModel):
    query: Annotated[str, Field(..., description="用户输入的问题")]
    session_id: Annotated[str, Field(..., description="会话ID")]


def ask_question_search(task_id, query, session_id):
    create_queue(task_id)  # 调用创建队列方法,队列字典里面已经有队列了,无需变量接收
    try:
        init_state = {
            "task_id": task_id,
            "original_query": query,
            "session_id": session_id
        }
        if is_task_cancelled(task_id):
            raise TaskCancelledError()
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        put_data(task_id, event="progress", data=get_task_info(task_id))
        QueryGraphRunner.create_and_run(init_state)
        if is_task_cancelled(task_id):
            raise TaskCancelledError()
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        put_data(task_id, event="progress", data=get_task_info(task_id))
    except TaskCancelledError:
        update_task_status(task_id, TASK_STATUS_CANCELLED, {"message": "查询已停止"})
        put_data(task_id, event="progress", data=get_task_info(task_id))
        put_data(task_id, "final", {"answer": "本次查询已停止。", "cancelled": True})
    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        put_data(task_id, event="error", data=get_task_info(task_id))
        # ==================== AI修改 开始 ====================
        # 兜底推一条可读的错误话术，前端至少能显示出来，而不是干等到超时
        put_data(task_id, "final", {"answer": "抱歉，处理您的问题时出现了错误，请稍后重试。"})
        # ==================== AI修改 结束 ====================
        raise e


@app.post("/query")
async def ask_question(
        bgt: BackgroundTasks,
        body: Annotated[BodyParams, Body(..., description="请求体参数")]):
    # 创建任务id,一定要加str不然就是个对象,前端根本拿不到任何东西
    task_id = str(uuid.uuid4())
    # ==================== AI修改 开始 ====================
    # 每次提问都刷新会话更新时间；已有标题由目录工具自动保留。
    try:
        upsert_session_catalog(body.session_id, body.query, preserve_title=True)
    except Exception as exc:
        logger.warning(f"刷新会话目录失败: {exc}")
    # ==================== AI修改 结束 ====================
    # 添加后台任务
    bgt.add_task(ask_question_search, task_id, body.query, body.session_id)
    return {
        "task_id": task_id,
        "query": body.query,
        "session_id": body.session_id
    }


# ==================== AI修改 开始 ====================
@app.post("/query/{task_id}/cancel")
async def cancel_query(task_id: Annotated[str, PathParam(..., description="查询任务ID")]):
    """设置协作式取消标记；正在进行的单次外部请求会在返回后退出。"""
    create_queue(task_id)
    cancel_task(task_id)
    put_data(task_id, event="progress", data=get_task_info(task_id))
    return {"task_id": task_id, "status": TASK_STATUS_CANCELLED}
# ==================== AI修改 结束 ====================


# ==================== AI修改 开始 ====================
# 教育数据导入与结构化查询接口。
# 课程资料和题目资料分开上传，后台任务完成后前端可以直接拿到统计结果。
@app.post("/education/import")
async def education_import_api(
        bg_task: BackgroundTasks,
        course_file: Annotated[UploadFile | None, File(description="课程介绍.md")] = None,
        question_file: Annotated[UploadFile | None, File(description="题目资料.md")] = None):
    # ==================== AI修改 开始 ====================
    # 课程和题库允许单独上传；至少有一份有效Markdown即可启动任务。
    if course_file is None and question_file is None:
        raise HTTPException(status_code=400, detail="请至少选择课程资料或题目资料中的一份")
    for upload_file, label in ((course_file, "课程资料"), (question_file, "题目资料")):
        if upload_file is not None and Path(upload_file.filename or "").suffix.lower() != ".md":
            raise HTTPException(status_code=400, detail=f"{label}必须是 .md 文件")
    # ==================== AI修改 结束 ====================

    task_id = str(uuid.uuid4())
    task_dir = build_runtime_temp_path(
        task_id,
        datetime.now().strftime("%Y-%m-%d"),
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    # ==================== AI修改 开始 ====================
    # 只为实际选择的文件落盘，未选择的一类传None给后台导入器。
    course_path = None
    question_path = None
    if course_file is not None:
        course_path = task_dir / Path(course_file.filename or "课程介绍.md").name
        with course_path.open("wb") as stream:
            shutil.copyfileobj(course_file.file, stream, 1024 * 1024)
    if question_file is not None:
        question_path = task_dir / Path(question_file.filename or "题目资料.md").name
        question_file.file.seek(0)
        with question_path.open("wb") as stream:
            shutil.copyfileobj(question_file.file, stream, 1024 * 1024)
    # ==================== AI修改 结束 ====================

    add_running_task(task_id, "education_upload")
    add_done_task(task_id, "education_upload")
    bg_task.add_task(
        run_education_import,
        task_id,
        str(course_path) if course_path else None,
        str(question_path) if question_path else None,
    )
    sources = []
    for source_name, source_path in (
        ("课程介绍", course_path),
        ("题目资料", question_path),
    ):
        if source_path:
            sources.append({
                "source_type": "education",
                "source_id": source_name,
                "source_path": source_name,
                "source_name": Path(source_path).name,
                "document_id": build_document_id(
                    "education",
                    source_name,
                    source_name,
                ),
            })
    return {"task_id": task_id, "sources": sources}


@app.get("/education/status/{task_id}")
async def education_status(task_id: Annotated[str, PathParam(..., description="教育导入任务ID")]):
    return get_education_status(task_id)


@app.get("/education/courses")
async def education_courses(keyword: str = "", limit: int = 50):
    return {"items": list_courses(keyword=keyword, limit=max(1, min(limit, 100)))}


@app.get("/education/questions")
async def education_questions(keyword: str = "", question_type: str = "", limit: int = 50):
    return {
        "items": list_questions(
            keyword=keyword,
            question_type=question_type,
            limit=max(1, min(limit, 100)),
        )
    }
# ==================== AI修改 结束 ====================


# api of 5
# sse接口,通过向队列添加数据实现状态的流转存储,然后通过sse一个个利用队列的特性取出来推送给前端
@app.get("/stream/{task_id}")
async def stream(task_id: Annotated[str, PathParam(..., description="任务ID")]):
    # 定义生成器函数,从队列中取数据并且一个个推送
    def generator_func(task_id):
        # 这里不需要再次判断队列字典里面是否存在队列,因为在get_data里面已经封装了
        while True:
            item = get_data(task_id)
            # 推送事件和数据给前端
            yield f"event: {item.get('event', '')}\n"
            yield f"data: {json.dumps(item.get('data', ''), ensure_ascii=False)}\n\n"  # 这是个状态列表,要转json字符串

    return StreamingResponse(generator_func(task_id), media_type="text/event-stream")


# ============================================================
# 以下为原 import_service.py 的全部接口(导入侧)
# ============================================================

# 定义后台任务
def run_graph(
        task_id: str,
        local_file_path: str,
        local_dir: str,
        # ==================== AI修改 开始 ====================
        # 这些字段来自原始来源，不能使用随机任务目录代替。
        source_type: str = "document",
        source_id: str = "local-upload",
        source_path: str = "",
        # ==================== AI修改 结束 ====================
):
    # main_graph中途若失败,则抛出异常被base接收,再被这里的try接收,返回前端上传失败状态
    document_id = build_document_id(source_type, source_id, source_path)

    # ==================== AI修改 开始 ====================
    # Mongo 只记录导入状态，不参与 Milvus/Neo4j 事务；状态写失败不能覆盖真正的导入异常。
    def record_sync(status, error=""):
        try:
            mark_knowledge_sync(
                document_id=document_id,
                status=status,
                task_id=task_id,
                source_type=source_type,
                source_id=source_id,
                source_path=source_path,
                error=error,
            )
        except Exception as sync_error:
            logger.warning(f"知识库同步状态写入失败: document_id={document_id}, error={sync_error}")
    # ==================== AI修改 结束 ====================

    try:
        record_sync("processing")
        init_state = {
            "task_id": task_id,
            "local_file_path": local_file_path,
            "local_dir": local_dir,
            # 临时落盘路径只负责运行；文档身份必须由原始来源元数据决定。
            "source_type": source_type,
            "source_id": source_id,
            "source_path": source_path,
        }
        # 更新总状态,执行main_graph之前总状态为process
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        ImportGraphRunner.create_runner(init_state)
        record_sync("synced")
        # 更新总状态,执行main_graph之后总状态为complete
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        record_sync("failed", str(e)[:1000])
        update_task_status(task_id, TASK_STATUS_FAILED)
        # ==================== AI修改 开始 ====================
        # 原来只打 {e}(异常str), 不打完整调用栈, 排查困难。
        # 改为 exc_info=True 输出完整 traceback。
        # 同时把错误摘要存进 task_info, /status 带着返回前端,
        # 用户不用在满屏 httpx DEBUG 日志里翻找报错。
        # ==================== AI修改 结束 ====================
        logger.error(f"任务ID为：{task_id}的文件上传失败，错误信息为：{e}", exc_info=True)
        # ==================== AI修改 开始 ====================
        # 截取异常类型+消息作为前端可读的错误摘要(完整 traceback 太长不适合前端展示)
        import traceback as _tb
        _err_lines = _tb.format_exception(type(e), e, e.__traceback__)
        # 取最后5行(通常是最关键的异常类型+消息行), 去掉空行
        _tail = [l.rstrip() for l in _err_lines[-5:] if l.strip()]
        set_task_error(task_id, "\n".join(_tail) or str(e))
        # ==================== AI修改 结束 ====================


# ==================== AI修改 开始 ====================
# 原逻辑: /upload 只接收单个文件。上传 md 文档时, 它引用的 images/ 图片
#         并不会跟着上到服务器 —— node_md_img 在 md 同级目录找不到 images 文件夹,
#         直接秒过(前端显示 0s), chunk 里留下 ![](images/xxx.png) 的死链接。
# 改进: 支持一次上传"1个主文档 + N张附属图片":
#   - 主文档(.md/.pdf/.docx) 存到 output/日期/任务ID/ 目录
#   - 附属图片存到该目录下的 images/ 子目录(node_md_img 正是在这里找图片)
#   - 前端会解析 md 引用, 只给每个文档带上它实际引用的图片, 不重复传
# 兼容: 老的单文件上传(只传一个文件)行为完全不变
# ==================== AI修改 结束 ====================
@app.post("/upload")
async def upload_api(
        bg_task: BackgroundTasks,
        files: Annotated[List[UploadFile], File(..., description="主文档和附属图片")]):

    # ==================== AI修改 开始 ====================
    # 主文档 = 第一个后缀为 md/pdf/docx 的文件; 其余视为附属图片
    doc_suffixes = {".md", ".pdf", ".docx"}
    image_suffixes = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    main_doc = None
    image_files = []
    for f in files:
        suffix = Path(_uploaded_basename(f.filename)).suffix.lower()
        if main_doc is None and suffix in doc_suffixes:
            main_doc = f
        elif suffix in image_suffixes:
            image_files.append(f)
    if main_doc is None:
        raise Exception("请至少上传一个 .md / .pdf / .docx 主文档")
    # ==================== AI修改 开始 ====================
    # 文件可能来自浏览器提交的路径字符串，落盘前只保留文件名，避免
    # 临时任务目录或用户传入的路径片段进入文档身份和保存路径。
    main_doc_name = _uploaded_basename(main_doc.filename) or "upload.md"
    upload_source_id, upload_source_path = build_upload_source_identity(main_doc.filename)
    # ==================== AI修改 结束 ====================

    # 1.生成task_id
    task_id = str(uuid.uuid4())

    # 加上节点流转状态监控,这是上传文件的起始位置,添加进字典列表
    add_running_task(task_id, "upload_file")
    # ==================== AI修改 开始 ====================
    # 先登记明确的排队状态；真正进入 run_graph 后才切换为 processing。
    update_task_status(task_id, TASK_STATUS_QUEUED, {"message": "文件已上传，等待导入线程执行"})
    # ==================== AI修改 结束 ====================

    # 2.接收文件并保存到指定位置 输出目录下加上时间目录
    # ==================== AI修改 开始 ====================
    # 每个任务一个独立子目录(任务ID命名), 隔离不同上传之间的文件
    # (原逻辑所有上传都堆在同一个日期目录, 两个同名 md 会互相覆盖;
    #  图片也只有独立目录才能与主文档的 images/ 约定对上)
    # ==================== AI修改 结束 ====================
    local_dir = fr"E:\AI大模型\第七阶段 掌柜智库\资料\05-设备手册汇总\output\{datetime.now().strftime('%Y-%m-%d')}"
    task_dir_obj = Path(local_dir) / task_id
    task_dir_obj.mkdir(parents=True, exist_ok=True)
    local_dir = str(task_dir_obj)
    local_file_path = str(task_dir_obj / main_doc_name)

    # shutil.copyfileobj(file.file, f, 1024*1021) 按缓冲区写入,比一次性read更合适
    # 3.文件流写入指定路径
    with open(local_file_path, "wb") as f:
        shutil.copyfileobj(main_doc.file, f, 1024 * 1024)
    logger.info(f"文件上传成功，保存路径为：{local_file_path}")

    # ==================== AI修改 开始 ====================
    # 附属图片保存到主文档同级的 images/ 目录(node_md_img 约定的查找位置)
    if image_files:
        images_dir_obj = task_dir_obj / "images"
        images_dir_obj.mkdir(parents=True, exist_ok=True)
        for img in image_files:
            img_name = _uploaded_basename(img.filename)
            if not img_name:
                continue
            img_path = images_dir_obj / img_name
            with open(img_path, "wb") as f:
                shutil.copyfileobj(img.file, f, 1024 * 1024)
        logger.info(f"附属图片保存成功，共{len(image_files)}张，目录为：{images_dir_obj}")
    # ==================== AI修改 结束 ====================

    # 备份到minio
    minio_client = create_minio_client()
    minio_client.fput_object(
        bucket_name=MinioConfig.MINIO_BUCKET_NAME,
        object_name=f"back_up/{datetime.now().strftime('%Y-%m-%d')}/{task_id}/{main_doc_name}",
        file_path=local_file_path
    )
    logger.info(f"文件已备份到：{MinioConfig.MINIO_BUCKET_NAME}, object_name为：back_up/{datetime.now().strftime('%Y-%m-%d')}/{task_id}/{main_doc_name}")

    # 添加节点状态流转监控:这里文件上传结束,将其加入done字典列表,并从running字典列表中删除
    add_done_task(task_id, "upload_file")
    # 文件保存和备份成功,接下来开启后台任务,调用graph,进行存库一系列操作
    bg_task.add_task(
        run_graph,
        task_id,
        local_file_path,
        local_dir,
        "document",
        upload_source_id,
        upload_source_path,
    )

    return {
        "task_id": task_id,
        "source_type": "document",
        "source_id": upload_source_id,
        "source_path": upload_source_path,
        "document_id": build_document_id(
            "document",
            upload_source_id,
            upload_source_path,
        ),
    }


# 定义前端访问状态接口,所有信息都在里面,前端拿到包含所有信息的字典后进行页面展示
@app.get("/status/{task_id}")
async def get_status(task_id: Annotated[str, PathParam(..., description="任务ID")]):
    return get_task_info(task_id)  # 里面包含了所有的信息,根据前端轮询发送请求,可以动态获得里面不同的信息


# ==================== AI修改 开始 ====================
# 本地文件夹导入接口: 用户输入服务器本机的一个文件夹路径, 后端直接扫磁盘导入。
#
# 为什么需要它(和 /upload 的本质区别):
#   - md 是纯文本文件, 图片只是其中一行链接 ![](images/xxx.png),
#     真正的图片字节在用户本地磁盘上。浏览器有安全沙箱, 网页不允许自行读取
#     磁盘文件, 所以走"上传"这条路时图片必须由用户手动一起选中才传得上来。
#   - 而后端就跑在用户自己电脑上, 可以直接读磁盘 —— 给个文件夹路径,
#     "找文档 → 解析md引用 → 自动配图片 → 拷贝到任务目录 → 走导入管线"
#     全部由代码自动完成, 用户一个文件都不用选。
#   - (PDF 不需要这个接口也是这个原因: 图片内嵌在 pdf 二进制里,
#      MinerU 解析时会自动抽出来下载到 images/ 目录, 天然全自动)
#
# 行为:
#   1. 递归扫描目录: .md/.pdf/.docx 是主文档, 图片按"文件名"建全局索引
#   2. 每个 md 解析出它引用的图片名, 只拷贝自己引用的那几张到 任务目录/images/
#      (19个md共用一个images文件夹时不会重复拷贝, 每个任务只带自己的图)
#   3. 每个文档独立 task_id + 独立任务目录, 之后与 /upload 走完全相同的
#      run_graph 后台管线(minio备份/节点监控/幂等入库全部一致)
class LocalImportParams(BaseModel):
    dir: Annotated[str, Field(..., description="服务器本地文件夹路径")]


# ==================== AI修改 开始 ====================
# md 图片引用解析: 项目文档(掌柜智库19篇)的123张图全部用 HTML <img src="images/xxx.jpg">
# 标签引用, 而非 Markdown ![](xxx.png) 语法。原正则只认 Markdown 语法导致一张图都配对不上。
# 现在同时支持两种语法, 提取出的"引用文件名"取 basename(忽略 src 里的相对路径前缀),
# 与磁盘 images_by_name 索引(按完整文件名建索引)做匹配。
# 两边正则规则必须一致, 确保配对不丢图。
#   1) Markdown: ![](path)        → 取 ](...path) 里的最后一段文件名
#   2) HTML img: <img src="path"> → 取 src 里的完整路径再取 basename(文件名可能含空格/中文)
_MD_MD_RE   = re.compile(r"\]\([^)]*?([^/()\s]+\.(?:png|jpe?g|gif|bmp|webp))\)", re.IGNORECASE)
_MD_HTML_RE = re.compile(r'<img[^>]*?src\s*=\s*["\']([^"\']+\.(?:png|jpe?g|gif|bmp|webp))["\']', re.IGNORECASE)


def _extract_img_refs(text: str) -> set:
    """从 md 文本里提取所有被引用的图片文件名(basename), 支持 markdown 和 html img 两种语法。"""
    refs: set = set()
    for m in _MD_MD_RE.finditer(text):          # markdown 语法, group(1) 已是排除路径的文件名
        refs.add(Path(m.group(1)).name)
    for m in _MD_HTML_RE.finditer(text):        # html img 语法, group(1) 是完整 src 路径, 取 basename
        refs.add(Path(m.group(1)).name)
    return refs


def _safe_obsidian_upload_path(filename: str | None) -> tuple[str, PurePosixPath]:
    """Return the Vault root and a safe path relative to that root."""
    normalized = str(filename or "").replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = path.parts
    if (
        len(parts) < 2
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or ":" in parts[0]
    ):
        raise HTTPException(status_code=400, detail="Obsidian 上传文件路径无效")
    return parts[0], PurePosixPath(*parts[1:])


@app.post("/upload_obsidian")
async def upload_obsidian(
        bg_task: BackgroundTasks,
        files: Annotated[List[UploadFile], File(..., description="Obsidian Vault 文件")],
        dry_run: bool = False):
    """Receive a browser-selected Vault and import only its Markdown/images."""
    allowed_suffixes = {".md", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    vault_name = None
    accepted_files = []
    markdown_count = 0

    for upload in files:
        root_name, relative_path = _safe_obsidian_upload_path(upload.filename)
        if vault_name is None:
            vault_name = root_name
        elif vault_name != root_name:
            raise HTTPException(status_code=400, detail="一次只能上传一个 Obsidian Vault")
        if relative_path.suffix.lower() not in allowed_suffixes:
            continue
        accepted_files.append((upload, relative_path))
        if relative_path.suffix.lower() == ".md":
            markdown_count += 1

    if not vault_name or markdown_count == 0:
        raise HTTPException(status_code=400, detail="上传的 Vault 中没有找到 Markdown 笔记")

    source_id = f"browser-upload:{vault_name}"
    with tempfile.TemporaryDirectory(prefix="obsidian-upload-") as temp_dir:
        vault_root = Path(temp_dir) / vault_name
        for upload, relative_path in accepted_files:
            destination = vault_root / Path(relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = getattr(upload, "file", None)
            if stream is None:
                continue
            stream.seek(0)
            with destination.open("wb") as output:
                shutil.copyfileobj(stream, output, 1024 * 1024)
        return await _import_directory(
            bg_task,
            str(vault_root),
            "obsidian",
            dry_run,
            source_id=source_id,
        )


# 本地批量导入用线程池并发跑导入管线(而非 FastAPI BackgroundTasks 的串行机制),
# 与前端"批量上传=并发多个请求"的体验保持一致; 限制并发数避免 embedding/显存打爆。
# ==================== AI修改 结束 ====================
# ==================== AI修改 开始 ====================
# max_workers=6: 文档级并发放开(切分/向量化/入Milvus全并行, 这些不占TPM)。
# 视觉模型限流由node_md_img内部的全局信号量(Semaphore(2))负责,
# 只有图片摘要生成这一步限流, 其他步骤不受影响。
# + 429指数退避重试兜底, 三重保险防TPM爆。
# ==================== AI修改 结束 ====================
_import_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="import-graph")

# ==================== AI修改 开始 ====================
# 防御性持有 run_in_executor 返回的 Future 强引用, 防止任务被 GC 取消
# (一旦被取消, run_graph 永不执行 → /status 永远空 → 前端状态不更新)。
# done_callback 在任务完成后自动从集合移除, 集合不会无限增长。
# ==================== AI修改 结束 ====================
_pending_futures: set = set()


@app.post("/import_local")
async def import_local(
        bg_task: BackgroundTasks,
        body: Annotated[LocalImportParams, Body(..., description="本地文件夹路径")],
        dry_run: bool = False):
    return await _import_directory(bg_task, body.dir, "document", dry_run)


async def _import_directory(
        bg_task: BackgroundTasks,
        raw_dir: str,
        source_type: str,
        dry_run: bool = False,
        source_id: str | None = None):
    # dry_run=True: 只扫描+配对图片+落盘, 不做minio备份也不跑导入管线(验证路径用)
    # 去掉两端引号/空格(用户从资源管理器复制路径时常带引号)
    root = Path(raw_dir.strip().strip('"').strip("'"))
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在或不是文件夹: {root}")

    doc_suffixes = {".md"} if source_type == "obsidian" else {".md", ".pdf", ".docx"}
    image_suffixes = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    # 递归收集目录下所有文件, 分成主文档和图片索引(图片按文件名索引, md里引用的就是文件名)
    all_files = [p for p in root.rglob("*") if p.is_file()]
    docs = [p for p in all_files if p.suffix.lower() in doc_suffixes]
    images_by_name = {p.name: p for p in all_files if p.suffix.lower() in image_suffixes}
    if not docs:
        raise HTTPException(status_code=400, detail=f"该目录下没有找到 .md / .pdf / .docx 文档")

    # ==================== AI修改 开始 ====================
    # 记录原始来源目录。后续任务目录带随机 task_id，但它不能参与文档身份。
    source_root = root.resolve()
    # ==================== AI修改 结束 ====================

    output_root = Path(fr"E:\AI大模型\第七阶段 掌柜智库\资料\05-设备手册汇总\output\{datetime.now().strftime('%Y-%m-%d')}")
    date_str = datetime.now().strftime('%Y-%m-%d')
    minio_client = create_minio_client()
    tasks = []

    for doc in sorted(docs):
        task_id = str(uuid.uuid4())
        task_dir = output_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # 主文档拷贝到任务目录(与 /upload 的落盘结构完全一致)
        local_file_path = str(task_dir / doc.name)
        # ==================== AI修改 开始 ====================
        # 文档身份使用来源实例和目录内相对路径；task_dir 只用于本次运行。
        if source_id:
            source_path = doc.resolve().relative_to(source_root).as_posix()
        else:
            source_id, source_path = build_local_source_identity(source_root, doc)
        document_id = build_document_id(source_type, source_id, source_path)
        # ==================== AI修改 结束 ====================
        shutil.copy2(doc, local_file_path)

        # md: 解析引用的图片名, 从全目录索引里找到源文件, 拷到同级 images/
        # 支持 markdown ![](path) 和 html <img src="path"> 两种引用语法
        n_images = 0
        if doc.suffix.lower() == ".md":
            text = doc.read_text(encoding="utf-8", errors="ignore")
            refs = _extract_img_refs(text)
            if refs:
                images_dir = task_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                for name in refs:
                    src = images_by_name.get(name)
                    if src:
                        shutil.copy2(src, images_dir / name)
                        n_images += 1
        logger.info(f"本地导入收集完成: {doc.name}, 配对图片{n_images}张, 任务目录: {task_dir}")

        # dry_run 模式到此为止(不备份不跑管线), 正常模式以下与 /upload 完全一致
        if dry_run:
            tasks.append({
                "task_id": task_id,
                "filename": doc.name,
                "images": n_images,
                "source_type": source_type,
                "source_id": source_id,
                "source_path": source_path,
                "document_id": document_id,
            })
            continue

        # 以下与 /upload 完全一致: minio备份 + 节点监控 + 后台graph
        add_running_task(task_id, "upload_file")
        # ==================== AI修改 开始 ====================
        # 本地批量导入使用线程池，线程未空闲前任务处于 queued，不再返回空状态。
        update_task_status(task_id, TASK_STATUS_QUEUED, {"message": "已提交，等待导入线程执行"})
        # ==================== AI修改 结束 ====================
        minio_client.fput_object(
            bucket_name=MinioConfig.MINIO_BUCKET_NAME,
            object_name=f"back_up/{date_str}/{task_id}/{doc.name}",
            file_path=local_file_path
        )
        add_done_task(task_id, "upload_file")
        # ==================== AI修改 开始 ====================
        # 原用 bg_task.add_task(run_graph, ...): FastAPI BackgroundTasks 在单个请求内
        # 是"串行"执行的(完成一个才跑下一个), 19篇文档要排队, 与前端"批量上传=并发
        # 多个请求"的体验不一致。改用模块级线程池并发提交 run_graph(max_workers=6),
        # run_in_executor 返回的 Future 不 await, 立即往下走 → 19个任务并发跑,
        # 响应瞬时返回。线程池限流避免 embedding/显存被多任务同时打爆。
        # ==================== AI修改 开始 ====================
        # asyncio.get_running_loop(): 在 async 路由里规范获取当前事件循环
        # (get_event_loop 在协程内虽能工作但有弃用警告, 改用 running loop 更稳)。
        # 保留 Future 强引用到 _pending_futures, 防止极端 GC 场景取消任务。
        # ==================== AI修改 结束 ====================
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(
            _import_pool,
            run_graph,
            task_id,
            local_file_path,
            str(task_dir),
            source_type,
            source_id,
            source_path,
        )
        _pending_futures.add(fut)
        fut.add_done_callback(_pending_futures.discard)
        # ==================== AI修改 结束 ====================
        tasks.append({
            "task_id": task_id,
            "filename": doc.name,
            "images": n_images,
            "source_type": source_type,
            "source_id": source_id,
            "source_path": source_path,
            "document_id": document_id,
        })

    logger.info(f"本地目录导入提交: {root}, 共{len(tasks)}个文档")
    return {
        "tasks": tasks,
        "total": len(tasks),
        "source_type": source_type,
        "source_id": source_id or source_root.as_posix(),
    }
# ==================== AI修改 结束 ====================


if __name__ == '__main__':
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8000
    )
