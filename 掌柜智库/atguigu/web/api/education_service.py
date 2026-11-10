# ==================== AI修改 开始 ====================
# 教育知识库服务：
# 1. 把教育导入脚本封装成后台任务，供前端按钮调用；
# 2. 从 Milvus 读取课程/题目结构化字段，供前端列表和详情卡片展示；
# 3. 保留纯函数 summary，便于没有启动 Milvus 时做单元测试。
# ==================== AI修改 结束 ====================
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import Any

from atguigu.config.config import MilvusConfig
from atguigu.edu_process.edu_import import (
    import_chunks,
    register_item_names,
    vectorize_chunks,
)
from atguigu.edu_process.edu_parsers import (
    extract_item_names,
    parse_course_md,
    parse_question_md,
)
from atguigu.tool.logger import logger
from atguigu.tool.mongo_client_tool import get_education_metadata
from atguigu.tool.milvus_client_create import (
    ensure_collection_loaded,
    get_milvus_client,
    select_existing_output_fields,
)
from atguigu.tool.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_running_task,
    get_task_info,
    set_task_error,
    update_task_status,
)

_EDUCATION_RESULTS: dict[str, dict[str, Any]] = {}
_EDUCATION_RESULTS_LOCK = threading.Lock()

EDUCATION_OUTPUT_FIELDS = [
    "id", "chunk_id", "document_id", "source_type", "source_id",
    "source_path", "item_name", "content_type", "chunk_index",
    "section_path", "content",
]


# ==================== AI修改 开始 ====================
def collect_education_import_inputs(
    course_path: str | None,
    question_path: str | None,
) -> list[tuple[str, str]]:
    """收集本次实际选择的教育资料，允许课程和题库单独导入。"""
    inputs: list[tuple[str, str]] = []
    if course_path:
        inputs.append(("course", course_path))
    if question_path:
        inputs.append(("question", question_path))
    return inputs


# ==================== AI修改 结束 ====================


def build_course_summary(chunk: dict[str, Any]) -> dict[str, Any]:
    """把 Milvus 课程 chunk 转成前端稳定使用的课程对象。"""
    metadata = chunk.get("domain_metadata") or chunk
    return {
        "course_name": metadata.get("course_name") or chunk.get("title") or chunk.get("section_path") or "",
        "course_code": metadata.get("course_code") or metadata.get("code") or "",
        "chapter_name": metadata.get("chapter_name") or "",
        "course_category": metadata.get("course_category") or "",
        "target_users": metadata.get("target_users") or "",
        "learning_goals": metadata.get("learning_goals") or "",
        "project_name": metadata.get("project_name") or "",
        "source_name": chunk.get("source_path") or "",
        "source_path": chunk.get("source_path") or "",
        "content": chunk.get("content") or "",
    }


def build_question_summary(chunk: dict[str, Any]) -> dict[str, Any]:
    """把 Milvus 题目 chunk 转成前端题目卡片对象。"""
    metadata = chunk.get("domain_metadata") or chunk
    return {
        "question_bank_name": metadata.get("question_bank_name") or chunk.get("item_name") or "",
        "question_bank_code": metadata.get("question_bank_code") or "",
        "question_code": metadata.get("question_code") or metadata.get("code") or "",
        "question_type": metadata.get("question_type") or metadata.get("q_type") or "",
        "course_name": metadata.get("course_name") or "",
        "source_name": chunk.get("source_path") or "",
        "source_path": chunk.get("source_path") or "",
        "content": chunk.get("content") or "",
    }


def run_education_import(
    task_id: str,
    course_path: str | None = None,
    question_path: str | None = None,
) -> None:
    """后台执行课程和题目导入，失败只标记当前任务。"""
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        add_running_task(task_id, "education_parse")
        # ==================== AI修改 开始 ====================
        # 前端可以只选课程或只选题库；未选择的一类按空列表处理。
        course_chunks = parse_course_md(course_path) if course_path else []
        question_chunks = parse_question_md(question_path) if question_path else []
        # ==================== AI修改 结束 ====================
        add_done_task(task_id, "education_parse")

        if not course_chunks and not question_chunks:
            raise ValueError("课程资料和题目资料都没有解析出有效内容")

        add_running_task(task_id, "education_embedding")
        course_chunks = vectorize_chunks(course_chunks)
        question_chunks = vectorize_chunks(question_chunks)
        add_done_task(task_id, "education_embedding")

        add_running_task(task_id, "education_import")
        item_names = extract_item_names(course_chunks, question_chunks)
        # ==================== AI修改 开始 ====================
        # 先写教育chunks再注册主体名，避免主体库异常导致教育表始终没有创建。
        # 课程/题目检索本身不依赖item_name过滤，主体名注册失败只记警告不阻断主数据。
        if course_chunks:
            import_chunks(course_chunks, "课程介绍")
        if question_chunks:
            import_chunks(question_chunks, "题目资料")
        if item_names:
            try:
                register_item_names(item_names, "教育课程与题库")
            except Exception as exc:
                logger.warning(f"教育主体名注册失败，已保留课程/题目数据: {exc}")
        # ==================== AI修改 结束 ====================
        add_done_task(task_id, "education_import")

        result = {
            "course_count": len(course_chunks),
            "question_count": len(question_chunks),
            "item_count": len(item_names),
        }
        with _EDUCATION_RESULTS_LOCK:
            _EDUCATION_RESULTS[task_id] = result
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as exc:
        update_task_status(task_id, TASK_STATUS_FAILED)
        error_lines = [line.rstrip() for line in traceback.format_exception(exc) if line.strip()]
        set_task_error(task_id, "\n".join(error_lines[-5:]) or str(exc))
        logger.error(f"教育数据导入失败: task_id={task_id}, error={exc}", exc_info=True)


def get_education_status(task_id: str) -> dict[str, Any]:
    """返回教育任务状态和成功统计。"""
    result = get_task_info(task_id)
    with _EDUCATION_RESULTS_LOCK:
        result["result"] = _EDUCATION_RESULTS.get(task_id, {})
    return result


def _query_education_chunks(content_type: str, limit: int = 100) -> list[dict[str, Any]]:
    """从 Mongo 取教育元数据，再按 chunk_id 回溯统一 Milvus Chunk。"""
    try:
        client = get_milvus_client()
        collection = MilvusConfig.milvus_chunks_collection
        if not client.has_collection(collection_name=collection):
            return []
        ensure_collection_loaded(collection)
        metadata_rows = get_education_metadata(content_type, limit=500)
        if not metadata_rows:
            return []
        chunk_ids = [row["chunk_id"] for row in metadata_rows]
        chunks = client.query(
            collection_name=collection,
            filter=f"chunk_id in {json.dumps(chunk_ids, ensure_ascii=False)}",
            output_fields=select_existing_output_fields(
                client, collection, EDUCATION_OUTPUT_FIELDS
            ),
            limit=max(1, min(limit, 500)),
        )
        chunks_by_id = {chunk.get("chunk_id"): chunk for chunk in chunks}
        return [
            {**chunks_by_id[row["chunk_id"]], "domain_metadata": row.get("domain_metadata", {})}
            for row in metadata_rows
            if row["chunk_id"] in chunks_by_id
        ]
    except Exception as exc:
        logger.warning(f"教育结构化查询失败: {exc}")
        return []


def list_courses(keyword: str = "", limit: int = 50) -> list[dict[str, Any]]:
    """按课程名/分类/目标筛选并去重返回课程列表。"""
    keyword = (keyword or "").strip().lower()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in _query_education_chunks("course_intro", limit=500):
        item = build_course_summary(chunk)
        haystack = " ".join(str(item.get(key, "")) for key in (
            "course_name", "course_category", "target_users", "learning_goals", "project_name",
        )).lower()
        if keyword and keyword not in haystack:
            continue
        identity = item["course_code"] or item["course_name"]
        if identity in seen:
            continue
        seen.add(identity)
        records.append(item)
        if len(records) >= limit:
            break
    return records


def list_questions(
    keyword: str = "",
    question_type: str = "",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """按题干关键词和题型筛选题目。"""
    keyword = (keyword or "").strip().lower()
    question_type = (question_type or "").strip()
    records: list[dict[str, Any]] = []
    for chunk in _query_education_chunks("question", limit=500):
        item = build_question_summary(chunk)
        if question_type and item["question_type"] != question_type:
            continue
        if keyword and keyword not in item["content"].lower():
            continue
        records.append(item)
        if len(records) >= limit:
            break
    return records


# ==================== AI修改 结束 ====================
