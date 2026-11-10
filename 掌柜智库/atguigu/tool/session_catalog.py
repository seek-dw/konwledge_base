# ==================== AI修改 开始 ====================
# 多会话目录工具。
# 聊天内容仍然放在原来的 chat_history 集合中；本文件只保存会话标题和更新时间，
# 这样增加“左侧会话列表”不会改变已有历史数据，也不会影响旧会话。
# ==================== AI修改 结束 ====================
import re
import threading
import time


def normalize_session_title(text: str, max_length: int = 32) -> str:
    """把首条问题整理成适合侧栏展示的短标题。"""
    title = re.sub(r"\s+", " ", (text or "")).strip()
    if not title:
        return "新会话"
    if len(title) <= max_length:
        return title
    return title[: max_length - 1].rstrip() + "…"


def build_session_record(session_id: str, title: str, updated_ts: float | None = None) -> dict:
    """构造可直接写入 MongoDB 的会话目录记录。"""
    return {
        "session_id": session_id,
        "title": normalize_session_title(title),
        "updated_ts": updated_ts or time.time(),
    }


# ==================== AI修改 开始 ====================
_LOCAL_SESSION_CATALOG: dict[str, dict] = {}
_LOCAL_SESSION_CATALOG_LOCK = threading.Lock()


def merge_session_catalog_rows(
    remote_rows: list[dict],
    local_rows: list[dict],
    limit: int = 50,
) -> list[dict]:
    """合并Mongo目录和进程内目录，保证数据库短暂不可用时不丢会话。"""
    merged: dict[str, dict] = {}
    for row in list(remote_rows or []) + list(local_rows or []):
        session_id = row.get("session_id")
        if session_id:
            candidate = {
                "session_id": session_id,
                "title": row.get("title") or "新会话",
                "updated_ts": row.get("updated_ts", 0),
            }
            current = merged.get(session_id)
            if current is None or candidate["updated_ts"] >= current["updated_ts"]:
                merged[session_id] = candidate
    return sorted(
        merged.values(),
        key=lambda row: row.get("updated_ts", 0),
        reverse=True,
    )[:limit]


def _get_catalog_collection():
    """懒加载会话目录集合，不复用 chat_history 的全局 collection 缓存。"""
    from atguigu.config.config import MongoConfig
    from atguigu.tool.mongo_client_tool import mongo_client_create

    collection = mongo_client_create()[MongoConfig.mongo_db_name]["session_catalog"]
    collection.create_index("session_id", unique=True)
    return collection


def upsert_session_catalog(
        session_id: str,
        title: str,
        preserve_title: bool = False,
) -> dict:
    """新建或刷新会话目录记录。"""
    record = build_session_record(session_id, title)
    if preserve_title:
        with _LOCAL_SESSION_CATALOG_LOCK:
            current = _LOCAL_SESSION_CATALOG.get(session_id)
        if current and current.get("title") not in {"新会话", "当前会话"}:
            record["title"] = current["title"]
        else:
            try:
                current = _get_catalog_collection().find_one({"session_id": session_id})
                if current and current.get("title") not in {"新会话", "当前会话"}:
                    record["title"] = current["title"]
            except Exception:
                pass
    # ==================== AI修改 开始 ====================
    # 先写入进程内目录，Mongo短暂断开时新会话仍能立即显示在左侧列表。
    with _LOCAL_SESSION_CATALOG_LOCK:
        _LOCAL_SESSION_CATALOG[session_id] = record
    try:
        _get_catalog_collection().update_one(
            {"session_id": session_id},
            {"$set": record},
            upsert=True,
        )
    except Exception:
        # Mongo恢复后下一次写入会重新同步；本次会话不因目录服务失败而丢失。
        pass
    # ==================== AI修改 结束 ====================
    return record


def list_session_catalog(limit: int = 50) -> list[dict]:
    """按最近使用时间倒序返回会话目录。"""
    # ==================== AI修改 开始 ====================
    # 远程目录和本地兜底目录合并，而不是Mongo失败时直接返回空列表，
    # 这样点击“+”创建的多个会话不会互相覆盖。
    with _LOCAL_SESSION_CATALOG_LOCK:
        local_rows = list(_LOCAL_SESSION_CATALOG.values())
    try:
        rows = _get_catalog_collection().find({}).sort("updated_ts", -1).limit(limit)
        remote_rows = list(rows)
    except Exception:
        remote_rows = []
    return merge_session_catalog_rows(remote_rows, local_rows, limit=limit)
    # ==================== AI修改 结束 ====================


def rename_session_catalog(session_id: str, title: str) -> dict:
    """重命名会话并返回最新记录。"""
    record = upsert_session_catalog(session_id, title)
    return record


def delete_session_catalog(session_id: str) -> int:
    """删除会话目录记录，聊天历史由 API 层同步删除。"""
    # ==================== AI修改 开始 ====================
    # 无论Mongo是否在线，先从本地兜底目录移除，保证界面删除动作即时生效。
    with _LOCAL_SESSION_CATALOG_LOCK:
        _LOCAL_SESSION_CATALOG.pop(session_id, None)
    try:
        result = _get_catalog_collection().delete_one({"session_id": session_id})
        return result.deleted_count
    except Exception:
        return 0
    # ==================== AI修改 结束 ====================
# ==================== AI修改 结束 ====================


# ==================== AI修改 结束 ====================
