# ==================== AI修改 开始 ====================
"""知识库删除业务服务。

HTTP 路由统一放在 ``app_service.py``；本模块只负责联动 Milvus 和 Neo4j，
让路由函数保持简洁，也避免删除接口各自实现一套清理逻辑。
"""

from atguigu.tool.knowledge_chunk_store import KnowledgeChunkStore
from atguigu.tool.graph_store import GraphStore
from atguigu.tool.logger import logger


def _delete_summary(raw_result) -> dict:
    """把不同 pymilvus 版本的删除返回值整理成稳定的 API 数据。"""
    if isinstance(raw_result, dict):
        delete_count = raw_result.get("delete_count")
    else:
        delete_count = getattr(raw_result, "delete_count", None)
    return {
        "delete_count": delete_count,
        "acknowledged": delete_count is not None,
    }


# ==================== AI修改 开始 ====================
def _delete_graph(method: str, *args) -> None:
    """同步清理 Neo4j；图服务暂时不可用时不影响 Milvus 删除。"""
    graph = None
    try:
        graph = GraphStore()
        getattr(graph, method)(*args)
    except Exception as exc:
        logger.warning(f"Neo4j 删除同步失败: {exc}")
    finally:
        if graph:
            graph.close()
# ==================== AI修改 结束 ====================


def delete_document_chunks(document_id: str) -> dict:
    # API 层只返回稳定的 JSON 摘要，不把 pymilvus 的版本相关对象直接暴露给前端。
    result = KnowledgeChunkStore().delete_document(document_id)
    _delete_graph("delete_document", document_id)
    return {
        "document_id": document_id,
        **_delete_summary(result),
    }


def delete_source_chunks(source_type: str, source_id: str | None = None) -> dict:
    result = KnowledgeChunkStore().delete_source(source_type, source_id)
    _delete_graph("delete_source", source_type, source_id)
    response = {
        "source_type": source_type,
        **_delete_summary(result),
    }
    if source_id:
        response["source_id"] = source_id
    return response


def delete_directory_chunks(
    source_path: str,
    source_type: str | None = None,
    source_id: str | None = None,
) -> dict:
    result = KnowledgeChunkStore().delete_path_prefix(source_path, source_type, source_id)
    _delete_graph("delete_path_prefix", source_path, source_type, source_id)
    response = {
        "source_path": source_path,
        **_delete_summary(result),
    }
    if source_type:
        response["source_type"] = source_type
    if source_id:
        response["source_id"] = source_id
    return response


# ==================== AI修改 结束 ====================
