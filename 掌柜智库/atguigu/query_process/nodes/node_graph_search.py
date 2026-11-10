# ==================== AI修改 开始 ====================
"""图检索节点：按实体关系找到 Chunk，再从 Milvus 取回原文。"""

import json
import re

from atguigu.config.config import MilvusConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.graph_store import (
    DEFAULT_GRAPH_HOPS,
    MAX_GRAPH_HOPS,
    MAX_GRAPH_PATHS,
    GraphStore,
)
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_create import (
    CHUNK_OUTPUT_FIELDS,
    ensure_collection_loaded,
    get_milvus_client,
    select_existing_output_fields,
)


class NodeGraphSearch(NodeBase):
    name = "node_graph_search"

    _PATH_QUERY_RE = re.compile(
        r"主路径|路径|如何连接|怎么连接|概念图|关系|关联|链路|流程|依赖|从.+到"
    )

    def __init__(self, graph_store=None, client=None):
        super().__init__()
        self.graph_store = graph_store
        self.client = client

    @classmethod
    def _select_hops(cls, state: QueryGraphState) -> int:
        query = " ".join(
            str(state.get(key) or "")
            for key in ("rewritten_query", "original_query")
        )
        if cls._PATH_QUERY_RE.search(query):
            return MAX_GRAPH_HOPS
        return DEFAULT_GRAPH_HOPS

    def process(self, state: QueryGraphState):
        entity_names = [
            str(name).strip()
            for name in (state.get("item_names") or [])
            if str(name).strip()
        ]
        if not entity_names:
            return {"graph_chunks": [], "graph_paths": []}

        # ==================== AI修改 开始 ====================
        # 教育查询和向量检索使用同一 content_type；Graph 只补同类 Chunk，
        # 普通文档查询不加类型限制，避免误伤已有通用资料。
        content_type = {
            "course": "course_intro",
            "question": "question",
        }.get(state.get("query_type"))
        # Neo4j 是增强召回路由，不应成为本地向量检索的硬依赖。
        try:
            graph_store = self.graph_store or GraphStore()
            graph_kwargs = {
                "hops": self._select_hops(state),
                "max_paths": MAX_GRAPH_PATHS,
            }
            if content_type:
                graph_kwargs["content_type"] = content_type
            hits = graph_store.find_related_chunks(entity_names, **graph_kwargs)
        except Exception as exc:
            logger.warning(f"Graph 检索失败，降级为向量检索: {exc}")
            return {"graph_chunks": [], "graph_paths": []}
        # ==================== AI修改 结束 ====================
        if not hits:
            return {"graph_chunks": [], "graph_paths": []}

        path_by_chunk = {}
        for hit in hits:
            chunk_id = hit.get("chunk_id")
            if chunk_id:
                paths = hit.get("path", [])
                if paths and isinstance(paths[0], list):
                    path_by_chunk.setdefault(chunk_id, []).extend(paths)
                else:
                    path_by_chunk.setdefault(chunk_id, []).append(paths)
        chunk_ids = list(path_by_chunk)
        client = self.client or get_milvus_client()
        # ==================== AI修改 开始 ====================
        # Graph 只返回 Chunk 指针，正文回溯失败时仍保留向量检索结果。
        try:
            ensure_collection_loaded(MilvusConfig.milvus_chunks_collection)
            output_fields = select_existing_output_fields(
                client, MilvusConfig.milvus_chunks_collection, CHUNK_OUTPUT_FIELDS
            )
            chunk_filter = (
                f"chunk_id in {json.dumps(chunk_ids, ensure_ascii=False)}"
            )
            if content_type:
                chunk_filter += f" and content_type == '{content_type}'"
            rows = client.query(
                collection_name=MilvusConfig.milvus_chunks_collection,
                filter=chunk_filter,
                output_fields=output_fields,
            )
        except Exception as exc:
            logger.warning(f"Graph Chunk 回溯失败，降级为向量检索: {exc}")
            return {"graph_chunks": [], "graph_paths": []}
        # ==================== AI修改 结束 ====================
        graph_chunks = []
        for row in rows:
            chunk = dict(row)
            chunk["score"] = 0.0
            chunk["source"] = "graph"
            chunk["graph_path"] = path_by_chunk.get(chunk.get("chunk_id"), [])
            graph_chunks.append(chunk)
        return {
            "graph_chunks": graph_chunks,
            "graph_paths": [path for paths in path_by_chunk.values() for path in paths],
        }
# ==================== AI修改 结束 ====================
