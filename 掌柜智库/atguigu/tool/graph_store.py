"""Neo4j 存储：只保存图结构和 Chunk 证据指针。"""

from __future__ import annotations

import os
# ==================== AI修改 开始 ====================
# 实体名称规范化和同文档图替换锁使用的标准库。
import re
import threading
import unicodedata
# ==================== AI修改 结束 ====================
from typing import Iterable, Mapping

from neo4j import GraphDatabase


# ==================== AI修改 开始 ====================
# 关系类型固定，避免模型随意生成无法查询和维护的关系名称。
RELATION_TYPES = {
    "CONTAINS", "NEXT", "MENTIONS", "PART_OF", "USES", "DEPENDS_ON",
    "REQUIRES", "IMPLEMENTS", "CAUSES", "RESOLVES", "TEACHES", "TESTS",
    "CONTRASTS_WITH", "RELATED_TO",
}

# 查询侧的默认和最大路径深度。普通事实问题不需要深路径，
# 路径问题才启用六跳；存储层仍做最后一道上限保护。
DEFAULT_GRAPH_HOPS = 3
MAX_GRAPH_HOPS = 6
# 限制每类 Graph 查询返回的路径数量，避免深路径结果挤爆后续融合池。
MAX_GRAPH_PATHS = 100


# ==================== AI修改 开始 ====================
# Neo4j 的替换同样是先删后写；按 document_id 串行化，避免两个同文档任务
# 交叉删除和写入，导致图中暂时混入两版数据。
_GRAPH_REPLACE_LOCKS = {}
_GRAPH_REPLACE_LOCKS_GUARD = threading.Lock()
_GRAPH_SCHEMA_READY = False
_GRAPH_SCHEMA_LOCK = threading.Lock()


def normalize_entity_name(value: str) -> str:
    """统一实体名称的 Unicode、空白和大小写表示。"""
    text = unicodedata.normalize("NFKC", str(value or "").strip())
    return re.sub(r"\s+", " ", text).casefold()


def _ensure_graph_schema(session) -> None:
    global _GRAPH_SCHEMA_READY
    if _GRAPH_SCHEMA_READY:
        return
    with _GRAPH_SCHEMA_LOCK:
        if _GRAPH_SCHEMA_READY:
            return
        # ==================== AI修改 开始 ====================
        # 三个稳定身份必须唯一，否则重复导入会在 Neo4j 生成重复文档、Chunk 和实体。
        for query in (
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.entity_key IS UNIQUE",
        ):
            session.run(query)
        _GRAPH_SCHEMA_READY = True
        # ==================== AI修改 结束 ====================
# ==================== AI修改 结束 ====================
# ==================== AI修改 结束 ====================


class GraphStore:
    """Neo4j 的最小适配层，所有关系都必须带原文 Chunk 证据。"""

    def __init__(self, driver=None):
        self.driver = driver or GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "neo4j"),
            ),
            # ==================== AI修改 开始 ====================
            # Neo4j 未启动或网络不可用时，最多等待 10 秒并让导入任务进入失败态。
            connection_timeout=10,
            # ==================== AI修改 结束 ====================
        )
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

    # ==================== AI修改 开始 ====================
    def close(self):
        self.driver.close()

    def replace_document(
        self,
        document: Mapping,
        chunks: Iterable[Mapping],
        entities: Iterable[Mapping],
        relations: Iterable[Mapping],
    ) -> None:
        """幂等替换一篇文档的图数据，关系证据必须指向本篇 Chunk。"""
        document_id = document["document_id"]
        chunks = list(chunks)
        entities = list(entities)
        relations = list(relations)
        with _GRAPH_REPLACE_LOCKS_GUARD:
            lock = _GRAPH_REPLACE_LOCKS.setdefault(document_id, threading.Lock())
        with lock, self.driver.session(database=self.database) as session:
            _ensure_graph_schema(session)
            tx = session.begin_transaction()
            try:
                # ==================== AI修改 开始 ====================
                # Entity 节点是跨文档复用的，旧关系没有 document_id 节点属性，
                # 所以必须先按关系属性删除本篇文档的语义边。
                tx.run(
                    "MATCH ()-[r {document_id: $document_id}]->() DELETE r",
                    document_id=document_id,
                )
                tx.run(
                    "MATCH (n {document_id: $document_id}) DETACH DELETE n",
                    document_id=document_id,
                )
                tx.run(
                    "MERGE (d:Document {document_id: $document_id}) "
                    "SET d.source_type=$source_type, d.source_id=$source_id, "
                    "d.source_path=$source_path, d.content_type=$content_type",
                    **{key: document.get(key, "") for key in (
                        "document_id", "source_type", "source_id", "source_path", "content_type"
                    )},
                )
                # 同一文档的节点和边使用 UNWIND 批量写入，避免每个节点单独往返 Neo4j。
                chunk_rows = [{
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "section_path": chunk.get("section_path", ""),
                } for chunk in chunks]
                if chunk_rows:
                    tx.run(
                        "MATCH (d:Document {document_id: $document_id}) "
                        "UNWIND $chunks AS chunk "
                        "MERGE (c:Chunk {chunk_id: chunk.chunk_id}) "
                        "SET c.document_id=$document_id, c.chunk_index=chunk.chunk_index, "
                        "c.section_path=chunk.section_path "
                        "MERGE (d)-[:CONTAINS]->(c)",
                        document_id=document_id,
                        chunks=chunk_rows,
                    )
                next_rows = [{
                    "previous_id": previous["chunk_id"],
                    "current_id": current["chunk_id"],
                } for previous, current in zip(chunks, chunks[1:])]
                if next_rows:
                    tx.run(
                        "UNWIND $next_rows AS item "
                        "MATCH (a:Chunk {chunk_id: item.previous_id}), "
                        "(b:Chunk {chunk_id: item.current_id}) "
                        "MERGE (a)-[:NEXT {document_id: $document_id}]->(b)",
                        next_rows=next_rows,
                        document_id=document_id,
                    )

                # 先按规范名/别名查找已有实体；类型冲突时不复用同名异义实体。
                entity_resolution = []
                resolution_entities = []
                for entity in entities:
                    resolution_entity = dict(entity)
                    resolution_entity["normalized_names"] = entity.get(
                        "normalized_names", [normalize_entity_name(entity.get("name", ""))]
                    )
                    resolution_entity["normalized_type"] = entity.get(
                        "normalized_type",
                        normalize_entity_name(entity.get("entity_type", "general")),
                    )
                    resolution_entities.append(resolution_entity)
                if resolution_entities:
                    entity_resolution = list(tx.run(
                        "UNWIND $entities AS incoming "
                        "OPTIONAL MATCH (existing:Entity) "
                        "WITH incoming, existing, "
                        "coalesce(existing.normalized_names, []) + "
                        "CASE WHEN existing.name IS NULL THEN [] "
                        "ELSE [toLower(existing.name)] END + "
                        "[alias IN coalesce(existing.aliases, []) | toLower(alias)] "
                        "AS known_names "
                        "WHERE any(value IN known_names "
                        "WHERE value IN incoming.normalized_names) "
                        "AND (coalesce(existing.normalized_type, toLower(existing.entity_type)) "
                        "= incoming.normalized_type "
                        "OR coalesce(existing.normalized_type, toLower(existing.entity_type)) = 'general' "
                        "OR incoming.normalized_type = 'general') "
                        "WITH incoming, collect(existing)[0] AS existing "
                        "RETURN incoming.entity_key AS incoming_key, "
                        "coalesce(existing.entity_key, incoming.entity_key) AS resolved_key, "
                        "coalesce(existing.entity_type, incoming.entity_type) AS resolved_type",
                        entities=resolution_entities,
                    ))
                resolved_keys = {}
                resolved_types = {}
                for record in entity_resolution:
                    data = record.data()
                    resolved_keys[data["incoming_key"]] = data["resolved_key"]
                    resolved_types[data["incoming_key"]] = data["resolved_type"]
                resolved_entities = []
                for entity in entities:
                    incoming_key = entity["entity_key"]
                    resolved = dict(entity)
                    resolved["entity_key"] = resolved_keys.get(incoming_key, incoming_key)
                    resolved["normalized_names"] = next(
                        item["normalized_names"] for item in resolution_entities
                        if item["entity_key"] == incoming_key
                    )
                    resolved["normalized_type"] = next(
                        item["normalized_type"] for item in resolution_entities
                        if item["entity_key"] == incoming_key
                    )
                    resolved["entity_type"] = resolved_types.get(
                        incoming_key, entity.get("entity_type", "general")
                    )
                    resolved_entities.append(resolved)
                entity_rows = [{
                    "entity_key": entity["entity_key"],
                    "name": entity["name"],
                    "entity_type": entity.get("entity_type", "general"),
                    "normalized_type": entity.get(
                        "normalized_type", normalize_entity_name(entity.get("entity_type", "general"))
                    ),
                    "aliases": entity.get("aliases", []),
                    "normalized_names": entity.get("normalized_names", []),
                    "chunk_ids": entity.get("chunk_ids", []),
                } for entity in resolved_entities]
                if entity_rows:
                    tx.run(
                        "UNWIND $entities AS entity "
                        "MERGE (e:Entity {entity_key: entity.entity_key}) "
                        "SET e.name=coalesce(e.name, entity.name), "
                        "e.entity_type=entity.entity_type, "
                        "e.normalized_type=entity.normalized_type, "
                        "e.aliases=coalesce(e.aliases, []) + "
                        "[alias IN entity.aliases WHERE NOT alias IN coalesce(e.aliases, [])], "
                        "e.normalized_names=coalesce(e.normalized_names, []) + "
                        "[value IN entity.normalized_names "
                        "WHERE NOT value IN coalesce(e.normalized_names, [])] "
                        "WITH e, entity "
                        "UNWIND entity.chunk_ids AS chunk_id "
                        "MATCH (c:Chunk {chunk_id: chunk_id}) "
                        "MERGE (c)-[:MENTIONS {document_id: $document_id}]->(e)",
                        entities=entity_rows,
                        document_id=document_id,
                    )
                for relation_type in RELATION_TYPES:
                    relation_rows = [
                        {key: relation.get(key, "") for key in (
                            "source_key", "target_key", "source_chunk_id",
                            "evidence", "confidence",
                        )}
                        for relation in relations
                        if relation.get("relation_type") == relation_type
                    ]
                    for relation in relation_rows:
                        relation["source_key"] = resolved_keys.get(
                            relation["source_key"], relation["source_key"]
                        )
                        relation["target_key"] = resolved_keys.get(
                            relation["target_key"], relation["target_key"]
                        )
                    if not relation_rows:
                        continue
                    tx.run(
                        f"UNWIND $relations AS relation "
                        f"MATCH (a:Entity {{entity_key: relation.source_key}}), "
                        f"(b:Entity {{entity_key: relation.target_key}}), "
                        f"(c:Chunk {{chunk_id: relation.source_chunk_id}}) "
                        f"MERGE (a)-[r:{relation_type} {{source_chunk_id: relation.source_chunk_id}}]->(b) "
                        "SET r.document_id=$document_id, r.evidence=relation.evidence, "
                        "r.confidence=relation.confidence",
                        relations=relation_rows,
                        document_id=document_id,
                    )
                # 文档更新后清理不再被任何文档引用的实体，保留共享实体。
                tx.run("MATCH (e:Entity) WHERE NOT (e)--() DELETE e")
                tx.commit()
            except Exception:
                tx.rollback()
                raise
            # ==================== AI修改 结束 ====================

    # ==================== AI修改 开始 ====================
    # Graph 与 Milvus 共用教育来源过滤；普通查询传 None 保持跨类型召回。
    def find_related_chunks(
        self,
        entity_names: list[str],
        hops: int = DEFAULT_GRAPH_HOPS,
        content_type: str | None = None,
        max_paths: int = MAX_GRAPH_PATHS,
    ) -> list[dict]:
        hops = min(max(hops, 1), MAX_GRAPH_HOPS)
        max_paths = min(max(max_paths, 1), MAX_GRAPH_PATHS)
        normalized_names = [
            normalize_entity_name(name) for name in entity_names if normalize_entity_name(name)
        ]
        with self.driver.session(database=self.database) as session:
            # ==================== AI修改 开始 ====================
            # 实体命中首先直接回溯 MENTIONS Chunk；关系证据作为补充，不能替代
            # 直接命中，否则只有实体没有语义边时 Graph RAG 会返回空列表。
            direct_result = session.run(
                "MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk) "
                "WHERE e.name IN $names OR any(alias IN coalesce(e.aliases, []) "
                "WHERE alias IN $names) "
                "OR any(value IN coalesce(e.normalized_names, []) "
                "WHERE value IN $normalized_names) "
                "OPTIONAL MATCH (d:Document)-[:CONTAINS]->(c) "
                "WHERE $content_type IS NULL OR d.content_type = $content_type "
                "RETURN c.chunk_id AS chunk_id, [e.name] AS path, "
                "1.0 AS evidence_score "
                "ORDER BY c.chunk_id LIMIT $max_paths",
                names=entity_names,
                normalized_names=normalized_names,
                content_type=content_type,
                max_paths=max_paths,
            )
            relation_result = session.run(
                "MATCH p=(e:Entity)-[*1..%d]-(other:Entity) "
                "WHERE e.name IN $names OR any(alias IN coalesce(e.aliases, []) "
                "WHERE alias IN $names) "
                "OR any(value IN coalesce(e.normalized_names, []) "
                "WHERE value IN $normalized_names) "
                "UNWIND relationships(p) AS r "
                "WITH r, p WHERE r.source_chunk_id IS NOT NULL "
                "OPTIONAL MATCH (d:Document)-[:CONTAINS]->"
                "(:Chunk {chunk_id: r.source_chunk_id}) "
                "WHERE $content_type IS NULL OR d.content_type = $content_type "
                "RETURN DISTINCT r.source_chunk_id AS chunk_id, "
                "[n IN nodes(p) | n.name] AS path, "
                "length(p) AS path_depth, "
                "coalesce(r.confidence, 0.0) AS evidence_score "
                "ORDER BY path_depth, evidence_score DESC, chunk_id "
                "LIMIT $max_paths" % hops,
                names=entity_names,
                normalized_names=normalized_names,
                content_type=content_type,
                max_paths=max_paths,
            )
            merged = {}
            for is_direct, result in ((True, direct_result), (False, relation_result)):
                for record in result:
                    data = record.data()
                    chunk_id = data.get("chunk_id")
                    path = data.get("path") or []
                    if not chunk_id or not path:
                        continue
                    item = merged.setdefault(
                        chunk_id,
                        {"paths": [], "direct": False, "evidence_score": 0.0},
                    )
                    item["direct"] = item["direct"] or is_direct
                    item["evidence_score"] = max(
                        item["evidence_score"], data.get("evidence_score", 0.0) or 0.0
                    )
                    if path not in item["paths"]:
                        item["paths"].append(path)
            return [
                {"chunk_id": chunk_id, "path": item["paths"]}
                for chunk_id, item in sorted(
                    merged.items(),
                    key=lambda pair: (
                        not pair[1]["direct"],
                        -pair[1]["evidence_score"],
                        min(len(path) for path in pair[1]["paths"]),
                        str(pair[0]),
                    ),
                )
            ]
    # ==================== AI修改 结束 ====================

    # ==================== AI修改 开始 ====================
    # Milvus 删除后同步清理图证据，避免已删除文档继续参与关系检索。
    def delete_document(self, document_id: str) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MATCH ()-[r {document_id: $document_id}]->() DELETE r",
                document_id=document_id,
            )
            session.run(
                "MATCH (n {document_id: $document_id}) DETACH DELETE n",
                document_id=document_id,
            )
            session.run("MATCH (e:Entity) WHERE NOT (e)--() DELETE e")

    def delete_source(self, source_type: str, source_id: str | None = None) -> None:
        query = (
            "MATCH (d:Document {source_type: $source_type}) "
            "WHERE $source_id IS NULL OR d.source_id = $source_id "
            "WITH collect(d.document_id) AS ids "
            "MATCH ()-[r]->() WHERE r.document_id IN ids DELETE r "
            "WITH ids MATCH (n) WHERE n.document_id IN ids DETACH DELETE n"
        )
        with self.driver.session(database=self.database) as session:
            session.run(query, source_type=source_type, source_id=source_id)
            session.run("MATCH (e:Entity) WHERE NOT (e)--() DELETE e")

    def delete_path_prefix(
        self,
        source_path: str,
        source_type: str | None = None,
        source_id: str | None = None,
    ) -> None:
        query = (
            "MATCH (d:Document) "
            "WHERE d.source_path STARTS WITH $prefix "
            "AND ($source_type IS NULL OR d.source_type = $source_type) "
            "AND ($source_id IS NULL OR d.source_id = $source_id) "
            "WITH collect(d.document_id) AS ids "
            "MATCH ()-[r]->() WHERE r.document_id IN ids DELETE r "
            "WITH ids MATCH (n) WHERE n.document_id IN ids DETACH DELETE n"
        )
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                prefix=source_path.rstrip("/") + "/",
                source_type=source_type,
                source_id=source_id,
            )
            session.run("MATCH (e:Entity) WHERE NOT (e)--() DELETE e")
    # ==================== AI修改 结束 ====================
# ==================== AI修改 结束 ====================
