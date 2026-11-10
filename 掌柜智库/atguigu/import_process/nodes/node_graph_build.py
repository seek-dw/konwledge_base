# ==================== AI修改 开始 ====================
"""从文档切片抽取实体和关系，并写入 Neo4j。"""

from __future__ import annotations

import hashlib
import json
import re

from langchain.chat_models import init_chat_model

from atguigu.config.config import ModelConfig
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
# ==================== AI修改 开始 ====================
# 统一使用 GraphStore 的实体名称规范化规则，保证构建和查询键一致。
from atguigu.tool.graph_store import GraphStore, RELATION_TYPES, normalize_entity_name
# ==================== AI修改 结束 ====================
from atguigu.tool.logger import logger


class NodeGraphBuild(NodeBase):
    """图构建节点：LLM 只负责抽取，节点负责校验和证据绑定。"""

    name = "node_graph_build"

    # ==================== AI修改 开始 ====================
    # 先尝试单次抽取；证据过长或单次输出异常时，再按批次串行兜底。
    PRIMARY_MAX_EVIDENCE_CHARS = 10000
    GRAPH_BATCH_SIZE = 16
    ENTITIES_PER_CHUNK = 8
    RELATIONS_PER_CHUNK = 12
    GRAPH_MAX_TOKENS = 16384
    GRAPH_TIMEOUT_SECONDS = 120
    # ==================== AI修改 结束 ====================

    def __init__(self, graph_store=None, llm=None):
        super().__init__()
        self.graph_store = graph_store
        self.llm = llm

    # ==================== AI修改 开始 ====================
    @staticmethod
    def _parse_model_json(response):
        """兼容模型返回的纯 JSON、代码围栏、解释前缀和结构化文本块。"""
        raw = getattr(response, "content", response)
        if not raw:
            additional = getattr(response, "additional_kwargs", {}) or {}
            # 部分思考模型把最终文本放在 additional_kwargs，而 content 为空。
            raw = additional.get("content") or additional.get("reasoning_content")
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list):
            raw = "".join(
                str(block.get("text") or block.get("content") or "")
                if isinstance(block, dict)
                else str(getattr(block, "text", block) or "")
                for block in raw
            )
        text = str(raw or "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("Graph Build 模型未返回有效 JSON")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("Graph Build 模型返回内容不是有效 JSON") from exc
    # ==================== AI修改 结束 ====================

    def process(self, state: ImportGraphState):
        chunks = state.get("chunks") or []
        document_id = state["document_id"]
        source_path = state.get("source_path") or state.get("local_file_path") or document_id
        chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
        fallback_batches = [
            (batch_start // self.GRAPH_BATCH_SIZE + 1,
             chunks[batch_start:batch_start + self.GRAPH_BATCH_SIZE])
            for batch_start in range(0, len(chunks), self.GRAPH_BATCH_SIZE)
        ]
        batch_count = len(fallback_batches)
        all_evidence = "\n\n".join(
            f"[chunk_id={chunk['chunk_id']}]\n"
            f"章节：{chunk.get('section_path', '')}\n"
            f"正文：{chunk.get('content', '')}"
            for chunk in chunks
        )
        # ==================== AI修改 开始 ====================
        # 记录总批次数，结合下面的 batch 日志即可定位具体卡在哪一批。
        logger.info(
            f"[GRAPH-BUILD] start document_id={document_id} "
            f"source_path={source_path} chunks={len(chunks)} batches={batch_count}"
        )
        # ==================== AI修改 结束 ====================

        if self.llm is None:
            self.llm = init_chat_model(
                model=ModelConfig.LLM_MODEL_NAME,
                model_provider="openai",
                api_key=ModelConfig.LLM_API_KEY,
                base_url=ModelConfig.LLM_BASE_URL,
                temperature=0,
                # ==================== AI修改 开始 ====================
                # 图抽取只需要结构化结果，限制输出并设置超时，避免导入任务
                # 一直停在 Graph Build 而不返回状态。
                max_tokens=self.GRAPH_MAX_TOKENS,
                timeout=self.GRAPH_TIMEOUT_SECONDS,
                # Qwen 的思考内容会占用输出预算；图抽取只需要最终 JSON，关闭思考避免 content 为空。
                extra_body={"enable_thinking": False},
                # ==================== AI修改 结束 ====================
            )
        # ==================== AI修改 开始 ====================
        # 单次请求优先，失败后才进入串行分批；每个批次只重试自身。
        def extract_request(batch_number, batch_total, batch_chunks,
                            entity_limit, relation_limit, retries, fallback=False):
            batch_evidence = "\n\n".join(
                f"[chunk_id={chunk['chunk_id']}]\n"
                f"章节：{chunk.get('section_path', '')}\n"
                f"正文：{chunk.get('content', '')}"
                for chunk in batch_chunks
            )
            messages = [{
                "role": "system",
                "content": (
                    "你是知识图谱抽取器。只从给定切片抽取实体和实体关系，"
                    "关系必须使用白名单：" + ",".join(sorted(RELATION_TYPES)) + "。"
                    "不要输出思考过程、解释或 Markdown 代码围栏，只输出 JSON。"
                    f"每个切片最多 {self.ENTITIES_PER_CHUNK} 个实体和 "
                    f"{self.RELATIONS_PER_CHUNK} 条关系，本批最多 "
                    f"{entity_limit} 个实体和 {relation_limit} 条关系。"
                    + ("这是单次请求。" if not fallback else "这是分批兜底请求。")
                    + "输出严格 JSON：{entities:[{name,entity_type,aliases,chunk_ids}],"
                    "relations:[{source,target,relation_type,source_chunk_id}]}。"
                ),
            }, {
                "role": "user",
                "content": batch_evidence,
            }]
            successful_entity_limit = entity_limit
            successful_relation_limit = relation_limit
            for attempt in range(retries):
                current_entity_limit = (
                    entity_limit if attempt == 0 else max(1, entity_limit // 2)
                )
                current_relation_limit = (
                    relation_limit if attempt == 0 else max(1, relation_limit // 2)
                )
                if attempt:
                    messages[0]["content"] += (
                        f"本次重试最多 {current_entity_limit} 个实体和 "
                        f"{current_relation_limit} 条关系。"
                    )
                logger.info(
                    f"[GRAPH-BUILD] LLM invoke start batch={batch_number}/{batch_total} "
                    f"attempt={attempt + 1} evidence_chars={len(batch_evidence)}"
                )
                response = self.llm.invoke(messages)
                metadata = getattr(response, "response_metadata", {}) or {}
                additional = getattr(response, "additional_kwargs", {}) or {}
                finish_reason = metadata.get("finish_reason") or additional.get("finish_reason")
                logger.info(
                    f"[GRAPH-BUILD] LLM invoke end batch={batch_number}/{batch_total} "
                    f"attempt={attempt + 1} "
                    f"response_chars={len(str(getattr(response, 'content', '') or ''))} "
                    f"finish_reason={finish_reason or 'unknown'}"
                )
                try:
                    extracted = self._parse_model_json(response)
                    if finish_reason in {"length", "max_tokens", "token_limit"}:
                        raise ValueError("Graph Build 模型输出达到 token 上限")
                    successful_entity_limit = current_entity_limit
                    successful_relation_limit = current_relation_limit
                    break
                except ValueError as exc:
                    if attempt == retries - 1:
                        message = f"{exc}；文档：{source_path}；批次：{batch_number}/{batch_total}"
                        logger.error(f"[GRAPH-BUILD] invalid model JSON after retry: {message}")
                        raise ValueError(message) from exc
                    logger.warning(
                        f"[GRAPH-BUILD] invalid model JSON, retry batch={batch_number}: {exc}"
                    )
                    messages[0]["content"] += (
                        "上一次返回格式无效。现在只返回一个完整 JSON 对象，"
                        "不要输出任何其他字符。实体只保留 name、entity_type、aliases、chunk_ids，"
                        "关系只保留 source、target、relation_type、source_chunk_id。"
                    )
            return (
                extracted.get("entities", [])[:successful_entity_limit],
                extracted.get("relations", [])[:successful_relation_limit],
            )

        # ==================== AI修改 开始 ====================
        # 当前文档证据未超过安全阈值时，先用 16384 输出预算尝试一次整篇抽取。
        batch_results = []
        primary_error = None
        if chunks and len(all_evidence) <= self.PRIMARY_MAX_EVIDENCE_CHARS:
            try:
                batch_results = [extract_request(
                    1, 1, chunks,
                    min(len(chunks), self.GRAPH_BATCH_SIZE) * self.ENTITIES_PER_CHUNK,
                    min(len(chunks), self.GRAPH_BATCH_SIZE) * self.RELATIONS_PER_CHUNK,
                    retries=1,
                )]
            except ValueError as exc:
                primary_error = exc
                logger.warning(f"[GRAPH-BUILD] single request failed, use sequential batches: {exc}")
        if not batch_results:
            if not chunks:
                batch_results = []
            else:
                if primary_error is None:
                    logger.info(
                        f"[GRAPH-BUILD] single request skipped evidence_chars={len(all_evidence)}"
                    )
                batch_results = []
                for batch_number, batch_chunks in fallback_batches:
                    batch_results.append(extract_request(
                        batch_number,
                        batch_count,
                        batch_chunks,
                        len(batch_chunks) * self.ENTITIES_PER_CHUNK,
                        len(batch_chunks) * self.RELATIONS_PER_CHUNK,
                        retries=2,
                        fallback=True,
                    ))
        extracted_entities = [
            item for entities, _ in batch_results for item in entities
        ]
        extracted_relations = [
            item for _, relations in batch_results for item in relations
        ]
        # ==================== AI修改 结束 ====================

        entity_map = {}
        entity_by_key = {}
        entities = []
        for item in extracted_entities:
            name = re.sub(r"\s+", " ", str(item.get("name") or "").strip())
            if not name:
                continue
            entity_type = str(item.get("entity_type") or "general").strip() or "general"
            normalized_name = normalize_entity_name(name)
            normalized_type = normalize_entity_name(entity_type) or "general"
            aliases = [
                str(alias).strip()
                for alias in item.get("aliases", [])
                if str(alias).strip()
            ]
            normalized_names = list(dict.fromkeys(
                normalize_entity_name(value)
                for value in [name, *aliases]
                if normalize_entity_name(value)
            ))
            key = f"entity_{hashlib.sha256((normalized_name + chr(31) + normalized_type).encode()).hexdigest()[:24]}"
            entity = {
                "entity_key": key,
                "name": name,
                "entity_type": entity_type,
                "normalized_type": normalized_type,
                "aliases": aliases,
                "normalized_names": normalized_names,
                "chunk_ids": [chunk_id for chunk_id in item.get("chunk_ids", []) if chunk_id in chunk_ids],
            }
            existing = entity_by_key.get(key)
            # ==================== AI修改 开始 ====================
            # 同一批中规范名不同但别名相同，且实体类型一致时视为同一实体；
            # 类型不同不合并，避免把同名课程、框架等异义实体混在一起。
            if existing is None:
                existing = next(
                    (
                        candidate for candidate in entities
                        if candidate["normalized_type"] == normalized_type
                        and set(candidate["normalized_names"]) & set(normalized_names)
                    ),
                    None,
                )
                if existing is not None:
                    entity_by_key[key] = existing
            # ==================== AI修改 结束 ====================
            if existing:
                existing["aliases"] = list(dict.fromkeys(existing["aliases"] + aliases))
                existing["normalized_names"] = list(dict.fromkeys(
                    existing["normalized_names"] + normalized_names
                ))
                existing["chunk_ids"] = list(dict.fromkeys(
                    existing["chunk_ids"] + entity["chunk_ids"]
                ))
            else:
                entity_by_key[key] = entity
                entities.append(entity)
            # ==================== AI修改 开始 ====================
            # 规范名和模型抽出的别名都指向同一个实体，关系端点写简称时也能解析。
            for value in normalized_names:
                entity_map.setdefault(value, entity_by_key[key])
            # ==================== AI修改 结束 ====================

        relations = []
        relation_keys = set()
        for item in extracted_relations:
            source = entity_map.get(normalize_entity_name(item.get("source")))
            target = entity_map.get(normalize_entity_name(item.get("target")))
            relation_type = str(item.get("relation_type") or "").upper()
            source_chunk_id = item.get("source_chunk_id")
            source_chunk = chunk_map.get(source_chunk_id)
            if not source or not target or relation_type not in RELATION_TYPES:
                continue
            if source_chunk is None:
                continue
            relation_key = (
                source["entity_key"], target["entity_key"], relation_type, source_chunk_id
            )
            if relation_key in relation_keys:
                continue
            relation_keys.add(relation_key)
            relations.append({
                "source_key": source["entity_key"],
                "target_key": target["entity_key"],
                "relation_type": relation_type,
                "source_chunk_id": source_chunk_id,
                # 证据来自本地原文，避免模型生成长文本把 JSON 撑爆或篡改证据。
                "evidence": str(source_chunk.get("content") or "")[:500],
                "confidence": float(item.get("confidence") or 1.0),
            })

        logger.info(
            f"[GRAPH-BUILD] parsed entities={len(entities)} relations={len(relations)}"
        )
        if self.graph_store is None:
            self.graph_store = GraphStore()
        logger.info("[GRAPH-BUILD] neo4j replace start")
        self.graph_store.replace_document(
            document={
                key: state.get(key, "")
                for key in (
                    "document_id", "source_type", "source_id", "source_path", "content_type"
                )
            },
            chunks=chunks,
            entities=entities,
            relations=relations,
        )
        logger.info("[GRAPH-BUILD] neo4j replace end")
        return {"graph_entities": entities, "graph_relations": relations}
# ==================== AI修改 结束 ====================
