# ==================== AI修改 开始 ====================
"""统一知识切片的 Milvus 持久化模块。

运行时只面向新的统一 schema；旧集合保留，不参与新导入和查询。
"""

from __future__ import annotations

import hashlib
# ==================== AI修改 开始 ====================
# 文档身份规范化和同文档替换锁使用的标准库。
import re
import threading
import unicodedata
# ==================== AI修改 结束 ====================
# ==================== AI修改 开始 ====================
from pathlib import Path
# ==================== AI修改 结束 ====================
from typing import Any, Callable, Mapping

from pymilvus import DataType

from atguigu.config.config import MilvusConfig
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_create import (
    ensure_collection_loaded,
    get_milvus_client,
)


# ==================== AI修改 开始 ====================
# 同一来源的路径只要语义相同，就必须生成同一个 document_id；Windows 路径
# 的大小写、斜杠写法和重复分隔符不应制造第二份文档。
_DOCUMENT_REPLACE_LOCKS = {}
_DOCUMENT_REPLACE_LOCKS_GUARD = threading.Lock()


def _canonical_identity_part(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "").strip())
    text = re.sub(r"/+", "/", text.replace("\\", "/"))
    return text.rstrip("/").casefold()


def _document_replace_lock(document_id: str):
    with _DOCUMENT_REPLACE_LOCKS_GUARD:
        return _DOCUMENT_REPLACE_LOCKS.setdefault(document_id, threading.Lock())
# ==================== AI修改 结束 ====================


# ==================== AI修改 开始 ====================
# Graph RAG 主表只保留通用检索和证据字段；教育专属字段进入 Mongo 元数据。
KNOWLEDGE_CHUNK_FIELDS = (
    "chunk_id",
    "document_id",
    "source_type",
    "source_id",
    "source_path",
    "item_name",
    "content_type",
    "chunk_index",
    "section_path",
    "content",
    "dense_vector",
    "sparse_vector",
)
# ==================== AI修改 结束 ====================

# ==================== AI修改 开始 ====================
# Milvus 的 VARCHAR 超长时会直接拒绝插入。这里做校验而不是截断正文，
# 避免为了满足 schema 悄悄丢掉回答依据。
_STRING_LIMITS = {
    "chunk_id": 128,
    "document_id": 128,
    "source_type": 64,
    "source_id": 500,
    "source_path": 2000,
    "item_name": 200,
    "content_type": 64,
    "content": 5000,
}
# ==================== AI修改 结束 ====================

# ==================== AI修改 开始 ====================
# 普通文档不会携带教育字段，因此由统一入库层补空值，保证所有来源使用同一行结构。
# 这是模板每个chunk都要包含的字段,如果没有值就用""填充
# 对比白名单少了稠密和稀疏向量字段,默认的字段,如果chunk里面是空,我默认给你添加这么多字段
_DEFAULTS: dict[str, Any] = {
    "chunk_id": "",
    "source_type": "document",
    "source_id": "",
    "source_path": "",
    "item_name": "",
    "content_type": "doc",
    "chunk_index": 0,
    "section_path": "",
    "content": "",
}
# ==================== AI修改 结束 ====================


def build_document_id(source_type: str, source_id: str, source_path: str) -> str:
    """根据来源和路径生成稳定的文档 ID。"""
    # source_type + source_id + source_path 共同构成文档身份；不能只用 file_title，
    # 因为不同目录中可以存在同名文件。
    # 生成器表达式,遍历三元组对每一个元素进行清洗,并且用分隔符链接清洗后的字符串
    # 流式处理数据,(expr for _ in xx)是惰性主导 , .join()逐个消费生成器,它不惰性,所以这里还是会全部遍历
    identity = "\x1f".join(
        _canonical_identity_part(value)
        for value in (source_type, source_id, source_path)
    )
    # 将identity通过utf-8编码为2进制,随后通过哈希函数创建哈希对象,再将结果转化为16进制字符串,最后只取前32个字符
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"doc_{digest}"


# ==================== AI修改 开始 ====================
# Chunk ID 同时供 Milvus 证据回溯和 Neo4j 关系引用，不能使用 Milvus 自增主键。
def build_chunk_id(
    document_id: str,
    chunk_index: int,
    section_path: str,
    content: str,
) -> str:
    identity = "\x1f".join(
        str(value or "").strip()
        for value in (document_id, chunk_index, section_path, content)
    )
    return f"chunk_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"
# ==================== AI修改 结束 ====================


# ==================== AI修改 开始 ====================
# 该函数用来获取上传文件的根目录以及相对目录,判断上传文件是否在目录中
def build_local_source_identity(source_root: str | Path, source_file: str | Path) -> tuple[str, str]:
    """把本地原始目录和文件转换为稳定的来源 ID 与相对路径。"""
    #expanduser()将路径中的~展开为完整路径 reslove()解析路径中的所有符号链接,返回绝对路径
    root = Path(source_root).expanduser().resolve()
    file_path = Path(source_file).expanduser().resolve()
    try:
        #计算file_path相对于root的路径
        relative_path = file_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"文件不在来源目录内: {file_path}") from exc
    # source_id 标识整个来源目录，source_path 只标识目录内的相对文件。
    # as_posix()将相对路径转化为posix格式字符串使用 / 的字符串 返回的就是根目录 和 根目录下一级到文件目录的路径
    return root.as_posix(), relative_path.as_posix()



def build_upload_source_identity(filename: str | None) -> tuple[str, str]:
    """把浏览器上传文件转换为不包含随机任务目录的稳定身份。"""
    normalized = str(filename or "").replace("\\", "/")
    # rsplit("/", 1)从右边开始分割字符串,只分割一次,[-1]取分割后的最后一个元素,相当于取到了路径的最后一个/的内容
    safe_filename = normalized.rsplit("/", 1)[-1].strip()
    # .和..相对目录不能当做正常文件名
    if not safe_filename or safe_filename in {".", ".."}:
        raise ValueError("上传文件名不能为空")
    # 浏览器上传没有原始目录，固定来源类型并用文件名做文档级替换键。
    return "browser-upload", safe_filename


# ==================== AI修改 结束 ====================


def escape_milvus_string(value: str) -> str:
    """转义 Milvus filter 字符串中的反斜杠和引号。"""
    return str(value or "").replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')


class KnowledgeChunkStore:
    """统一知识切片的写入和批量删除接口。"""

    def __init__(
        self,
        client=None,
        collection_name: str | None = None,
        # Callable代表可调用的东西常见就是一个函数或者可调用对象,这个函数接收一个str参数,并且返回None
        ensure_loaded: Callable[[str], None] | None = None,
    ):
        self.client = client or get_milvus_client()
        self.collection_name = collection_name or MilvusConfig.milvus_chunks_collection
        self._ensure_loaded = ensure_loaded or ensure_collection_loaded

    def ensure_collection(self, dim: int) -> None:
        # 统一表只在第一次导入时创建；后续导入复用同一个 schema 和索引。
        if not self.client.has_collection(collection_name=self.collection_name):
            schema = _build_schema_for_client(self.client, dim)
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 120, "nprobe": 12},
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "NORMALIZE": True,
                    "quantization": "none",
                },
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"统一知识切片集合创建完成: {self.collection_name}")
        # 删除和查询都要求 Collection 已加载，统一在存储层保证这个前置条件。
        self._ensure_loaded(self.collection_name)

    # 当用户上传同一个文档的时候会删除所有旧文档,插入新文档的所哟chunks
    def replace_document(
        self,
        document_id: str,
        # Mapping只读映射接口,告诉调用者,只读取数据,不会进行修改
        chunks: list[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not str(document_id or "").strip():
            raise ValueError("document_id不能为空")
        if not chunks:
            raise ValueError("chunks不能为空")

        metadata = dict(metadata or {})
        prepared_chunks = []
        for index, chunk in enumerate(chunks):
            # 先验证向量，再合并文档元数据；这样失败时不会先删除旧文档。
            if not chunk.get("dense_vector"):
                raise ValueError("chunk缺少dense_vector")
            if not chunk.get("sparse_vector"):
                raise ValueError("chunk缺少sparse_vector")
            # 复制一份默认字段返回一个新的对象,浅拷贝
            prepared = dict(_DEFAULTS)
            # update 存在即更新键值对,不存在则创建
            prepared.update(dict(chunk))
            prepared.update(
                {
                    key: value
                    for key, value in metadata.items()
                    if value is not None
                }
            )
            prepared["document_id"] = document_id
            # 兼容旧切片输入，但统一只保留 section_path/chunk_index。
            prepared["section_path"] = prepared.get("section_path") or chunk.get("title", "")
            prepared["chunk_index"] = int(chunk.get("chunk_index", index))
            prepared["chunk_id"] = prepared.get("chunk_id") or build_chunk_id(
                document_id,
                prepared["chunk_index"],
                prepared["section_path"],
                prepared.get("content", ""),
            )
            self._validate_chunk(prepared)
            # 只选取新 schema 认识的字段，避免来源适配器携带临时字段导致插入失败。
            prepared_chunks.append(
                {
                    field: prepared.get(field, _DEFAULTS.get(field))
                    for field in KNOWLEDGE_CHUNK_FIELDS
                }
            )

        dim = len(prepared_chunks[0]["dense_vector"])
        self.ensure_collection(dim)
        # 只有整篇新文档已经完成校验并准备好后，才删除同 document_id 的旧切片。
        # 这保证向量化或字段校验失败时，旧版本不会先被清空。
        # ==================== AI修改 开始 ====================
        # Milvus 没有按 document_id 的唯一约束；锁住 delete + insert，避免同一
        # 文档的两个导入线程交叉执行后各插入一份。
        # ==================== AI修改 结束 ====================
        with _document_replace_lock(document_id):
            self.client.delete(
                collection_name=self.collection_name,
                filter=f"document_id == '{escape_milvus_string(document_id)}'",
            )
            result = self.client.insert(
                collection_name=self.collection_name,
                data=prepared_chunks,
            )
        ids = result.get("ids") or []
        # Milvus 返回 ID 数量异常意味着写入结果不能和 chunk 一一对应，必须显式失败。
        if len(ids) != len(prepared_chunks):
            raise RuntimeError(
                f"Milvus返回ID数量异常: expected={len(prepared_chunks)}, actual={len(ids)}"
            )

        result_chunks = []
        for original, prepared, chunk_id in zip(chunks, prepared_chunks, ids):
            # 复制原始chunk
            enriched = dict(original)
            # 除了两个向量之外其他都放回chunk
            enriched.update(
                {
                    key: value
                    for key, value in prepared.items()
                    if key not in {"dense_vector", "sparse_vector"}
                }
            )
            enriched["id"] = chunk_id
            result_chunks.append(enriched)
        return {"chunks": result_chunks, "ids": ids}

    #根据文件id进行删除每个文件都会创建一个独一无二的document_id
    def delete_document(self, document_id: str):
        if not str(document_id or "").strip():
            raise ValueError("document_id不能为空")
        self._ensure_loaded(self.collection_name)
        # 一篇文档的全部切片共用 document_id，因此一次 filter 可以完成整篇删除。
        return self.client.delete(
            collection_name=self.collection_name,
            filter=f"document_id == '{escape_milvus_string(document_id)}'",
        )
    #根据文件来源类型进行删除
    def delete_source(self, source_type: str, source_id: str | None = None):
        if not str(source_type or "").strip():
            raise ValueError("source_type不能为空")
        # source_id 可选：传入时删除某个 Vault/项目，不传时删除整个来源类型。
        conditions = [f"source_type == '{escape_milvus_string(source_type)}'"]
        if source_id:
            conditions.append(f"source_id == '{escape_milvus_string(source_id)}'")
        return self._delete_by_filter(" and ".join(conditions))

    # 根据某个目录删除下面所有的文档,并且支持指定来源类型
    def delete_path_prefix(
        self,
        source_path: str,
        source_type: str | None = None,
        source_id: str | None = None,
    ):
        prefix = str(source_path or "").strip().replace("\\", "/").rstrip("/")
        if not prefix:
            raise ValueError("source_path不能为空")
        # 用目录前缀批量命中目录下所有文档，避免用户逐个寻找文件切片。
        conditions = [f"source_path like '{escape_milvus_string(prefix)}/%'"]
        if source_type:
            conditions.append(f"source_type == '{escape_milvus_string(source_type)}'")
        if source_id:
            conditions.append(f"source_id == '{escape_milvus_string(source_id)}'")
        return self._delete_by_filter(" and ".join(conditions))


    def _delete_by_filter(self, filter_expression: str):
        self._ensure_loaded(self.collection_name)
        return self.client.delete(
            collection_name=self.collection_name,
            filter=filter_expression,
        )

    #验证检查一个chunk是否符合milvus的长度要求
    @staticmethod
    def _validate_chunk(chunk: Mapping[str, Any]) -> None:
        for field, max_length in _STRING_LIMITS.items():
            value = chunk.get(field, "")
            if not isinstance(value, str):
                continue
            if len(value) > max_length:
                raise ValueError(
                    f"{field}超过统一 schema 限制: {len(value)} > {max_length}；"
                    "请在切片或来源解析阶段处理，不在入库层静默截断"
                )


def _build_schema_for_client(client, dim: int):
    """使用传入客户端建 schema，便于测试和多客户端场景复用。"""
    schema = client.create_schema(auto_id=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    # ==================== AI修改 开始 ====================
    # 主表只保留稳定 Chunk 身份和章节信息。
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="section_path", datatype=DataType.VARCHAR, max_length=2000)
    # ==================== AI修改 结束 ====================
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="source_path", datatype=DataType.VARCHAR, max_length=2000)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=5000)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=64)
    return schema


# ==================== AI修改 结束 ====================
