# ==================== AI修改 开始 ====================
# 整个文件为教育实战新增：教育数据导入器(CLI命令行工具,不走前端)
#
# 【定位】edu_parsers.py负责"拆"(文件->chunks),本文件负责"入"(chunks->数据库)。
# 它就是一条缩水版导入管线,跳过了docx解析/AI切分/LLM商品名识别这些
# 通用管线的步骤,直接做剩下三件事:
#   向量化 -> 注册主体名 -> chunks入库
#
# 【用法】
#   干跑验证(只解析不入库,几秒出结果):
#     python -m atguigu.edu_process.edu_import --course 课程介绍.md --question 题目资料.md --dry-run
#   真正导入(去掉--dry-run,预计十几分钟):
#     python -m atguigu.edu_process.edu_import --course 课程介绍.md --question 题目资料.md
#
# 【注意】正文 chunks 统一写入 knowledge_chunks_graph；旧集合保留不动，
#         首次使用新集合需要重新导入资料。
# ==================== AI修改 结束 ====================
import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import DataType

from atguigu.config.config import MilvusConfig
from atguigu.edu_process.edu_parsers import (
    extract_item_names,
    parse_course_md,
    parse_question_md,
)
# 关键复用: chunks入库直接调用统一知识存储模块
# (建表/按document_id替换/字段校验/批量插入和前端导入使用同一条路)
from atguigu.tool.knowledge_chunk_store import (
    KnowledgeChunkStore,
    build_chunk_id,
    build_document_id,
)
from atguigu.tool.bgem3_create_tool import vectorize_texts
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_create import get_milvus_client, ensure_collection_loaded
from atguigu.tool.mongo_client_tool import replace_education_metadata
from atguigu.import_process.nodes.node_graph_build import NodeGraphBuild

# 每批向量化3条。BGE-M3对批量大小敏感,3是与node_bge_embedding节点一致的稳妥值,
# 太大容易触发embedding服务限频/超时
BATCH_SIZE = 3
EDUCATION_CHUNK_SIZE = 4500


# ==================== AI修改 开始 ====================
def split_education_chunks(chunks):
    """在向量化前把超长教育正文切到统一 Milvus 上限以内。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=EDUCATION_CHUNK_SIZE,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", ";", " ", ""],
    )
    result = []
    for chunk in chunks:
        content = str(chunk.get("content") or "")
        pieces = splitter.split_text(content) if len(content) > EDUCATION_CHUNK_SIZE else [content]
        for piece in pieces:
            item = dict(chunk)
            item["content"] = piece
            item["section_path"] = item.get("section_path") or item.get("title", "")
            result.append(item)
    return result
# ==================== AI修改 结束 ====================


def vectorize_chunks(chunks):
    """
    批量给chunks补上稠密+稀疏两个向量。

    【为什么要两个向量】Milvus里做的是混合检索:
      dense_vector(稠密)  = 语义向量,"换个说法也能搜到"(语义泛化能力)
      sparse_vector(稀疏) = 关键词向量,"字面对上就搜到"(专有名词/编码精确匹配)
    课程编码、题目编码这种字符串靠稀疏向量兜底,自然语言问法靠稠密向量。

    【嵌入文本格式必须与查询侧一致】
    线上NodeBGEEmbedding入库时用的是 f"{item_name} {content}",
    检索时问题向量化的对象自然携带了主体名语义,两边格式一致相似度才有意义。
    """
    chunks = split_education_chunks(chunks)
    # range(0, len, BATCH_SIZE)生成每批的起点,经典分批套路
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]  # 切片取一批(最后一批可能不足3条,切片自动处理)
        texts = [
            f"{c.get('section_path') or c.get('title', '')} "
            f"{c.get('item_name', '')} {c.get('content', '')}"
            for c in batch
        ]
        vectors = vectorize_texts(texts)  # 一次调用同时返回dense和sparse两个列表
        # 向量按顺序回填到每个chunk上(enumerate对位回填)
        for idx, chunk in enumerate(batch):
            chunk["dense_vector"] = vectors.get("dense")[idx]
            chunk["sparse_vector"] = vectors.get("sparse")[idx]
        # 每50批打一条进度日志,2000条chunk不至于刷屏也不至于看不到进度
        if (i // BATCH_SIZE) % 50 == 0:
            logger.info(f"向量化进度: {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")
    logger.info(f"向量化完成,共{len(chunks)}条")
    return chunks


def ensure_item_collection(milvus_client, dim=1024):
    """
    确保item主体库存在——"没有就建,有就跳过"(幂等)。

    为什么要在这里兜底建表? 因为教育数据可能比第一批普通文档更早导入,
    那时items表还没被node_item_name_recognition创建过,不建直接插会报错。
    表结构与NodeItemNameRecognition节点里完全一致,避免两处建出两个不同的表。
    """
    collection_name = MilvusConfig.milvus_item_collection
    if milvus_client.has_collection(collection_name=collection_name):
        return  # 已存在,什么都不做(这就是幂等: 重复执行无副作用)
    # auto_id=True: 主键id由Milvus自动生成,插入时不用传
    schema = milvus_client.create_schema(auto_id=True)
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True
    ).add_field(
        field_name="item_name",        # 主体名本身(题库名/系列名/文档商品名)
        datatype=DataType.VARCHAR,
        max_length=200
    ).add_field(
        field_name="file_title",       # 该主体是从哪个文件注册来的(溯源用)
        datatype=DataType.VARCHAR,
        max_length=200
    ).add_field(
        field_name="dense_vector",     # 主体名的语义向量(问题向量拿它来匹配)
        datatype=DataType.FLOAT_VECTOR,
        dim=dim                        # 1024 = BGE-M3的输出维度
    ).add_field(
        field_name="sparse_vector",    # 主体名的关键词向量
        datatype=DataType.SPARSE_FLOAT_VECTOR
    )
    # 索引决定检索速度和精度,参数与主管线保持一致
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="IVF_FLAT",         # 聚类索引,数据量不大时精度速度兼顾
        metric_type="COSINE",          # 余弦相似度(语义相似的标准度量)
        params={"nlist": 200, "nprobe": 20}  # 200个聚类中心,查询时探查20个
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",  # 稀疏倒排索引(关键词精确匹配)
        metric_type="IP",                    # 内积
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "normalize": True,
            "quantization": "none"
        }
    )
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params  # 建表同时建索引,一步到位
    )
    logger.info(f"item主体库{collection_name}创建完成")


def register_item_names(item_names, file_title):
    """
    把课程系列名/题库名逐个注册到item主体库。

    【逐个"先删后插"= 幂等】
    对每个名字: 先按item_name条件删除旧记录,再插入新记录。
    好处: 脚本重复跑多少次,items表里都不会出现同名主体的重复行。
    (这里没法像chunks那样按file_title一把删——一个文件贡献多个主体名,
    而items表里还混着其他文件注册的主体,只能按名单个清理)

    【性能权衡】每个名字单独调一次vectorize_texts,73个主体=73次调用,
    没有批量是因为删插逻辑按名字隔离,批量向量化和逐个删插交错写起来
    更容易出错,主体数量少(73个)不值得优化
    """
    milvus_client = get_milvus_client()
    if not milvus_client:
        raise Exception("milvus_client 初始化失败")
    collection_name = MilvusConfig.milvus_item_collection
    ensure_item_collection(milvus_client)

    # ==================== AI修改 开始 ====================
    # 删除前确保items表已load进内存,否则报collection not loaded(code=101)
    # (Milvus的collection必须load才能检索/条件删除,建表默认不load)
    ensure_collection_loaded(collection_name)
    # ==================== AI修改 结束 ====================

    for i, name in enumerate(item_names, start=1):
        # filter表达式是字符串拼接进Milvus查询的,名字里的引号/反斜杠
        # 必须转义,否则轻则语法错、重则filter注入(和SQL注入同一个道理)
        safe_name = name.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        milvus_client.delete(
            collection_name=collection_name,
            filter=f"item_name == '{safe_name}'"
        )
        vectors = vectorize_texts([name])  # 主体名单条向量化
        milvus_client.insert(
            collection_name=collection_name,
            data={
                "item_name": name,
                "file_title": file_title,
                "dense_vector": vectors.get("dense")[0],
                "sparse_vector": vectors.get("sparse")[0],
            }
        )
        if i % 50 == 0:
            logger.info(f"主体注册进度: {i}/{len(item_names)}")
    logger.info(f"item主体库注册完成,共{len(item_names)}个主体")


def import_chunks(chunks, file_title, source_path=None, source_id=None):
    """
    教育 chunk 直接调用统一知识存储模块。

    教育来源和普通文档使用同一张 knowledge_chunks_graph 表，课程/题目差异
    只保留在 content_type 和 source_type 元数据中。
    """
    # ==================== AI修改 开始 ====================
    source_path = str(source_path or file_title)
    source_id = str(source_id or file_title)
    document_id = build_document_id("education", source_id, source_path)
    content_type = chunks[0].get("content_type") or "doc"
    metadata_rows = []
    for index, chunk in enumerate(chunks):
        section_path = chunk.get("section_path") or chunk.get("title") or file_title
        chunk["document_id"] = document_id
        chunk["chunk_index"] = index
        chunk["section_path"] = section_path
        chunk["chunk_id"] = chunk.get("chunk_id") or build_chunk_id(
            document_id, index, section_path, chunk.get("content", "")
        )
        metadata_rows.append({
            "document_id": document_id,
            "chunk_id": chunk["chunk_id"],
            "chunk_index": index,
            "content_type": content_type,
            "domain_metadata": {
                key: chunk.get(key, "")
                for key in (
                    "course_name", "course_code", "chapter_name", "course_category",
                    "target_users", "learning_goals", "project_name",
                    "question_bank_name", "question_bank_code", "question_code",
                    "question_type", "code", "q_type",
                )
                if chunk.get(key, "")
            },
        })
    KnowledgeChunkStore(
        collection_name=MilvusConfig.milvus_chunks_collection,
    ).replace_document(
        document_id=document_id,
        chunks=chunks,
        metadata={
            "document_id": document_id,
            "source_type": "education",
            "source_id": source_id,
            "source_path": source_path,
            "content_type": content_type,
        },
    )
    replace_education_metadata(document_id, metadata_rows)
    NodeGraphBuild().process({
        "document_id": document_id,
        "source_type": "education",
        "source_id": source_id,
        "source_path": source_path,
        "content_type": content_type,
        "chunks": chunks,
    })
    # ==================== AI修改 结束 ====================
    logger.info(f"{file_title}导入完成,共{len(chunks)}条chunk")


def main():
    """
    CLI入口。argparse自动解析命令行参数并生成--help文档。
    --dry-run是action="store_true"型开关: 带上就是True,不带就是False
    """
    parser = argparse.ArgumentParser(description="教育数据导入工具")
    parser.add_argument("--course", required=True, help="课程介绍.md路径")
    parser.add_argument("--question", required=True, help="题目资料.md路径")
    parser.add_argument("--dry-run", action="store_true", help="只解析不入库")
    args = parser.parse_args()

    # ---- 第1步: 确定性解析(纯本地文件操作,不碰任何数据库) ----
    course_chunks = parse_course_md(args.course)
    question_chunks = parse_question_md(args.question)
    total = len(course_chunks) + len(question_chunks)
    logger.info(f"解析完成: 课程系列{len(course_chunks)}个, 题目{len(question_chunks)}道, 共{total}条chunk")

    # dry-run到此为止: 只验证解析结果对不对,零风险
    if args.dry_run:
        logger.info("dry-run模式,跳过向量化与入库")
        return

    # ---- 第2步: 批量向量化(耗时大头,2000条x双向量) ----
    logger.info("开始向量化课程chunk...")
    course_chunks = vectorize_chunks(course_chunks)
    logger.info("开始向量化题目chunk...")
    question_chunks = vectorize_chunks(question_chunks)

    # ---- 第3步: 主体名注册(课程系列名+题库名进items表) ----
    item_names = extract_item_names(course_chunks, question_chunks)
    logger.info(f"待注册主体{len(item_names)}个")
    register_item_names(item_names, "教育课程与题库")

    # ---- 第4步: chunks入库(分两次调,课程/题目各自挂独立的file_title,
    #      这样"课程介绍"或"题目资料"可以单独重导而互不影响) ----
    import_chunks(course_chunks, "课程介绍", args.course, "course")
    import_chunks(question_chunks, "题目资料", args.question, "question")

    logger.info(f"教育数据全部导入完成: 主体{len(item_names)}个, chunk {total}条")


if __name__ == "__main__":
    main()
