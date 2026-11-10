# atguigu/query_process/nodes/node_search_embedding.py
from atguigu.config.config import MilvusConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.bgem3_create_tool import vectorize_texts
from atguigu.tool.convert_to_json import convert_to_json
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_create import (
    # ==================== AI修改 开始 ====================
    CHUNK_OUTPUT_FIELDS,
    build_item_name_expr,
    # ==================== AI修改 结束 ====================
    create_reqs,
    get_milvus_client,
    my_hybrid_search,
    select_existing_output_fields,
)


# ==================== AI修改 开始 ====================
def education_collection_is_available(client, collection_name: str) -> bool:
    """查询前确认教育表存在，避免Milvus 100 collection not found直接冒泡。"""
    try:
        return bool(client and client.has_collection(collection_name=collection_name))
    except Exception as exc:
        logger.warning(f"检查教育collection失败: {collection_name}, {exc}")
        return False


# ==================== AI修改 结束 ====================


class NodeSearchEmbedding(NodeBase):
    """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    实现思路:
        1.item_names,rewritten_query二者不为空,说明有值,说明answer为空,
        需要根据重写的问题和意图识别进行多路搜索,这是第一个向量搜索
        2.根据二者进行向量搜索,拿到自己需要的向量库的数据和字段
        3.将数据整理成统一的key方便后续rrf倒排融合

    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState):
        # 拿到上一轮的意图识别和改写后的用户问题
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names", "")
        # ==================== AI修改 开始 ====================
        # 宽松校验：只强制要求rewritten_query
        # course/question等宽泛问题可能没有具体主体名(如"有哪些Python课程"),item_names允许为空
        if not rewritten_query:
            logger.info("向量搜索节点接收意图识别数据失败")
            raise Exception("向量搜索节点接收意图识别数据失败")
        query_type = state.get("query_type", "")
        # ==================== AI修改 结束 ====================

        # 虽然只有一个rewritten_query ,但是vectorize_texts函数需要列表
        embed_query = vectorize_texts([rewritten_query])
        # 获取重写的问题的稀疏稠密
        dense_data = embed_query.get("dense")[0]
        sparse_data = embed_query.get("sparse")[0]
        # 拿到要检索的表名
        # ==================== AI修改 开始 ====================
        # 所有正文来源都走统一新集合，课程/题目只通过 content_type 过滤。
        collection_name = MilvusConfig.milvus_chunks_collection
        # ==================== AI修改 结束 ====================

        # ==================== AI修改 开始 ====================
        # 教育资料尚未成功导入时，返回空召回让后续节点走友好兜底，
        # 不再把“表不存在”变成整条问答任务的Milvus异常。
        if not education_collection_is_available(get_milvus_client(), collection_name):
            logger.warning(
                f"知识切片集合尚未创建: {collection_name}，请先完成资料导入"
            )
            return {"embedding_chunks": []}
        # ==================== AI修改 结束 ====================

        # 开始检索,首先创建混合检索请求
        # expr过滤字段需要字符串,所以item_names需要整理将里面的特殊字符转义
        # (意图识别出来的item_names经过item_name向量库匹配搜索到的所有主体识别名称)
        # ==================== AI修改 开始 ====================
        # expr 组合构建：item_name过滤(有主体时) + content_type过滤(课程/题目意图时)
        # 关键修复(2026-08-20): question/course意图下, 教育数据chunks的item_name是
        # 题库名(如"尚硅谷大模型技术之Python课后练习题"), 而item_db向量匹配返回的
        # 可能是课程简称(如"Python")——两者不一致时 item_name in [...] 会把所有
        # 题目chunks全部过滤掉, 返回0条 → node_rrf两路全空 → 崩溃/"未找到"。
        # 修复: question/course意图不叠加item_name过滤, 仅用content_type缩小范围,
        # 由向量相似度+rerank保证相关性。item_name过滤仅用于doc/knowledge意图
        # (查特定产品文档时, item_name精确匹配是必要的)。
        expr_conditions = []
        item_expr = None
        if query_type in ("doc", "knowledge") and item_names:
            item_expr = build_item_name_expr(item_names)
            if item_expr:
                expr_conditions.append(item_expr)
        if query_type == "course":
            expr_conditions.append("content_type == 'course_intro'")
        elif query_type == "question":
            expr_conditions.append("content_type == 'question'")
        expr = " and ".join(expr_conditions) if expr_conditions else None
        if item_expr:
            fallback_expr = " and ".join(
                condition for condition in expr_conditions if condition != item_expr
            ) or None
        else:
            fallback_expr = expr
        # ==================== AI修改 结束 ====================
        reqs = create_reqs(
            dense_data=dense_data,
            sparse_data=sparse_data,
            dense_anns_field="dense_vector",
            sparse_anns_field="sparse_vector",
            dense_param={
                "metric_type": "COSINE"
            },
            sparse_param={
                "metric_type": "IP"
            },
            # ==================== AI修改 开始 ====================
            # expr由原来的必传item_name过滤改为组合过滤(可为None)
            expr=expr
            # ==================== AI修改 结束 ====================
        )

        # ==================== AI修改 开始 ====================
        output_fields = select_existing_output_fields(
            get_milvus_client(), collection_name, CHUNK_OUTPUT_FIELDS
        )
        # ==================== AI修改 结束 ====================
        res = my_hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=[0.9, 0.1],
            # ==================== AI修改 开始 ====================
            # output_fields 增加教育元数据字段,供答案生成时引用来源(课程名/题库名/题目编码/题型)
            output_fields=output_fields,
            # ==================== AI修改 结束 ====================
            # ==================== AI修改 开始 ====================
            # limit 10 -> 20 (提升召回):
            # rerank节点的断崖检测上限RERANK_MAX_TOPK=20,但每路只召回10条,
            # 两路RRF去重后本地候选经常不足20,断崖检测"有力没处使"。
            # 初召回扩大到20,给重排模型更大的候选池,正确答案更容易进入最终TopK
            limit=20
            # ==================== AI修改 结束 ====================
        )
        # ==================== AI修改 开始 ====================
        # item_name 是文档归属名，而主体识别返回的是技术实体名；两者不一致时，
        # 精确过滤会把本地相关 Chunk 全部过滤掉，零命中时退回无过滤向量检索。
        if item_expr and not res[0]:
            logger.warning(
                f"本地向量检索主体过滤零命中，放宽为全库检索: item_names={item_names}"
            )
            reqs = create_reqs(
                dense_data=dense_data,
                sparse_data=sparse_data,
                dense_anns_field="dense_vector",
                sparse_anns_field="sparse_vector",
                dense_param={"metric_type": "COSINE"},
                sparse_param={"metric_type": "IP"},
                expr=fallback_expr,
            )
            res = my_hybrid_search(
                collection_name=collection_name,
                reqs=reqs,
                ranker=[0.9, 0.1],
                output_fields=output_fields,
                limit=20,
            )
        logger.info(
            f"本地向量检索完成: collection={collection_name}, "
            f"filter={expr or 'none'}, hits={len(res[0])}"
        )
        # ==================== AI修改 结束 ====================

        embedding_chunks = [
            {
                #把entity解包得到所有的output_fields
                **search_result.get("entity",{}),
                "score":search_result.get("distance"),
                "source":"local"
            }
            for search_result in res[0]
        ]

        return {
            "embedding_chunks":embedding_chunks
        }

if __name__ == '__main__':
    init_state = {
        "rewritten_query":"CNN是什么",
        "item_names":["CNN"]
    }
    node = NodeSearchEmbedding()
    res = node.process(init_state)
    logger.info(convert_to_json(res))
