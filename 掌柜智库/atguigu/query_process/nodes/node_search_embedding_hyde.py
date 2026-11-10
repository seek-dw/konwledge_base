# atguigu/query_process/nodes/node_search_embedding_hyde.py
from langchain.chat_models import init_chat_model

from atguigu.config.config import ModelConfig, MilvusConfig
from atguigu.config.prompt import HYDE_PROMPT
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


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    实现思路:
        1.拿到item_name和rewritten_query之后判断并校验
        2.交给llm生成假设性答案
        3.将rewritten_query再次重写
        4.送入向量库查询得到搜索结果
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    # ==================== AI修改 开始 ====================
    # LLM客户端懒加载缓存: 原来每次process都init_chat_model重新建立客户端
    # (每次查询白白多一次连接/鉴权开销), 改成类级缓存只初始化一次, 后续复用。
    # 与 node_item_name_confirm / node_answer_output 同款优化。
    _llm = None
    # ==================== AI修改 结束 ====================

    def process(self, state: QueryGraphState):
        #校验
        item_names = state.get("item_names","")
        rewritten_query = state.get("rewritten_query","")
        # ==================== AI修改 开始 ====================
        # 1.宽松校验：宽泛问题可能没有具体主体名,只强制要求rewritten_query
        if not rewritten_query:
            logger.error("向量搜索节点接收意图识别数据失败")
            raise Exception("向量搜索节点接收意图识别数据失败")
        # 2.HyDE条件化：课程/题目检索不走HyDE
        # 假设性答案适合"知识点问答",对"课程列表/题目查询"这种结构化检索反而会跑偏
        query_type = state.get("query_type", "")
        if query_type in ("course", "question"):
            return {
                "hyde_embedding_chunks": []
            }
        # ==================== AI修改 结束 ====================

        #生成假设性答案
        # ==================== AI修改 开始 ====================
        # 类级懒加载缓存, 首次初始化后复用(省掉每轮重建HTTP客户端的开销)
        if self._llm is None:
            self._llm = init_chat_model(
                model=ModelConfig.LLM_MODEL_NAME,
                model_provider="openai",
                # ==================== AI修改 开始 ====================
                # HyDE 假设文档生成是文本调用，跟随当前平台的语言模型配置。
                api_key=ModelConfig.LLM_API_KEY,
                base_url=ModelConfig.LLM_BASE_URL,
                # ==================== AI修改 结束 ====================
                temperature=ModelConfig.MODEL_TEMPERATURE
            )
        llm = self._llm
        # ==================== AI修改 结束 ====================

        messages = [{
            "role":"user","content":HYDE_PROMPT.format(rewritten_query= rewritten_query)
        }]

        res = llm.invoke(input = messages)
        #Hypothetical Document Embeddings 假设性文档嵌入
        hyde_answer = res.content
        #重写rewritten_query
        merge_query = f"{rewritten_query}/{hyde_answer}"




        #向量化问题,进行混合搜索
        embed_query = vectorize_texts([merge_query])
        dense_data = embed_query.get("dense")[0]
        sparse_data = embed_query.get("sparse")[0]
        # 开始检索,首先创建混合检索请求
        # expr过滤字段需要字符串,所以item_names需要整理将里面的特殊字符转义
        # (意图识别出来的item_names经过item_name向量库匹配搜索到的所有主体识别名称)
        collection_name = MilvusConfig.milvus_chunks_collection

        # ==================== AI修改 开始 ====================
        # expr组合构建(与node_search_embedding保持一致,item_names可能为空)
        # 课程/题目已在上方跳过 HyDE；这里仅处理普通知识问题。
        expr_conditions = []
        if item_names:
            item_expr = build_item_name_expr(item_names)
            if item_expr:
                expr_conditions.append(item_expr)
        expr = " and ".join(expr_conditions) if expr_conditions else None
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
            expr=expr
            # ==================== AI修改 结束 ====================
        )

        res = my_hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=[0.9, 0.1],
            # ==================== AI修改 开始 ====================
            # HyDE 与普通向量检索共用新集合和统一字段。
            output_fields=select_existing_output_fields(
                get_milvus_client(), collection_name, CHUNK_OUTPUT_FIELDS
            ),
            # ==================== AI修改 结束 ====================
            # ==================== AI修改 开始 ====================
            # limit 10 -> 20 (提升召回):与node_search_embedding保持一致,
            # 扩大HyDE这一路的候选池,让RRF融合和rerank有足够的候选可选
            limit=20
            # ==================== AI修改 结束 ====================
        )
        # ==================== AI修改 开始 ====================
        # HyDE 同样不能把技术实体名当成 Chunk 文档归属名；过滤零命中时放宽一次。
        if expr and not res[0]:
            logger.warning(
                f"HyDE 主体过滤零命中，放宽为全库检索: item_names={item_names}"
            )
            reqs = create_reqs(
                dense_data=dense_data,
                sparse_data=sparse_data,
                dense_anns_field="dense_vector",
                sparse_anns_field="sparse_vector",
                dense_param={"metric_type": "COSINE"},
                sparse_param={"metric_type": "IP"},
                expr=None,
            )
            res = my_hybrid_search(
                collection_name=collection_name,
                reqs=reqs,
                ranker=[0.9, 0.1],
                output_fields=select_existing_output_fields(
                    get_milvus_client(), collection_name, CHUNK_OUTPUT_FIELDS
                ),
                limit=20,
            )
        logger.info(
            f"HyDE 本地向量检索完成: collection={collection_name}, "
            f"filter={expr or 'none'}, hits={len(res[0])}"
        )
        # ==================== AI修改 结束 ====================

        hyde_embedding_chunks = [
            {
                # 把entity解包得到所有的output_fields
                **search_result.get("entity", {}),
                "score": search_result.get("distance"),
                "source": "local"
            }
            for search_result in res[0]
        ]

        return {
            "hyde_embedding_chunks": hyde_embedding_chunks
        }



if __name__ == '__main__':
    init_state = {
        "rewritten_query": "CNN是什么",
        "item_names": ["CNN"]
    }
    node = NodeSearchEmbeddingHyde()
    res = node.process(init_state)
    logger.info(convert_to_json(res))
