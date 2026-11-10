import json

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from atguigu.config.config import MilvusConfig
from atguigu.tool.logger import logger

milvus_client = None

# ==================== AI修改 开始 ====================
# 正文检索和 HyDE 共用统一字段；教育专属信息从 MongoDB 按 document_id/chunk_id 读取。
CHUNK_OUTPUT_FIELDS = (
    "id", "chunk_id", "document_id", "source_type", "source_id",
    "source_path", "item_name", "content_type", "chunk_index",
    "section_path", "content",
)


def build_item_name_expr(item_names):
    """构造 Milvus 的主体过滤表达式，兼容字符串和名称列表。"""
    if isinstance(item_names, str):
        values = [item_names]
    else:
        values = [str(item) for item in (item_names or [])]
    values = [
        value.replace("\\", "\\\\").replace("'", "\\'")
        for value in values
        if value.strip()
    ]
    return f"item_name in {json.dumps(values, ensure_ascii=False)}" if values else None
# ==================== AI修改 结束 ====================


def get_milvus_client():
    global milvus_client
    if not milvus_client:
        milvus_client = MilvusClient(
            uri= MilvusConfig.milvus_url
        )
    return milvus_client


# ==================== AI修改 开始 ====================
# 幂等的"确保collection已加载"工具函数
# 背景: Milvus的collection必须load进内存才能做向量检索和按条件删除,
# 刚create_collection的新表、被release的表、Milvus重启后的表都处于
# "not loaded"状态,此时直接delete/search会报:
#   MilvusException: (code=101, message=collection not loaded)
# 修复策略:
# 1.先查get_load_state,只有非Loaded状态才真正发起load,避免每次调用
#   都多一次load请求(get_load_state是轻量元数据查询,开销极小)
# 2.任何异常都不向上抛——加载失败时让后续真正的业务操作去报错,
#   那里的错误信息更具体,不会误导排查方向
_loaded_check_cache = set()  # 本进程内已确认Loaded的表,跳过重复的状态查询

def ensure_collection_loaded(collection_name):
    """确保collection已加载进内存,未加载则自动load,可重复调用无副作用"""
    client = get_milvus_client()
    if collection_name in _loaded_check_cache:
        return
    try:
        state = client.get_load_state(collection_name=collection_name)
        # state为LoadState枚举: NotLoad / Loading / Loaded
        # (空值兼容: 某些pymilvus版本查询不到状态时按未加载处理)
        if state is None or "Loaded" not in str(state):
            client.load_collection(collection_name=collection_name)
            logger.info(f"collection {collection_name} 未加载,已自动load进内存")
        else:
            _loaded_check_cache.add(collection_name)
    except Exception as e:
        # 兜底: 状态查询失败(老版本无该方法等)就直接尝试load,
        # 已加载的表重复load在Milvus侧是幂等的,不会报错
        try:
            client.load_collection(collection_name=collection_name)
        except Exception as load_err:
            logger.warning(f"collection {collection_name} 加载检查失败: {load_err}, 原始异常: {e}")


# ==================== AI修改 开始 ====================
def select_existing_output_fields(client, collection_name, desired_fields):
    """按Milvus当前schema筛选查询字段，兼容旧集合缺少新元数据字段。"""
    try:
        description = client.describe_collection(collection_name=collection_name)
        existing_fields = {
            field.get("name")
            for field in description.get("fields", [])
            if field.get("name")
        }
        selected = [field for field in desired_fields if field in existing_fields]
        # 极少数旧Milvus版本describe_collection返回不完整时保留原列表，
        # 让底层报出真实错误，而不是在这里静默丢掉所有输出字段。
        return selected or list(desired_fields)
    except Exception as exc:
        logger.warning(f"读取collection字段失败，暂按目标字段查询: {collection_name}, {exc}")
        return list(desired_fields)
# ==================== AI修改 结束 ====================
# ==================== AI修改 结束 ====================

#创建混合检索请求
def create_reqs(dense_data,
                sparse_data,
                dense_anns_field = None,
                sparse_anns_field = None,
                dense_param = None,
                sparse_param = None ,
                limit=10,
                expr=None):


    #创建稠密向量请求
    dense_req = AnnSearchRequest(
    data= [dense_data],
    anns_field= dense_anns_field,#要请求的字段
    param= dense_param,
    limit= limit,
    expr= expr
    )
    #expr过滤表达式,类似于mysql里面的where
    sparse_req = AnnSearchRequest(
        data=[sparse_data],
        anns_field=sparse_anns_field,  # 要请求的字段
        param=sparse_param,
        limit=limit,
        expr=expr
    )
    return [dense_req,sparse_req]

#创建混合检索方法:
def my_hybrid_search(collection_name,reqs,ranker=(0.5,0.5),limit=10,output_fields=None):
    milvus_client = get_milvus_client()

    # ==================== AI修改 开始 ====================
    # 检索前确保collection已加载: 新建的表/Milvus重启后的表处于not loaded状态,
    # 直接检索会报collection not loaded。在此统一收口,所有调用
    # my_hybrid_search的查询节点(检索/主体确认/HyDE/评测)都自动受到保护
    ensure_collection_loaded(collection_name)
    # ==================== AI修改 结束 ====================

    """
    norm_score归一化,统一量纲:稠密向量的重排序分数和稀疏向量的重排序分数
    ranker[0],rank[1],对权重进行分配
    """
    #*nums权重分配顺序对应reqs的请求顺序
    weight_ranker = WeightedRanker(ranker[0],ranker[1],norm_score=True)

    res = milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs = reqs,
        ranker = weight_ranker,
        limit= limit,
        output_fields = output_fields #返回会包含id 和 距离,除了这些你想自定义返回的字段
    )
    return res
