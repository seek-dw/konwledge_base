import threading
import time

from pymongo import MongoClient

from atguigu.config.config import MongoConfig
# ==================== AI修改 开始 ====================
# 引入logger: 写操作返回值校验失败时记录日志(原QUESTION的完善动作之一)
from atguigu.tool.logger import logger
# ==================== AI修改 结束 ====================



mongo_client = None
def mongo_client_create():
    global mongo_client
    if not mongo_client:
        mongo_client = MongoClient(MongoConfig.mongo_url)
    return mongo_client

collection = None
db = None
_education_metadata_collection = None
_knowledge_sync_collection = None

#定义获取mongo集合的函数,传入的参数就是集合的名字,默认值为"chat_history"
def get_mongo_collection(collection_name="chat_history"):
    global db
    global collection

    mongo_client = mongo_client_create()
    if db is None:
        #创建数据库
        db = mongo_client[MongoConfig.mongo_db_name]
    if collection is None:
        #创建表
        collection = db[collection_name]
        #给表创建索引
        #创建联合索引:代表_id 升序,ts降序,session_id升序索引结构
        collection.create_index([("_id",1),("ts",-1),("session_id",1)])
    return collection


# ==================== AI修改 开始 ====================
# 教育字段属于领域元数据，不进入 Milvus 主 Chunk 表。
def get_education_metadata_collection():
    global _education_metadata_collection
    if _education_metadata_collection is None:
        client = mongo_client_create()
        education_db = client[MongoConfig.mongo_db_name]
        _education_metadata_collection = education_db["education_metadata"]
        _education_metadata_collection.create_index(
            [("document_id", 1), ("chunk_id", 1)], unique=True
        )
    return _education_metadata_collection


def replace_education_metadata(document_id, rows):
    collection = get_education_metadata_collection()
    collection.delete_many({"document_id": document_id})
    if rows:
        collection.insert_many(rows)


def get_education_metadata(content_type, limit=500):
    return list(
        get_education_metadata_collection()
        .find({"content_type": content_type}, {"_id": 0})
        .sort("chunk_index", 1)
        .limit(limit)
    )
# ==================== AI修改 结束 ====================


# ==================== AI修改 开始 ====================
# 记录同一 document_id 在 Milvus/Neo4j 导入链中的最后同步状态，便于失败后定位。
def mark_knowledge_sync(
    document_id,
    status,
    task_id="",
    source_type="",
    source_id="",
    source_path="",
    error="",
):
    global _knowledge_sync_collection
    if _knowledge_sync_collection is None:
        client = mongo_client_create()
        _knowledge_sync_collection = client[MongoConfig.mongo_db_name]["knowledge_sync"]
        _knowledge_sync_collection.create_index("document_id", unique=True)
    _knowledge_sync_collection.update_one(
        {"document_id": document_id},
        {"$set": {
            "document_id": document_id,
            "status": status,
            "task_id": task_id,
            "source_type": source_type,
            "source_id": source_id,
            "source_path": source_path,
            "error": error,
            "updated_at": time.time(),
        }},
        upsert=True,
    )
# ==================== AI修改 结束 ====================

def get_recent_history_list(session_id,limit = 10):
    #获取mongo里面指定的集合,默认为"chat_history"
    collection = get_mongo_collection()
    #从该集合里面查找指定session_id的聊天记录,并按ts降序排列,取前limit条
    result = collection.find({"session_id":session_id}).sort("ts",-1).limit(limit)
    #返回的时一个Cursor对象,需要遍历才能获取具体内容
    """
    增删改查的每一步操作的返回值都是一个对象
    查询的时候返回的是游标对象Cursor,防止查询大量数据导致内存崩溃,需要遍历才能获取具体内容
    新增的时候返回的是InsertOneResult对象,里面包含了重要属性就是mongodb自动生成的_id,显示为inserted_id
    更新的时候返回的是UpdateResult对象,里面包含了matched_count,和modify_count,二者分别表示匹配到的文档数量和修改的文档数量
    删除的时候返回的是DeleteResult对象,里面包含了delete_count,表示的是删除的数量
    """
    # print(result)
    return list(result)


def add_or_update_history(session_id,role,text,rewritten_query=None,item_names=None,image_urls = None,msg_type=None,ts=None,_id=None):
    collection = get_mongo_collection()
    #如果_id存在,说明该次函数调用传入了_id,即更新操作
    if _id:
        data = {
            "_id":_id,
            "session_id":session_id,
            "role":role,
            "text":text,
            "rewritten_query":rewritten_query,
            "item_names":item_names,
            "ts":ts or time.time(),
            "image_urls":image_urls
        }
        #update_one进行更新操作:update_one(filter,update)
        collection.update_one({"_id":_id},{"$set":data})
        return _id
    #如果_id不存在,说明该次函数调用没有传入_id,即添加操作
    else:
        data = {
            "session_id":session_id,
            "role":role,
            "text":text,
            "rewritten_query":rewritten_query,
            "item_names":item_names,
            "ts":ts or time.time(),
            "image_urls": image_urls,
            # ==================== AI修改 开始 ====================
            # msg_type: 给每条历史打类型标签("qa"=知识库问答, "chat"=闲聊/引导话术),
            # 用于A档记忆分层注入时过滤闲聊噪声(更早的对话只保留qa类)。
            # 旧数据无此字段, 读取时用 .get("msg_type","qa") 兜底为qa, 兼容不丢历史。
            # ==================== AI修改 结束 ====================
            "msg_type": msg_type

        }
        # ==================== AI修改 开始 ====================
        # 原QUESTION: 添加的时候返回对象?更新会不会返回对象,查询和删除呢?返回什么对象?有什么用?
        # 统一回答(并顺手把所有写操作的返回值补全,调用方可以据此判断是否真的写成功):
        # - insert_one → InsertOneResult: .inserted_id是mongo自动生成的_id(本函数已返回),
        #   .acknowledged表示服务端是否确认写入
        # - update_one/update_many → UpdateResult: .matched_count匹配到的条数,
        #   .modified_count实际修改的条数(二者可能不等:值相同就不算修改)
        # - delete_one/delete_many → DeleteResult: .deleted_count删除的条数
        # - find/find_one → Cursor/None: Cursor是懒加载游标,遍历时才真正拉数据
        # 用处: 写库后校验(如deleted_count==0说明本来就没有数据可删,能提前发现
        # session_id传错之类的bug),下方的返回值改进就是按这个思路做的
        result = collection.insert_one(data)
        # AI修改: 返回前校验写入是否被服务端确认,失败时打日志(原来静默吞掉)
        if not result.acknowledged:
            logger.error(f"历史记录写入未被MongoDB确认: session_id={session_id}")
        return result.inserted_id

#通过session_id来及逆行数据库会话历史的删除操作
def clear_history(session_id):
    collection = get_mongo_collection()
    # ==================== AI修改 开始 ====================
    # AI修改: 原来不接收返回值,删除是否生效调用方无从知晓
    # delete_many返回DeleteResult.deleted_count(实际删除条数),0说明该会话本来就没有历史
    result = collection.delete_many({"session_id":session_id})
    return result.deleted_count
    # ==================== AI修改 结束 ====================


#重要节点:在大模型意图识别完成后返回的主体识别和问题重写豆芽更新到原来的历史记录当中
def update_item_names_and_rewritten_query(ids,item_names=None,rewritten_query = None):
    collection = get_mongo_collection()
    data = {
        "item_names":item_names,
        "rewritten_query":rewritten_query
    }
    # ==================== AI修改 开始 ====================
    # AI修改: 返回UpdateResult的matched_count/modified_count,回填是否命中一目了然
    # (matched=0说明message_id不对,能在测试阶段就暴露"回填落空"的问题)
    result = collection.update_many({"_id":{"$in":ids}},{"$set":data})
    return result.matched_count, result.modified_count
    # ==================== AI修改 结束 ====================


# ==================== AI修改 开始 ====================
# 长期记忆: 滚动摘要。短期记忆(get_recent_history_list)只取最近N条,更早的对话
# 直接丢失; 长期记忆把超阈值的早期对话压缩成一段摘要存到 chat_summary 集合,
# 注入prompt时用 [长期摘要 + 最近N轮原文], 让模型记得更久又不爆上下文。
# 注意: 不能复用 get_mongo_collection(name) —— 它的全局 collection 缓存有bug,
# 第一次建 chat_history 后, 传别的 name 仍返回 chat_history, 所以这里单独缓存。
_summary_col = None
def _get_summary_collection():
    global _summary_col
    if _summary_col is None:
        client = mongo_client_create()
        db = client[MongoConfig.mongo_db_name]
        _summary_col = db["chat_summary"]
        _summary_col.create_index("session_id", unique=True)
    return _summary_col

def get_session_summary(session_id):
    """读取某会话的长期记忆摘要, 没有则返回None。"""
    doc = _get_summary_collection().find_one({"session_id": session_id})
    return doc.get("summary") if doc else None

def upsert_session_summary(session_id, summary):
    """存/更新某会话的长期记忆摘要。"""
    _get_summary_collection().update_one(
        {"session_id": session_id},
        {"$set": {"session_id": session_id, "summary": summary, "updated_ts": time.time()}},
        upsert=True
    )

# ==================== AI修改 开始 ====================
# 压缩并发锁: 多轮快速问答时 answer_output 每轮都会异步启动一个压缩线程,
# 若上一轮的压缩线程还没跑完(LLM压缩耗时数秒), 下一轮又启动一个,
# 两个线程同时 count+取最早6条+删除 → 竞态可能把同一批删两次或删到错批,
# 严重时"上一轮"的记录被误删, 记忆断裂。加全局锁串行化压缩。
_compress_lock = threading.Lock()
# ==================== AI修改 开始 ====================
# 压缩失败退避: LLM(云端API)故障时(余额不足/网络异常/限流), 每轮问答都会异步启动
# 压缩线程, 若不做退避会每轮都失败并刷ERROR日志, 毫无意义且污染日志。
# 记录每个会话最近一次失败时间, 失败后 _COMPRESS_FAIL_BACKOFF_SECONDS(5分钟)内
# 跳过压缩尝试; 超过退避期后自动恢复正常压缩, 无需人工干预。
_COMPRESS_FAIL_BACKOFF: dict = {}
_COMPRESS_FAIL_BACKOFF_SECONDS = 300
# ==================== AI修改 结束 ====================


def compress_session_history(session_id, llm, keep_recent=12, compress_batch=6):
    """把超出 keep_recent 条的早期对话压缩成摘要。
    keep_recent: 保留最近多少条原文不压缩(默认12=6轮);
    compress_batch: 每次压缩多少条早期对话(默认6=3轮)。
    压缩成功后删除已压缩的原文(摘要已保存), 控制注入prompt的总长度。
    压缩失败不影响主流程, 只记日志。
    并发安全: 全局 _compress_lock 串行化, 防止多轮并发压缩竞态误删近期记录。
    """
    # blocking=false如果锁被其他线程占用,不等待
    if not _compress_lock.acquire(blocking=False):
        # 上一个压缩还在进行中(LLM压缩通常要几秒), 直接放弃本次,
        # 下次回答触发时再压, 避免堆积的压缩线程并发删除
        logger.info(f"长期记忆压缩已在执行中, 跳过本次: session={session_id}")
        return
    try:
        # 失败退避: 上次压缩失败后的退避期内直接跳过, 避免API故障时反复无效重试
        last_fail = _COMPRESS_FAIL_BACKOFF.get(session_id, 0)
        if time.time() - last_fail < _COMPRESS_FAIL_BACKOFF_SECONDS:
            return
        col = get_mongo_collection()
        total = col.count_documents({"session_id": session_id})
        if total <= keep_recent + compress_batch:
            return  # 还没到压缩门槛, 继续用全量原文
        # 取最早的一批(按ts升序)作为待压缩对象
        old_msgs = list(col.find({"session_id": session_id}).sort("ts", 1).limit(compress_batch))
        if not old_msgs:
            return
        existing = get_session_summary(session_id) or ""
        batch_text = "\n".join(f"{m.get('role')}: {m.get('text','')}" for m in old_msgs)
        compress_prompt = (
            "请把以下对话历史浓缩成一段简洁的要点摘要"
            "(保留关键事实、实体名称、用户意图、已得出的结论, 丢弃寒暄客套), "
            "中文, 不超过300字:\n\n" + batch_text
        )
        res = llm.invoke([{"role": "user", "content": compress_prompt}])
        new_summary = (existing + "\n" + res.content).strip() if existing else res.content.strip()
        upsert_session_summary(session_id, new_summary)
        # 摘要已保存, 删除已压缩的原文, 避免后续再次注入
        col.delete_many({"_id": {"$in": [m["_id"] for m in old_msgs]}})
        logger.info(f"长期记忆压缩完成: session={session_id}, 压缩{len(old_msgs)}条, 摘要{len(new_summary)}字")
    except Exception as e:
        # 压缩失败不阻塞主流程: 原文不删、摘要不写, 记忆不会丢;
        # 记WARNING并进入退避, 避免每轮问答都失败刷屏
        _COMPRESS_FAIL_BACKOFF[session_id] = time.time()
        logger.warning(f"长期记忆压缩失败({_COMPRESS_FAIL_BACKOFF_SECONDS}s内不再重试): session={session_id}, {e}")
    finally:
        _compress_lock.release()
# ==================== AI修改 结束 ====================
# ==================== AI修改 结束 ====================


if __name__ == '__main__':
    add_or_update_history("test_001", "user", "咨询下烫金机。")
    # add_or_update_history("test_001", "assistant", "您好。请问是哪个型号")
    # result = add_or_update_history("test_001", "user", "hak180")
    # print(result,type(result))
    # add_or_update_history("test_001", "assistant", "具体有什么问题呢？")

    # result = get_recent_history_list("test_001")
    # print(result)

    # clear_history("test_001")
