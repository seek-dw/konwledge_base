# atguigu/query_process/nodes/node_item_name_confirm.py

import json
import re
from typing import Any
from langchain.chat_models import init_chat_model
from atguigu.config.config import ModelConfig, MilvusConfig
from atguigu.config.prompt import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, ITEM_NAME_EXTRACT_TEMPLATE
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.convert_to_json import convert_to_json
from atguigu.tool.bgem3_create_tool import vectorize_texts
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_create import create_reqs, my_hybrid_search
from atguigu.tool.mongo_client_tool import add_or_update_history, get_recent_history_list,update_item_names_and_rewritten_query


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称
    实现思路:
        1. 保存本轮用户问题到 MongoDB
        2. 获取最近 HISTORY_LIMIT 条历史
        3. 判断是否属于“关于历史对话本身”的元问题
        4. LLM 进行意图识别 + 主体识别 + 问题改写
        5. 如果是闲聊 → 直接返回 chitchat
        6. 如果识别出了主体 → 对主体进行 Milvus 软匹配
        7. score >= 0.5 的标准主体全部作为过滤条件
        8. 如果没有匹配到 ≥0.5 的主体 → 不加主体过滤，允许全库检索
        9. 本轮历史只回填当前 message_id
        10. 返回 query_type / rewritten_query / item_names 等状态
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    # ==================== AI修改 开始 ====================
    # 历史窗口大小统一读取配置，默认20条，约等于10轮完整问答。
    # 统一配置后，意图识别和答案生成不会出现记忆窗口不一致。
    HISTORY_LIMIT: int = ModelConfig.QUERY_HISTORY_LIMIT
    # ==================== AI修改 结束 ====================

    # LLM客户端懒加载缓存: 原来每次process都init_chat_model重新建立客户端
    # (每次查询白白多一次连接开销),改成类级缓存只初始化一次,后续复用
    _llm = None

    # ==================== AI修改 开始 ====================
    # 元对话问题检测: "我刚才问了什么/我们聊了什么/你还记得吗" 这类关于"对话本身"的问题
    # 直接路由 chitchat, 既不调LLM也不检索——否则会被知识库检索内容干扰答非所问
    # (用户实测: 问"我刚才问了什么"被检索到产品手册内容, 回答跑偏到好几轮前)。
    # 命中后由 answer_output 的闲聊分支带着完整历史回答, 准确又省时。
    _META_PATTERNS = [
        r"我刚才问(的|了|过)?(是)?(什么|啥)",
        r"(刚才|之前|上一轮)(我|我们|咱们)?(问|说|聊)(的|了|过)?(是)?(什么|啥)",
        r"(我们|咱们)(刚才|之前|刚刚)(聊|说|谈)(的|了|过)?(是)?(什么|啥)",
        r"刚才(聊|说)(的|了|过)?(是)?(什么|啥)",
        r"上一句(说|问)(的|过)?(是)?(什么|啥)",
        r"(你还记得|你记得)(我刚才|之前|我们|吗)",
        r"前面(问|聊)(的|了|过)?(是)?(什么|啥)",
        r"(上一轮|之前|前面)(说|问|聊)(的|过)?(那个|那些|的事|的问题|的话题)",
    ]
    _META_Q_RE = re.compile("|".join(_META_PATTERNS))
    # ==================== AI修改 结束 ====================

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 一.获取最近的20条历史会话记录,整理成一个字符串
        history_content, message_id, original_query, session_id = self.make_history_to_str(state)
        # ==================== AI修改 开始 ====================
        # 元对话问题短路(在LLM调用之前判断): 命中直接返回chitchat,
        # 省掉一次LLM调用+检索, answer_output会带完整历史回答
        if self._META_Q_RE.search(original_query or ""):
            logger.info(f"元对话问题命中, 直接走闲聊分支: {original_query[:30]}")
            return {
                "message_id": message_id,
                "original_query": original_query,
                "rewritten_query": original_query,
                "item_names": [],
                "answer": "",
                "query_type": "chitchat",
                # ==================== AI修改 开始 ====================
                # 元对话也读取扩大后的记忆窗口，才能回答“刚才聊了什么”。
                "history": get_recent_history_list(session_id, limit=self.HISTORY_LIMIT)
                # ==================== AI修改 结束 ====================
            }
        # ==================== AI修改 结束 ====================
        # 二.交给大模型进行产品名称确认,并判断输出结果格式,并进行解析拿到item_name和rewritten_query
        # ==================== AI修改 开始 ====================
        # 解析函数升级为三元组返回(query_type, item_name_list, rewritten_query)
        # 1) 输出query_type意图类型 2) JSON解析失败降级为chitchat而不是raise 3) LLM调用失败也降级
        query_type, item_name_list, rewritten_query = self.send_to_llm_and_parse_item_name(history_content, original_query)

        # ==================== AI修改 开始 ====================
        # 只有模型明确判定为闲聊才跳过检索；knowledge/doc 没有主体时仍走宽泛检索。
        # 否则“数据库连接失败怎么排查”这类问题会因没有产品名而被误当闲聊。
        if query_type == "chitchat":
        # ==================== AI修改 结束 ====================
            return {
                "message_id": message_id,          # 反馈信息的id
                "original_query": original_query,  # 原始问题
                "rewritten_query": original_query, # 闲聊不需要改写
                "item_names": [],                  # 无主体
                "answer": "",                      # answer留空,让answer_output走流式聊天分支
                "query_type": "chitchat",
                # ==================== AI修改 开始 ====================
                # 闲聊分支与知识库分支共用同一记忆窗口。
                "history": get_recent_history_list(session_id, limit=self.HISTORY_LIMIT)  # 历史记录
                # ==================== AI修改 结束 ====================
            }

        # 三.意图识别item_name匹配混合搜索主体识别,拿到{意图item_name,主体item_name,匹配分数}
        # ==================== AI修改 开始 ====================
        # 所有意图统一软匹配(分数>=0.5采纳作为过滤条件,匹配不上就不加过滤
        # 走全库自由检索),让检索和重排去决定相关性,绝不拦截用户
        # answer永远为空,代表一定会走检索分支
        if item_name_list:
            # llm识别出来的意图去milvus里面进行向量搜索返回一个列表
            all_searched_item_name_list = self.item_match_and_get_all_search_item_list(item_name_list)
            # 软匹配,只留下分数高于0.5的匹配结果
            matched = [item.get("search_item_name") for item in all_searched_item_name_list
                       if item.get("score", 0.0) >= 0.5]
            final_item_names = list(dict.fromkeys(matched))  # 去重保序
        else:
            # 没识别出主体:宽泛检索,不加item_name过滤
            final_item_names = []
        answer = ""  # 永不产生确认类反馈,answer为空让流程正常进检索
        # ==================== AI修改 结束 ====================

        # 四.对齐和过滤:根据匹配的分数, 根据是否存在, 以及分数大小, 决定最终返回的item_name以及answer变量,判断是否反馈给用户需要确认
        # answer, final_item_names = self.align_and_filter_for_distance(all_searched_item_name_list)
        # ==================== AI修改 开始 ====================
        # 原IMPROVE: 限制了只能在一个会话窗口问一种问题,否则会污染历史
        # 已修复: judge_answer_and_backfill_history已改为只回填本轮message_id
        # (不再批量覆盖最近20条),换主体提问不会把历史里的旧主体改掉
        # 五.保存反馈历史,以及回填本轮历史记录
        message_id = self.judge_answer_and_backfill_history(answer, final_item_names, message_id, rewritten_query,session_id)


        # ==================== AI修改 开始 ====================
        return {
            "message_id":message_id,#反馈信息的id
            "original_query":original_query,#原始问题
            "rewritten_query":rewritten_query,#重写的问题
            "item_names":final_item_names,#最终确定的商品名称
            "answer":answer,#反馈信息
            "query_type":query_type,#意图类型,路由器据此分流
            # ==================== AI修改 开始 ====================
            # 返回状态中的历史也使用统一配置，避免前端/后续节点拿到旧窗口。
            "history": get_recent_history_list(session_id, limit=self.HISTORY_LIMIT)  # 历史记录
            # ==================== AI修改 结束 ====================
        }
        # ==================== AI修改 结束 ====================

    def make_history_to_str(self, state: QueryGraphState) -> tuple[str, str, str, Any]:


        # 要进入该节点,说明用户已经提问,所以已经有了历史记录,所以拿去session_id以及原始问题origin_query
        session_id = state.get("session_id", "")
        if not session_id:
            logger.error("session_id 不存在")
            raise Exception("session_id 不存在")
        original_query = state.get("original_query", "")
        if not original_query:
            logger.error("用户原始问题不存在")
            raise Exception("用户原始问题不存在,请输入您的问题")

        # 将该问题存入历史记录,返回的是InsertOne的inserted_id也就是mongo自动生成的_id
        message_id = add_or_update_history(session_id, "user", original_query, msg_type="qa")

        # ==================== AI修改 开始 ====================
        # 原IMPROVE: 从当前session_id中抽取前20条历史消息,局限性很高,换个窗口历史就是0
        # 说明与改进: "换窗口历史归零"是多轮会话的语义设计——session_id就是一个会话,
        # 新开窗口=新会话,历史隔离是正确的(否则A会话的问题会污染B会话的意图识别)。
        # 真正的问题是10这个魔法数字散落各处且不可调,现统一抽成类常量HISTORY_LIMIT:
        # 1)意图识别原料的取数条数 2)答案生成时的历史窗口,两处保持一致
        # 想让模型"记得更久"改这一个常量即可
        history_list = get_recent_history_list(session_id=session_id, limit=self.HISTORY_LIMIT)

        history_content = ""
        for history in history_list:
            # 从history取出要给大模型意图识别的内容,这里取"role","text"
            content = f"{history.get("role", "")}:{history.get("text", "")}\n"
            # 字符串拼接
            history_content += content
        return history_content, message_id, original_query, session_id

    # ==================== AI修改 开始 ====================
    # 稳定解析LLM返回的JSON：去markdown壳 -> json.loads -> 正则提取兜底
    def extract_json_safe(self, text: str):
        if not text:
            return None
        text = text.strip()
        # 去掉 ```json / ```JSON / ``` 三种壳(开头的和结尾的分开去,大小写都防)
        if text.startswith("```"):
            text = re.sub(r"^```(json|JSON)?\s*", "", text)
            text = re.sub(r"```\s*$", "", text).strip()
        # 第一次直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 兜底：正则抓取第一个 {...} JSON对象(不管前后有什么废话都能抓到)
        # [\s\S]匹配空白字符或者非空白字符,比.多一个换行,所有任意都能匹配
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                # group(0)代表整个匹配到的字符串,group(1)表示()捕获的
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def send_to_llm_and_parse_item_name(self, history_content: str, original_query: str) -> tuple[str, list[Any], Any]:
        # ==================== AI修改 开始 ====================
        # 原IMPROVE: 每次运行都要重新初始化
        # 已改进: init_chat_model改为类级懒加载缓存,首次调用初始化,之后直接复用,
        # 省掉每次查询重建HTTP客户端/鉴权的开销
        if self._llm is None:
            self._llm = init_chat_model(
                model=ModelConfig.LLM_MODEL_NAME,
                model_provider="openai",
                # ==================== AI修改 开始 ====================
                # 主体确认是文本调用，跟随当前平台切换语言模型地址和 key。
                api_key=ModelConfig.LLM_API_KEY,
                base_url=ModelConfig.LLM_BASE_URL,
                # ==================== AI修改 结束 ====================
                temperature=ModelConfig.MODEL_TEMPERATURE
            )
        llm = self._llm
        # ==================== AI修改 结束 ====================
        messages = [
            {"role": "system", "content": ITEM_NAME_EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_content,
                original_query=original_query)
             }]

        try:
            res = llm.invoke(input=messages)
            res_dict = self.extract_json_safe(res.content)
        except Exception as e:
            # LLM调用本身失败(网络/超时),降级为闲聊而不是整链崩溃
            logger.error(f"意图识别LLM调用失败,降级为chitchat: {e}")
            return "chitchat", [], original_query

        # 解析失败同样降级为闲聊——大不了当普通对话回答,绝不报错
        if not isinstance(res_dict, dict):
            logger.warning(f"意图识别JSON解析失败,原始返回:{res.content},降级为chitchat")
            return "chitchat", [], original_query

        # 拿到意图类型并校验合法性
        query_type = res_dict.get("query_type", "knowledge")
        if query_type not in ("chitchat", "course", "question", "doc", "knowledge"):
            query_type = "knowledge"

        # 拿到意图识别和重写问题
        item_names = res_dict.get("item_names")
        if item_names:
            item_name_list = [item_name.replace(" ", "").replace("\n", "").replace("\t", "")
                              for item_name in item_names]
        else:
            item_name_list = []

        rewritten_query = res_dict.get("rewritten_query")
        if not rewritten_query:
            rewritten_query = original_query

        return query_type, item_name_list, rewritten_query
    # ==================== AI修改 结束 ====================

    def item_match_and_get_all_search_item_list(self, item_name_list: list[Any]) -> list[Any]:
        # 1.将准备好的意图识别列表向量化,返回的是一个字典列表,{"dense":[[],[],[]],"sparse":[[],[],[]]}
        embed_item_dict = vectorize_texts(item_name_list)
        # print(embed_item_dict)
        collection_name = MilvusConfig.milvus_item_collection

        all_searched_item_name_list = []
        # 遍历意图识别列表,并取出每一个的稠密和稀疏向量
        for idx, item_name in enumerate(item_name_list):
            dense_data = embed_item_dict.get("dense")[idx]
            sparse_data = embed_item_dict.get("sparse")[idx]
            # 准备好了query向量,准备混合搜索
            reqs = create_reqs(
                dense_data=dense_data,
                sparse_data=sparse_data,
                dense_anns_field="dense_vector",
                sparse_anns_field="sparse_vector",
                dense_param={"metric_type": "COSINE"},
                sparse_param={"metric_type": "IP"}
            )

            res = my_hybrid_search(
                collection_name=collection_name,
                reqs=reqs,
                ranker=(0.6, 0.4),
                limit=10,
                output_fields=["item_name"]
            )
            # print(convert_to_json(res))
            # 混合搜索的结果是一个二维列表,
            # data: [[{'id': 468284006386632269, 'distance': 0.3981636166572571, 'entity': {'item_name': 'HAK180烫金机'}}]]
            # res[0]取到一个意图识别的数据,是一个列表字典,里面包含了多个字典,代表一个意图识别里面的多个匹配,id 不同,得分不同
            # searched_item_names 是一个列表字典,里面包含多个字典
            search_item_names = [{
                "original_item_name": item_name,
                "search_item_name": item.get("entity", {}).get("item_name", ""),
                "score": item.get("distance")
            } for item in res[0]]
            # extend,将所有循环的列表字典拆出来一个个字典添加到searched_item_name_list中
            all_searched_item_name_list.extend(search_item_names)
        return all_searched_item_name_list

    # 旧的对齐逻辑,现在直接改版成answer永远为空,宽泛检索,意图识别到闲聊自动到输出节点
    # def align_and_filter_for_distance(self, all_searched_item_name_list: list[Any]) -> tuple[str, list[Any]]:
    #     # 拿到所有意图识别的匹配item_name以及评分,接下来对齐过滤
    #     # 确定没问题的搜索结果,可以直接用
    #     confirm_item_names = [
    #         item.get("search_item_name") for item in all_searched_item_name_list
    #         if item.get("score") >= 0.8
    #     ]
    #     # 模糊结果,得分不高不低,需要用户确认
    #     option_item_names = [
    #         item.get("search_item_name") for item in all_searched_item_name_list
    #         if item.get("score") >= 0.5 and item.get("score") < 0.8
    #     ]
    #     # 如果存在则赋值给最终的item_name
    #     if confirm_item_names:
    #         final_item_names = confirm_item_names
    #         answer = ""
    #     elif option_item_names:
    #         # ==================== AI修改 开始 ====================
    #         # 原question: 这里item为啥为空?只能为空?
    #         # 答案与改进: 不是只能为空。原设计置空是为了"先确认再检索"的保守策略,
    #         # 但置空意味着本轮完全不检索,用户必须再回一句才能拿到东西,体验割裂。
    #         # 改进: 直接把模糊候选放进final_item_names,检索时对所有候选同时过滤,
    #         # 用户确认的同时下一轮已经能拿到相关内容,确认与检索并行不冲突
    #         # (注: 当前主流程已改用统一软匹配,此方法保留作为兼容备用)
    #         final_item_names = option_item_names
    #         answer = f"请确认您要咨询的商品是哪一个{','.join(option_item_names)}"
    #     else:
    #         final_item_names = []
    #         answer = "无法识别您咨询的问题,请重新提问"
    #
    #     # 判断answer有没有值来处理历史记录
    #     # 如果answer有值,则代表向用户反馈了信息,则会存在历史记录,要添加历史记录
    #     # 如果没有,则证明意图识别匹配到了高分答案,直接输出,不添加历史记录
    #     # 不管有没有,最好都要重新更新item_names和rewritten_query
    #     return answer, final_item_names

    def judge_answer_and_backfill_history(self, answer: str, final_item_names: list[Any], message_id, rewritten_query,session_id) -> Any:
        if answer:
            # 保存这个反馈信息到历史记录中
            message_id = add_or_update_history(session_id, "assistant", answer, msg_type="chat")
        # ==================== AI修改 开始 ====================
        # 原QUESTION: 查询20条历史记录,为啥取20条?这样回填不会污染历史?为什么不是根据message_id更新
        # 已修复: 1)20条抽成HISTORY_LIMIT常量(见类属性,附选型理由)
        # 2)回填从"批量覆盖最近20条"改为只更新本轮message_id,彻底消灭历史污染
        # 只回填当前问题的主体名称和重写问题,避免了历史污染
        if message_id:
            update_item_names_and_rewritten_query([message_id], item_names=final_item_names, rewritten_query=rewritten_query)
        # ==================== AI修改 结束 ====================
        return message_id




if __name__ == '__main__':
    session_id = "test_001"
    add_or_update_history(session_id, "user", "咨询下烫金机。")
    add_or_update_history(session_id, "assistant", "您好。请问是哪个型号")
    add_or_update_history(session_id, "user", "hak180")
    add_or_update_history(session_id, "assistant", "具体有什么问题呢？")
    node = NodeItemNameConfirm()
    init_state = {
        "session_id": 1,
        "original_query": "如何购买ipad"
    }
    res = node(init_state)
    logger.info(convert_to_json(res))
