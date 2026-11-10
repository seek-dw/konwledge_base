from langgraph.constants import END
from langgraph.graph import StateGraph

import os
from atguigu.query_process.nodes.node_answer_output import NodeAnswerOutput
from atguigu.query_process.nodes.node_item_name_confirm import NodeItemNameConfirm
from atguigu.query_process.nodes.node_rerank import NodeRerank
from atguigu.query_process.nodes.node_rrf import NodeRrf
from atguigu.query_process.nodes.node_search_embedding import NodeSearchEmbedding
from atguigu.query_process.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from atguigu.query_process.nodes.node_graph_search import NodeGraphSearch
from atguigu.query_process.nodes.node_web_search_mcp import NodeWebSearchMcp
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.convert_to_json import convert_to_json
from atguigu.tool.logger import logger
# ==================== AI修改 开始 ====================
# 共享查询 Runner 的初始化锁。
import threading
# ==================== AI修改 结束 ====================


class MainGraphRunner:
    # ==================== AI修改 开始 ====================
    # 查询图结构和节点可跨请求复用，避免每次请求都重新构建并编译 LangGraph。
    _shared_runner = None
    _runner_lock = threading.Lock()
    # ==================== AI修改 结束 ====================

    def __init__(self):
        self.builder = StateGraph(state_schema=QueryGraphState)
        self.add_nodes()
        self.add_edges()
        self.graph = None

    def add_nodes(self):
        # 添加节点，此处省略具体实现细节...
        self.builder.add_node(NodeItemNameConfirm.name,NodeItemNameConfirm())
        self.builder.add_node(NodeSearchEmbedding.name,NodeSearchEmbedding())
        self.builder.add_node(NodeSearchEmbeddingHyde.name,NodeSearchEmbeddingHyde())
        # ==================== AI修改 开始 ====================
        # 图检索只补充本地关系候选，正文仍从统一 Milvus 集合回溯。
        self.builder.add_node(NodeGraphSearch.name, NodeGraphSearch())
        # ==================== AI修改 结束 ====================
        self.builder.add_node(NodeWebSearchMcp.name,NodeWebSearchMcp())
        self.builder.add_node(NodeRrf.name,NodeRrf())
        self.builder.add_node(NodeRerank.name,NodeRerank())
        self.builder.add_node(NodeAnswerOutput.name,NodeAnswerOutput())

    def add_edges(self):
        # 添加边，此处省略具体实现细节...
        self.builder.set_entry_point(NodeItemNameConfirm.name)
        self.builder.add_conditional_edges(NodeItemNameConfirm.name,self.after_confirm_router)
        self.builder.add_edge(NodeSearchEmbedding.name,NodeRrf.name)
        self.builder.add_edge(NodeSearchEmbeddingHyde.name,NodeRrf.name)
        self.builder.add_edge(NodeGraphSearch.name, NodeRrf.name)
        self.builder.add_edge(NodeWebSearchMcp.name,NodeRrf.name)
        self.builder.add_edge(NodeRrf.name,NodeRerank.name)
        self.builder.add_edge(NodeRerank.name,NodeAnswerOutput.name)
        self.builder.add_edge(NodeAnswerOutput.name,END)


    def after_confirm_router(self, state: QueryGraphState):
        # ==================== AI修改 开始 ====================
        # 意图路由分流：
        # 1.已有反馈话术(需要用户确认商品/拒答引导) → 直达答案节点吐出去
        # 2.闲聊(chitchat) → 直达答案节点走流式聊天分支,跳过全部检索
        # 3.其余(course/question/doc/knowledge) → 并行检索, 按 .env 开关动态组合:
        #    - embedding 本地检索必走(快, 毫秒级)
        #    - web_search 默认关闭(ENABLE_WEB_SEARCH=false): 每轮重建MCP连接+网络
        #      往返是回答慢的元凶之一, 本地知识库为主时收益低风险大
        #    - hyde 默认关闭(ENABLE_HYDE=false): HyDE每轮调用一次LLM生成假设性
        #      答案再检索, 额外3-5秒延迟, 是回答慢的元凶。本地知识库问答场景
        #      embedding检索已足够, 如需开启在.env设 ENABLE_HYDE=true
        answer = state.get("answer","")
        if answer:
            return NodeAnswerOutput.name
        if state.get("query_type") == "chitchat":
            return NodeAnswerOutput.name
        # ==================== AI修改 开始 ====================
        # 图检索是本地检索的固定补充路由；没有实体时节点自行返回空列表。
        routes = [NodeSearchEmbedding.name, NodeGraphSearch.name]
        # ==================== AI修改 结束 ====================
        if os.getenv("ENABLE_WEB_SEARCH", "false").strip().lower() in ("1", "true", "yes", "on"):
            routes.append(NodeWebSearchMcp.name)
        if os.getenv("ENABLE_HYDE", "false").strip().lower() in ("1", "true", "yes", "on"):
            routes.append(NodeSearchEmbeddingHyde.name)
        return routes
        # ==================== AI修改 结束 ====================

    def run(self,state):
        if self.graph is None:
            self.graph = self.builder.compile()
        return self.graph.invoke(state)
    @classmethod
    def create_and_run(cls,state):
        # ==================== AI修改 开始 ====================
        # 首次请求初始化共享 Runner，后续请求只复用已编译的工作流。
        with cls._runner_lock:
            if cls._shared_runner is None:
                cls._shared_runner = cls()
        return cls._shared_runner.run(state)
        # ==================== AI修改 结束 ====================



if __name__ == '__main__':
    init_state = {
        "answer":"asjkdhaksj"
    }
    res = MainGraphRunner.create_and_run(init_state)
    logger.info(convert_to_json(res))

