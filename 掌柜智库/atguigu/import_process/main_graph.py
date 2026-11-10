"""
编译节点:放在和node节点平齐的位置,用于将所有节点串联,此处使用类来整合并实现(优雅)
    1.init里面放入所有的方法调用函数
    2,定义多个方法,封装节点的添加,边的添加,图的运行
"""
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from atguigu.import_process.nodes.node_bge_embedding import NodeBGEEmbedding
from atguigu.import_process.nodes.node_document_split import NodeDocumentSplit
from atguigu.import_process.nodes.node_entry import NodeEntry
from atguigu.import_process.nodes.node_import_milvus import NodeImportMilvus
from atguigu.import_process.nodes.node_graph_build import NodeGraphBuild
from atguigu.import_process.nodes.node_item_name_recognition import NodeItemNameRecognition
from atguigu.import_process.nodes.node_md_img import NodeMDImg
from atguigu.import_process.nodes.node_pdf_to_md import NodePDFToMD
# ==================== AI修改 开始 ====================
# 教育实战新增：docx 转换节点
from atguigu.import_process.nodes.node_docx_to_md import NodeDocxToMD
# ==================== AI修改 结束 ====================
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger


class MainGraphRunner:
    def __init__(self):
        self.builder = StateGraph(state_schema=ImportGraphState)
        self.nodes = self.add_nodes()
        self.edges = self.add_edges()
        self.graph = None

    def add_nodes(self):
        self.builder.add_node(NodeEntry.name, NodeEntry())
        self.builder.add_node(NodePDFToMD.name, NodePDFToMD())
        self.builder.add_node(NodeMDImg.name, NodeMDImg())
        self.builder.add_node(NodeDocumentSplit.name, NodeDocumentSplit())
        self.builder.add_node(NodeItemNameRecognition.name, NodeItemNameRecognition())
        self.builder.add_node(NodeBGEEmbedding.name, NodeBGEEmbedding())
        self.builder.add_node(NodeImportMilvus.name, NodeImportMilvus())
        # ==================== AI修改 开始 ====================
        # Milvus 写入成功后建立图数据，保证图中的 Chunk 证据已经存在。
        self.builder.add_node(NodeGraphBuild.name, NodeGraphBuild())
        # ==================== AI修改 结束 ====================
        # ==================== AI修改 开始 ====================
        # 教育实战新增：docx 转换节点
        self.builder.add_node(NodeDocxToMD.name, NodeDocxToMD())
        # ==================== AI修改 结束 ====================

    def after_entry_router(self, state: ImportGraphState):
        if state.get("is_md_read_enabled"):
            return NodeMDImg.name
        if state.get("is_pdf_read_enabled"):
            return NodePDFToMD.name
        # ==================== AI修改 开始 ====================
        # 教育实战新增：docx 分支
        if state.get("is_docx_read_enabled"):
            return NodeDocxToMD.name
        # ==================== AI修改 结束 ====================
        else:
            return END

    # 涉及到了条件边,先写条件路由方法
    def add_edges(self):
        self.builder.add_edge(START, NodeEntry.name)
        self.builder.add_conditional_edges(NodeEntry.name, self.after_entry_router, {
            NodeMDImg.name: NodeMDImg.name,
            NodePDFToMD.name: NodePDFToMD.name,
            # ==================== AI修改 开始 ====================
            # 教育实战新增：docx 路由映射
            NodeDocxToMD.name: NodeDocxToMD.name
            # ==================== AI修改 结束 ====================
        })
        self.builder.add_edge( NodePDFToMD.name, NodeMDImg.name)
        # ==================== AI修改 开始 ====================
        # 教育实战新增：docx转换完md后与pdf一样,汇入 node_md_img 统一走后续链路
        self.builder.add_edge(NodeDocxToMD.name, NodeMDImg.name)
        # ==================== AI修改 结束 ====================
        self.builder.add_edge(NodeMDImg.name, NodeDocumentSplit.name)
        self.builder.add_edge(NodeDocumentSplit.name, NodeItemNameRecognition.name)
        self.builder.add_edge(NodeItemNameRecognition.name, NodeBGEEmbedding.name)
        self.builder.add_edge(NodeBGEEmbedding.name, NodeImportMilvus.name)
        # ==================== AI修改 开始 ====================
        self.builder.add_edge(NodeImportMilvus.name, NodeGraphBuild.name)
        self.builder.add_edge(NodeGraphBuild.name, END)
        # ==================== AI修改 结束 ====================

    def graph_run(self, state: ImportGraphState):
        # 懒加载,所以这里设置一个判断
        if not self.graph:
            self.graph = self.builder.compile()
        res = self.graph.invoke(state)
        return res

    # 直接类名cls.create_runner()调用
    @classmethod
    def create_runner(cls, state: ImportGraphState):
        return cls().graph_run(state)


if __name__ == '__main__':
    init_state = {
        "local_file_path": r"E:\AI大模型\第七阶段 掌柜智库\资料\05-设备手册汇总\doc\hak180产品安全手册.pdf",
        "local_dir" : r"E:\AI大模型\第七阶段 掌柜智库\资料\05-设备手册汇总\doc"
    }
    res = MainGraphRunner.create_runner(init_state)
    logger.info(res)
