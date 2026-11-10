# atguigu/import_process/nodes/node_bge_embedding.py
import json

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.bgem3_create_tool import vectorize_texts
from atguigu.tool.convert_to_json import convert_to_json
from atguigu.tool.logger import logger


class NodeBGEEmbedding(NodeBase):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    1.获取加入了主体识别的chunks
    2.批量处理chunks进行向量化
    3.取出向量化后的稠密向量和稀疏向量
    4.将向量化后的结果添加到chunks中进行保存
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        # 1.获取加入了主体识别的chunks
        chunks = state.get("chunks")
        if not chunks:
            logger.error("chunks不能为空")
            raise Exception("chunks不能为空")

        # 2.批量处理chunks进行向量化
        for i in range(0,len(chunks),3):
            #拿到批量的chunks
            batch_chunks = chunks[i:i+3]
            #拿到批量chunks内容
            # ==================== AI修改 开始 ====================
            # 嵌入文本包含章节层级、主体和正文，保证每个切片自带结构上下文。
            batch_chunks_content = [
                f"{chunk.get('section_path', '')} {chunk.get('item_name', '')} {chunk.get('content')}"
                for chunk in batch_chunks
            ]
            # ==================== AI修改 结束 ====================
            #向量化批量chunks的内容
            batch_chunk_content_embed = vectorize_texts(batch_chunks_content)
            #取出稠密向量和稀疏向量
            for idx , chunk in enumerate(batch_chunks):
                chunk["dense_vector"] = batch_chunk_content_embed.get("dense")[idx]
                chunk["sparse_vector"] = batch_chunk_content_embed.get("sparse")[idx]

        # with open(r"E:\AI大模型\第七阶段 掌柜智库\资料\05-设备手册汇总\doc\chunk_vector.json","w",encoding="utf-8") as f:
        #     f.write(convert_to_json(chunks))

        return{
                "chunks":chunks
            }




if __name__ == '__main__':
    node=NodeBGEEmbedding()
    with open("./data/item_chunk.json","r",encoding="utf-8") as f:
        item_chunks = json.load(f)
    init_state={
            "chunks":item_chunks
    }
    res = node(init_state)
    logger.info(convert_to_json(res))
