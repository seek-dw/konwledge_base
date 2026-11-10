# atguigu/import_process/nodes/node_item_name_recognition.py
import json
# ==================== AI修改 开始 ====================
import re
# ==================== AI修改 结束 ====================

from langchain.chat_models import init_chat_model
from pymilvus import DataType
from atguigu.config.config import ModelConfig, MilvusConfig
from atguigu.config.prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.bgem3_create_tool import  vectorize_texts
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_create import get_milvus_client, ensure_collection_loaded


class NodeItemNameRecognition(NodeBase):
    """
    主体识别节点：主体识别与标签提取
    上节通过获取图片清洗后的md文件进行切分后的到了该文件的所有的chunks
    本节的目的就是给整个文章增加一个主体识别
    具体实现过程:
    1.获取chunks
    2.对chunks进行自定义切片,拿到最有可能包含主体信息的chunks
    3.将切分后的chunks拼接成一个新的大chunks字符串
    4.将大的字符串chunks交给大模型并设置提示词让其识别其主体名称
    5.创建milvus客户端并创建表
    6.将识别获取的主体名称向量化后插入milvus库

    """

    # ==================== AI修改 开始 ====================
    # 主体识别的输入和输出都先经过纯函数处理，避免模型一次不稳定的返回
    # 直接污染所有 chunk 或破坏 Milvus 过滤表达式。使用静态方法是因为
    # 这些函数不依赖节点实例状态，但仍属于主体识别节点的内部契约。
    _ITEM_NAME_PREFIX_RE = re.compile(
        r"^(?:主体名称|核心主体|商品名称|名称|item[_ -]?name)\s*[:：]\s*",
        re.IGNORECASE,
    )
    _UNKNOWN_ITEM_NAMES = {"不确定", "无法确定", "无法识别", "未知", "unknown", "none", "null"}

    @staticmethod
    def build_item_name_evidence(chunks, file_title: str, max_chars: int = 12000) -> str:
        """Build bounded evidence from document metadata, headings, and representative chunks."""
        parts = [f"文件名：{file_title}"]
        # 给标题去重
        seen_titles = set()
        titles = []
        for chunk in chunks or []:
            section_path = str(
                chunk.get("section_path") or chunk.get("title") or ""
            ).strip()
            if section_path and section_path not in seen_titles:
                seen_titles.add(section_path)
                titles.append(section_path)
        # 拼标题
        if titles:
            parts.append("章节标题索引：\n" + "\n".join(f"- {title}" for title in titles))

        total = len(chunks or [])
        if total:
            # 选择代表性较强的chunk
            indexes = [0, 1, 2, total // 4, total // 2, (total * 3) // 4, total - 3, total - 2, total - 1]
            # dict.fromkeys 内置dict类的类方法,接收可迭代对象和值,返回一个字典,因为字典的Key不能重复所以可以去重+保持顺序
            selected_indexes = list(dict.fromkeys(index for index in indexes if 0 <= index < total))
            for index in selected_indexes:
                chunk = chunks[index]
                title = str(
                    chunk.get("section_path") or chunk.get("title") or ""
                ).strip()
                content = str(chunk.get("content") or "").strip()
                parts.append(f"片段 {index + 1}\n标题：{title}\n正文：{content}")

        evidence = ""
        for part in parts:
            if len(evidence) >= max_chars:
                break
            separator = "\n\n" if evidence else ""
            remaining = max_chars - len(evidence) - len(separator)
            if remaining <= 0:
                break
            evidence += separator + part[:remaining]
        return evidence

    @staticmethod
    def _compact_item_name_text(value: str) -> str:
        # casefold()大小写归一,和lower几乎没区别,只不过对ASCII字符处理的更彻底
        return re.sub(r"\s+", "", value).casefold()

    @staticmethod
    def normalize_item_name(raw_item_name, fallback: str, evidence: str, max_length: int = 100) -> str:
        """规范化模型输出，以后遇到了直接复制,并拒绝那些无法从提供的文档证据中得到支持的名称."""
        # fallback有值的话用fallback
        fallback_text = re.sub(r"\s+", " ", str(fallback or "").strip())[:max_length].rstrip()
        if isinstance(raw_item_name, (list, tuple)):
            raw_item_name = next((item for item in raw_item_name if item), "")
        # 转字符串并且去两侧空白
        text = str(raw_item_name or "").strip()
        # splitlines换行切割成列表, 只取模型输出中的第一行非空内容 next取第一个
        text = next((line.strip() for line in text.splitlines() if line.strip()), "")
        # 把这些符合正则的名称前缀都删掉替换为""只留下真正的名称
        text = NodeItemNameRecognition._ITEM_NAME_PREFIX_RE.sub("", text).strip()
        # 去除字符串两端的空格啥的
        text = text.strip("`\"'“”‘’ ")
        # 把连续空白变成一个空格
        text = re.sub(r"\s+", " ", text)
        # 去除字符串两端的句号、感叹号、问号、分号
        text = re.sub(r"[。！？；;]+$", "", text).strip()
        # 模型输出是不是无效名称
        if not text or text.casefold() in NodeItemNameRecognition._UNKNOWN_ITEM_NAMES:
            return fallback_text
        # 模型生成的名称是否存在于给模型的证据中?防止模型瞎编
        if NodeItemNameRecognition._compact_item_name_text(text) not in NodeItemNameRecognition._compact_item_name_text(str(evidence or "")):
            return fallback_text
        return text[:max_length].rstrip()

    @staticmethod
    def escape_milvus_string(value: str) -> str:
        """Escape a value embedded in a Milvus string filter expression."""
        return str(value).replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    # ==================== AI修改 结束 ====================

    name = "node_item_name_recognition"
    def process(self, state: ImportGraphState):

        # 1.获取chunks
        chunks = state.get("chunks","")
        if not chunks:
            logger.error("chunks为空")
            raise Exception("chunks为空")
        file_title = state.get("file_title","")
        if not file_title:
            logger.error("file_title is  empty")
            raise Exception("file_title is empty")


        # ==================== AI修改 开始 ====================
        # 2-3.同时提供章节标题索引和分布式正文证据；标题能覆盖主体只出现于
        # 中间章节的文档，固定上限避免长文档撑爆模型上下文。
        content_str = self.build_item_name_evidence(chunks, file_title)
        # ==================== AI修改 结束 ====================

        # 4.将大的字符串chunks交给大模型并设置提示词让其识别其主体名称

        # ==================== AI修改 开始 ====================
        # 模型调用失败时使用文件标题继续导入；单个文档的识别服务波动不应
        # 阻断整个导入任务。规范化同时保留名称内部空格并验证正文证据。
        try:
            llm = init_chat_model(
                model=ModelConfig.LLM_MODEL_NAME,
                model_provider="openai",
                # ==================== AI修改 开始 ====================
                # 主体识别是文本调用，使用当前平台的语言模型地址和 key。
                api_key=ModelConfig.LLM_API_KEY,
                base_url=ModelConfig.LLM_BASE_URL,
                # ==================== AI修改 结束 ====================
                temperature=ModelConfig.MODEL_TEMPERATURE,
            )
            messages = [
                {"role": "system", "content": ITEM_NAME_SYSTEM_PROMPT},
                {"role": "user", "content": ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                    file_title=file_title,
                    context=content_str,
                )},
            ]
        #原来这里使用提示词json格式,然后这里会做一个简单的json格式输出防御
        #现在替换为了正则,直接使用正则拿item_name,并加上N个防御手段
            response = llm.invoke(input=messages)
            # getattr(对象,属性名,默认值) 相当于res.content,如果没有content返回 "" 不报错
            raw_item_name = getattr(response, "content", "")
        except Exception as exc:
            logger.error(f"主体识别模型调用失败，回退到文件名: {exc}")
            raw_item_name = ""
        item_name = self.normalize_item_name(raw_item_name, file_title, content_str)
        # ==================== AI修改 结束 ====================

        #创建milvus客户端
        milvus_client = get_milvus_client()
        #校验
        if not milvus_client:
            logger.error("milvus_client 初始化失败")
            raise Exception("milvus_client 初始化失败")
        #创建表名
        collection_name = MilvusConfig.milvus_item_collection
        #判断表名是否已经存在,若不存在,则创建
        if not milvus_client.has_collection(collection_name):

            #创建结构
            schema = milvus_client.create_schema(
                auto_id = True,
            )
            schema.add_field(
                field_name = "id",
                datatype = DataType.INT64,
                is_primary = True
            ).add_field(
                field_name = "item_name",
                datatype = DataType.VARCHAR,
                max_length = 100
            ).add_field(
                field_name = "file_title",
                datatype = DataType.VARCHAR,
                max_length = 100
            ).add_field(
                field_name = "dense_vector",
                datatype = DataType.FLOAT_VECTOR,
                dim = 1024
            ).add_field(
                field_name = "sparse_vector",
                datatype = DataType.SPARSE_FLOAT_VECTOR
            )

            # 创建索引
            index_params = milvus_client.prepare_index_params()
            index_params.add_index(
                field_name = "dense_vector",
                index_type = "IVF_FLAT", #暴力检索
                metric_type = "COSINE",
                params = {"nlist":200,"nrobe":20}
            )
            index_params.add_index(
                field_name = "sparse_vector",
                index_type = "SPARSE_INVERTED_INDEX", #倒排索引
                metric_type = "IP",
                params = {

                    #稀疏检索算法
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    #归一化,让内积等价于余弦相似度
                    "normalize": True,
                    #关闭量化,保持原始精度, "sq8": 8-bit量化
                    "quantization": "none"
                }
            )
            #创建表完成
            milvus_client.create_collection(
                collection_name = collection_name,
                schema = schema,
                index_params = index_params
            )


        # ==================== AI修改 开始 ====================
        # 先完成向量化和字段校验，再删除旧记录。这样模型或 embedding 失败时
        # 不会先把已有主体记录删掉；过滤值统一转义，避免特殊名称破坏表达式。
        item_name_vector = vectorize_texts([item_name])
        dense_vectors = item_name_vector.get("dense") or []
        sparse_vectors = item_name_vector.get("sparse") or []
        if not dense_vectors or not sparse_vectors:
            raise ValueError("主体名称向量化结果为空")
        data = {
            "item_name": item_name,
            "file_title": file_title,
            "dense_vector": dense_vectors[0],
            "sparse_vector": sparse_vectors[0],
        }
        safe_item_name = self.escape_milvus_string(item_name)
        ensure_collection_loaded(collection_name)
        milvus_client.delete(collection_name, filter=f"item_name == '{safe_item_name}'")
        milvus_client.insert(collection_name=collection_name, data=data)
        # ==================== AI修改 结束 ====================

        #
        for chunk in chunks:
            chunk["item_name"] = item_name

        # with open(r"E:\AI大模型\第七阶段 掌柜智库\资料\05-设备手册汇总\doc\chunks_with_item_name.json", "w", encoding="utf-8") as f:
        #     f.write(json.dumps(chunks,ensure_ascii=False,indent=4))

        return {
            "item_name":item_name,
            "chunks" : chunks
        }



if __name__ == '__main__':
    node = NodeItemNameRecognition()

    # with open("./chunk.json","r",encoding="utf-8") as f:
    #     chunks_json = f.read()
    # chunks = json.loads(chunks_json)
    #
    with open("data/chunk.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    # print(chunks,type(chunks))
    init_state={
        "chunks":chunks,
        "file_title":"07【掌柜智库】【导入】文档切片"
    }
    res = node(init_state)
    logger.info(res)
