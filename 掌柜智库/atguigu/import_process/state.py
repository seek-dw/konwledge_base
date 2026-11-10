# atguigu/import_process/state.py

from typing import TypedDict

class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有节点产生和消费的数据字段
    """
    task_id: str # 任务唯一ID，用于追踪日志

    #流程控制标记
    is_md_read_enabled: bool    # 是否启用 Markdown 读取路径
    is_pdf_read_enabled: bool   # 是否启用 PDF 读取路径
    # ==================== AI修改 开始 ====================
    # 教育实战：docx 支持（项目文档/课程文档大量为docx格式）
    is_docx_read_enabled: bool  # 是否启用 DOCX 读取路径
    docx_path: str              # DOCX 文件路径 (如果输入是docx)
    # ==================== AI修改 结束 ====================

    # 路径相关
    local_dir: str  # 当前工作目录或输出目录
    local_file_path: str    # 原始输入文件路径
    file_title: str # 文件标题（文件名去后缀）
    # ==================== AI修改 开始 ====================
    # 统一知识表的文档级身份；同一篇文档产生的所有 chunk 共用 document_id。
    document_id: str
    source_type: str  # document / education / obsidian / 后续扩展来源
    source_id: str  # 来源实例，例如某个 Vault、项目或数据源
    source_path: str  # 来源内部的稳定路径
    # ==================== AI修改 结束 ====================
    pdf_path: str   # PDF 文件路径 (如果输入是PDF)
    md_path: str    # Markdown 文件路径 (转换后或直接输入的)

    # 内容数据
    md_content: str # Markdown 的全文内容
    chunks: list    # 切片列表
    item_name:str # 识别主体的名称（例如：万用表）

    # 数据库关联
    embeddings_content: list # 包含向量数的列表 ，准备写入 Milvus
