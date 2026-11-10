import os
import tempfile
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv
# 扫描当前目录寻找.env文件.解析文件中的key=value,将这些键值对写入os.environ字典中
load_dotenv(override=True)

# ==================== AI修改 开始 ====================
# 将环境变量安全转换为正整数，避免配置为空或写错时导致服务启动失败。
def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default
# ==================== AI修改 结束 ====================

class MinerUConfig:
    """
    MinerU 配置类
    """
    MINERU_API_KEY = os.getenv("MINERU_API_KEY")

# ==================== AI修改 开始 ====================
# 语言模型和视觉模型共享“当前平台”选择，但保留各自的模型、地址和 key 字段。
# 这样只改 MODEL_PROVIDER 就能在魔搭/OpenRouter 间切换，业务节点不再绑定某个平台。
def resolve_model_profiles(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """根据环境变量解析当前语言模型和视觉模型的有效配置。"""
    # env存在用env不存在用环境变量
    values = os.environ if env is None else env
    # 在env或者环境变量中读取"MODEL_PROVIDER"开关,默认值设为魔搭
    provider = (values.get("MODEL_PROVIDER") or "modelscope").strip().lower()
    # 给魔搭的值进行一些其他写法的兼容
    provider = {"model_scope": "modelscope", "model-scope": "modelscope"}.get(
        provider, provider
    )
    if provider not in {"modelscope", "openrouter"}:
        raise ValueError(
            "MODEL_PROVIDER 只能是 modelscope 或 openrouter，"
            f"当前值为: {provider}"
        )
    # 辅助函数根据env里面的键名获取值
    def value(name: str, default: str = "") -> str:
        return (values.get(name) or default).strip()
    # 魔搭的配置
    modelscope = {
        "base_url": value(
            "MODELSCOPE_BASE_URL",
            #兼容默认值
            value("VL_MODEL_BASE_URL", "https://api-inference.modelscope.cn/v1"),
        ),
        "api_key": value("MODELSCOPE_API_KEY", value("MODA_API_KEY")),
        "llm_model": value("MODELSCOPE_LLM_MODEL", value("LLM_MODEL_NAME")),
        "vl_model": value("MODELSCOPE_VL_MODEL", value("VL_MODEL_NAME")),
    }
    # openrouter的配置
    openrouter = {
        "base_url": value("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key": value("OPENROUTER_API_KEY"),
        # 用户指定的 OpenRouter 模型同时承担文本和图片输入。
        "llm_model": value(
            "OPENROUTER_LLM_MODEL", "dots-studio/dots-3-note-preview:free"
        ),
        "vl_model": value(
            "OPENROUTER_VL_MODEL", "dots-studio/dots-3-note-preview:free"
        ),
    }
    selected = modelscope if provider == "modelscope" else openrouter
    return {
        "provider": provider,
        "llm_model": selected["llm_model"],
        "llm_base_url": selected["base_url"],
        "llm_api_key": selected["api_key"],
        "vl_model": selected["vl_model"],
        "vl_base_url": selected["base_url"],
        "vl_api_key": selected["api_key"],
    }
# ==================== AI修改 结束 ====================


class ModelConfig:
    """
    MODEL_VL_AND_LLM 配置类
    """
    # 重排序模型api -> openrouter
    OPENROUTER_API_KEY= os.getenv("OPENROUTER_API_KEY")
    # ==================== AI修改 开始 ====================
    # 读取一次当前平台配置；修改 .env 后重启服务即可切换，避免运行中混用两个平台。
    # 不传参 env为None 自动读取os.environ
    _ACTIVE_MODEL_PROFILE = resolve_model_profiles()
    MODEL_PROVIDER = _ACTIVE_MODEL_PROFILE["provider"]

    # 文本调用(答案、主体识别、HyDE、主体确认)使用 LLM_* 配置。
    LLM_BASE_URL = _ACTIVE_MODEL_PROFILE["llm_base_url"]
    LLM_API_KEY = _ACTIVE_MODEL_PROFILE["llm_api_key"]
    LLM_MODEL_NAME = _ACTIVE_MODEL_PROFILE["llm_model"]

    # 视觉调用(图片摘要)使用 VL_* 配置；当前两平台共用同一地址和 key，字段仍分开保留。
    VL_BASE_URL = _ACTIVE_MODEL_PROFILE["vl_base_url"]
    # 兼容旧节点使用的完整字段名；视觉节点统一读取 VL_BASE_URL。
    VL_MODEL_BASE_URL = VL_BASE_URL
    VL_API_KEY = _ACTIVE_MODEL_PROFILE["vl_api_key"]
    VL_MODEL_NAME = _ACTIVE_MODEL_PROFILE["vl_model"]

    # 兼容尚未迁移的旧节点；活动节点会直接使用上面的 LLM_*/VL_* 字段。
    MODA_API_KEY = LLM_API_KEY
    # ==================== AI修改 结束 ====================
    # ==================== AI修改 开始 ====================
    #视觉模型温度
    MODEL_TEMPERATURE = os.getenv("MODEL_TEMPERATURE") or os.getenv(
        "VL_MODEL_TEMPERATURE"
    )
    # 回答输出预算保留详细回答空间；模型根据提示词在内容完整处主动收束。
    # 下面参数均可在根目录 .env 中覆盖，方便不同模型按上下文窗口调节。
    LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 32768)
    # provider 硬截断时自动续写，避免答案停在半句话或代码块中。
    LLM_MAX_CONTINUATIONS = _env_int("LLM_MAX_CONTINUATIONS", 1)
    LLM_CONTINUATION_CONTEXT_CHARS = _env_int(
        "LLM_CONTINUATION_CONTEXT_CHARS", 24000
    )
    # ==================== AI修改 结束 ====================
    # 历史消息从10条扩大到20条，约保留10轮问答，减少多轮对话遗忘。
    QUERY_HISTORY_LIMIT = _env_int("QUERY_HISTORY_LIMIT", 20)
    # 检索上下文预算从10000字符扩大到18000字符，给详细回答更多依据。
    ANSWER_MAX_CONTEXT_CHARS = _env_int("ANSWER_MAX_CONTEXT_CHARS", 48000)
    # 历史文本设置独立上限，避免记忆扩大后挤占当前问题和回答输出空间。
    ANSWER_MAX_HISTORY_CHARS = _env_int("ANSWER_MAX_HISTORY_CHARS", 24000)
    # 前端上下文进度条的展示上限，与实际模型输出预算分开控制。
    ANSWER_CONTEXT_DISPLAY_LIMIT = _env_int("ANSWER_CONTEXT_DISPLAY_LIMIT", 80000)
    # 模型总上下文窗口；输出预算、提示词安全余量会从这里扣除后再分配资料。
    LLM_CONTEXT_WINDOW_TOKENS = _env_int("LLM_CONTEXT_WINDOW_TOKENS", 262144)
    # 给系统提示词、格式开销和token估算误差预留的安全空间。
    LLM_PROMPT_SAFETY_TOKENS = _env_int("LLM_PROMPT_SAFETY_TOKENS", 4096)
    # ==================== AI修改 结束 ====================

class MinioConfig:
    """
    MINIO配置类
    """

    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
    # 访问密钥
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
    # 私有密钥
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
    # 存储桶名称
    MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
    # 图片上传目录
    MINIO_IMG_DIR = os.getenv("MINIO_IMG_DIR")

class EmbeddingConfig:
    """
    EMBEDDING_MODEL配置类
    """

    # 嵌入模型
    EMBEDDING_MODEL =  os.getenv("EMBEDDING_MODEL")
    # 向量化设备
    DEVICE = os.getenv("DEVICE")
    # 精度训练 这读出来是字符串,所以一定要判断然后变成布尔
    USE_FP16 = True if os.getenv("USE_FP16") in ["True"] else False

class MilvusConfig:
    """
    MILVUS配置类
    """
    #创建客户端url地址
    milvus_url = os.getenv("MILVUS_URL")
    # 主体识别集合表仍单独保留，因为它和正文切片的检索用途不同。
    milvus_item_collection = os.getenv("ITEM_COLLECTION") or "item_collection"
    # ==================== AI修改 开始 ====================
    # Graph RAG 固定使用新集合；旧 knowledge_chunks 保留作回滚和对照评测。
    milvus_chunks_collection = "knowledge_chunks_graph"
    # 教育专属字段不再写入主 Chunk 集合。
    education_chunks_collection = milvus_chunks_collection
    # ==================== AI修改 结束 ====================

class MongoConfig:
    """
    Mongo配置类
    """

    # mongo的链接地址
    mongo_url = os.getenv("MONGO_URL")
    # 数据库名
    mongo_db_name = os.getenv("MONGO_DB_NAME")

class McpConfig:
    """
    Mcp配置类
    """

    # mcp连接地址
    mcp_server = os.getenv("MCP_SERVER")
    mcp_api_key = os.getenv("OPENAI_API_KEY")

# ==================== AI修改 开始 ====================
# 运行时临时目录统一放到项目目录之外，避免教育导入、测试或调试过程
# 默认不在源码目录生成 temp_data、日志和中间文件；可通过环境变量覆盖默认位置。
# ==================== AI修改 结束 ====================
class RuntimeConfig:
    RUNTIME_TEMP_ROOT = Path(os.getenv(
        "KNOWLEDGE_RUNTIME_TEMP_ROOT",
        str(Path(tempfile.gettempdir()) / "Knowledge_Database"),
    ))


def build_runtime_temp_path(task_id: str, date_label: str) -> Path:
    """Build a dated task directory outside the project workspace."""
    return RuntimeConfig.RUNTIME_TEMP_ROOT / "education_imports" / date_label / task_id
