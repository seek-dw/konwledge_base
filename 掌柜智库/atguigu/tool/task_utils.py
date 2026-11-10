import time
# ==================== AI修改 开始 ====================
# 取消事件与查询任务同生命周期，供前后端协作终止工作流。
import threading
# ==================== AI修改 结束 ====================
from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional

# ---------------------------
# 内存态任务追踪（单进程）
# ---------------------------
# key: task_id
# value: 节点名列表（原始英文/节点ID）

# defaultdict(list): 只要访问不存在的 key，自动帮你初始化 []
_tasks_running_list: Dict[str, List[str]] = defaultdict(list)
_tasks_done_list: Dict[str, List[str]] = defaultdict(list)
_tasks_duration: Dict[str, Dict[str, float]] = defaultdict(dict)

# key: task_id
# value: status 字符串（如 processing/completed/failed）
_tasks_status: Dict[str, str] = {}

# ==================== AI修改 开始 ====================
# 限流等待不是未知任务：单独保存状态详情，让前端知道正在等待第几次重试、
# 预计等待多久以及当前是哪一个节点触发了限流。
_tasks_status_detail: Dict[str, Dict[str, Any]] = defaultdict(dict)
# ==================== AI修改 结束 ====================

# key: task_id
# value: 任务结果（例如 query 的 answer）

# 只要访问不存在的 key，自动帮你初始化 {}
_tasks_result: Dict[str, Dict[str, str]] = defaultdict(dict)

# ==================== AI修改 开始 ====================
# key: task_id
# value: 错误信息字符串(导入/查询失败时的异常原因)
# 原来失败时只设 status=failed, /status 只返回 "failed" 三个字,
# 前端只能显示"❌ 工作流执行失败中止", 用户看不到真实原因(如段错误前的
# 具体异常、Milvus 连接失败等), 报告"控制台没有找到报错信息"。
# 现在把异常摘要存进来, /status 带着一起返回, 前端展示给用户。
_tasks_error_msg: Dict[str, str] = {}
# ==================== AI修改 结束 ====================


TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
# ==================== AI修改 开始 ====================
# queued 表示任务已经登记但尚未拿到导入线程；waiting_response 表示任务仍在运行，
# 只是因模型限流暂时等待下一次重试。两者都不应该被前端解释成空状态。
TASK_STATUS_QUEUED = "queued"
TASK_STATUS_WAITING_RESPONSE = "waiting_response"
# ==================== AI修改 开始 ====================
# 用户主动停止与服务失败是两种不同结果，前端需要单独展示 cancelled。
TASK_STATUS_CANCELLED = "cancelled"
_cancel_events: Dict[str, threading.Event] = {}


class TaskCancelledError(Exception):
    """查询被用户停止后，用于终止 LangGraph 后续节点。"""


# ==================== AI修改 结束 ====================
# ==================== AI修改 结束 ====================

# 节点名 -> 中文名映射（用于前端展示）
# 说明：这里的 key 应与 LangGraph 的 add_node("xxx", ...) 中的节点名一致。
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",  
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_milvus": "导入向量库",
    # ==================== AI修改 开始 ====================
    # Graph Build 是导入链的最后一步，必须和前端保持同一中文名称。
    "node_graph_build": "构建知识图谱",
    # ==================== AI修改 结束 ====================
    # --- Query 流程节点---
    "node_item_name_confirm": "确认问题产品",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_join": "多路搜索合并",
    # ==================== AI修改 开始 ====================
    # 教育导入任务的三个阶段，用中文展示给前端进度卡片。
    "education_upload": "教育资料上传",
    "education_parse": "教育资料解析",
    "education_embedding": "教育数据向量化",
    "education_import": "写入教育知识库",
    # ==================== AI修改 结束 ====================
}


def _to_cn(node_name: str) -> str:
    """将节点名转换为中文展示名；若无映射则返回原名。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)

def add_running_task(task_id: str, node_name: str) -> None:
    """
    添加“正在运行”的节点任务。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    # _ensure_task(task_id)

    # 1. 获取当前任务的运行节点列表（利用 defaultdict 自动初始化特性）
    running = _tasks_running_list[task_id]

    # 2. 将当前节点加入运行列表（并做去重判断，防止重复添加）
    if node_name not in running:
        running.append(node_name)

def add_done_task(task_id: str, node_name: str) -> None:
    """
    添加“已完成”的节点任务。
    注意：添加已完成任务时，会把同名的“正在运行”任务删除。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """

    # 1. 如果该节点还在运行列表中，则将其移出（表示该节点已结束运行）
    if node_name in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].remove(node_name)

    # 2. 获取当前任务的已完成节点列表
    done = _tasks_done_list[task_id]

    # 3. 将当前节点加入已完成列表（做去重判断，防止重复标记）
    if node_name not in done:
        done.append(node_name)


# ==================== AI修改 开始 ====================
def remove_running_task(task_id: str, node_name: str) -> None:
    """取消时移除运行中的节点，但不把它伪装成已完成。"""
    if node_name in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].remove(node_name)


def cancel_task(task_id: str) -> None:
    """设置协作式取消标记，并立即让状态接口可观察。"""
    _cancel_events.setdefault(task_id, threading.Event()).set()
    update_task_status(task_id, TASK_STATUS_CANCELLED, {"message": "查询已停止"})


def is_task_cancelled(task_id: str) -> bool:
    return _cancel_events.get(task_id, threading.Event()).is_set()
# ==================== AI修改 结束 ====================



def get_running_task_list(task_id: str) -> List[str]:
    """
    获取正在运行节点列表（中文展示）。
    """
    # 获取指定任务运行中的节点列表，并统一转换为中文名返回
    running = _tasks_running_list.get(task_id, [])

    # 把运行中的节点名转换为中文名
    return [ _to_cn(n)  for n in running]


def get_done_task_list(task_id: str) -> List[str]:
    """
    获取已完成节点列表（中文展示）。
    """
    # 获取指定任务已完成的节点列表，并统一转换为中文名返回
    done = _tasks_done_list.get(task_id, [])
    return [_to_cn(n) for n in done]


def get_task_status(task_id: str ) -> str:
    """
    获取当前任务状态。

    参数：
    - task_id: 任务ID

    返回：
    - str: 状态名称；如果未设置过则返回空字符串
    """
    # 安全获取指定任务的总体运行状态，若不存在则返回空字符串
    return _tasks_status.get(task_id, "")


def update_task_status(
        task_id: str,
        status_name: str,
        status_detail: Optional[Dict[str, Any]] = None,
) -> None:
    """
    更新任务状态。

    参数：
    - task_id: 任务ID
    - status_name: 状态名称（字符串）
    """

    # ==================== AI修改 开始 ====================
    # 更新指定任务的总体运行状态，并同步替换本次状态对应的详情。
    # 普通 processing/completed/failed 状态没有详情时会清空旧的等待提示，
    # 避免任务恢复处理后前端仍显示上一轮限流信息。
    _tasks_status[task_id] = status_name
    _tasks_status_detail[task_id] = dict(status_detail or {})
    # ==================== AI修改 结束 ====================


# ==================== AI修改 开始 ====================
def set_task_waiting_response(
        task_id: str,
        retry_count: int,
        retry_after: int,
        message: str,
) -> None:
    """记录一次可恢复的模型限流等待，供状态接口直接返回给前端。"""
    update_task_status(
        task_id,
        TASK_STATUS_WAITING_RESPONSE,
        {
            "message": message,
            "retry_count": retry_count,
            "retry_after": retry_after,
        },
    )


def get_task_status_detail(task_id: str) -> Dict[str, Any]:
    """获取当前任务状态的附加详情，返回副本避免调用方修改全局状态。"""
    return dict(_tasks_status_detail.get(task_id, {}))


def set_task_error(task_id: str, error_msg: str) -> None:
    """记录任务的错误信息(失败时调用), 供 /status 返回给前端展示。"""
    _tasks_error_msg[task_id] = error_msg


def get_task_error(task_id: str) -> str:
    """获取任务的错误信息(若无则返回空串)。"""
    return _tasks_error_msg.get(task_id, "")
# ==================== AI修改 结束 ====================


def add_node_duration(task_id: str, node_name: str, duration: float) -> None:
    """记录节点耗时（秒）"""
    cn_name = _to_cn(node_name)
    _tasks_duration[task_id][cn_name] = round(duration, 2)

def get_node_durations(task_id: str) -> Dict[str, float]:
    """获取所有节点的耗时"""
    return dict(_tasks_duration.get(task_id, {}))

def get_task_info(task_id: str) -> Dict[str, any]:
    """
    获取任务的全局信息（状态 + 运行中节点 + 已完成节点）
    :param task_id: 任务ID
    :return: 包含 status、running_list、done_list 的字典
    """
    return {
        "status": get_task_status(task_id),
        # ==================== AI修改 开始 ====================
        # 将 waiting_response 的重试信息作为结构化字段返回，前端不再猜测空状态。
        "status_detail": get_task_status_detail(task_id),
        # ==================== AI修改 结束 ====================
        "running_list": get_running_task_list(task_id),
        "done_list": get_done_task_list(task_id),
        "durations": get_node_durations(task_id),
        # ==================== AI修改 开始 ====================
        # 失败时带上错误原因, 前端展示给用户, 不用翻控制台找
        "error_msg": get_task_error(task_id),
        # ==================== AI修改 结束 ====================
    }
"""{
    "status": "completed",

    "running_list": [],

    "done_list": [
        "开始上传文件",
        "检查文件",
        "PDF转Markdown",
        "Markdown图片处理",
        "文档切分",
        "向量生成",
        "导入向量库"
    ],

    "durations": {
        "开始上传文件": 1.2,
        "检查文件": 0.1,
        "PDF转Markdown": 8.6,
        "Markdown图片处理": 3.2,
        "文档切分": 0.5,
        "向量生成": 15.3,
        "导入向量库": 2.1
    }
}"""

"""
1.将队列视作全局变量
2.从队列中取数据和添加数据的方法都作为一个函数写进该文件
-----#{方便后续所有的其他模块进行状态流的更新或者删除}#------
"""

from collections import deque

#定义全局的队列字典,里面会同时更新 {整个节点的流转状态}、{大模型的流式返回答案}
queue_dict:Dict[str,Queue]= {}

#创建队列
def create_queue(task_id):
    if not queue_dict.get(task_id):
        queue_dict[task_id] = Queue()
    _cancel_events.setdefault(task_id, threading.Event())

#向队列中添加数据
def put_data(task_id,event,data):
    while not queue_dict.get(task_id):
        time.sleep(1)
    queue_dict.get(task_id).put({"event":event,"data":data})

#向队列中获取数据
def get_data(task_id,):
    while not queue_dict.get(task_id):
        time.sleep(1)
    return queue_dict.get(task_id).get()

