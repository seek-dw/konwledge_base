# atguigu/query_process/base.py

"""
查询流程节点基类

定义统一的节点接口规范，提供通用功能
"""
import time
from abc import abstractmethod, ABC

from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger
from atguigu.tool.task_utils import (
    put_data,
    get_task_info,
    add_running_task,
    add_done_task,
    add_node_duration,
    # ==================== AI修改 开始 ====================
    remove_running_task,
    is_task_cancelled,
    TaskCancelledError,
    # ==================== AI修改 结束 ====================
)


class NodeBase(ABC):

    name: str = "node_base"

    def __init__(self):
        """
        强制子类设置name
        """
        if self.name == "node_base":
            raise ValueError(f"{self.__class__.__name__} 必须设置 name 属性")

    def __call__(self, state: QueryGraphState):
        """
        节点执行入口
        """
        try:
            logger.info(f"{self.name} 开始执行...")

            task_id = state.get("task_id")
            # ==================== AI修改 开始 ====================
            # 查询节点从进入到结束统一计时，供SSE进度和前端时间线展示真实耗时。
            start_time = time.perf_counter()
            # ==================== AI修改 结束 ====================
            # ==================== AI修改 开始 ====================
            # 节点开始前拦截已取消任务，避免并行分支继续发起下一次检索。
            if is_task_cancelled(task_id):
                raise TaskCancelledError()
            # ==================== AI修改 结束 ====================
            add_running_task(task_id, self.name)
            put_data(task_id, event="progress", data=get_task_info(task_id))

            result = self.process(state)

            # ==================== AI修改 开始 ====================
            # 外部检索返回后再次检查，当前节点不会把结果继续传给后续节点。
            if is_task_cancelled(task_id):
                raise TaskCancelledError()
            # ==================== AI修改 结束 ====================

            # ==================== AI修改 开始 ====================
            # 先记录耗时再推送完成事件，确保前端收到done_list时同步拿到durations。
            add_node_duration(task_id, self.name, time.perf_counter() - start_time)
            # ==================== AI修改 结束 ====================
            add_done_task(task_id, self.name)
            put_data(task_id,event="progress",data = get_task_info(task_id))

            logger.info(f"{self.name} 结束执行...")
            return result
        except TaskCancelledError:
            remove_running_task(task_id, self.name)
            put_data(task_id, event="progress", data=get_task_info(task_id))
            raise
        except Exception as e:
            # ==================== AI修改 开始 ====================
            # 失败节点也保留耗时，方便用户知道卡在哪一步、耗了多久。
            if "start_time" in locals():
                add_node_duration(task_id, self.name, time.perf_counter() - start_time)
            # ==================== AI修改 结束 ====================
            logger.error(f"{self.name} 执行失败: {e}")
            raise

    @abstractmethod
    def process(self, state: QueryGraphState):
        """
        节点的核心处理逻辑
        :return:
        """
        pass
