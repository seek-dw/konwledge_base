# atguigu/query_process/nodes/node_web_search_mcp.py
import json
import os

from sympy import print_rcode

from atguigu.config.config import McpConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.convert_to_json import convert_to_json
from atguigu.tool.logger import logger
import asyncio
from agents.mcp import MCPServerStreamableHttp


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    此处使用openai-sdk封装后的mcp调用模式,省去了自己手动调用工具,处理工具返回结果,传回messages,agent循环等
    实现思路:
        1.获取重写的问题
        2.mcp连接,并调用工具
        3.解析结果
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    # ==================== AI修改 开始 ====================
    # 回答慢的元凶之一: 每轮都重建MCP连接+网络搜索, 且失败重试3次,
    # MCP服务不可达/网络差时一次回答要多等10-30秒。
    # 修复: 增加 .env 开关 ENABLE_WEB_SEARCH(默认false) —— 本地知识库为主,
    # 网络搜索默认关闭; 开启时也快速失败(单次尝试,失败返回空,不重试不等待)。
    @staticmethod
    def _enabled() -> bool:
        return os.getenv("ENABLE_WEB_SEARCH", "false").strip().lower() in ("1", "true", "yes", "on")

    def process(self, state: QueryGraphState):
        # ==================== AI修改 开始 ====================
        if not self._enabled():
            # 开关关闭: 返回空结果, 不碰MCP, 不拖慢链路
            logger.info("web_search 已关闭(ENABLE_WEB_SEARCH!=true), 返回空结果")
            return {"web_search_docs": []}
        # ==================== AI修改 结束 ====================
        # 获取重写的问题
        rewritten_query = state.get("rewritten_query", "")
        if not rewritten_query:
            logger.error("rewritten_query不能为空")
            raise ValueError("rewritten_query不能为空")

        #调用mcp
        try:
            result = asyncio.run(self.web_search(rewritten_query,10))
        except Exception as e:
            # 快速失败: 网络/MCP异常不阻塞主链路, 返回空结果让RRF/rerank正常走
            logger.error(f"web_search 调用失败, 返回空结果继续: {e}")
            return {"web_search_docs": []}
        # ==================== AI修改 结束 ====================
        print(result)
        #解析结果
        try:
            data = json.loads(result.content[0].text).get("pages")
        except Exception as e:
            logger.error(f"web_search 结果解析失败, 返回空结果继续: {e}")
            return {"web_search_docs": []}
        return {
            "web_search_docs": [
                {
                    "title": item.get("title"),
                    "content": item.get("snippet"),
                    "url": item.get("url"),
                    "source": "web"
                }
                for item in data
            ]
        }



    async def web_search(self,query,limit) -> None:
        token = McpConfig.mcp_api_key
        #创建一个mcp_client,并建立mcp_server连接,cs架构
        #async with 异步上下文管理器,建立连接,初始化等,获取工具,使用和关闭异步操作,并在关闭的时候清理资源
        async with MCPServerStreamableHttp(
                #标识名称
                name="Web_Search_Server",
                #mcp_server的连接参数
                params={
                    "url": McpConfig.mcp_server,
                    "headers": {"Authorization": f"Bearer {token}"},
                    "timeout": 10,
                },
                #缓存获取到的tool列表
                cache_tools_list=True,
                #最大尝试次数
                max_retry_attempts=3,
                client_session_timeout_seconds=10
        ) as server:
            result = await server.call_tool("bailian_web_search",arguments=
                                            {"query":query,
                                             "count":limit})


        return result

        #创建助手
        # 这个agent只支持openai原生的api_key,使用其他网站的api_key就是用不了agent
        # agent = Agent(
        #     name="Assistant",
        #     #系统提示词
        #     instructions="Use the MCP tools to answer the questions.",
        #     mcp_servers=[server],
        #     #要求必须使用工具
        #     model_settings=ModelSettings(tool_choice="required"),
        # )
        # #运行,放入agent和问题,Runner.run会获取提示词,读取工具列表,模型判断,是否需要调用add ,调用mcp tool,等待
        # result = await Runner.run(agent, "Add 7 and 22.")
        # print(result.final_output)

if __name__ == '__main__':
    init_state = {
        "rewritten_query": "关于HAK180烫金机如何使用",
        "item_names": ["HAK180烫金机"]
    }
    node = NodeWebSearchMcp()
    res = node.process(init_state)
    logger.info(convert_to_json(res))