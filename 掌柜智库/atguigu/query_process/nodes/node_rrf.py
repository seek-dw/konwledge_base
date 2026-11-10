# atguigu/query_process/nodes/node_rrf.py

from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.convert_to_json import convert_to_json
from atguigu.tool.logger import logger

class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将向量、HyDE 和 Graph 三路召回结果进行加权融合排序。
    实现思路:
        1.获取多路召回的结果(普通搜索和假设搜索)
        2.给每一路设置权重
        3.遍历每一路,遍历每一路里面的每一个答案,将他们的评分按照RRF公式进行更新:s = s + w/(k+排名)
        4.优先根据稳定 chunk_id 判断多路是否命中同一结果，分数直接相加
        5.最后根据分数再次进行排序
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def __init__(
        self,
        embedding_weight: float = 1.0,
        hyde_weight: float = 0.8,
        graph_weight: float = 1.5,
        rrf_k: int = 60,
    ):
        super().__init__()
        if embedding_weight < 0 or hyde_weight < 0 or graph_weight < 0:
            raise ValueError("RRF 路由权重不能为负数")
        if rrf_k < 0:
            raise ValueError("RRF_K 不能为负数")
        self.embedding_weight = embedding_weight
        self.hyde_weight = hyde_weight
        self.graph_weight = graph_weight
        self.rrf_k = rrf_k

    def process(self, state: QueryGraphState):
        #校验数据
        embedding_chunks = state.get("embedding_chunks","")
        hyde_embedding_chunks = state.get("hyde_embedding_chunks","")
        graph_chunks = state.get("graph_chunks", "")
        # ==================== AI修改 开始 ====================
        # 宽松校验：HyDE在课程/题目意图下被跳过返回空列表是正常情况
        # 只要有一路有结果就继续融合,两路都空才报错
        # 2026-08-20 改: 两路都空时不再 raise 整链崩溃(用户实测知识库数据缺失时
        # 直接报"倒排融合节点获取数据失败"错误), 改为降级返回空列表,
        # 让 answer_output 走"无结果"兜底分支, 给用户友好回复而不是报错。
        if not embedding_chunks and not hyde_embedding_chunks and not graph_chunks:
            logger.warning("倒排融合节点: 两路检索均为空(可能知识库缺少该数据), 降级返回空结果")
            return {"rrf_chunks": []}
        # 过滤掉空的路,只融合有数据的多路
        weighted_chunks = [
            ("vector", embedding_chunks, self.embedding_weight),
            ("hyde", hyde_embedding_chunks, self.hyde_weight),
            ("graph", graph_chunks, self.graph_weight),
        ]
        weighted_chunks = [
            (route_name, chunks, weight)
            for route_name, chunks, weight in weighted_chunks
            if chunks and weight > 0
        ]
        # ==================== AI修改 结束 ====================

        #倒排融合
        #逻辑:
            # 1. 遍历每一路的每一个chunk前提是已经排好名的chunk
            # 2. 保存每个chunk的id以及使用RRF公式进行计算RRF分数
            # 3. 优先根据稳定 chunk_id 判断多路召回是否重复，并保留完整 Chunk
            # 4. 如果id重复,拿现有的RRF分数加上重复的RRF分数
            # 5. 如果id不重复,保存原始分数并单独记录RRF分数
            # 6. 对新的字典取值并按RRF融合分数排序
        final_chunk_dict = {}
        for route_name, chunks, weight in weighted_chunks:
            for idx,chunk in enumerate(chunks):
                # 新 schema 的 chunk_id 是跨向量库和图数据库共享的稳定身份。
                # 只有旧结果缺少 chunk_id 时才回退到 Milvus id，避免同一 Chunk
                # 的 Graph 回溯结果和向量结果被当成两条候选。
                stable_chunk_id = str(chunk.get("chunk_id") or "").strip()
                if stable_chunk_id:
                    chunk_key = ("chunk_id", stable_chunk_id)
                elif chunk.get("id") is not None:
                    chunk_key = ("legacy_id", str(chunk["id"]))
                else:
                    # 缺少任何身份的异常结果不能把所有候选合并成一条。
                    chunk_key = ("anonymous", route_name, idx)
                original_score = chunk.get("score", 0.0)
                chunk_score = weight / (self.rrf_k + idx + 1)
                if chunk_key in final_chunk_dict:
                    existing = final_chunk_dict[chunk_key]
                    existing["rrf_score"] += chunk_score
                    source = chunk.get("source") or route_name
                    retrieval_sources = existing.setdefault("retrieval_sources", [])
                    if source not in retrieval_sources:
                        retrieval_sources.append(source)
                    if chunk.get("graph_path"):
                        graph_paths = existing.setdefault("graph_path", [])
                        for path in chunk["graph_path"]:
                            if path not in graph_paths:
                                graph_paths.append(path)
                else:
                    merged_chunk = dict(chunk)
                    merged_chunk["sim_score"] = original_score
                    merged_chunk["rrf_score"] = chunk_score
                    merged_chunk["retrieval_sources"] = [
                        chunk.get("source") or route_name
                    ]
                    final_chunk_dict[chunk_key] = merged_chunk


        rrf_chunks = sorted(
            final_chunk_dict.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        # ==================== AI修改 开始 ====================
        # 原Question: 为什么图片里面的摘要和url都没了?
        # 答案: 摘要和url并没有丢。图片处理节点(node_md_img)在导入时已把md里的
        # 图片原文 ![xxx](images/xxx.jpg) 替换成了 ![摘要文字](minio的url),
        # 摘要和url都嵌在content字符串内部随chunk一起存储;
        # RRF按id去重时整个chunk字典原样保留(见上方final_chunk_dict[chunk_id]=chunk),
        # 所有字段都不会丢。前端拿图片的方式是node_answer_output用正则从content里
        # 提取 ![..](url) 的url并放进image_urls返回,而不是靠独立字段。
        # 另: 删除了调试残留的print(len(rrf_chunks)),生产代码不应print
        # ==================== AI修改 结束 ====================
        return {
            "rrf_chunks":rrf_chunks
        }


if __name__ == '__main__':
    node = NodeRrf()
    mock_state ={
         "embedding_chunks": [
        {
            "item_name": "HAK180烫金机",
            "id": 468344320359437535,
            "title": "## HAK 180 烫金机",
            "file_title": "hak180产品安全手册",
            "content": "## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",
            "score": 0.8551799058914185,
            "source": "local"
        },
        {
            "item_name": "HAK180烫金机",
            "id": 468344320359437536,
            "title": "## HAK 180 烫金机",
            "file_title": "hak180产品安全手册",
            "content": "## HAK 180 烫金机\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出的任何索赔，我们不承担任何责任。",
            "score": 0.8501793146133423,
            "source": "local"
        },
        {
            "item_name": "HAK180烫金机",
            "id": 468344320359437539,
            "title": "## 设备",
            "file_title": "hak180产品安全手册",
            "content": "## 设备\n\n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。",
            "score": 0.8495146036148071,
            "source": "local"
        },
        {
            "item_name": "HAK180烫金机",
            "id": 468344320359437534,
            "title": "自定义标题",
            "file_title": "hak180产品安全手册",
            "content": "![](images/677a08ee041965bbbdb6b483d6c17d5aaa36a26b6dc96870a2019f0307b8616f.jpg)  \nD01WD7001-00\n\nSCHN\n",
            "score": 0.8392753005027771,
            "source": "local"
        },
        {
            "item_name": "HAK180烫金机",
            "id": 468344320359437541,
            "title": "## 设备",
            "file_title": "hak180产品安全手册",
            "content": "## 设备\n\n•\t请勿在卡纸或有纸张散落在设备内部的情况下尝试使用本设备。纸张与定影单元长时间接触可能导致火灾。\n\n请勿使用任何易燃物品、任何类型的喷雾剂包含酒精或氨水的有机溶剂/液体来清洁本设备的内部或外部。否则可能导致火灾。请改用无绒干抹布。有关如何清洁本设备的说明，请参阅 。\n\n•\t请勿将本设备放在化学品附近，或者将本设备放置在可能会泼溅到化学品的位置。万一化学品接触本设备，则存在火灾或触电的风险。特别是有机溶剂或液体（如苯、油漆稀释剂、抛光剂或除臭剂）可能导致塑料盖和/或电缆溶解或分解，从而产生火灾或触电的风险。这些化学品或其他化学品可能导致本设备故障或褪色。",
            "score": 0.8377050161361694,
            "source": "local"
        },
        {
            "item_name": "HAK180烫金机",
            "id": 468344320359437549,
            "title": "## 为设备选择一个安全的位置",
            "file_title": "hak180产品安全手册",
            "content": "## 为设备选择一个安全的位置\n\n•\t提起本设备时，请使用双手抓稳本设备的两侧。如果抓住的是进纸托板和出纸盒，它们可能会掉下来。必须通过将双手放在本设备下面来搬运本设备。\n\n![](images/cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg)  \n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人身伤害以及设备严重损坏。",
            "score": 0.8368643522262573,
            "source": "local"
        },
        {
            "item_name": "HAK180烫金机",
            "id": 468344320359437544,
            "title": "## 电源线",
            "file_title": "hak180产品安全手册",
            "content": "## 电源线\n\n•\t本设备通过 AC 220 V-240 V 50/60 Hz 电源供电。\n\n请勿将本设备连接到直流电源或逆变器（直流交流变换器）。存在火灾或触电的风险。\n\n•\t请勿用湿手触摸插头。这样可能导致触电。如果不确定您拥有哪种类型的电源，请联系合格的电工。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。\n\n•\t设备内部有高压电极。\n\n先拔掉电源线，再清洁设备内部。拔出电源线时，不要拉电线，而是捏住插头往外拔。存在发生火灾、触电或设备故障的风险。\n\n•\t请勿将任何物体压在电源线上。\n\n•\t请勿将本设备放在人们可能踏过电源线的位置。",
            "score": 0.836472749710083,
            "source": "local"
        },
        {
            "item_name": "HAK180烫金机",
            "id": 468344320359437547,
            "title": "## 设备",
            "file_title": "hak180产品安全手册",
            "content": "## 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![](images/c61a7f4e923881679f747508ae309c39dc221685344b068009256b1b3a40cc00.jpg)\n\n![](images/5067b2891ca4f761e2874921e0eb433aa742afbf38ca8dc509afecbf0aa6a6b5.jpg)\n",
            "score": 0.8029515147209167,
            "source": "local"
        },
        {
            "item_name": "HAK180烫金机",
            "id": 468344320359437538,
            "title": "## 设备",
            "file_title": "hak180产品安全手册",
            "content": "## 设备\n\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。",
            "score": 0.784317672252655,
            "source": "local"
        },
        {
            "item_name": "HAK180烫金机",
            "id": 468344320359437545,
            "title": "## 电源线",
            "file_title": "hak180产品安全手册",
            "content": "## 电源线\n\n•\t请勿将本设备放置在会使得拉伸或拉紧电源线的位置，否则电源线可能会磨损或损坏。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。如果拔出设备的电源插头，请勿触摸损坏 磨损的部分。\n\n•\t请勿让设备压在电源线上。\n\n•\t请勿在雷暴天气期间使用本设备。存在闪电导致触电的潜在风险。\n\n•\t请勿使用任何非指定的电缆。否则可能导致火灾或人员受伤。必须按照 正确安装。\n\n•\t请勿让任何金属硬件或任何类型的液体落在设备的电源插头上。否则可能导致触电或火灾。\n\n•\tBrother 强烈建议您不要使用任何类型的延长线。",
            "score": 0.7795202732086182,
            "source": "local"
        }
    ],
        "hyde_embedding_chunks": [
            {
                "title": "## 设备",
                "file_title": "hak180产品安全手册",
                "content": "## 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![](images/c61a7f4e923881679f747508ae309c39dc221685344b068009256b1b3a40cc00.jpg)\n\n![](images/5067b2891ca4f761e2874921e0eb433aa742afbf38ca8dc509afecbf0aa6a6b5.jpg)\n",
                "item_name": "HAK180烫金机",
                "id": 468344320359437547,
                "score": 0.8760473132133484,
                "source": "local"
            },
            {
                "title": "## 设备",
                "file_title": "hak180产品安全手册",
                "content": "## 设备\n\n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。",
                "item_name": "HAK180烫金机",
                "id": 468344320359437539,
                "score": 0.8580772876739502,
                "source": "local"
            },
            {
                "title": "## HAK 180 烫金机",
                "file_title": "hak180产品安全手册",
                "content": "## HAK 180 烫金机\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出的任何索赔，我们不承担任何责任。",
                "item_name": "HAK180烫金机",
                "id": 468344320359437536,
                "score": 0.8559046387672424,
                "source": "local"
            },
            {
                "title": "## 电源线",
                "file_title": "hak180产品安全手册",
                "content": "## 电源线\n\n•\t本设备通过 AC 220 V-240 V 50/60 Hz 电源供电。\n\n请勿将本设备连接到直流电源或逆变器（直流交流变换器）。存在火灾或触电的风险。\n\n•\t请勿用湿手触摸插头。这样可能导致触电。如果不确定您拥有哪种类型的电源，请联系合格的电工。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。\n\n•\t设备内部有高压电极。\n\n先拔掉电源线，再清洁设备内部。拔出电源线时，不要拉电线，而是捏住插头往外拔。存在发生火灾、触电或设备故障的风险。\n\n•\t请勿将任何物体压在电源线上。\n\n•\t请勿将本设备放在人们可能踏过电源线的位置。",
                "item_name": "HAK180烫金机",
                "id": 468344320359437544,
                "score": 0.8549141883850098,
                "source": "local"
            },
            {
                "title": "## 为设备选择一个安全的位置",
                "file_title": "hak180产品安全手册",
                "content": "## 为设备选择一个安全的位置\n\n•\t提起本设备时，请使用双手抓稳本设备的两侧。如果抓住的是进纸托板和出纸盒，它们可能会掉下来。必须通过将双手放在本设备下面来搬运本设备。\n\n![](images/cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg)  \n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人身伤害以及设备严重损坏。",
                "item_name": "HAK180烫金机",
                "id": 468344320359437549,
                "score": 0.8548282980918884,
                "source": "local"
            },
            {
                "title": "## 设备",
                "file_title": "hak180产品安全手册",
                "content": "## 设备\n\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。",
                "item_name": "HAK180烫金机",
                "id": 468344320359437538,
                "score": 0.8532317876815796,
                "source": "local"
            },
            {
                "title": "## HAK 180 烫金机",
                "file_title": "hak180产品安全手册",
                "content": "## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",
                "item_name": "HAK180烫金机",
                "id": 468344320359437535,
                "score": 0.8152989149093628,
                "source": "local"
            },
            {
                "title": "自定义标题",
                "file_title": "hak180产品安全手册",
                "content": "![](images/677a08ee041965bbbdb6b483d6c17d5aaa36a26b6dc96870a2019f0307b8616f.jpg)  \nD01WD7001-00\n\nSCHN\n",
                "item_name": "HAK180烫金机",
                "id": 468344320359437534,
                "score": 0.7948704361915588,
                "source": "local"
            },
            {
                "title": "## 设备",
                "file_title": "hak180产品安全手册",
                "content": "## 设备\n\n![](images/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ\n\n![](images/501bb8d2d681e4502d87badb15a68939eadfa086d309c3599f1c36b0bc559177.jpg)",
                "item_name": "HAK180烫金机",
                "id": 468344320359437543,
                "score": 0.7925941348075867,
                "source": "local"
            },
            {
                "title": "## 设备",
                "file_title": "hak180产品安全手册",
                "content": "## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n",
                "item_name": "HAK180烫金机",
                "id": 468344320359437551,
                "score": 0.7907807230949402,
                "source": "local"
            }
        ]
    }

    res = node(mock_state)
    logger.info(convert_to_json(res))
