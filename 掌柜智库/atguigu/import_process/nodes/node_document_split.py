# atguigu/import_process/nodes/node_document_split.py
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger
from atguigu.tool.convert_to_json import convert_to_json
from atguigu.tool.knowledge_chunk_store import build_chunk_id


class NodeDocumentSplit(NodeBase):
    """
    目的:将处理好图片的md文档按"标题树"切分成chunk(树形切分)
    一.拿到文档内容
        1.拿路径,拿标题,文件流拿文档
    二、树形粗切:按标题层级解析成"面包屑"区块
        1.统一换行符,按行切分
        2.用栈维护当前标题层级路径:遇到N级标题就把栈中>=N级的标题弹出再压入,
          这样每个区块都能拿到完整的祖先路径,如"第一章 安装 > 1.2 电源线"
        3.代码块内的#号不当作标题(沿用原有的代码块开关逻辑)
    三、兄弟小节合并
        正文极短的兄弟小节(如只有一句话的小节)单独成chunk会导致向量信息量不足,
        同一父标题下的连续小节自动合并
    四、递归精切(每个区块内部再切)
        1.表格不切(保持表格完整性)
        2.递归切割器切分并加入part索引(溯源)
        3.每个chunk的content = "面包屑标题路径 + 正文",
          向量化时自带完整层级上下文,查询"电源线相关"也能命中"安装>电源线"
    """

    name = "node_document_split"

    # ==================== AI修改 开始 ====================
    # 树形切分核心参数
    # 精切chunk上限(字符),与原逻辑一致
    CHUNK_SIZE = 300
    # 相邻chunk重叠字符数:原来是2,等于没有重叠——句子被切断处上下文直接丢失,
    # 查询恰好落在切断点时两边都召回不全。50字符约1~2句话,保证边界信息双份保留
    CHUNK_OVERLAP = 50
    # 正文短于该值的兄弟小节会被合并进前一个同级小节,避免碎片chunk
    MERGE_MIN_BODY = 60
    # 区块"整块保留"的绝对上限(字符)。超过该值即使是表格也必须再切。
    # 教训(06文档翻车现场): 06文档内嵌了文档切分节点的完整源码(12985字符),
    # 代码里恰好有 "<table" 字面量,命中表格豁免条件被整块保留——
    # 12985字符的巨块送进BGE-M3(最长8192token)后,在6G显存的显卡上
    # 直接CUDA OOM甚至原生段错误(Windows事件日志0xc0000005,进程无任何报错就死了),
    # 导致06文件反复导入失败而其他18篇文档全部成功。
    # 该上限远大于CHUNK_SIZE,小表格仍然整块保留不受影响。
    MAX_WHOLE_SECTION = 1500
    # ==================== AI修改 结束 ====================

    def process(self, state: ImportGraphState):

        # 一.拿到文档内容
        md_path = state.get("md_path", "")
        if not md_path:
            logger.error("路径不存在")
            raise Exception("路径不存在")
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            logger.error("路径不存在")
            raise Exception("路径不存在")

        file_title = state.get("file_title")
        if not file_title:
            #.stem去掉路径下最后一段的文件名的扩展名
            file_title = md_path_obj.stem

        # ==================== AI修改 开始 ====================
        # 图片输出断链修复: 优先使用 node_md_img 已经替换好图片的 md_content
        # (原文里的 <img src="images/xxx.jpg"> 已被替换成 ![摘要](minio_url))。
        # 原逻辑(课程遗留bug)无视 state["md_content"], 每次都从磁盘重读原始
        # md_path —— 图片摘要和MinIO URL只写进了 _backup.md 和返回值,
        # 从未进入切分链路, 导致 chunk 里全是 <img src="images/xxx.jpg"> 死链,
        # 答案生成节点用 markdown 正则一张URL都提取不到, 前端永远不显示图。
        # 现在优先取 state["md_content"], 取不到(如单独调试本节点)才回退读磁盘。
        md_content = state.get("md_content") or ""
        if not md_content:
            with md_path_obj.open("r", encoding="utf-8") as f:
                md_content = f.read()
        # ==================== AI修改 结束 ====================
        # 统一换行符,所有的操作系统都支持\n换行
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        md_lines = md_content.split("\n")

        # ==================== AI修改 开始 ====================
        # 二、树形粗切:解析标题树,产出带面包屑路径的区块列表
        sections = self._parse_sections(md_lines, file_title)

        # 三、兄弟小节合并:消除信息量不足的碎片区块
        sections = self._merge_small_sections(sections)

        # 四、递归精切:区块内部超过上限再切,切完拼上面包屑前缀
        final_section_list = self._fine_split(
            sections,
            state.get("document_id") or file_title,
        )
        # ==================== AI修改 结束 ====================

        return {"chunks": final_section_list}

    # ==================== AI修改 开始 ====================
    def _parse_sections(self, md_lines, file_title):
        """
        树形粗切:逐行扫描,用栈维护标题层级路径。
        返回区块列表,每个区块:
            breadcrumb: 面包屑路径,如"第一章 安装 > 1.2 电源线"(纯文本,不带#号)
            parent_key: 父路径(去掉最后一级),用于判断两个区块是不是"兄弟"
            body:       该标题下的正文(不含标题行本身)
        """
        code_pattern = r"(`{3,}|~{3,})" #至少出现3次
        # 捕获两组:第1组是#号个数(即标题层级),第2组是标题文字
        heading_pattern = r"^(#{1,6})\s+(.+)$"

        sections = []
        # 栈元素: (标题层级, 标题文字)。栈底到栈顶就是当前的"祖先路径"
        heading_stack = []
        body_lines = []
        is_in_block = False
        marker = None

        def _flush():
            """把当前累积的正文收尾成一个区块"""
            body = "\n".join(body_lines).strip()
            # 空正文(纯空白)的区块直接丢弃,如标题紧跟标题的情况
            if not body:
                return
            if heading_stack:
                # 解包遍历,因为heading_stack里面存放的是元组, _代表这个值我不需要可读性约定
                # breadcrumb就是面包屑路径,如"第一章 安装 > 1.2 电源线"
                breadcrumb = " > ".join(t for _, t in heading_stack)
                # 兄弟判定键:父路径。len>1才有父,否则父路径为空串
                # parent_key判断是否存在父路径,先把最后一个元素去掉,遍历剩余面包屑路径就是父路径
                # 后续使用它来判断哪些区块属于同一个父标题,从而进行合并
                parent_key = " > ".join(t for _, t in heading_stack[:-1]) if len(heading_stack) > 1 else ""
            else:
                # 如果当前的文档没有标题层级,那么代码给它人为创建一个归属
                # 第一个标题之前的内容(文档引言),用文件名当面包屑
                breadcrumb = file_title
                # 引言区单独一个parent_key,防止和一级标题区块误合并
                parent_key = file_title
            sections.append({
                "breadcrumb": breadcrumb,
                "parent_key": parent_key,
                "body": body,
            })

        for raw_line in md_lines:
            line = raw_line.strip()
            # 代码块开关逻辑(沿用原实现):进入/退出```或~~~包裹的代码块
            code_match = re.match(code_pattern, line)
            if code_match:
                if not is_in_block:
                    is_in_block = True
                    marker = code_match.group(1)
                elif marker == code_match.group(1):
                    is_in_block = False
                    marker = None
                # 代码块的内容行原样保留在正文里
                body_lines.append(raw_line)
                continue

            # 代码块内的#号不是标题
            # 三元表达式,等价于 if is_in_block: heading_match = None else: heading_match = re.match(heading_pattern, line)
            heading_match = None if is_in_block else re.match(heading_pattern, line)
            if heading_match:
                # 先收尾上一个区块
                _flush()
                body_lines = []
                # 正则获取标题的层级就是#的数量
                level = len(heading_match.group(1))
                # 正则捕获标题内容
                title_text = heading_match.group(2).strip()
                # 核心树形逻辑:遇到N级标题,弹出栈中所有>=N级的标题再压入自己
                # 例:栈为[1章,2节],来了一个2级标题->弹出2节压入新2节(同级替换)
                #     来了一个3级标题->直接压入(成为2节的子标题)
                # 栈列表存放(层级,标题内容)元组,当新标题入栈时弹出所有平级和更低级的标题
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title_text))
            else:
                body_lines.append(raw_line)

        # 收尾最后一个区块(原有的"漏掉最后一块"问题在这里统一解决)
        _flush()
        return sections

    def _merge_small_sections(self, sections):
        """
        兄弟小节合并:同一父路径下,正文极短的区块并入前一个兄弟区块。
        碎片chunk(比如整节只有一句"详见附录")单独向量化时信息量不足,
        相似度必然偏低,检索时永远排不上来,还会挤占TopK名额。
        表格区块不参与合并,保持表格粒度独立完整。
        """
        merged = []
        for sec in sections:
            #返回的是一个布尔值,只有当以下条件全部满足的时候才可以进行合并
            can_merge = (
                merged
                #条件1 当前区块的父路径和前一个的父路径相同
                and sec["parent_key"] == merged[-1]["parent_key"]
                #条件2 当前区块正文长度小于阈值
                and len(sec["body"]) < self.MERGE_MIN_BODY
                #条件3 当前区块不能是表格
                and "<table" not in sec["body"]
                #条件4 前一个也不能是表格 防止正文和表格混合
                and "<table" not in merged[-1]["body"]
            )
            #如果条件满足那么合并列表最后一个元素的正文就等于之前最后一个元素的正文+当前区域的正文
            if can_merge:
                merged[-1]["body"] = (merged[-1]["body"] + "\n\n" + sec["body"]).strip()
            else:
                # dict()浅拷贝,防止合并时改到原始解析结果
                # 浅拷贝只复制外壳,不会复制里面的对象,因为这里只有一层字符串所以用浅拷贝就够了
                # 使用浅拷贝后,最外层对象不是同一个地址,但是里面的嵌套对象,还是指向同一个地址
                # 深拷贝会递归复制里面所有的对象,并开辟新的地址
                # 此处为了不影响原始的sec对齐进行浅拷贝后进行合并,用深拷贝属于大炮打蚊子没必要
                merged.append(dict(sec))
        return merged

    def _fine_split(self, sections, document_id):
        """
        递归精切:区块内部超过CHUNK_SIZE再细切,每个chunk的content都带上面包屑前缀。
        content = "面包屑路径\n\n正文片段"
        这样向量化/重排/LLM阅读时,每个chunk都自带"我从哪来"的层级上下文。
        """
        # ==================== AI修改 开始 ====================
        # 分隔符在 "\n" 之后、句读之前插入 "</tr>":
        # 真实的大表格(超上限必须切时)优先在行边界</tr>处断开,
        # 不会把表格从单元格中间拦腰截断;普通文本碰不到</tr>不受影响。
        spliter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "</tr>", "。", "！", "？", "；", ".", "!", "?", ";", " "],
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
        )
        # ==================== AI修改 开始 ====================
        # chunk_index 在整篇文档内连续递增，chunk_id 因此能稳定用于图谱证据回溯。
        final_section_list = []
        for sec in sections:
            breadcrumb = sec["breadcrumb"]
            body = sec["body"]
            # ==================== AI修改 开始 ====================
            # 整块保留的条件收紧为两条,同时满足才行:
            #   1.区块确实足够短(<=CHUNK_SIZE) —— 原有逻辑
            #   2.含表格的区块额外放宽到 MAX_WHOLE_SECTION(小表格保持完整性)
            # 修复06文档bug: 原条件是 "<table" in body 就整块保留且【无长度上限】,
            # 06文档4.4节内嵌了切分节点完整源码(12985字符),源码文本里恰好含
            # "<table"字面量 → 被误判成表格 → 整块保留 → 巨块打爆嵌入模型。
            # 现在超过MAX_WHOLE_SECTION的区块无论含不含<table都必须再切。
            is_small = len(body) <= self.CHUNK_SIZE
            is_small_table = "<table" in body and len(body) <= self.MAX_WHOLE_SECTION
            if is_small or is_small_table:
                content = f"{breadcrumb}\n\n{body}"
                final_section_list.append({
                    "chunk_id": build_chunk_id(
                        document_id, len(final_section_list), breadcrumb, content
                    ),
                    "document_id": document_id,
                    "chunk_index": len(final_section_list),
                    "section_path": breadcrumb,
                    "content": content,
                })
                continue
            # ==================== AI修改 结束 ====================
            split_chunk_list = spliter.split_text(body)
            for split_chunk in split_chunk_list:
                content = f"{breadcrumb}\n\n{split_chunk}"
                final_section_list.append({
                    "chunk_id": build_chunk_id(
                        document_id, len(final_section_list), breadcrumb, content
                    ),
                    "document_id": document_id,
                    "chunk_index": len(final_section_list),
                    "section_path": breadcrumb,
                    "content": content,
                })
        # ==================== AI修改 结束 ====================
        return final_section_list
    # ==================== AI修改 结束 ====================


if __name__ == '__main__':
    node = NodeDocumentSplit()
    init_state={
        "md_path": r"E:\AI大模型\第七阶段 掌柜智库\掌柜智库05\07【掌柜智库】【导入】文档切片.md"
    }
    res = node(init_state)
    logger.info(convert_to_json(res))
