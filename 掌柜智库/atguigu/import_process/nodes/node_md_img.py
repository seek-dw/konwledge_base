# atguigu/import_process/nodes/node_md_img3.py
import base64
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

from langchain.chat_models import init_chat_model
from minio.deleteobjects import DeleteObject

from atguigu.config.config import ModelConfig, MinioConfig
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger
from atguigu.tool.minio_client_tool import create_minio_client
# ==================== AI修改 开始 ====================
# 图片摘要节点需要把可恢复的限流等待写入统一任务状态，前端才能展示真实阶段。
from atguigu.tool.task_utils import (
    TASK_STATUS_PROCESSING,
    set_task_waiting_response,
    update_task_status,
)
# ==================== AI修改 结束 ====================
# ==================== AI修改 开始 ====================
# 统一使用共享图片 URL 构造器，避免 endpoint 已带 http:// 时重复拼接协议，
# 同时对中文、空格等文件名进行 URL 编码，保证前端 img.src 可以正常访问。
# ==================== AI修改 结束 ====================
from atguigu.tool.image_url_tool import build_minio_image_url, extract_image_references

# ==================== AI修改 开始 ====================
# 全局视觉模型信号量: 限制同时调用视觉模型(图片摘要生成)的并发数为2。
# 文档级并发可以放开(max_workers=6, 切分/向量化/入Milvus全并行),
# 但视觉模型调用受API的TPM限制, 必须单独限流。
# 信号量是进程级全局的, 所有node_md_img实例共享, 保证最多2张图同时调API。
_vl_semaphore = threading.Semaphore(2) #设置线程的信号量为2,代表一个进程中同时只能有2个线程进行视觉模型调用
# ==================== AI修改 结束 ====================

# ==================== AI修改 开始 ====================
def is_retryable_rate_error(message: str) -> bool:
    """只把临时频率限制判为可重试，余额/额度错误直接交给上层处理。"""
    normalized = (message or "").lower()
    non_retryable_markers = (
        "insufficient balance",
        "insufficient_balance",
        "insufficient quota",
        "insufficient_quota",
        "quota exceeded",
        "free-models-per-day",
        "openrouter_free_tier_daily",
        "daily reset",
        "余额不足",
        "额度不足",
    )
    if any(marker in normalized for marker in non_retryable_markers):
        return False
    return (
        "429" in normalized
        or "too many requests" in normalized
        or "rate limit" in normalized
        or "tpm" in normalized
        or "rpm" in normalized
    )
# ==================== AI修改 结束 ====================


class NodeMDImg(NodeBase):
    """
    上一个节点拿到了md_path 和 md_content将pdf转化为md
    这一节主要处理一下md文件中的图片,目的获得图片的摘要和url替换原图
        摘要是给模型看的,防止后续rag检索丢失图片信息
        url是给前端用的,检索到相关内容前端可以直接拿到url返回原图

    #关键要素:图片附近上下文(per_content、follow_content)、图片的内容(base64_str)

    1.拿到md文件内容并创造图片存放目录和所有图片文件列表
        1.防御
        2.文件读写流读出内容 (防御)
        3.构造图片目录,并列出目录下每一个文件,判断是否满足图片后缀要求
    2.拿到上下文和图片的内容(二进制)
        1.防御,判断图片的格式是否支持
        2.使用正则解析匹配图片拿到图片的跨度索引
        3.拿到索引后通过切片拿到图片附近的上下文
    3.根据上下文和图片内容获得摘要
        1.准备视觉大模型
        2.利用滑动窗口算法规避大模型的限频
        3.调用大模型(因为是视觉大模型,提示词官网找)
    4.获得url并将全文内容替换并备份
        1.构建上传目录,构建客户端获取工具获取客户端,对上传目录幂等性删除
        2.上传图片,拼接url,构造新的带有url列表
        3.遍历该列表,使用正则替换摘要和url,备份为新文件
    """

    name = "node_md_img"

    # ==================== AI修改 开始 ====================
    # 原improve: 上下文拿取过于随意没有限制了
    # 改进: 抽出专门的方法,做三件事——
    # 1.去掉切片窗口里"其他图片"的引用(图片A的上下文不该再塞图片B的base64摘要噪音)
    # 2.压缩空白字符(原文连续换行/缩进浪费宝贵的上下文窗口)
    # 3.按长度上限裁剪,尽量在句子边界断开,避免把句子截一半让视觉模型误读
    @staticmethod
    def _clean_context(raw: str, limit: int) -> str:
        # 去掉上下文里其他图片的md引用: ![...](...) 整段替换为占位符
        cleaned = re.sub(r"!\[.*?\]\(.*?\)", "[图片]", raw)
        # 压缩连续空白(换行/制表符/多空格)为单个空格
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) <= limit:
            return cleaned
        # 超长: 优先在句读处断开,找不到就硬截
        cut = cleaned[:limit]
        for sep in ("。", "；", "，", " ", "."):
            # rfind从右向左查找,找到第一个匹配的索引
            pos = cut.rfind(sep)
            if pos > limit // 2:  # 断点不能太靠前,否则丢的内容太多
                return cut[:pos + 1]
        return cut
    # ==================== AI修改 结束 ====================

    def process(self, state: ImportGraphState):

    #1.拿到md文件内容
        #1.防御
        # ==================== AI修改 开始 ====================
        # 保存当前任务 ID，供视觉模型退避时更新 waiting_response 状态。
        task_id = state.get("task_id", "")
        # ==================== AI修改 结束 ====================
        md_path = state.get("md_path", "")
        if not md_path:
            logger.error("路径不存在")
            raise Exception("请提供md文件路径")
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            logger.error("路径错误")
            raise Exception("请提供正确的md文件路径")

    #2.读取文件内容
        #with open()后面既可以接path对象,也可以接字符串路径
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        #防御
        if not md_content:
            logger.error("md文件内容为空")
            raise Exception("文件内容为空")

        logger.info("md文件内容读取成功")


    #3.构造图片目录,并且列出目录下所有图片
        #构造图片存放目录
        images_dir_path_obj = md_path_obj.parent / "images"
        #防御
        if not images_dir_path_obj.exists():
            os.makedirs(images_dir_path_obj)
        #列出该目录下所有文件,os.listdir()遍历该目录下的所有文件,遍历出来后返回的娥是一个列表
        #os.listdir传递path对象或者是字符串都可以,但是返回的就是一个字符串列表
        # ==================== AI修改 开始 ====================
        # 只把真正的图片文件纳入处理；附件带相对路径时，不能把子目录当图片。
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        image_name_list = [
            path.name for path in images_dir_path_obj.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ]
        relative_references = {
            Path(reference.split("?", 1)[0].split("#", 1)[0].replace("\\", "/")).name
            for reference in extract_image_references(md_content)
            if not reference.startswith(("http://", "https://", "data:"))
        }
        missing_references = sorted(relative_references - set(image_name_list))
        if missing_references:
            raise FileNotFoundError(
                "文档引用了图片但上传附件缺失: " + ", ".join(missing_references)
                + "；请把 Markdown 和对应图片一起上传"
            )
        if not relative_references:
            return {"md_content": md_content}
        image_name_list = [name for name in image_name_list if name in relative_references]
        # ==================== AI修改 结束 ====================
        # print(image_name_list)
        #如果没有图片,不需要处理图片,直接返回md文件内容
        if not image_name_list:
            return {
                "md_content":md_content
            }

    #3.根据正则匹配拿到图片的跨度索引,再进行切片拿到图片附近的上下文
        #遍历所有图片
        #所有图片格式
        image_with_context_str_list = []
        length = 200
        for image_name in image_name_list:
            #先判断该图片是否符合图片格式
            if Path(image_name).suffix.lower() not in image_extensions:
                logger.warning("图片格式错误")
                continue
            #图片格式符合要求,进行正则匹配
            # ==================== AI修改 开始 ====================
            # 原正则只认 Markdown 图片语法 ![](path)，但项目文档(掌柜智库19篇)
            # 的123张图全部用 HTML <img src="images/xxx.jpg"> 标签引用，
            # 导致一张图都匹配不上、被当作"未被引用"跳过、不生成摘要。
            # 现在同时支持两种引用语法，任一命中即可定位引用在文档中的位置。
            """
            定位图片在文档中的位置，支持两种引用语法：
              1) Markdown: ![描述](路径)        形如 ![](images/a.png)
              2) HTML img: <img src="路径" ...> 形如 <img src="images/1.整体架构图.jpg" style="zoom:67%;" />
            用 re.escape 转义图片名中的特殊字符(点、括号、空格等)，
            通过 match.span() 获取引用索引，用于截取图片上下文辅助视觉模型生成摘要。
            """
            md_pat   = r"!\[.*?\]\(.*?" + re.escape(image_name) + r"\)"
            # []的定义是内部一组候选字符,只匹配单个字符,只要字符串中的当前字符与[]中的某一个字符相同,就匹配成功
            # [^>] ^放在[]开头表示"非"或者"排除" 匹配任意的非>字符,捕获<img>除了结束符外的所有属性 *?非贪婪匹配,尽可能少的匹配字符,避免跨标签
            # ["\']匹配src值的开始引号,同时支持单引号和双引号
            html_pat = r'<img[^>]*?src\s*=\s*["\'].*?' + re.escape(image_name) + r'["\']'
            # 组合正则,就是说两种正则表达式满足其一都会产生正则对象,re.IGNORECASE忽略大小写
            pattern = re.compile(md_pat + "|" + html_pat, re.IGNORECASE)
            # ==================== AI修改 结束 ====================
            #match返回要给对象,对象中包含了span跨度索引
            match = pattern.search(md_content)
            # print(match)
            if not match:
                logger.error(f"图片{image_name}未被引用")
                continue
            #解包
            start, end = match.span()
            # ==================== AI修改 开始 ====================
            # 原improve: 上下文拿取过于随意没有限制了
            # 已改进: 原来直接max(0,start-200):start盲切200字符,可能截断句子、
            # 还会把相邻其他图片的引用一起带给视觉模型当"上下文"(纯噪音)。
            # 现在经过_clean_context: 去其他图片引用、压空白、句读处断开
            # 优化后
            # 1.去除其他的图片url噪声
            # 2.压缩空白字符
            # 3.按长度上限裁剪,尽量在句子边界断开,避免把句子截一半让视觉模型误读
            #上文
            pre_context = self._clean_context(md_content[max(0,start-length):start], length)
            #下文
            follow_context = self._clean_context(md_content[end:min(len(md_content),end+length)], length)
            # ==================== AI修改 结束 ====================

            # 拿到每个图片的路径
            # 此处的image_name是字符串, / 的左边必须是Path对象,右边可以任意拼接
            image_single_path = images_dir_path_obj / image_name

            #图片的内容
            with open(image_single_path,"rb") as f:
                image_content = f.read() #这里拿到的是<class bytes>一堆二进制内容
                #base64将二进制内容转化为可打印的字符串
                # base64模块.b64encode(二进制编码内容)方法,把内容从二进制转化为字符串但是还是<class bytes>
                # .decode("utf-8")方法,继续把<class bytes>转化为<class str>
                #输出base64_str就是图片经过base64编码解码后的文本表示
                every_image_base64_str = base64.b64encode(image_content).decode("utf-8")
                # print(base64_str)

            #拿到带有上下文和图片内容的列表
            image_with_context_str_list.append(
                {
                    "image_single_path":image_single_path,
                    "pre_context":pre_context,
                    "follow_context":follow_context,
                    "image_name":image_name,
                    "every_image_base64_str":every_image_base64_str
                }
            )


    #5.根据上下文和图片内容获得摘要
        #因为大模型会进行限频,考虑这一因素,在发送请求的时候采用滑动窗口算法限制单位时间内给大模型发送请求的次数
        #创建一个双向队列(同时满足左出右进,左进右出)(此处没用到只用到了单向)
        dq = deque(maxlen=20)
        llm = None
        image_with_summary_list = []
        for image_with_context_str in image_with_context_str_list:
            #每张图片的发送请求的当前时间
            current_time = time.time()
            #判断队列是否为空,并判断当前时间离第一个请求是否超过单位规定时间
            #如果超过规定时间则代表该图片可以发送请求,否则只能等待
            while dq and current_time - dq[0] > 60:
                #超过单位规定时间,清楚过期的请求记录
                dq.popleft()
            #来到这里说明while条件不满足,则需视情况是否等待
            #如果队列满员,则需要进行等待
            if dq and len(dq) == dq.maxlen:
                #current_time-dq[0]代表dq[0]离现在已经过去多长时间
                #60-(current_time - dq[0])代表距离规定单位时间还剩多少时间,就是需要等待的时间
                need_wait_time = max(
                    0,
                    60 - (current_time - dq[0])
                )
                time.sleep(need_wait_time)
                #睡眠完成,重置现在时间
                current_time = time.time()
                #等待时间已到,再次判断,这次应该是大于60了,所以清除旧记录,更新新的记录
                while dq and current_time - dq[0] > 60:
                    dq.popleft()
            #记录所有请求的时间戳
            dq.append(current_time)

            #构造提示词并调用大模型
            #优先基于官方文档的提示词进行编写
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                # 这个格式就是base64在使用的时候的规定
                                "url": "data:image/jpeg;base64," + every_image_base64_str,
                            },
                        },
                        {"type": "text", "text": f"""
                                                    这是一张图片，图片上文部分为"{image_with_context_str.get("pre_context")}"，
                                                    下文部分为"{image_with_context_str.get("follow_context")}"，
                                                    请用中文简要总结这张图片的摘要,字数在50字以内。"""
                         },
                    ],
                },
            ]
            if not llm :
                llm = init_chat_model(
                    model = ModelConfig.VL_MODEL_NAME,
                    model_provider = "openai",
                    # ==================== AI修改 开始 ====================
                    # 图片摘要按当前平台读取视觉模型专用配置，支持与文本模型一起切换。
                    api_key = ModelConfig.VL_API_KEY,
                    base_url = ModelConfig.VL_BASE_URL,
                    # ==================== AI修改 结束 ====================
                    # ==================== AI修改 开始 ====================
                    temperature = ModelConfig.MODEL_TEMPERATURE
                    # ==================== AI修改 结束 ====================
                )
            # ==================== AI修改 开始 ====================
            # 全局信号量限流 + 429指数退避重试(双保险):
            # 1.信号量: 最多2个线程同时调视觉模型(不管多少文档在并发跑),
            #   切分/向量化/入Milvus不受限, 只有图片摘要生成这一步限流
            # 2.重试: 万一信号量+deque双重限流下仍偶发429, 自动退避重试
            # 信号量限制跨文件并发，deque 限制单文件请求速率，职责不同。
            res = None
            _wait_sec = 30


            # 这个for是在上面的拿出每一张图片的for里面的,意味着每张图片都会进行vlmodel的请求,都会走这个信号量的限频率
            # 如果拿到了res 就直接退出for循环,如果发生了限流错误,那么就开始进行指数退避
            for _attempt in range(3): # 指数退避的次数,如果2次之内线程没拿到锁,那就会进行指数退避,如果第3次依然没拿到直接抛出异常
                try:
                    # 因为_vl_semaphore是一个进程间的全局变量,所以每一个线程之间拿到的是同一个信号量,不管有多少图片请求走过这里
                    # 它们拿到的都是同一个信号量,所以这里使用信号量进行限流
                    _vl_semaphore.acquire() # 获得一个互斥锁,总数只有2个 #这个操作决定了这个线程什么时候占用资源
                    try:
                        res = llm.invoke(input = messages)
                    finally:
                        _vl_semaphore.release() # 释放一个互斥锁 #这个操作决定了这个线程什么时候释放资源
                    break # 只要拿到答案立刻退出不再进行for循环
                except Exception as _e:
                    _msg = str(_e)
                    # ==================== AI修改 开始 ====================
                    # 余额/额度不足虽然也可能返回429，但等待不会恢复；只有临时
                    # RPM/TPM/Too Many Requests 才进入指数退避。
                    _is_rate = is_retryable_rate_error(_msg)
                    # ==================== AI修改 结束 ====================
                    if _is_rate and _attempt < 2:
                        # ==================== AI修改 开始 ====================
                        # 进入 sleep 前先写入可恢复状态，前端显示等待响应中；
                        # 不使用空字符串表达限流，避免被误判成状态丢失。
                        if task_id:
                            set_task_waiting_response(
                                task_id,
                                retry_count=_attempt + 1,
                                retry_after=_wait_sec,
                                message=f"触发模型限流，{_wait_sec}秒后自动重试",
                            )
                        # ==================== AI修改 结束 ====================
                        logger.warning(f"图片摘要生成触发限流(429), 第{_attempt + 1}次重试, 等待{_wait_sec}秒后重试")
                        time.sleep(_wait_sec)
                        _wait_sec *= 2 # 指数退避,假如信号量+令牌桶依然触发限流,说明请求过于频繁,开始睡眠,并随着指数递增
                        # ==================== AI修改 开始 ====================
                        # 退避窗口结束后恢复 processing，前端重新显示正常处理阶段。
                        if task_id:
                            update_task_status(
                                task_id,
                                TASK_STATUS_PROCESSING,
                                {"message": "限流等待结束，正在重试图片摘要"},
                            )
                        # ==================== AI修改 结束 ====================
                        continue
                    raise
            # ==================== AI修改 结束 ====================
            image_with_summary_list.append(
                {
                    "image_name":image_with_context_str.get("image_name"),
                    "image_single_path":image_with_context_str.get("image_single_path"),
                    "summary":res.content
                }
            )

    # 6.获得url地址
        # ==================== AI修改 开始 ====================
        # 原逻辑: 所有文档的图片共用同一个上传目录(MINIO_IMG_DIR),且每次导入前
        #         幂等删除会清空"整个目录"
        # 隐患: 连续导入多个文档时(如19个项目文档),后一个文档的导入会把前一个
        #       文档刚上传的图片全部删掉,导致前面文档chunk里已入库的图片URL全部404
        # 改进: 每个文档使用独立子目录(以md文件名stem命名),幂等删除只清自己目录——
        #       既保留"同一文档重导时不残留旧图"的幂等性,又不会误删其他文档的图片
        upload_dir = MinioConfig.MINIO_IMG_DIR.rstrip("/") + "/" + md_path_obj.stem # rstrip("/") 去除右侧的/
        # ==================== AI修改 结束 ====================
        #获取minio客户端
        minio_client = create_minio_client()

        #幂等性删除,上传新数据之前先删除旧数据
        #先列出目录下面的老数据,返回的是一个生成器对象,遍历出来之后是一个个的object对象,只找前缀为upload_dir的对象
        old_image_list = minio_client.list_objects(
            bucket_name = MinioConfig.MINIO_BUCKET_NAME,
            prefix = upload_dir,
            recursive = True)

        #然后删除旧数据.返回一个生成器对象
        errors = minio_client.remove_objects(
            bucket_name = MinioConfig.MINIO_BUCKET_NAME,
            #删除数据只支持delete_object类型,所以要转一下类型
            delete_object_list= [DeleteObject(obj.object_name) for obj in old_image_list]
        )
        #真正执行删除
        for error in errors:
            logger.error( error)

        #准备上传图片

        image_with_summary_and_url_list=[]
        for image_with_summary in image_with_summary_list:
            minio_client.fput_object(
                bucket_name = MinioConfig.MINIO_BUCKET_NAME,
                object_name = upload_dir + "/"+image_with_summary.get("image_name"),
                file_path = image_with_summary.get("image_single_path")
            )

        # ==================== AI修改 开始 ====================
        # 图片已经通过 fput_object 写入 MinIO；这里生成与对象名完全一致的
        # 浏览器访问地址，后续会随 Markdown 一起进入切片并写入 Milvus。
            #这里其实和手动拼接是得到了一样的结果,只不过通过工具的封装,将每一个拼接元素都做了安全性处理,使其接收文件名的泛化能力更强
            url = build_minio_image_url(
                MinioConfig.MINIO_ENDPOINT,
                MinioConfig.MINIO_BUCKET_NAME,
                f"{upload_dir}/{image_with_summary.get('image_name')}",
            )
        # ==================== AI修改 结束 ====================
            image_with_summary_and_url_list.append(
                {
                    **image_with_summary,
                    "url":url
                }
            )


        #url+摘要都已经获取,现在替换原md文件中的 图片并重命名备份文件
        #使用正则匹配到每一站图片然后进行替换
        for image_with_summary_and_url in image_with_summary_and_url_list:
            img_name = image_with_summary_and_url.get("image_name")
            summary  = image_with_summary_and_url.get("summary")
            url      = image_with_summary_and_url.get("url")
            # ==================== AI修改 开始 ====================
            # 原正则只认 Markdown ![](path) 语法, 但项目文档(掌柜智库19篇)的123张图
            # 全部用 HTML <img src="images/xxx.jpg" style="zoom:67%;" /> 标签引用,
            # 匹配不到 → 摘要和MinIO URL 没回填进 md_content → chunk 里既没有摘要也
            # 没有 url → 答案生成节点用 r'!\[.*?\]\((.*?)\)' 提取图片URL时一张都提不到
            # → 前端不显示图。这是图片输出链路的断点(与"配对""反查"两处同根因)。
            # 现在同时支持两种引用语法, 统一替换成 Markdown ![summary](url) 格式,
            # 下游 answer_output 的 markdown 提取正则无需改动即可拿到 MinIO URL。
            md_pat   = r"!\[.*?\]\(.*?" + re.escape(img_name) + r"\)"
            # html img 标签可能带 style/alt 等属性,且可能自闭合 <img .../> 或 <img ...>
            html_pat = r'<img[^>]*?src\s*=\s*["\'].*?' + re.escape(img_name) + r'["\'][^>]*?/?>'
            pattern = re.compile(md_pat + "|" + html_pat, re.IGNORECASE)
            # ==================== AI修改 结束 ====================
            #pattern.sub(新内容,要替换的原文),pattern中自带匹配条件
            md_content = pattern.sub(
                f"![{summary}]({url})",
                md_content
            )
            #备份新的md文件
            new_md_path_obj = md_path_obj.parent / (md_path_obj.stem + "_backup.md")
            with open(new_md_path_obj, "w", encoding="utf-8") as f:
                f.write(md_content)
                logger.info(f"{md_path_obj}备份成功,备份文件路径为:{new_md_path_obj}")

        # ==================== AI修改 开始 ====================
        # 记录导入阶段实际替换的图片数量，后续如果查询没有图片，
        # 可以快速判断是“导入没发现图”还是“查询/前端丢图”。
        logger.info(
            f"图片处理完成: 发现{len(image_name_list)}个文件，"
            f"成功上传并替换{len(image_with_summary_and_url_list)}张图片"
        )
        # ==================== AI修改 结束 ====================
        return {"md_content":md_content}



if __name__ == '__main__':
    node = NodeMDImg()
    pdf_list = [
        r"E:\AI大模型\第七阶段 掌柜智库\资料\05-设备手册汇总\doc\hak180产品安全手册\hak180产品安全手册.md"
    ]
    results = []  # 收集所有文件的结果
    for pdf_path in pdf_list:
        state = {
            "md_path": pdf_path,
        }
        res = node(state)
        results.append(res)
    # ============ 汇总 ============
    logger.info(f"共 {len(pdf_list)} 个文件，成功 {len(results)} 个")
    for r in results:
        logger.info(r)
