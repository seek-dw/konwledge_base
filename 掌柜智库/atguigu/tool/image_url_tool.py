# ==================== AI修改 开始 ====================
# 新增统一图片 URL 工具：导入、查询和测试都复用这里的规则，
# 让图片从 MinIO 到聊天界面的地址格式保持一致。
# ==================== AI修改 结束 ====================

"""图片引用解析与 MinIO URL 构造工具。"""

import re
from pathlib import PurePosixPath
from urllib.parse import quote, unquote, urlparse


# ==================== AI修改 开始 ====================
# 图片后缀判断集中在一个地方，答案节点和测试都复用同一规则，
# 避免前端认为是图片、后端却把同一个 URL 过滤掉。
# ==================== AI修改 结束 ====================
_IMAGE_SUFFIX_RE = re.compile(
    r"\.(?:png|jpe?g|gif|bmp|webp|svg)(?:[?#].*)?$",
    re.IGNORECASE,
)

# ==================== AI修改 开始 ====================
# 图片 URL 后可能被模型或前端粘上下一句中文；图片后缀是可靠边界，
# 只保留后缀和合法 ASCII 查询串，避免把整串误判成非图片。
# ==================== AI修改 结束 ====================
_IMAGE_URL_PREFIX_RE = re.compile(
    r"^(.*?\.(?:png|jpe?g|gif|bmp|webp|svg)"
    r"(?:[?#][A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?)",
    re.IGNORECASE,
)

# ==================== AI修改 开始 ====================
# 同时支持 Markdown 图片和 HTML img 标签，因为项目历史文档两种写法都存在。
# ==================== AI修改 结束 ====================
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_HTML_IMAGE_RE = re.compile(
    r"<img[^>]*?src\s*=\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

# ==================== AI修改 开始 ====================
# 只有用户明确索要图片时才扩大候选范围，普通问答仍只展示最终相关切片里的图，
# 避免每个答案都带出无关图片。
# ==================== AI修改 结束 ====================
_IMAGE_REQUEST_KEYWORDS = (
    "图片", "流程图", "示意图", "架构图", "数据流", "截图",
    "外观", "结构图", "接线图", "展示出来", "给我图",
)


def _endpoint_base(endpoint: str) -> str:
    """把 MinIO endpoint 规范化为带协议、无尾斜杠的地址。"""
    value = str(endpoint or "").strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value


def build_minio_image_url(endpoint: str, bucket: str, object_name: str) -> str:
    """根据 MinIO endpoint、bucket 和对象名生成浏览器访问 URL。"""
    object_parts = [
        # quote 对url中的特殊字符进行百分号编码,核心作用就是将一些危险的字符(空格,中文,斜杠,问号)转化为%xx格式,确保被当做普通数据处理
        # "safe= "所有非字幕数字字符都将被编码
        quote(part, safe="")
        for part in str(object_name or "").strip("/").split("/")
        if part
    ]
    object_path = "/".join(object_parts) #object_name拆分全部quote安全性处理后重新合并
    bucket_path = quote(str(bucket or "").strip("/"), safe="")
    return f"{_endpoint_base(endpoint)}/{bucket_path}/{object_path}"


def extract_image_references(text: str) -> list[str]:
    """提取文本中出现的图片地址，按出现顺序去重。"""
    result: list[str] = []
    for pattern in (_MARKDOWN_IMAGE_RE, _HTML_IMAGE_RE):
        for raw in pattern.findall(str(text or "")):
            value = raw.strip()
            if value and value not in result:
                result.append(value)
    return result


# ==================== AI修改 开始 ====================
# 最终重排结果可能只有“流程图说明”，真正带图片引用的切片在本轮召回候选中。
# 这里在图片请求场景下合并两组切片，只返回去重后的原始引用，URL 映射仍由查询节点统一处理。
# ==================== AI修改 结束 ====================
def collect_image_references_from_chunks(
    primary_chunks: list[dict] | None,
    fallback_chunks: list[dict] | None,
    query: str,
) -> list[str]:
    source_chunks = list(primary_chunks or [])
    query_text = str(query or "")
    if any(keyword in query_text for keyword in _IMAGE_REQUEST_KEYWORDS):
        source_chunks.extend(fallback_chunks or [])

    result: list[str] = []
    for chunk in source_chunks:
        for reference in extract_image_references(chunk.get("content") or ""):
            if reference not in result:
                result.append(reference)
    return result


def _is_image_reference(value: str) -> bool:
    parsed = urlparse(value)
    return bool(_IMAGE_SUFFIX_RE.search(parsed.path or value))


def resolve_image_reference(
    reference: str,
    file_title: str,
    endpoint: str,
    bucket: str,
    image_dir: str,
) -> str:
    """把绝对图片 URL 原样保留，把旧相对路径映射到文档专属 MinIO 目录。"""
    value = unquote(str(reference or "").strip())
    if value.startswith(("http://", "https://")):
        return value

    # 旧切片只记录 images/xxx.jpg，真正的文件名取路径最后一段即可。
    filename = PurePosixPath(value.replace("\\", "/")).name
    object_name = (
        f"{str(image_dir or '').strip('/')}/"
        f"{str(file_title or '').strip('/')}/{filename}"
    )
    return build_minio_image_url(endpoint, bucket, object_name)


def normalize_image_urls(values: list[str] | None) -> list[str]:
    """过滤非图片地址并按原顺序去重。"""
    result: list[str] = []
    for raw in values or []:
        value = str(raw or "").strip()
        match = _IMAGE_URL_PREFIX_RE.match(value)
        if match:
            value = match.group(1)
        if value and _is_image_reference(value) and value not in result:
            result.append(value)
    return result


# ==================== AI修改 开始 ====================
# 本文件新增逻辑到此结束；后续如果扩展图片格式，请同步补充上面的
# 后缀规则和对应测试，避免只改一层导致图片再次在链路中丢失。
# ==================== AI修改 结束 ====================
