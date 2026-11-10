# atguigu/query_process/nodes/node_answer_output.py
import threading
from pathlib import Path

from langchain.chat_models import init_chat_model

# ==================== AI修改 开始 ====================
# 同时读取 MinIO 配置，旧切片中的相对图片路径需要在答案阶段补成完整 URL。
# ==================== AI修改 结束 ====================
from atguigu.config.config import MinioConfig, ModelConfig
# ==================== AI修改 开始 ====================
# 新增导入：闲聊提示词CHAT_PROMPT、无结果兜底提示词NO_RESULT_PROMPT
from atguigu.config.prompt import ANSWER_PROMPT, CHAT_PROMPT, NO_RESULT_PROMPT
# ==================== AI修改 结束 ====================
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger
from atguigu.tool.mongo_client_tool import add_or_update_history, get_recent_history_list, get_session_summary, compress_session_history
# ==================== AI修改 开始 ====================
# 答案流在每个增量处检查取消标记，主动停止时关闭模型流。
from atguigu.tool.task_utils import put_data, is_task_cancelled
# ==================== AI修改 结束 ====================
# ==================== AI修改 开始 ====================
# 图片解析规则与导入节点共用，确保 Markdown/HTML 两种写法处理一致。
# ==================== AI修改 结束 ====================
from atguigu.tool.image_url_tool import (
    collect_image_references_from_chunks,
    extract_image_references,
    normalize_image_urls,
    resolve_image_reference,
)

# ==================== AI修改 开始 ====================
# 只裁剪最早历史，保留最新一轮对话。
def trim_history_content(history_content: str, max_chars: int) -> str:
    if not history_content or max_chars <= 0:
        return ""
    if len(history_content) <= max_chars:
        return history_content
    marker = "【更早的历史已折叠】\n"
    keep_chars = max(0, max_chars - len(marker))
    return marker + history_content[-keep_chars:]
    # ==================== AI修改 结束 ====================


# ==================== AI修改 开始 ====================
# 按模型总窗口分配检索资料和历史，输出预算与安全余量优先保留。
def fit_prompt_fields(
    prompt_template: str,
    *,
    context: str,
    history: str,
    question: str,
    item_names: str,
) -> tuple[str, str]:
    fixed_prompt = prompt_template.format(
        context="",
        history="",
        question=question or "",
        item_names=item_names or "",
    )
    available_tokens = max(
        2048,
        ModelConfig.LLM_CONTEXT_WINDOW_TOKENS
        - ModelConfig.LLM_MAX_TOKENS
        - ModelConfig.LLM_PROMPT_SAFETY_TOKENS,
    )
    prompt_char_budget = max(8000, int(available_tokens * 0.85))
    fields_budget = max(0, prompt_char_budget - len(fixed_prompt))

    desired_context = min(
        len(context or ""), ModelConfig.ANSWER_MAX_CONTEXT_CHARS
    )
    desired_history = min(
        len(history or ""), ModelConfig.ANSWER_MAX_HISTORY_CHARS
    )

    # 检索资料优先占约65%，剩余空间给历史，再回填双方未用空间。
    context_limit = min(desired_context, int(fields_budget * 0.65))
    history_limit = min(desired_history, fields_budget - context_limit)
    remaining = fields_budget - context_limit - history_limit
    if remaining > 0:
        extra_context = min(remaining, desired_context - context_limit)
        context_limit += extra_context
        remaining -= extra_context
        history_limit += min(remaining, desired_history - history_limit)
    fitted_context = (context or "")[:context_limit]
    fitted_history = trim_history_content(history or "", history_limit)
    return fitted_context, fitted_history

# 判断 provider 是否因输出预算耗尽而结束。
def is_length_finish_reason(reason: object) -> bool:
    """判断流式模型是否因为达到输出长度上限而结束。"""
    return str(reason or "").lower() in {
        "length",
        "max_tokens",
        "token_limit",
    }

# provider 截断时构造一次只追加尾部的续写请求。
def build_continuation_prompt(prompt: str, answer: str) -> str:

    question = ""
    for marker in ("【用户问题】", "【用户消息】"):
        if marker in prompt:
            question = prompt.split(marker, 1)[1]
            question = question.split("请回答：", 1)[0]
            question = question.split("请回应：", 1)[0]
            break
    if not question.strip():
        question = prompt[-2000:]
    answer_tail = (answer or "")[-ModelConfig.LLM_CONTINUATION_CONTEXT_CHARS:]
    return (
        "上一条回答因服务端输出上限被截断。请从‘上一条回答末尾’的最后一个字符继续，"
        "只输出需要追加的内容，不要重复开头、标题、已经出现的句子，也不要解释你在续写。"
        "先补完当前句子、列表项或代码块，再在完整自然的边界结束；如果原回答尚未完成"
        "【来源】或【图片】区块，也要补完整后再结束。\n\n"
        f"原始用户问题：\n{question.strip()}\n\n"
        f"上一条回答末尾：\n{answer_tail}"
    )
    # ==================== AI修改 结束 ====================


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案生成
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_answer_output"

    # ==================== AI修改 开始 ====================
    # 复用 LLM 客户端，避免每轮重新建立连接。
    _llm = None
    # ==================== AI修改 结束 ====================

    # ==================== AI修改 开始 ====================
    # 统一处理流式输出、上下文占用上报和 provider 截断续写。
    def stream_llm_answer(self, task_id: str, prompt: str, *,
                          context_chars=0, history_chars=0, question_chars=0,
                          summary_active=False) -> str:
        # 上报总量和检索/记忆/问题的分项长度，供前端显示。
        try:
            _chars = len(prompt)
            _tokens = int(_chars / 1.5)
            # 展示上限与实际模型预算分开，避免进度条一开始满格。
            _limit = ModelConfig.ANSWER_CONTEXT_DISPLAY_LIMIT
            put_data(task_id, "context_info", {
                "chars": _chars,
                "tokens": _tokens,
                "limit": _limit,
                "pct": min(100, int(_chars * 100 / _limit)),
                # 让进度条能区分检索资料和对话记忆。
                "context_chars": context_chars,   # 检索到的知识库内容
                "history_chars": history_chars,   # 对话记忆(含长期摘要)
                "question_chars": question_chars,  # 当前问题
                "summary_active": summary_active,  # 长期记忆摘要是否已激活
            })
        except Exception:
            pass
        # 懒加载并复用 LLM 客户端。
        if self._llm is None:
            # 输出上限和续写次数统一由配置控制。
            self._llm = init_chat_model(
                model = ModelConfig.LLM_MODEL_NAME,
                model_provider="openai",
                # 文本回答使用当前平台的统一配置。
                api_key = ModelConfig.LLM_API_KEY,
                base_url = ModelConfig.LLM_BASE_URL,
                temperature = ModelConfig.MODEL_TEMPERATURE,
                # 所有回答请求使用同一个输出上限。
                max_tokens = ModelConfig.LLM_MAX_TOKENS
            )
        llm = self._llm
        messages = [{"role": "user", "content": prompt}]
        answer = ""
        continuation_count = 0
        try:
            while True:
                # ==================== AI修改 开始 ====================
                if is_task_cancelled(task_id):
                    break
                # ==================== AI修改 结束 ====================
                response_stream = None
                response_text = ""
                finish_reason = None
                try:
                    # 每次只把本次新增内容推送给前端；硬截断时循环请求续写。
                    response_stream = llm.stream(input=messages)
                    for response in response_stream:
                        # ==================== AI修改 开始 ====================
                        if is_task_cancelled(task_id):
                            break
                        # ==================== AI修改 结束 ====================
                        delta = response.content or ""
                        if delta:
                            put_data(task_id, "delta", {"delta": delta})
                            response_text += delta
                        # 读取 provider 返回的结束原因。
                        metadata = getattr(response, "response_metadata", {}) or {}
                        additional = getattr(response, "additional_kwargs", {}) or {}
                        # 从左往右找到第一个有值的reason
                        finish_reason = (
                            metadata.get("finish_reason")
                            or additional.get("finish_reason")
                            or finish_reason
                        )
                finally:
                    close_stream = getattr(response_stream, "close", None)
                    if callable(close_stream):
                        try:
                            close_stream()
                        except Exception as close_error:
                            logger.warning(f"关闭答案生成流失败: {close_error}")

                answer += response_text
                # ==================== AI修改 开始 ====================
                if is_task_cancelled(task_id):
                    break
                # ==================== AI修改 结束 ====================
                # 只有达到 provider 长度上限时才续写。
                if not is_length_finish_reason(finish_reason):
                    break
                if continuation_count >= ModelConfig.LLM_MAX_CONTINUATIONS:
                    logger.warning(
                        "答案连续触发 provider 输出上限，已达到续写保护次数"
                    )
                    if answer:
                        notice = "\n\n（回答生成达到服务限制，以上为已生成内容。）"
                        put_data(task_id, "delta", {"delta": notice})
                        answer += notice
                    break

                continuation_count += 1
                messages = [{
                    "role": "user",
                    "content": build_continuation_prompt(prompt, answer),
                }]
        except Exception as e:
            # LLM流式调用失败(网络/超时)：已生成的部分照常返回,一段都没有则给出兜底话术,绝不静默崩溃
            logger.error(f"答案生成LLM流式调用失败: {e}")
            if not answer:
                answer = "抱歉，回答生成时出现了问题，请稍后重试。"
            else:
                # 已有部分内容时显式收尾，避免前端停在半句话。
                notice = "\n\n（生成连接中途结束，以上为已完成内容。）"
                put_data(task_id, "delta", {"delta": notice})
                answer += notice
        return answer
        # ==================== AI修改 结束 ====================

    # 组装历史：长期摘要 + 最近若干条原文。
    # ==================== AI修改 开始 ====================
    def make_history_content(self, state: QueryGraphState) -> str:
        session_id = state.get("session_id", "")
        try:
            # ==================== AI修改 开始 ====================
            # 最近历史条数由统一配置控制。
            raw = get_recent_history_list(
                session_id,
                limit=ModelConfig.QUERY_HISTORY_LIMIT,
            )
            # ==================== AI修改 结束 ====================
        except Exception:
            raw = []
        # reversed内置反转函数,接收字符串,不能用于数字和生成器,返回迭代器,作用是反转列表
        raw_asc = list(reversed(raw)) if raw else []
        recent_content = "".join(f"{m.get('role','')}: {m.get('text','')}\n\n" for m in raw_asc)
        # 长期摘要承接更早的对话。
        long_term = ""
        try:
            summary = get_session_summary(session_id)
            if summary:
                long_term = f"【长期记忆摘要(较早对话的要点)】\n{summary}\n\n"
        except Exception:
            pass
        # ==================== AI修改 开始 ====================
        # 超限时只折叠最早内容，保留最近对话。
        history_content = (long_term + "【近期对话】\n" + recent_content) if long_term else recent_content
        return trim_history_content(history_content, ModelConfig.ANSWER_MAX_HISTORY_CHARS)
        # ==================== AI修改 结束 ====================
    # ==================== AI修改 结束 ====================

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 反馈话术也要写入历史，避免下一轮丢失确认上下文。
        answer = state.get("answer")
        task_id = state.get("task_id")
        session_id = state.get("session_id")
        original_query = state.get("original_query")

        # ========== 分支1：需要用户确认/引导的反馈话术，直接推送，不调LLM ==========
        if answer:
            add_or_update_history(
                session_id=session_id, role="assistant", text=answer,
                rewritten_query=original_query, item_names=[], image_urls=[],
                msg_type="chat"
            )
            put_data(task_id, "final", {"answer": answer, "image_urls": []})
            return {"answer": answer}

        #拿取长期记忆和短期记忆
        history_content = self.make_history_content(state)
        query_type = state.get("query_type", "")
        chunks = state.get("reranked_docs") or []

        # 闲聊或无检索结果时使用对应的通用提示词。
        if query_type == "chitchat" or not chunks:
            template = CHAT_PROMPT if query_type == "chitchat" else NO_RESULT_PROMPT
            # ==================== AI修改 开始 ====================
            # 闲聊和无结果分支也服从总上下文预算。
            _, history_content = fit_prompt_fields(
                template,
                context="",
                history=history_content,
                question=original_query,
                item_names="",
            )
            prompt = template.format(history=history_content, question=original_query)
            # ==================== AI修改 结束 ====================
            # ==================== AI修改 开始 ====================
            # 该分支没有检索资料，只上报记忆和问题长度。
            _summary_active = history_content.startswith("【长期记忆摘要")
            answer = self.stream_llm_answer(
                task_id, prompt,
                context_chars=0,
                history_chars=len(history_content),
                question_chars=len(original_query or ""),
                summary_active=_summary_active,
            )
            # 无论生成是否为空都写入 assistant 记录，保持历史成对。
            # ==================== AI修改 结束 ====================
            _mt = "chat" if query_type == "chitchat" else "qa"
            add_or_update_history(
                session_id=session_id, role="assistant",
                text=answer or "抱歉，回答生成时出现了问题，请稍后重试。",
                rewritten_query=original_query, item_names=[], image_urls=[],
                msg_type=_mt
            )
            # ==================== AI修改 开始 ====================
            # final 始终携带答案，前端无需依赖 delta 拼接。
            put_data(task_id, "final", {
                "answer": answer or "",
                "image_urls": [],
            })
            # ==================== AI修改 结束 ====================
            # 长期记忆: 异步压缩超阈值的早期对话(不阻塞响应)
            threading.Thread(target=compress_session_history, args=(session_id, self._llm), daemon=True).start()
            return {"answer": answer}

        # 正常检索生成。
        chunk_content = ""
        # ==================== AI修改 开始 ====================
        # 回答上下文只保留章节、来源、正文和图路径。
        def format_chunk(chunk):
            section_path = chunk.get("section_path") or chunk.get("item_name") or "未命名章节"
            source_path = chunk.get("source_path") or chunk.get("url") or ""
            graph_paths = chunk.get("graph_path") or []
            graph_text = "；".join(
                " -> ".join(path) for path in graph_paths if path
            )
            relation = f"\n关系路径：{graph_text}" if graph_text else ""
            return f"[章节：{section_path}] [来源：{source_path}]{relation}\n{chunk.get('content') or ''}\n\n"

        chunk_content = "".join(format_chunk(chunk) for chunk in chunks)
        MAX_CONTEXT_CHARS = ModelConfig.ANSWER_MAX_CONTEXT_CHARS
        while len(chunk_content) > MAX_CONTEXT_CHARS and len(chunks) > 1:
            chunks = chunks[:-1]
            chunk_content = "".join(format_chunk(chunk) for chunk in chunks)
        if len(chunk_content) > MAX_CONTEXT_CHARS:
            chunk_content = chunk_content[:MAX_CONTEXT_CHARS]
        # ==================== AI修改 结束 ====================

        item_names = state.get("item_names") or []
        item_names_str = ",".join(item_names)
        rewritten_query = state.get("rewritten_query")

        # ==================== AI修改 开始 ====================
        # 最终组装前再次按总窗口压缩检索和历史。
        chunk_content, history_content = fit_prompt_fields(
            ANSWER_PROMPT,
            context=chunk_content,
            history=history_content,
            question=rewritten_query,
            item_names=item_names_str,
        )
        # ==================== AI修改 结束 ====================

        prompt = ANSWER_PROMPT.format(
            context = chunk_content,
            history = history_content,
            question = rewritten_query,
            item_names = item_names_str
            )

        #调用大模型(改为公共流式方法)
        # ==================== AI修改 开始 ====================
        # 传入组成分解: 检索context + 记忆history + 问题question 三段
        # summary_active 判断长期记忆摘要是否已激活(与记忆压缩直接挂钩)
        _summary_active = history_content.startswith("【长期记忆摘要")
        answer = self.stream_llm_answer(
            task_id, prompt,
            context_chars=len(chunk_content),
            history_chars=len(history_content),
            question_chars=len(rewritten_query or original_query or ""),
            summary_active=_summary_active,
        )
        # ==================== AI修改 结束 ====================

        # ==================== AI修改 开始 ====================
        # 图片请求不能只看最终 rerank 结果：被断崖裁掉的向量/Graph 切片中
        # 可能仍有用户明确要看的图片。合并本轮各路候选并按 chunk 去重。
        retrieval_chunks = []
        seen_chunk_ids = set()
        for route in (
            "reranked_docs",
            "rrf_chunks",
            "embedding_chunks",
            "hyde_embedding_chunks",
            "graph_chunks",
        ):
            for candidate in state.get(route) or []:
                chunk_id = candidate.get("chunk_id") or candidate.get("id") or id(candidate)
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    retrieval_chunks.append(candidate)
        # 统一补全旧切片中的相对图片路径；图片请求时从全候选补抓图片。
        fallback_chunks = retrieval_chunks
        primary_references = collect_image_references_from_chunks(
            chunks, [], original_query or rewritten_query or ""
        )
        fallback_references = collect_image_references_from_chunks(
            [], fallback_chunks, original_query or rewritten_query or ""
        )
        allowed_references = set(primary_references + fallback_references)
        image_source_chunks = list(chunks)
        if fallback_references:
            image_source_chunks = retrieval_chunks
        # ==================== AI修改 结束 ====================

        image_candidates = []
        for doc in image_source_chunks:
            text = doc.get("content") or ""
            # ==================== AI修改 开始 ====================
            # 新切片有 source_path，旧切片可能只保留 file_title；两者都能定位图片目录。
            # ==================== AI修改 结束 ====================
            file_title = Path(str(
                doc.get("source_path") or doc.get("file_title") or ""
            )).stem
            for reference in extract_image_references(text):
                if reference not in allowed_references:
                    continue
                if reference.startswith(("http://", "https://")):
                    image_candidates.append(reference)
                elif file_title:
                    image_candidates.append(resolve_image_reference(
                        reference,
                        file_title,
                        MinioConfig.MINIO_ENDPOINT,
                        MinioConfig.MINIO_BUCKET_NAME,
                        MinioConfig.MINIO_IMG_DIR,
                    ))
        images = normalize_image_urls(image_candidates)
        # 记录召回切片中的图片数量，便于区分“没有召回图片”和“前端加载失败”。
        logger.info(f"答案图片提取完成: 召回{len(chunks)}个切片，返回{len(images)}张图片")

        #保存历史
        # ==================== AI修改 开始 ====================
        # 无论生成是否为空都写入 assistant 记录，保持历史链完整。
        add_or_update_history(
            session_id = session_id,
            role = "assistant",
            text = answer or "抱歉，回答生成时出现了问题，请稍后重试。",
            rewritten_query = rewritten_query,
            item_names = item_names,
            image_urls = images,
            msg_type = "qa"
        )
        # ==================== AI修改 结束 ====================
        # final 同时携带完整答案和图片列表。
        # ==================== AI修改 开始 ====================
        put_data(task_id, "final", {
            "answer": answer or "",
            "image_urls": images,
        })
        # 长期记忆: 异步压缩超阈值的早期对话(不阻塞响应, self._llm已在stream_llm_answer初始化)
        if self._llm is not None:
            threading.Thread(target=compress_session_history, args=(session_id, self._llm), daemon=True).start()
        return {
            "answer":answer
        }
        # ==================== AI修改 结束 ====================


