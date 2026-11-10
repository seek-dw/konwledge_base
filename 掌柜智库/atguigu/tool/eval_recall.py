# atguigu/tool/eval_recall.py
# ==================== AI修改 开始 ====================
"""
召回率评测脚本
用法(在项目根目录,用项目venv运行):
    # 模式1: 自动从向量库生成评测集并评测(推荐,先跑这个)
    python -m atguigu.tool.eval_recall

    # 模式2: 评测前N个主体(快速抽检)
    python -m atguigu.tool.eval_recall --sample 20

    # 模式3: 使用自定义评测集(JSON文件)
    python -m atguigu.tool.eval_recall --file my_eval.json

    # 模式4: 只评估 RRF，排查 Rerank 是否改变召回
    python -m atguigu.tool.eval_recall --file my_eval.json --skip-rerank

    自定义评测集格式(JSON数组,每个元素):
    {
        "query": "python数据分析课程讲什么",
        "expect_item_names": ["Python数据分析系列"],   # 可选: 期望命中的主体名
        "expect_keywords": ["pandas"],                  # 可选: 期望命中的内容关键词
        "expect_chunk_ids": ["chunk_x"],                # 推荐: 真实 Chunk ground truth
        "expect_document_ids": ["doc_x"]                # 可选: 真实 Document ground truth
    }

指标说明:
    Hit@K     : Top-K条结果中至少有1条命中期望 → 该query记1次命中
    Recall@K  : 期望的主体名/关键词被召回的比例(多期望时按条平均)
    MRR       : 第一条命中结果的排名倒数的平均值(排名越靠前分越高,1.0=全在首位)
    自动生成的评测集只是“主体名冒烟测试”，不能代表真实问答召回率。
    真实评测应在自定义 JSON 中填写 expect_chunk_ids/expect_document_ids。
    评测原理: 固定意图/主体后复用线上向量、Graph、RRF、Rerank节点，
    分别报告 vector、graph、fused、final 四个阶段的指标；没有填写真实
    expect_chunk_ids/expect_document_ids 时只标记为 smoke，不冒充真实召回率。
"""
import argparse
import json
import os
import time

from atguigu.config.config import MilvusConfig
from atguigu.tool.bgem3_create_tool import vectorize_texts
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_create import (
    CHUNK_OUTPUT_FIELDS,
    create_reqs,
    ensure_collection_loaded,
    get_milvus_client,
    my_hybrid_search,
    select_existing_output_fields,
)
# ==================== AI修改 开始 ====================
# 评测直接复用线上召回节点，避免脚本只测 Milvus 而线上实际走了 Graph/RRF/Rerank。
from atguigu.query_process.nodes.node_graph_search import NodeGraphSearch
from atguigu.query_process.nodes.node_rerank import NodeRerank
from atguigu.query_process.nodes.node_rrf import NodeRrf
from atguigu.query_process.nodes.node_search_embedding import NodeSearchEmbedding
from atguigu.query_process.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
# ==================== AI修改 结束 ====================

# 评测默认参数(与线上node_search_embedding完全一致)
RANKER_WEIGHTS = [0.9, 0.1]
# ==================== AI修改 开始 ====================
# 线上检索limit已从10提升到20(扩大初召回池喂饱rerank断崖检测),评测同步
SEARCH_LIMIT = 20          # 线上检索取Top20
EVAL_TOP_K = [1, 3, 5, 10, 20] # 在这几个K上分别算指标
# ==================== AI修改 结束 ====================
# 自动生成评测集的query模板(贴近用户真实问法,避免直接抄主体名造成虚高)
QUERY_TEMPLATES = {
    "course_intro": "{name}这门课主要讲什么内容",
    "question": "{name}题库里有哪几道题",
    "doc": "{name}的核心要点有哪些",
}
DEFAULT_CONTENT_TYPE_TEMPLATE = "doc"


def build_eval_set_from_milvus(sample: int = 0):
    """从chunks表自动生成评测集: 每个主体(item_name)生成一条query,
    ground truth就是该主体名下已有的chunk——这是最可靠的监督信号"""
    client = get_milvus_client()
    if not client:
        raise Exception("Milvus客户端初始化失败,请检查服务是否启动")
    collection = MilvusConfig.milvus_chunks_collection
    if not client.has_collection(collection):
        raise Exception(f"集合 {collection} 不存在,请先导入数据再评测")
    client.load_collection(collection)

    # 查出所有(主体名,内容类型)组合及其chunk数
    res = client.query(
        collection_name=collection,
        filter='item_name != ""',
        output_fields=["item_name", "content_type"],
        limit=16384,
    )
    counter = {}
    for row in res:
        name = row.get("item_name", "")
        ct = row.get("content_type", "doc")
        counter[(name, ct)] = counter.get((name, ct), 0) + 1

    eval_set = []
    items = sorted(counter.items(), key=lambda x: -x[1])  # chunk多的主体排前面
    if sample > 0:
        items = items[:sample]
    for (name, ct), cnt in items:
        template = QUERY_TEMPLATES.get(ct, QUERY_TEMPLATES[DEFAULT_CONTENT_TYPE_TEMPLATE])
        eval_set.append({
            "query": template.format(name=name),
            "expect_item_names": [name],
            "expect_content_type": ct,
            "chunk_count": cnt,
            "evaluation_mode": "self_generated_smoke",
        })
    return eval_set


def search_one(query: str, expr: str = None):
    """保留一个纯向量检索入口，便于排查向量路由本身。"""
    ensure_collection_loaded(MilvusConfig.milvus_chunks_collection)
    embed = vectorize_texts([query])
    dense_data = embed.get("dense")[0]
    sparse_data = embed.get("sparse")[0]
    reqs = create_reqs(
        dense_data=dense_data,
        sparse_data=sparse_data,
        dense_anns_field="dense_vector",
        sparse_anns_field="sparse_vector",
        dense_param={"metric_type": "COSINE"},
        sparse_param={"metric_type": "IP"},
        expr=expr,
    )
    res = my_hybrid_search(
        collection_name=MilvusConfig.milvus_chunks_collection,
        reqs=reqs,
        ranker=RANKER_WEIGHTS,
        output_fields=select_existing_output_fields(
            get_milvus_client(), MilvusConfig.milvus_chunks_collection, CHUNK_OUTPUT_FIELDS
        ),
        limit=SEARCH_LIMIT,
    )
    return [
        {**item.get("entity", {}), "score": item.get("distance")}
        for item in res[0]
    ]


# ==================== AI修改 开始 ====================
# 固定评测输入中的意图和主体，跳过线上 LLM 意图识别，只测真实召回链路。
def search_pipeline_one(case: dict, skip_rerank: bool = False) -> dict:
    state = {
        "original_query": case["query"],
        "rewritten_query": case.get("rewritten_query") or case["query"],
        "query_type": case.get("query_type") or {
            "course_intro": "course",
            "question": "question",
        }.get(case.get("expect_content_type"), "knowledge"),
        "item_names": case.get("item_names") or case.get("expect_item_names") or [],
    }
    vector = NodeSearchEmbedding().process(state).get("embedding_chunks", [])
    graph = NodeGraphSearch().process(state).get("graph_chunks", [])
    hyde = []
    if os.getenv("ENABLE_HYDE", "false").strip().lower() in ("1", "true", "yes", "on"):
        hyde = NodeSearchEmbeddingHyde().process(state).get("hyde_embedding_chunks", [])
    fused = NodeRrf().process({
        "embedding_chunks": vector,
        "hyde_embedding_chunks": hyde,
        "graph_chunks": graph,
    }).get("rrf_chunks", [])
    final = fused
    if not skip_rerank:
        final = NodeRerank().process({
            "rrf_chunks": fused,
            "web_search_docs": [],
            "rewritten_query": state["rewritten_query"],
        }).get("reranked_docs", [])
    return {"vector": vector, "graph": graph, "hyde": hyde, "fused": fused, "final": final}
# ==================== AI修改 结束 ====================


def _match_target(result: dict, target_type: str, target: str) -> bool:
    if target_type == "chunk_id":
        return result.get("chunk_id") == target
    if target_type == "document_id":
        return result.get("document_id") == target
    if target_type == "item_name":
        return target in (result.get("item_name") or "")
    return target in (result.get("content") or "").lower()


def _route_result(results, targets):
    hit_ranks = []
    for target_type, target in targets:
        rank = next(
            (index + 1 for index, result in enumerate(results)
             if _match_target(result, target_type, target)),
            None,
        )
        if rank:
            hit_ranks.append(rank)
    per_k = {
        k: sum(rank <= k for rank in hit_ranks) / max(1, len(targets))
        for k in EVAL_TOP_K
    }
    first_rank = min(hit_ranks) if hit_ranks else None
    return {
        "recall": per_k,
        "hit": bool(hit_ranks),
        "first_rank": first_rank,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


# ==================== AI修改 开始 ====================
# 真实评测使用 Chunk/Document ID；主体名和关键词仅作为 smoke 兼容指标。
def evaluate_one(case: dict, skip_rerank: bool = False):
    expect_chunks = set(case.get("expect_chunk_ids") or [])
    expect_documents = set(case.get("expect_document_ids") or [])
    targets = [("chunk_id", target) for target in expect_chunks]
    targets += [("document_id", target) for target in expect_documents]
    if not targets:
        targets = [("item_name", target) for target in set(case.get("expect_item_names") or [])]
        targets += [("keyword", target.lower()) for target in case.get("expect_keywords") or []]

    pipeline = search_pipeline_one(case, skip_rerank=skip_rerank)
    route_results = {
        route: _route_result(pipeline.get(route, []), targets)
        for route in ("vector", "graph", "hyde", "fused", "final")
    }
    final = route_results["final"]
    final_results = pipeline.get("final", [])
    return {
        "query": case["query"],
        "evaluation_mode": "ground_truth" if expect_chunks or expect_documents else "smoke",
        "has_ground_truth": bool(expect_chunks or expect_documents),
        "expected_chunk_ids": sorted(expect_chunks),
        "expected_document_ids": sorted(expect_documents),
        "n_results": len(final_results),
        "hit": final["hit"],
        "first_rank": final["first_rank"],
        "recall_per_k": final["recall"],
        "route_recall": {
            route: data["recall"] for route, data in route_results.items()
        },
        "route_mrr": {
            route: data["mrr"] for route, data in route_results.items()
        },
        "top1_item": final_results[0].get("item_name") if final_results else "",
        "top1_title": final_results[0].get("title") if final_results else "",
    }
# ==================== AI修改 结束 ====================


def run_eval(eval_set, out_path="eval_recall_report.json", skip_rerank=False):
    total = len(eval_set)
    logger.info(f"开始评测,共{total}条query,每条检索Top{SEARCH_LIMIT}...")
    route_stats = {
        route: {k: 0.0 for k in EVAL_TOP_K}
        for route in ("vector", "graph", "hyde", "fused", "final")
    }
    route_mrr = {route: 0.0 for route in route_stats}
    details, miss_cases = [], []
    case_error_count = 0
    t0 = time.time()
    for i, case in enumerate(eval_set):
        try:
            result = evaluate_one(case, skip_rerank=skip_rerank)
        except Exception as exc:
            case_error_count += 1
            logger.error(f"第{i + 1}条评测异常: {exc}")
            details.append({"query": case.get("query", ""), "error": str(exc)})
            continue
        details.append(result)
        for route in route_stats:
            for k in EVAL_TOP_K:
                route_stats[route][k] += result["route_recall"][route][k]
            route_mrr[route] += result["route_mrr"][route]
        if not result["hit"]:
            miss_cases.append({"query": result["query"], "top1_item": result["top1_item"],
                               "top1_title": result["top1_title"]})
        if (i + 1) % 10 == 0:
            logger.info(f"进度 {i + 1}/{total}")

    evaluated = len(eval_set) - case_error_count
    denominator = max(1, evaluated)
    route_metrics = {
        route: {
            **{f"Recall@{k}": round(route_stats[route][k] / denominator, 4)
               for k in EVAL_TOP_K},
            "MRR": round(route_mrr[route] / denominator, 4),
        }
        for route in route_stats
    }
    ground_truth_count = sum(
        result.get("has_ground_truth", False)
        for result in details if "error" not in result
    )
    final_metrics = route_metrics["final"]
    summary = {
        "总query数": total,
        "实际评测query数": evaluated,
        "评测异常数": case_error_count,
        "真实ground_truth数量": ground_truth_count,
        "指标": final_metrics,
        "各阶段指标": route_metrics,
        "评测耗时(秒)": round(time.time() - t0, 1),
        "评测配置": {
            "ranker权重(稠密,稀疏)": RANKER_WEIGHTS,
            "检索条数limit": SEARCH_LIMIT,
            "向量库": MilvusConfig.milvus_chunks_collection,
            "是否跳过Rerank": skip_rerank,
            "评测性质": "真实 ID 评测" if ground_truth_count else "主体/关键词 smoke，不能代表真实召回",
        },
        "未命中query清单": miss_cases[:50],
        "明细": details,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 52)
    print("           掌柜智库 召回率评测报告")
    print("=" * 52)
    print(f"  评测query数 : {total} (成功 {evaluated}, 异常 {case_error_count})")
    for route in ("vector", "graph", "fused", "final"):
        print(f"  {route:<7} Recall@{SEARCH_LIMIT}: {route_metrics[route][f'Recall@{SEARCH_LIMIT}']:.2%}")
    print(f"  final MRR  : {final_metrics['MRR']:.2%}")
    print(f"  完整报告已保存: {os.path.abspath(out_path)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="掌柜智库召回率评测")
    parser.add_argument("--sample", type=int, default=0, help="只抽前N个主体评测(0=全部)")
    parser.add_argument("--file", type=str, default="", help="自定义评测集JSON路径")
    parser.add_argument("--out", type=str, default="eval_recall_report.json", help="报告输出路径")
    parser.add_argument("--skip-rerank", action="store_true", help="跳过Rerank，只评估RRF结果")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            eval_set = json.load(f)
        logger.info(f"加载自定义评测集: {args.file},共{len(eval_set)}条")
    else:
        eval_set = build_eval_set_from_milvus(sample=args.sample)
        logger.info(f"从向量库自动生成评测集{len(eval_set)}条(每个主体一条query)")
        if not eval_set:
            logger.error("向量库中没有带item_name的数据,无法自动生成评测集")
            return
    run_eval(eval_set, out_path=args.out, skip_rerank=args.skip_rerank)


if __name__ == "__main__":
    main()
# ==================== AI修改 结束 ====================
