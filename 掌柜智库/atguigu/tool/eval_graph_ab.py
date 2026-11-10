# ==================== AI修改 开始 ====================
"""离线比较 Graph RAG 开关前后的召回效果。

运行位置：掌柜智库目录
    python -m atguigu.tool.eval_graph_ab --sample 20
    python -m atguigu.tool.eval_graph_ab --file tests/graph_eval_fixture.json
"""

import argparse
import json
import os
import time
from pathlib import Path

from atguigu.config.config import MilvusConfig
from atguigu.query_process.nodes.node_graph_search import NodeGraphSearch
from atguigu.query_process.nodes.node_rerank import NodeRerank
from atguigu.query_process.nodes.node_rrf import NodeRrf
from atguigu.query_process.nodes.node_search_embedding import NodeSearchEmbedding
from atguigu.query_process.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from atguigu.tool.eval_recall import (
    EVAL_TOP_K,
    SEARCH_LIMIT,
    build_eval_set_from_milvus,
)
from atguigu.tool.logger import logger


# None 表示跟随 .env；命令行 --disable-hyde 只影响本次离线评测。
ENABLE_HYDE_OVERRIDE = None


def _state_for_case(case: dict) -> dict:
    return {
        "original_query": case["query"],
        "rewritten_query": case.get("rewritten_query") or case["query"],
        "query_type": case.get("query_type") or {
            "course_intro": "course",
            "question": "question",
        }.get(case.get("expect_content_type"), "knowledge"),
        "item_names": case.get("item_names") or case.get("expect_item_names") or [],
    }


def _hyde_enabled() -> bool:
    if ENABLE_HYDE_OVERRIDE is not None:
        return ENABLE_HYDE_OVERRIDE
    return os.getenv("ENABLE_HYDE", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _targets_for_case(case: dict) -> list[tuple[str, object]]:
    targets = [("chunk_id", value) for value in case.get("expect_chunk_ids") or []]
    targets += [("document_id", value) for value in case.get("expect_document_ids") or []]
    if targets:
        return targets
    targets = [("item_name", value) for value in case.get("expect_item_names") or []]
    return targets + [("keyword", value) for value in case.get("expect_keywords") or []]


def _matches(result: dict, target_type: str, target: object) -> bool:
    if target_type in {"chunk_id", "document_id"}:
        field = target_type
        return result.get(field) is not None and str(result.get(field)) == str(target)
    if target_type == "item_name":
        return str(target) in (result.get("item_name") or "")
    return str(target).casefold() in (result.get("content") or "").casefold()


def _route_result(results: list[dict], targets: list[tuple[str, object]]) -> dict:
    ranks = []
    matched_targets = []
    for target_type, target in targets:
        rank = next(
            (index + 1 for index, result in enumerate(results)
             if _matches(result, target_type, target)),
            None,
        )
        if rank:
            ranks.append(rank)
            matched_targets.append(f"{target_type}:{target}")
    return {
        "recall": {
            k: sum(rank <= k for rank in ranks) / max(1, len(targets))
            for k in EVAL_TOP_K
        },
        "hit": bool(ranks),
        "first_rank": min(ranks) if ranks else None,
        "mrr": 1.0 / min(ranks) if ranks else 0.0,
        "matched_targets": matched_targets,
    }


def _compact(result: dict) -> dict:
    return {
        key: result.get(key, default)
        for key, default in (
            ("chunk_id", ""),
            ("document_id", ""),
            ("source", ""),
            ("score", 0.0),
            ("item_name", ""),
            ("title", ""),
            ("graph_path", []),
        )
    }


def _timed(callable_):
    started = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - started


def collect_route_results(case: dict, skip_rerank: bool = False) -> dict:
    """执行一次共享召回，再从同一批结果派生 Graph off/on 两条链。"""
    state = _state_for_case(case)
    started = time.perf_counter()
    vector = NodeSearchEmbedding().process(state).get("embedding_chunks", [])
    hyde = []
    if _hyde_enabled():
        hyde = NodeSearchEmbeddingHyde().process(state).get("hyde_embedding_chunks", [])
    shared_seconds = time.perf_counter() - started

    graph, graph_seconds = _timed(
        lambda: NodeGraphSearch().process(state).get("graph_chunks", [])
    )
    inputs = {
        "embedding_chunks": vector,
        "hyde_embedding_chunks": hyde,
    }
    baseline_fused, baseline_fusion_seconds = _timed(
        lambda: NodeRrf().process({**inputs, "graph_chunks": []}).get("rrf_chunks", [])
    )
    graph_fused, graph_fusion_seconds = _timed(
        lambda: NodeRrf().process({**inputs, "graph_chunks": graph}).get("rrf_chunks", [])
    )

    baseline_final = baseline_fused
    graph_final = graph_fused
    baseline_rerank_seconds = 0.0
    graph_rerank_seconds = 0.0
    if not skip_rerank:
        baseline_final, baseline_rerank_seconds = _timed(
            lambda: NodeRerank().process({
                "rrf_chunks": baseline_fused,
                "web_search_docs": [],
                "rewritten_query": state["rewritten_query"],
            }).get("reranked_docs", [])
        )
        graph_final, graph_rerank_seconds = _timed(
            lambda: NodeRerank().process({
                "rrf_chunks": graph_fused,
                "web_search_docs": [],
                "rewritten_query": state["rewritten_query"],
            }).get("reranked_docs", [])
        )

    return {
        "shared": {"vector": vector, "hyde": hyde},
        "vector": vector,
        "hyde": hyde,
        "graph": graph,
        "baseline_fused": baseline_fused,
        "graph_fused": graph_fused,
        "baseline_final": baseline_final,
        "graph_final": graph_final,
        "timing": {
            "shared_seconds": shared_seconds,
            "graph_seconds": graph_seconds,
            "baseline_fusion_seconds": baseline_fusion_seconds,
            "graph_fusion_seconds": graph_fusion_seconds,
            "baseline_rerank_seconds": baseline_rerank_seconds,
            "graph_rerank_seconds": graph_rerank_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }


def evaluate_pipeline(case: dict, final_results: list[dict], route_results: dict) -> dict:
    """计算单条 query 的 A/B 指标；允许测试传入简化的 route_results。"""
    shared = route_results.get("shared") or {}
    vector = route_results.get("vector", shared.get("vector", []))
    hyde = route_results.get("hyde", shared.get("hyde", []))
    graph = route_results.get("graph", [])
    baseline_fused = route_results.get("baseline_fused", route_results.get("off", []))
    graph_fused = route_results.get("graph_fused", route_results.get("on", []))
    baseline_final = route_results.get("baseline_final", route_results.get("off", []))
    graph_final = final_results or route_results.get(
        "graph_final", route_results.get("on", [])
    )
    targets = _targets_for_case(case)
    routes = {
        "vector": vector,
        "hyde": hyde,
        "graph": graph,
        "fused": baseline_fused,
        "final": baseline_final,
    }
    graph_routes = {
        "vector": vector,
        "hyde": hyde,
        "graph": graph,
        "fused": graph_fused,
        "final": graph_final,
    }
    baseline_eval = {name: _route_result(items, targets) for name, items in routes.items()}
    graph_eval = {name: _route_result(items, targets) for name, items in graph_routes.items()}
    baseline_targets = set(baseline_eval["final"]["matched_targets"])
    graph_targets = set(graph_eval["final"]["matched_targets"])
    graph_route_targets = set(graph_eval["graph"]["matched_targets"])

    return {
        "query": case["query"],
        "evaluation_mode": "ground_truth" if any(
            case.get(key) for key in ("expect_chunk_ids", "expect_document_ids")
        ) else "smoke",
        "has_ground_truth": bool(
            case.get("expect_chunk_ids") or case.get("expect_document_ids")
        ),
        "baseline": {
            "metrics": {
                name: {
                    "recall": data["recall"],
                    "MRR": data["mrr"],
                }
                for name, data in baseline_eval.items()
            },
            "details": {name: [_compact(item) for item in items] for name, items in routes.items()},
        },
        "graph_on": {
            "metrics": {
                name: {
                    "recall": data["recall"],
                    "MRR": data["mrr"],
                }
                for name, data in graph_eval.items()
            },
            "details": {name: [_compact(item) for item in items] for name, items in graph_routes.items()},
        },
        "matched_targets": {
            "baseline_final": sorted(baseline_targets),
            "graph_route": sorted(graph_route_targets),
            "graph_final": sorted(graph_targets),
        },
        "timing": route_results.get("timing", {}),
        "graph_unique_hit": bool(graph_route_targets - baseline_targets),
        "graph_final_gain": bool(graph_targets - baseline_targets),
        "regression": bool(baseline_targets - graph_targets),
    }


def _format_metrics(metrics: dict) -> dict:
    return {
        route: {
            **{f"Recall@{k}": round(data["recall"][k], 4) for k in EVAL_TOP_K},
            "MRR": round(data["MRR"], 4),
        }
        for route, data in metrics.items()
    }


def _average_metrics(details: list[dict], branch: str) -> dict:
    routes = ("vector", "hyde", "graph", "fused", "final")
    totals = {
        route: {"recall": {k: 0.0 for k in EVAL_TOP_K}, "MRR": 0.0}
        for route in routes
    }
    for detail in details:
        for route, data in detail[branch]["metrics"].items():
            for k in EVAL_TOP_K:
                totals[route]["recall"][k] += data["recall"][k]
            totals[route]["MRR"] += data["MRR"]
    denominator = max(1, len(details))
    for data in totals.values():
        for k in EVAL_TOP_K:
            data["recall"][k] /= denominator
        data["MRR"] /= denominator
    return _format_metrics(totals)


def run_ab(eval_set: list[dict], out_path: str = "graph_ab_report.json", skip_rerank: bool = False) -> dict:
    started = time.perf_counter()
    baseline_details = []
    graph_details = []
    deltas = []
    timing_totals = {}
    errors = 0

    for index, case in enumerate(eval_set, start=1):
        try:
            routes = collect_route_results(case, skip_rerank=skip_rerank)
            result = evaluate_pipeline(case, routes.get("graph_final", []), routes)
        except Exception as exc:
            errors += 1
            logger.error(f"第{index}条 Graph A/B 评测异常: {exc}")
            baseline_details.append({"query": case.get("query", ""), "error": str(exc)})
            graph_details.append({"query": case.get("query", ""), "error": str(exc)})
            continue

        baseline_details.append({"query": result["query"], **result["baseline"]})
        graph_details.append({"query": result["query"], **result["graph_on"]})
        deltas.append({
            "query": result["query"],
            "evaluation_mode": result["evaluation_mode"],
            "graph_unique_hit": result["graph_unique_hit"],
            "graph_final_gain": result["graph_final_gain"],
            "regression": result["regression"],
            "matched_targets": result["matched_targets"],
            "timing": result["timing"],
        })
        for name, value in result["timing"].items():
            timing_totals[name] = timing_totals.get(name, 0.0) + value

    baseline_metric_inputs = [
        {"baseline": {"metrics": item["metrics"]}}
        for item in baseline_details if "metrics" in item
    ]
    graph_metric_inputs = [
        {"graph_on": {"metrics": item["metrics"]}}
        for item in graph_details if "metrics" in item
    ]
    baseline_metrics = _average_metrics(baseline_metric_inputs, "baseline") if baseline_metric_inputs else {}
    graph_metrics = _average_metrics(graph_metric_inputs, "graph_on") if graph_metric_inputs else {}
    gain = {
        f"Recall@{k}": round(
            graph_metrics.get("final", {}).get(f"Recall@{k}", 0.0)
            - baseline_metrics.get("final", {}).get(f"Recall@{k}", 0.0),
            4,
        )
        for k in EVAL_TOP_K
    }
    summary = {
        "total_queries": len(eval_set),
        "evaluated_queries": len(deltas),
        "ground_truth_queries": sum(
            detail.get("evaluation_mode") == "ground_truth" for detail in deltas
        ),
        "evaluation_errors": errors,
        "graph_final_recall_gain": gain,
        "graph_final_mrr_gain": round(
            graph_metrics.get("final", {}).get("MRR", 0.0)
            - baseline_metrics.get("final", {}).get("MRR", 0.0),
            4,
        ),
        "graph_unique_hit_queries": sum(item["graph_unique_hit"] for item in deltas),
        "graph_final_gain_queries": sum(item["graph_final_gain"] for item in deltas),
        "regression_queries": sum(item["regression"] for item in deltas),
        "timing_seconds": {
            **{name: round(value, 3) for name, value in timing_totals.items()},
            "total_seconds": round(time.perf_counter() - started, 3),
        },
        "config": {
            "collection": MilvusConfig.milvus_chunks_collection,
            "search_limit": SEARCH_LIMIT,
            "hyde_enabled": _hyde_enabled(),
            "skip_rerank": skip_rerank,
            "web_search": "excluded from both branches",
        },
    }
    report = {
        "summary": summary,
        "baseline": {"metrics": baseline_metrics, "details": baseline_details},
        "graph_on": {"metrics": graph_metrics, "details": graph_details},
        "per_query_delta": deltas,
    }
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 56)
    print("             掌柜智库 Graph RAG A/B 评测")
    print("=" * 56)
    print(f"  query数: {len(eval_set)} (成功 {len(deltas)}, 异常 {errors})")
    print(f"  baseline final: {baseline_metrics.get('final', {})}")
    print(f"  graph_on final: {graph_metrics.get('final', {})}")
    print(f"  Graph 独有命中: {summary['graph_unique_hit_queries']} 条")
    print(f"  Graph 最终增益: {summary['graph_final_gain_queries']} 条")
    print(f"  最终回退: {summary['regression_queries']} 条")
    print(f"  报告已保存: {Path(out_path).resolve()}")
    return report


def load_eval_set(file_path: str = "", sample: int = 0) -> list[dict]:
    if file_path:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("评测集必须是 JSON 数组")
        return data
    return build_eval_set_from_milvus(sample=sample)


def main() -> None:
    global ENABLE_HYDE_OVERRIDE
    parser = argparse.ArgumentParser(description="掌柜智库 Graph RAG A/B 评测")
    parser.add_argument("--sample", type=int, default=0, help="自动评测前N个主体")
    parser.add_argument("--file", default="", help="人工评测集 JSON 路径")
    parser.add_argument("--out", default="graph_ab_report.json", help="报告输出路径")
    parser.add_argument("--disable-hyde", action="store_true", help="本次评测关闭 HyDE")
    parser.add_argument("--skip-rerank", action="store_true", help="只比较 RRF，不执行 Rerank")
    args = parser.parse_args()
    if args.disable_hyde:
        ENABLE_HYDE_OVERRIDE = False
    eval_set = load_eval_set(args.file, args.sample)
    run_ab(eval_set, args.out, skip_rerank=args.skip_rerank)


if __name__ == "__main__":
    main()
# ==================== AI修改 结束 ====================
