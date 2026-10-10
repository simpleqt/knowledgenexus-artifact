#!/usr/bin/env python3
"""kNN 近似召回损失探针（20261003_knn_efsearch 前置诊断）。

量同一查询向量在不同 ef_search 深度下的 top-k 重叠率：以最深 ef 为参照，
浅 ef（含缺省不设=插件默认 16）与参照的重叠缺口 = HNSW 近似损失上界——
它直接决定 KNN_EF_SEARCH 补偿件的期望收益（MRL 1024 表征损失之外的部分）。

只读、无金标依赖：不跑检索服务，直打 OpenSearch（与 dense_search 同形 filter：
kb_id + visible）。生产栈（5M 级 chunk）与测试栈（5 万级）对照——小索引上
预期接近满重叠，差异主要在生产规模显形。

用法：
  .venv/bin/python knn_probe.py --stack test
  .venv/bin/python knn_probe.py --stack prod
产物：runs/knn_probe_<stack>.json（含逐查询明细）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import BASE, DATA, NE_JOURNAL_KB, REPO, load_questions, load_services_env  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "libs", "engine"))

# 测试栈覆盖（run_services_test.sh 同源常量）
TEST_OS = "https://127.0.0.1:29200"
TEST_KB = json.load(open(os.path.join(DATA, "teststack_ne", "kb.json")))["kb_id"]


def setup_env(stack: str) -> str:
    """嵌入/OS 环境自装配：嵌入三件套+MRL 维度取 .env.services（knc_env.sh 不导
    MRL，漏设则查询向量 4096 打不进 1024 索引）；OS 指向目标栈（生产凭据取
    deploy/.env，已有 env 不覆写——便于 knc_env.sh 包裹复用）。"""
    svc = load_services_env()
    for k in ("EMBEDDING_API_BASE", "EMBEDDING_API_KEY", "EMBEDDING_MODEL"):
        os.environ.setdefault(k, svc.get(k, ""))
    os.environ["EMBEDDING_MRL_DIM"] = os.environ.get(
        "EMBEDDING_MRL_DIM") or svc.get("EMBEDDING_MRL_DIM") or "1024"
    if stack == "test":
        os.environ.update(OPENSEARCH_URL=TEST_OS, OPENSEARCH_USER="admin",
                          OPENSEARCH_PASSWORD="KncTest#2026")
    else:
        dot: dict[str, str] = {}
        for line in open(os.path.join(REPO, "deploy", ".env"), encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                dot[k.strip()] = v.strip()
        os.environ.setdefault("OPENSEARCH_URL", "https://127.0.0.1:19200")
        os.environ.setdefault("OPENSEARCH_USER", "admin")
        os.environ.setdefault("OPENSEARCH_PASSWORD", dot.get("OPENSEARCH_ADMIN_PASSWORD", ""))
    return TEST_KB if stack == "test" else NE_JOURNAL_KB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=["prod", "test"], default="prod")
    ap.add_argument("--index", default="knc-chunk")
    ap.add_argument("--field", default="chunk_vector")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--efs", default="0,64,128,256,512",
                    help="逗号分隔 ef 深度；0=缺省不设（插件默认）")
    ap.add_argument("--limit", type=int, default=0, help="限制查询数（0=全部）")
    args = ap.parse_args()

    kb = setup_env(args.stack)
    from knc_engine import embedding, search  # env 装配后再导入
    c = search.client()
    n_docs = c.count(index=args.index, body={"query": {"term": {"kb_id": kb}}})["count"]
    n_all = c.count(index=args.index)["count"]
    print(f"[{args.stack}] index={args.index} kb={kb[:12]}… chunks(kb)={n_docs} / total={n_all}")

    queries = [q["query"] for s in ("nea", "ner") for q in load_questions(s)]
    if args.limit:
        queries = queries[:args.limit]
    vecs = embedding.embed(queries)
    print(f"queries={len(queries)} dim={len(vecs[0])} (EMBEDDING_MRL_DIM={os.environ['EMBEDDING_MRL_DIM']})")

    efs = [int(x) for x in args.efs.split(",")]
    flt = {"bool": {"filter": [{"term": {"kb_id": kb}}, {"term": {"visible": True}}]}}

    def knn(vec: list[float], ef: int) -> tuple[list[str], float]:
        q: dict = {"vector": vec, "k": args.topk, "filter": flt}
        if ef > 0:
            q["method_parameters"] = {"ef_search": ef}
        t0 = time.perf_counter()
        r = c.search(index=args.index, body={
            "size": args.topk, "query": {"knn": {args.field: q}}, "_source": False})
        return [h["_id"] for h in r["hits"]["hits"]], (time.perf_counter() - t0) * 1000

    knn(vecs[0], efs[-1])  # 预热（首查含连接/缓存建立）
    detail: list[dict] = []
    ref_ef = max(efs) if max(efs) > 0 else 0
    per_ef_hits: dict[int, list[list[str]]] = {ef: [] for ef in efs}
    per_ef_ms: dict[int, list[float]] = {ef: [] for ef in efs}
    for i, vec in enumerate(vecs):
        row = {}
        for ef in efs:
            ids, ms = knn(vec, ef)
            per_ef_hits[ef].append(ids)
            per_ef_ms[ef].append(ms)
        ref = per_ef_hits[ref_ef][i]
        for ef in efs:
            row[f"ov_{ef}"] = len(set(per_ef_hits[ef][i]) & set(ref)) / min(args.topk, len(ref) or 1)
        detail.append(row)

    print(f"\nef 深度 vs 参照(ef={ref_ef}) top-{args.topk} 重叠率（0=插件缺省）：")
    print(f"{'ef':>6} {'mean_ov':>8} {'min_ov':>8} {'ov<1 题数':>10} {'ms_mean':>8} {'ms_p95':>8}")
    out = {"stack": args.stack, "index": args.index, "kb": kb, "chunks_kb": n_docs,
           "chunks_total": n_all, "topk": args.topk, "ref_ef": ref_ef,
           "n_queries": len(queries), "summary": {}, "detail": detail}
    for ef in efs:
        ovs = [d[f"ov_{ef}"] for d in detail]
        mss = sorted(per_ef_ms[ef])
        p95 = mss[int(len(mss) * 0.95) - 1] if len(mss) >= 20 else mss[-1]
        s = {"mean_overlap": round(sum(ovs) / len(ovs), 4),
             "min_overlap": round(min(ovs), 4),
             "queries_below_full": sum(1 for o in ovs if o < 1.0),
             "ms_mean": round(sum(mss) / len(mss), 1), "ms_p95": round(p95, 1)}
        out["summary"][str(ef)] = s
        print(f"{ef:>6} {s['mean_overlap']:>8.4f} {s['min_overlap']:>8.2f} "
              f"{s['queries_below_full']:>10} {s['ms_mean']:>8.1f} {s['ms_p95']:>8.1f}")
    dest = os.path.join(BASE, "runs", f"knn_probe_{args.stack}.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[written] {dest}")


if __name__ == "__main__":
    main()
