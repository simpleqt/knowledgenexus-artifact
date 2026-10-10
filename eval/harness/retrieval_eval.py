#!/usr/bin/env python3
"""检索评测（§7.1/§7.2 验收门禁底座）：题集 × /api/retrieve → Recall@k / MRR@10。

题集格式（jsonl）：{"query": str, "gold_doc_ids": [hex...], "gold_chunk_ids": [hex...]}
口径：Recall@k = 命中 gold 文档的题比例；MRR@10 = 首个 gold 文档排名倒数均值。
用法：python retrieval_eval.py --dataset x.jsonl --api http://127.0.0.1:18121/api/retrieve
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request


def retrieve(api: str, kb: str, query: str, top_k: int,
             channels: list[str] | None = None, strategy: str | None = None) -> list[dict]:
    body = {"kb_id": kb, "query": query, "top_k": top_k}
    if channels:
        body["channels"] = channels
    if strategy:
        body["strategy"] = strategy
    payload = json.dumps(body).encode()
    req = urllib.request.Request(api, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        return json.loads(resp.read())["chunks"]


def evaluate(rows: list[dict], api: str, kb: str, ks: list[int],
             channels: list[str] | None = None, strategy: str | None = None) -> dict:
    per_query, recalls, mrr, lat = [], {k: [] for k in ks}, [], []
    ndcg, gold_evidence = [], []
    for row in rows:
        gold = set(row["gold_doc_ids"])
        gold_chunks = set(row.get("gold_chunk_ids") or [])
        t0 = time.time()
        hits = retrieve(api, kb, row["query"], max(ks), channels, strategy)
        lat.append(time.time() - t0)
        docs = [h["doc_id"] for h in hits]
        rank = next((i + 1 for i, d in enumerate(docs[:10]) if d in gold), None)
        r_at = {f"recall@{k}": int(any(d in gold for d in docs[:k])) for k in ks}
        for k in ks:
            recalls[k].append(r_at[f"recall@{k}"])
        if rank:
            mrr.append(1.0 / rank)
        # nDCG@10（文档级 gold，按去重后文档位次折减 1/log2(rank+1)；二值相关理想 DCG=1）
        seen_docs: set[str] = set()
        udocs = [d for d in docs if not (d in seen_docs or seen_docs.add(d))]
        dcg = sum(1.0 / math.log2(r + 2) for r, d in enumerate(udocs[:10]) if d in gold)
        ndcg.append(dcg)
        # Gold Evidence Recall（块级：gold_chunk_ids 命中率；未标 gold_chunk 的题跳过）
        if gold_chunks:
            got = {h.get("chunk_id") for h in hits}
            gold_evidence.append(len(gold_chunks & got) / len(gold_chunks))
        per_query.append({"query": row["query"], **r_at,
                          "mrr@10": round(1.0 / rank, 4) if rank else 0.0,
                          "ndcg@10": round(dcg, 4),
                          "first_gold_rank": rank, "returned": len(hits)})
    n = len(rows)
    return {"n": n,
            "metrics": {**{f"recall@{k}": round(sum(v) / n, 4) for k, v in recalls.items()},
                        "mrr@10": round(sum(mrr) / n, 4),
                        "ndcg@10": round(sum(ndcg) / n, 4),
                        **({"gold_evidence_recall": round(sum(gold_evidence) / len(gold_evidence), 4)}
                           if gold_evidence else {}),
                        "latency_p50_s": round(sorted(lat)[n // 2], 3),
                        "latency_p95_s": round(sorted(lat)[min(n - 1, int(n * 0.95))], 3)},
            "per_query": per_query}


def paired_bootstrap_ci(a: list[dict], b: list[dict], metric: str, iters: int = 10000,
                        seed: int = 20260912) -> dict:
    """配对差 bootstrap CI（§7.1 统计纪律：负效应判定一律 CI 上界<0）。
    a/b 为同题集两次运行的 per_query；返回 mean_diff 与 95% CI。"""
    import random
    rng = random.Random(seed)
    da = [q[metric] for q in a]
    db = [q[metric] for q in b]
    n = len(da)
    diffs = [x - y for x, y in zip(da, db)]
    means = []
    for _ in range(iters):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo, hi = means[int(iters * 0.025)], means[int(iters * 0.975)]
    return {"metric": metric, "n": n, "mean_diff": round(sum(diffs) / n, 4),
            "ci95": [round(lo, 4), round(hi, 4)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:18121/api/retrieve")
    ap.add_argument("--kb", required=True)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--out", default=None)
    ap.add_argument("--channels", default=None, help="e.g. lexical / lexical,dense")
    ap.add_argument("--strategy", default=None,
                    help="Planner 策略 A/B/C/D/E（D/E=topic 路由实验通道；默认 auto）")
    ap.add_argument("--compare", default=None,
                    help="另一运行的 per_query JSON（同题集）——输出配对差 bootstrap CI")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.dataset, encoding="utf-8") if line.strip()]
    chans = args.channels.split(',') if args.channels else None
    result = evaluate(rows, args.api, args.kb, [int(k) for k in args.ks.split(",")], chans,
                      args.strategy)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    if args.compare:
        other = json.load(open(args.compare, encoding="utf-8"))
        other_pq = other["per_query"] if "per_query" in other else other
        cis = [paired_bootstrap_ci(result["per_query"], other_pq, m)
               for m in ("recall@5", "mrr@10") if m in result["per_query"][0]
               and m in other_pq[0]]
        result["paired_ci"] = cis
        for c in cis:
            print(f"paired {c['metric']}: diff={c['mean_diff']} ci95={c['ci95']}")
    if args.out:
        json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
