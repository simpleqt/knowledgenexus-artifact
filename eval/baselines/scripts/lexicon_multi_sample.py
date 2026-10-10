#!/usr/bin/env python3
"""词典隔离对照 10 次重采样（2026-10-10 三轮审查补 #3：mean±sd 防 cherry-pick）。

首版 fetch_chunks 只取 chunk_id 序前 2,000 块（前缀偏置）；本版改为
random_score 定种子的独立均匀随机抽样（每样本 2,000 块、跨样本去重重抽，
一步带出 content_text）。KB1 实测 5.83M 块——sort _id 全量分页每页重排
5.8M 集合不可行，random_score 单请求即出。复现性：结果 JSON 全量存档
各样本 chunk_id 清单（random_score 依赖段内 doc 序，seed 跨段合并不稳）。
每样本 × 5 臂全确定性扫描（无 LLM），报 links/entities/C10/每文档概念数/
泛词头质量占比 的 mean±sd。任务无状态（词典+样本块随任务 pickle，
compiled regex 不可 pickle 故 worker 内现编译），8 进程并行。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/lexicon_multi_sample.py \
  --out eval/baselines/results/20261010_lexicon_multisample'
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import load_lexicon, precompile, scan  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402  (OS client)

# 与 run_arm 同一泛词集合（v2.0 类别词），保证跨版可比
V20_HEAD = {"method", "energy", "parameter", "material", "model",
            "term", "time", "field", "phase", "temperature"}


def fetch_all_ids(kb_hex: str) -> list[str]:
    """保留给计数核对（_count 走 curl 亦可）。"""
    c = M.search.client()
    r = c.count(index="knc-chunk", body={"query": {"bool": {"filter": [
        {"term": {"kb_id": kb_hex}}, {"term": {"visible": True}}]}}})
    return [""] * r["count"]  # 占位：仅取 total，id 全量分页已废弃（见下）


def random_sample(kb_hex: str, seed: str, per: int) -> list[tuple[str, str]]:
    """均匀随机抽 per 块（function_score random_score 定种子；一步带出
    content_text，无 mget）。random_score 依赖段内 doc序，seed 复现性以
    存档采样 chunk_id 清单为准（结果 JSON 全记）。"""
    c = M.search.client()
    body = {"size": per, "query": {"bool": {"filter": [
        {"term": {"kb_id": kb_hex}}, {"term": {"visible": True}}],
        "must": [{"function_score": {"query": {"match_all": {}},
                                     "random_score": {"seed": seed,
                                                      "field": "_seq_no"}}}]}},
        "_source": ["chunk_id", "doc_id", "content_text"]}
    r = c.search(index="knc-chunk", body=body)
    out = []
    for h in r["hits"]["hits"]:
        s = h["_source"]
        if s.get("content_text"):
            out.append((s["chunk_id"], s["doc_id"], s["content_text"]))
    return out


def one_task(args) -> tuple[str, int, dict]:
    """单臂单样本：worker 内编译模式（regex 不可 pickle）→ 扫描 → 指标。"""
    arm, si, lex, case_short, stop_block, chunks = args
    pats = precompile(lex, case_short, stop_block)
    links: dict[tuple, int] = {}
    doc_ents: dict[str, set] = {}
    for did, text in chunks:
        for surface, etype, s, e in scan(text, pats):
            ent = (surface.lower(), etype)
            links[ent] = links.get(ent, 0) + 1
            doc_ents.setdefault(did, set()).add(ent)
    total = sum(links.values())
    top10 = sorted(links.values(), reverse=True)[:10]
    per_doc = [len(v) for v in doc_ents.values()]
    head = sum(n for (f, t), n in links.items() if f in V20_HEAD)
    return arm, si, {
        "links": total,
        "entities": len(links),
        "c10": round(sum(top10) / total, 3) if total else 0.0,
        "conc_per_doc": round(statistics.mean(per_doc), 2) if per_doc else 0,
        "generic_head_share": round(head / total, 3) if total else 0.0,
        "docs": len(doc_ents),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="104112e63d9d4aa5a6c21e1c1549ef0f")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--per-sample", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lexica = {
        "v2.0": load_lexicon("5ec5756:schemas/nuclear_electronics.yaml"),
        "v2.5": load_lexicon("589af20:schemas/nuclear_electronics.yaml"),
        "v2.1": load_lexicon(os.path.join(REPO, "schemas", "nuclear_electronics.yaml")),
    }
    arm_defs = [
        ("v2.0-collapse", "v2.0", False, False),
        ("v2.5-mid", "v2.5", False, False),
        ("v2.1-repair", "v2.1", False, False),
        ("v2.1+caseShort", "v2.1", True, False),
        ("v2.1+stopwordBlock", "v2.1", False, True),
    ]

    total = len(fetch_all_ids(args.kb))
    print(f"[corpus] {total} visible chunks in KB", flush=True)
    k = args.samples
    sample_ids: list[set[str]] = []
    sample_chunks: list[list[tuple[str, str]]] = []
    seen: set[str] = set()
    for i in range(k):
        # 跨样本期望重叠 ~2000*10/5.8M ≈ 0.3%——超采 5% 后过滤取前 N，
        # 不够再超采一轮（seen 只增不减）
        rows = random_sample(args.kb, f"20261010-{i}",
                             args.per_sample + args.per_sample // 20)
        rows = [r for r in rows if r[0] not in seen][: args.per_sample]
        if len(rows) < args.per_sample:
            extra = random_sample(args.kb, f"20261010-{i}-x",
                                  args.per_sample)
            rows += [r for r in extra
                     if r[0] not in seen and r[0] not in
                     {x[0] for x in rows}][: args.per_sample - len(rows)]
        assert len(rows) == args.per_sample, f"sample {i} 补抽后仍不足"
        ids = {r[0] for r in rows}
        assert not (ids & seen), f"sample {i} 过滤后仍重叠"
        seen |= ids
        sample_ids.append(ids)
        sample_chunks.append([(r[1], r[2]) for r in rows])
        print(f"[sample {i}] {len(rows)} chunks", flush=True)
    union = sum(len(v) for v in sample_chunks)
    print(f"[samples] {union} chunks across {k} disjoint random samples",
          flush=True)

    tasks = [(arm, si, lexica[lexkey], cs, sb, sample_chunks[si])
             for si in range(k)
             for arm, lexkey, cs, sb in arm_defs]
    results: dict[str, dict[int, dict]] = {a: {} for a, *_ in arm_defs}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        done = 0
        for arm, si, m in pool.map(one_task, tasks):
            results[arm][si] = m
            done += 1
            if done % 10 == 0:
                print(f"[progress] {done}/{len(tasks)}", flush=True)

    agg = {}
    for a, *_ in arm_defs:
        rows = [results[a][s] for s in sorted(results[a])]
        agg[a] = {}
        for key in ("links", "entities", "c10", "conc_per_doc",
                    "generic_head_share", "docs"):
            vals = [r[key] for r in rows]
            agg[a][key] = {"mean": round(statistics.mean(vals), 3),
                           "sd": round(statistics.stdev(vals), 3)}
    out = {"kb": args.kb, "total_chunks": total, "samples": k,
           "per_sample": args.per_sample,
           "sampling": "independent uniform random samples, "
                       "random_score seed 20261010-i, cross-sample deduped",
           "sample_chunk_ids": {str(i): sorted(sample_ids[i])
                                for i in range(k)},
           "per_run": results, "aggregate": agg}
    json.dump(out, open(os.path.join(args.out, "multisample.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(agg, ensure_ascii=False, indent=1))
    print("[saved]", os.path.join(args.out, "multisample.json"))


if __name__ == "__main__":
    main()
