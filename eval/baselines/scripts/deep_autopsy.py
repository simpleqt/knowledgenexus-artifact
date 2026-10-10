#!/usr/bin/env python3
"""deep 端点失败靶题逐层归因（2026-09-28）：学习型聚合的第一手需求文档。

对象：probe_arms.json 中 deep_rank 为 None 的靶题（正常轨 0 命中且 deep 亦未恢复）。
分层（deep 管线 = 通道 top-50 超采 → Qwen3 重排 → PPR → top-20）：

  not-in-candidates      gold 块不在任一通道 top-50 并集（候选生成层缺失）
  rerank-low             gold 在候选中但重排名次 >20（重排语义打分不足）
  rerank-ok-displaced    gold 重排名次 ≤20 但 deep 最终缺席（PPR/融合位移）
  附 PPR 起点诊断：查询实体面形在词典中的命中数（<2 = 种子无从谈起）

用法：测试栈 env 同 miss_autopsy；产物 deep_autopsy.json + stdout 报告。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, REPO  # noqa: E402

KB = json.load(open(os.path.join(DATA, "teststack_ne", "kb.json")))["kb_id"]
PROBE = Path(REPO) / "eval/baselines/results/20260928_stressset/probe_arms.json"
OUT = Path(REPO) / "eval/baselines/results/20260928_stressset/deep_autopsy.json"
DEPTH = 50  # deep 端点超采深度 min(top_k*3, 50)


def rerank_scores(query: str, docs: list[str]) -> list[tuple[int, float]]:
    base = os.environ["RERANK_API_BASE"].rstrip("/")
    payload = {"model": os.environ["RERANK_MODEL"], "query": query,
               "documents": [d[:4000] for d in docs]}
    req = urllib.request.Request(f"{base}/rerank", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['RERANK_API_KEY']}"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310
        results = json.loads(r.read())["results"]
    return sorted(((r["index"], float(r["relevance_score"])) for r in results),
                  key=lambda t: -t[1])


def main() -> None:
    for line in open(os.path.join(REPO, "deploy", ".env.services"), encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)
    from knc_engine import db, embedding, search

    probe = json.load(open(PROBE, encoding="utf-8"))
    failed = [r for r in probe["rows"] if not r["deep_rank"]]

    # 词典面形（PPR 起点诊断）
    ent_names: set[str] = set()
    for r in db.exec("SELECT canonical_name n FROM canonical_entity"
                     " WHERE kb_id=UNHEX(%s) AND status='ACTIVE'", (KB,), fetch=True):
        ent_names.add(r["n"].lower())
    for r in db.exec("SELECT alias_text t FROM entity_alias"
                     " WHERE kb_id=UNHEX(%s)", (KB,), fetch=True):
        ent_names.add(r["t"].lower())

    records = []
    for qi, row in enumerate(failed):
        q = row["query"]
        gold = set(row["gold"])
        qvec = embedding.embed([q])[0]
        ch = {"lexical": search.lexical_search(KB, q, DEPTH),
              "dense": search.dense_search(KB, qvec, DEPTH),
              "synq": search.synq_search(KB, qvec, DEPTH),
              "docvec": search.doc_vec_search(KB, qvec, DEPTH)}
        # 候选并集（chunk 级，记录每通道最优名次）
        cand: dict[str, dict] = {}
        for name, hits in ch.items():
            for r, h in enumerate(hits, 1):
                cid = h.get("chunk_id") or "?"
                e = cand.setdefault(cid, {"item": h, "ch": {}})
                e["ch"][name] = min(e["ch"].get(name, 99), r)
        gold_cids = [cid for cid, e in cand.items()
                     if (e["item"].get("doc_id")) in gold]
        surfaces = sorted({s for s in ent_names if s in q.lower()})
        rec = {"query": q[:60], "gold_doc": row["gold"][0],
               "n_candidates": len(cand),
               "gold_in_candidates": bool(gold_cids),
               "gold_channel_best": min((min(e["ch"].values())
                                         for e in (cand[c] for c in gold_cids)),
                                        default=None),
               "ppr_surfaces": surfaces[:6],
               "n_surfaces": len(surfaces)}
        if gold_cids:
            docs = [e["item"].get("content_text", "")
                    for e in cand.values()]
            cids = list(cand)
            scored = rerank_scores(q, docs)
            g_rerank = next(((rk + 1, sc) for rk, (idx, sc) in enumerate(scored)
                             if cids[idx] in set(gold_cids)), None)
            rec["gold_rerank_rank"] = g_rerank[0] if g_rerank else None
            rec["gold_rerank_score"] = round(g_rerank[1], 4) if g_rerank else None
            rec["rerank_20th_score"] = round(scored[19][1], 4) if len(scored) >= 20 else None
            if g_rerank is None:
                rec["class"] = "rerank-error"
            elif g_rerank[0] > 20:
                rec["class"] = "rerank-low"
            else:
                rec["class"] = "rerank-ok-displaced"
        else:
            rec["class"] = "not-in-candidates"
        records.append(rec)
        print(f"[{qi+1}/{len(failed)}] {rec['class']} cand={rec['n_candidates']} "
              f"gold_ch={rec['gold_channel_best']} rr={rec.get('gold_rerank_rank')} "
              f"surfs={rec['n_surfaces']}", flush=True)

    from collections import Counter
    cls = Counter(r["class"] for r in records)
    json.dump({"classes": dict(cls), "records": records},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(cls), ensure_ascii=False))
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
