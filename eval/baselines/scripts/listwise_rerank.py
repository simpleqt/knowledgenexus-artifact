#!/usr/bin/env python3
"""深度轨排序升级离线消融（2026-09-28）：pointwise 重排之上叠加 listwise LLM 精排。

候选：复用 knc-mq-base_{set}_L.json 的 scoped 前 20 候选（与扇出消融同源配对）。
臂：
  base   原融合序（不重排）
  pw     Qwen3-Reranker-4B pointwise（批量 /rerank）
  lw     qwen3.5-35B listwise 单窗置换（前 20 一次排完）
  pwlw   pointwise 序 → listwise 精排前 10（TREC 2025 两级管线模式）
产物：runs/knc-lw-{arm}_{set}_L.json ×4，score.py --ref base 直接配对。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python eval/baselines/scripts/listwise_rerank.py --set nea --set ner'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import RUNS, load_questions  # noqa: E402

LISTWISE_SYSTEM = (
    "你是搜索结果重排器。给定查询与一组候选片段（带编号），"
    "按片段对查询的相关性从高到低输出全部编号，不得遗漏、不得新增。"
    '只输出 JSON：{"order": [编号从大到小排列]}')


def llm_json(system: str, user: str, max_tokens: int = 500) -> dict:
    base = os.environ["LLM_API_BASE"].rstrip("/")
    body = json.dumps({
        "model": os.environ["LLM_MODEL"], "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310
                text = json.loads(r.read())["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            if attempt == 3:
                return {}
            time.sleep(5 * (attempt + 1))
    return {}


def pointwise(query: str, docs: list[str]) -> list[tuple[int, float]]:
    base = os.environ["RERANK_API_BASE"].rstrip("/")
    payload = {"model": os.environ["RERANK_MODEL"], "query": query,
               "documents": [d[:4000] for d in docs]}
    req = urllib.request.Request(
        f"{base}/rerank", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['RERANK_API_KEY']}"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        results = json.loads(resp.read())["results"]
    return sorted(((r["index"], float(r["relevance_score"])) for r in results),
                  key=lambda t: -t[1])


def listwise_permute(query: str, cands: list[dict], ids: list[int]) -> list[int] | None:
    """对 cands 的 ids 子集做 listwise 置换；失败返回 None（调用方保持原序）。"""
    blocks = "\n\n".join(
        f"[{c['_lw_id']}] {c['content_text'][:300]}" for c in cands)
    out = llm_json(LISTWISE_SYSTEM, f"查询：{query}\n\n候选片段：\n{blocks}")
    order = out.get("order")
    if not isinstance(order, list):
        return None
    try:
        perm = [int(x) for x in order]
    except (TypeError, ValueError):
        return None
    idset = set(ids)
    perm = [i for i in perm if i in idset] + [i for i in ids if i not in perm]
    return perm if sorted(perm) == sorted(ids) else None


def rerank_hits(hits: list[dict], order_idx: list[int]) -> list[dict]:
    return [hits[i] for i in order_idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True)
    args = ap.parse_args()

    for set_name in args.set:
        questions = {q["query"]: q for q in load_questions(set_name)}
        base_file = os.path.join(RUNS, f"knc-mq-base_{set_name}_L.json")
        pred = json.load(open(base_file, encoding="utf-8"))
        arms: dict[str, list] = {a: [] for a in ("pw", "lw", "pwlw")}

        for qi, row in enumerate(pred["per_query"]):
            q = row["query"]
            gold = set(row["gold_doc_ids"])
            hits = row["hits"]  # base 序（scoped 融合序），chunk 级
            # 候选文本需取回 content：base 预测只存了 id——从原查询重取一份
            # （scoped 直调无缓存，成本 ~1s；且与 base 同码同参，序一致）
            from multiquery_fanout import scoped_retrieve  # noqa: E402
            from common import NE_JOURNAL_KB, load_slice_ids  # noqa: E402
            slice_ids = load_slice_ids("L")
            chunks = scoped_retrieve(q, slice_ids)
            cands = [{**c, "_lw_id": i + 1} for i, c in enumerate(chunks)]
            assert len(cands) == len(hits), "候选数不一致"

            docs = [c["content_text"] for c in cands]
            ids = [c["_lw_id"] for c in cands]

            # pw：pointwise 全量重排
            try:
                pw_order = [i for i, _ in pointwise(q, docs)]
                pw_order += [i for i in ids if i not in pw_order]
            except Exception:  # noqa: BLE001
                pw_order = list(ids)
            arms["pw"].append(rerank_hits(hits, [i - 1 for i in pw_order]))

            # lw：listwise 单窗全排
            lw_order = listwise_permute(q, cands, ids) or list(ids)
            arms["lw"].append(rerank_hits(hits, [i - 1 for i in lw_order]))

            # pwlw：pointwise 前 10 listwise 精排
            top10 = pw_order[:10]
            tail = [i for i in pw_order if i not in top10]
            pwlw_top = listwise_permute(q, cands, top10) or top10
            pwlw_order = pwlw_top + tail
            arms["pwlw"].append(rerank_hits(hits, [i - 1 for i in pwlw_order]))

            fg = lambda hs: next((i + 1 for i, h in enumerate(hs)  # noqa: E731
                                  if h["doc_id"] in gold), None)
            print(f"[{set_name} {qi+1}/{len(pred['per_query'])}] "
                  f"base={fg(hits)} pw={fg(arms['pw'][-1])} "
                  f"lw={fg(arms['lw'][-1])} pwlw={fg(arms['pwlw'][-1])}", flush=True)

        for arm, rows in arms.items():
            out = {"system": f"knc-lw-{arm}", "slice": "L", "set": set_name,
                   "corpus_docs": pred["corpus_docs"],
                   "index_cost": {"wall_s": 0, "note": f"离线重排臂 {arm}（候选=base 前20）"},
                   "per_query": [{"query": r0["query"], "gold_doc_ids": r0["gold_doc_ids"],
                                  "hits": hits, "latency_s": None}
                                 for r0, hits in zip(pred["per_query"], rows)]}
            path = os.path.join(RUNS, f"knc-lw-{arm}_{set_name}_L.json")
            json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"[written] {path}")


if __name__ == "__main__":
    main()
