#!/usr/bin/env python3
"""多查询扇出消融（2026-09-28）：查询侧语义增强——LLM 为每题生成 3 个改写变体，
原查询 + 变体分别走 scoped 全通道检索，uniform RRF(k=60) 融合候选池后取前 20。

- 语料/题集/口径与七系统对比一致：L 切片（300 篇）× nea 48 / ner 39，
  进程内直调 _retrieve 规划器（Redis 旁路 stub，无缓存污染，与 run_knc_scoped 同法）
- 产物：
  runs/knc-mq-base_{set}_L.json      原查询基线（同会话重取，配对）
  runs/knc-mq-fanout_{set}_L.json    扇出融合臂
  results/20260928_query_enhance/paraphrase_{set}.jsonl
                                     每题 3 变体 + 逐变体首金标名次（改写鲁棒性题集）
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python eval/baselines/scripts/multiquery_fanout.py --set nea --set ner'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from common import NE_JOURNAL_KB, REPO, load_questions, load_slice_ids, write_predictions

BASE = os.path.join(REPO, "eval", "baselines")
RESULT_DIR = Path(BASE) / "results" / "20260928_query_enhance"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402

# Redis 旁路 stub（与 run_knc_scoped 一致：准入/缓存/指标均与排序无关）
M.cache.admit = lambda *a, **k: (True, "")
M.cache.session_admit = lambda *a, **k: (True, 0)
M.cache.singleflight_lock = lambda *a, **k: True
M.cache.singleflight_release = lambda *a, **k: None
M.cache.result_get = lambda *a, **k: None
M.cache.record_latency_ms = lambda *a, **k: None
M.cache.bump_metric = lambda *a, **k: None
M.cache.acl_version = lambda *a, **k: 0

RRF_K = 60
N_VARIANTS = 3

VARIANT_SYSTEM = (
    "你是学术文献检索的查询改写器。对给定查询生成 3 个改写变体："
    "信息需求与原查询完全一致（不改问题对象、限定条件与问点数量），"
    "仅改变措辞、句式、详略或术语表达（可中英互换）。"
    '只输出 JSON：{"variants": ["变体1", "变体2", "变体3"]}')


def llm_json(system: str, user: str) -> dict:
    base = os.environ["LLM_API_BASE"].rstrip("/")
    body = json.dumps({
        "model": os.environ["LLM_MODEL"], "temperature": 0, "max_tokens": 300,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                text = json.loads(r.read())["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            if attempt == 3:
                return {}
            time.sleep(5 * (attempt + 1))
    return {}


def scoped_retrieve(query: str, doc_ids: list[str], top_k: int = 20):
    req = M.RetrieveRequest.model_construct(
        kb_id=NE_JOURNAL_KB, kb_ids=None, query=query, top_k=60, channels=None,
        doc_ids=list(doc_ids), strategy="A", session_id=None,
        tenant_id="public", acl_scopes=["public"])
    deadline = time.monotonic() + 120.0
    resp = M._retrieve(req, [NE_JOURNAL_KB], deadline,
                       lambda: time.monotonic() < deadline, "knc-mq-eval")
    sliceset = set(doc_ids)
    return [c for c in resp.chunks if c.get("doc_id") in sliceset][:top_k]


def rrf_fuse(lists: list[list[dict]], top_k: int = 20) -> list[dict]:
    scores: dict[str, float] = {}
    keep: dict[str, dict] = {}
    for lst in lists:
        for r, c in enumerate(lst):
            key = c.get("chunk_id") or f"{c.get('doc_id')}:{len(keep)}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + r + 1)
            keep.setdefault(key, c)
    return [keep[k] for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:top_k]


def to_pred_rows(rows: list[tuple[str, list[str], list[dict], float]]) -> list[dict]:
    return [{"query": q, "gold_doc_ids": gold,
             "hits": [{"rank": i + 1, "doc_id": c.get("doc_id"),
                       "chunk_id": c.get("chunk_id"), "score": c.get("_score")}
                      for i, c in enumerate(chunks)],
             "latency_s": round(dt, 3)} for q, gold, chunks, dt in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True)
    args = ap.parse_args()
    slice_ids = load_slice_ids("L")

    variants_log: dict[str, list[str]] = {}
    for set_name in args.set:
        questions = load_questions(set_name)
        base_rows, fan_rows = [], []
        para_path = RESULT_DIR / f"paraphrase_{set_name}.jsonl"
        para_f = open(para_path, "a", encoding="utf-8")

        for qi, q in enumerate(questions):
            gold = q["gold_doc_ids"]
            gset = set(gold)

            # ① 变体生成（带缓存：改写结果冻结复用）
            if q["query"] not in variants_log:
                out = llm_json(VARIANT_SYSTEM, f"查询：{q['query']}")
                vs = [v.strip() for v in (out.get("variants") or [])[:N_VARIANTS]
                      if isinstance(v, str) and v.strip()]
                variants_log[q["query"]] = vs
            variants = variants_log[q["query"]]

            # ② 基线：原查询 scoped 检索
            t0 = time.time()
            base_chunks = scoped_retrieve(q["query"], slice_ids)
            base_dt = time.time() - t0
            base_rows.append((q["query"], gold, base_chunks, base_dt))

            # ③ 扇出：原查询 + 变体分别检索后 RRF 融合
            t0 = time.time()
            lists = [base_chunks]
            per_var_rank = []
            for v in variants:
                vchunks = scoped_retrieve(v, slice_ids)
                lists.append(vchunks)
                fr = next((i + 1 for i, c in enumerate(vchunks)
                           if c.get("doc_id") in gset), None)
                per_var_rank.append(fr)
            fused = rrf_fuse(lists) if variants else base_chunks
            fan_dt = time.time() - t0
            fan_rows.append((q["query"], gold, fused, fan_dt))

            # ④ 改写鲁棒性记录（逐变体首金标名次；冻结为新题集）
            base_first = next((i + 1 for i, c in enumerate(base_chunks)
                               if c.get("doc_id") in gset), None)
            para_f.write(json.dumps({
                "query": q["query"], "gold_doc_ids": gold, "source_set": set_name,
                "variants": variants,
                "first_gold_rank": {"original": base_first, "variants": per_var_rank},
            }, ensure_ascii=False) + "\n")
            para_f.flush()
            print(f"[{set_name} {qi+1}/{len(questions)}] vars={len(variants)} "
                  f"base_first={base_first} var_ranks={per_var_rank}", flush=True)
        para_f.close()

        write_predictions("knc-mq-base", "L", set_name,
                          {"wall_s": 0, "note": "scoped L 切片 全通道 原查询"},
                          to_pred_rows(base_rows), len(slice_ids))
        write_predictions("knc-mq-fanout", "L", set_name,
                          {"wall_s": 0,
                           "note": f"scoped L 切片 全通道 原查询+{N_VARIANTS}变体 RRF(k={RRF_K})"},
                          to_pred_rows(fan_rows), len(slice_ids))
    json.dump(variants_log, open(RESULT_DIR / "variants_cache.json", "w",
                                 encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
