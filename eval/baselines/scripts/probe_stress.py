#!/usr/bin/env python3
"""高难靶题机制探针（2026-09-28）：10 题 v2 靶题 × 4 个候选恢复机制。

靶题定义（stress_expand_v2.jsonl）：金标通道 top-100 可见 ∧ 融合 top-10 缺席——
"证据系统看得见但捞不回"。探针回答：每个机制各能捞回几题。

臂：
  base    正常端点 top-20（对照）
  deep    深度端点（超采 3×50 + Qwen3 重排 + PPR——生产为难题设计的路径）
  fanout  LLM 3 变体 + uniform RRF(k=60)（已被规模消融否决的机制，靶题上复查）
  lift    散票追加救援（离线模拟）：≥2 通道 top-60 可见且不在 base top-20 的文档，
          其最优块按 RRF 分追加在 21 名后——严格追加，top-20 不动，按构造无排序风险

用法：bash -c 'source deploy/.env.services 取 LLM_* 后 .venv/bin/python probe_stress.py'
产物：results/20260928_stressset/probe_arms.json + stdout 表
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, REPO  # noqa: E402

API = "http://127.0.0.1:28121/api/retrieve"
DEEP = API + "/deep"
KB = json.load(open(os.path.join(DATA, "teststack_ne", "kb.json")))["kb_id"]
IDMAP = json.load(open(os.path.join(DATA, "teststack_ne", "idmap.json")))
REV = {v: k for k, v in IDMAP.items()}
OUT = Path(REPO) / "eval/baselines/results/20260928_stressset/probe_arms.json"
N_VARIANTS = 3

VARIANT_SYSTEM = (
    "你是学术文献检索的查询改写器。对给定查询生成 3 个改写变体：信息需求完全一致，"
    "仅改变措辞、句式、详略或术语表达（可中英互换）。"
    '只输出 JSON：{"variants": ["变体1", "变体2", "变体3"]}')


def llm_json(system: str, user: str) -> dict:
    base = os.environ["LLM_API_BASE"].rstrip("/")
    body = json.dumps({"model": os.environ["LLM_MODEL"], "temperature": 0,
                       "max_tokens": 300,
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


def api_retrieve(query: str, deep: bool = False, top_k: int = 20) -> list[dict]:
    url = DEEP if deep else API
    req = urllib.request.Request(url, data=json.dumps(
        {"kb_id": KB, "query": query, "top_k": top_k}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
        return json.loads(r.read())["chunks"]


def rrf_fuse(lists: list[list[dict]], top_k: int = 20) -> list[dict]:
    scores: dict[str, float] = {}
    keep: dict[str, dict] = {}
    for lst in lists:
        for r, c in enumerate(lst):
            key = c.get("chunk_id") or f"{c.get('doc_id')}:{len(keep)}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + r + 1)
            keep.setdefault(key, c)
    return [keep[k] for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:top_k]


def gold_rank(docs: list[str], gold: set[str]):
    return next((i + 1 for i, d in enumerate(docs) if d in gold), None)


def main() -> None:
    for line in open(os.path.join(REPO, "deploy", ".env.services"), encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)

    from knc_engine import embedding, search  # 测试栈 env 由外层注入
    targets = [json.loads(l) for l in
               open(Path(REPO) / "eval/baselines/results/20260928_stressset/"
                    "stress_expand_v2.jsonl", encoding="utf-8") if l.strip()]

    rows = []
    for qi, t in enumerate(targets):
        gold = {t["gold_doc_ids"][0]}  # stable id 空间（与 REV 译回后的序列同域）
        q = t["query"]

        base_chunks = api_retrieve(q)
        base_docs = [REV.get(c["doc_id"], c["doc_id"]) for c in base_chunks]

        deep_chunks = api_retrieve(q, deep=True)
        deep_docs = [REV.get(c["doc_id"], c["doc_id"]) for c in deep_chunks]

        out = llm_json(VARIANT_SYSTEM, f"查询：{q}")
        variants = [v.strip() for v in (out.get("variants") or [])[:N_VARIANTS]
                    if isinstance(v, str) and v.strip()]
        lists = [base_chunks] + [api_retrieve(v) for v in variants]
        fan_docs = [REV.get(c["doc_id"], c["doc_id"]) for c in rrf_fuse(lists)]

        # 散票追加救援（严格追加模拟）
        qvec = embedding.embed([q])[0]
        ch = {"lexical": search.lexical_search(KB, q, 60),
              "dense": search.dense_search(KB, qvec, 60),
              "synq": search.synq_search(KB, qvec, 60),
              "docvec": search.doc_vec_search(KB, qvec, 60)}
        per_doc: dict[str, dict[str, float]] = {}
        for name, hits in ch.items():
            for r, h in enumerate(hits, 1):
                did = h.get("doc_id") or "?"
                td = per_doc.setdefault(did, {})
                td[name] = min(td.get(name, 99), 1.0 / (60 + r))
        base_set = {c.get("doc_id") for c in base_chunks}
        lifted = []
        for did, tickets in per_doc.items():
            if did in base_set or len(tickets) < 2:
                continue
            lifted.append((sum(tickets.values()), did))
        lifted.sort(key=lambda x: -x[0])
        lift_docs = [REV.get(did, did) for _, did in lifted[:10]]

        rows.append({
            "query": q, "gold": list(gold), "n_variants": len(variants),
            "base_rank": gold_rank(base_docs, gold),
            "deep_rank": gold_rank(deep_docs, gold),
            "fanout_rank": gold_rank(fan_docs, gold),
            "lift_rescued": gold in set(lift_docs),
            "n_lifted": len(lift_docs),
        })
        print(f"[{qi+1}/{len(targets)}] base={rows[-1]['base_rank']} "
              f"deep={rows[-1]['deep_rank']} fan={rows[-1]['fanout_rank']} "
              f"lift={rows[-1]['lift_rescued']}({rows[-1]['n_lifted']})", flush=True)

    n = len(rows)
    summary = {
        "n": n,
        "gold@20": {a: sum(1 for r in rows if (r[f"{a}_rank"] or 99) <= 20)
                    for a in ("base", "deep", "fanout")},
        "gold@10": {a: sum(1 for r in rows if (r[f"{a}_rank"] or 99) <= 10)
                    for a in ("base", "deep", "fanout")},
        "lift_rescued": sum(1 for r in rows if r["lift_rescued"]),
        "lift_per_q_mean": round(sum(r["n_lifted"] for r in rows) / n, 1),
    }
    json.dump({"summary": summary, "rows": rows}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
