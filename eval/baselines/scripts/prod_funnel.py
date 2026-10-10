#!/usr/bin/env python3
"""生产全库难题出题漏斗（2026-09-29，两级口径：全量走生产只读）。

语料：ne-journal 生产库全量 3,479 篇（corpus_docs 300 + corpus_docs_extra 3,179），
doc_id 即生产 KB 主键（无 idmap 翻译——测试栈教训不适用）。

漏斗（继承 v2 三重验收 + 生产规模）：
  ① 非摘要随机片段 LLM 起草（措辞脱离指令）
  ② 8-gram 成句照抄率 ≤15%
  ③ 通道 top-100 可见（进程内引擎直调生产存储，只读）
  ④ 生产融合 top-10 缺席（生产只读 API 18121 回环）
  ⑤ 与既有题集去重（nea/ner/stress_v1/v2）

产物：results/20260929_hardtargets/hard_targets_prod.jsonl（断点续跑）
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python prod_funnel.py --drafts 600'
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, REPO, load_questions  # noqa: E402

KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
PROD_API = "http://127.0.0.1:18121/api/retrieve"   # 回环免认证
OUT_DIR = Path(REPO) / "eval/baselines/results/20260929_hardtargets"
OUT = OUT_DIR / "hard_targets_prod.jsonl"

DRAFT_SYS = ("你是学科文献测评题的出题人，专门构造高难度检索问题。基于给定文档"
             "片段写一条中文检索问题，要求：问题能由该片段回答；措辞必须与片段"
             "措辞脱离——不照抄原文连续 8 字以上的词串，用同义改写与不同句式，"
             "不得出现'该研究/本文/该文献'等指代；涉及具体内容（方法/参数/"
             "结论/比较）。只输出问题本身。")


def llm(system: str, user: str, max_tokens: int = 200) -> str:
    base = os.environ["LLM_API_BASE"].rstrip("/")
    body = json.dumps({"model": os.environ["LLM_MODEL"], "temperature": 0.3,
                       "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception:  # noqa: BLE001
            if attempt == 3:
                return ""
            time.sleep(5 * (attempt + 1))
    return ""


def _section_excerpt(text: str, rng: random.Random) -> str:
    if len(text) <= 3600:
        return text[:3000]
    start = rng.randint(len(text) // 3, max(len(text) // 3 + 1, len(text) - 3000))
    return text[start:start + 3000]


def _gram_overlap(q: str, text: str, n: int = 8) -> float:
    qg = {q[i:i + n] for i in range(len(q) - n + 1)}
    if not qg:
        return 1.0
    return sum(1 for g in qg if g in text) / len(qg)


def fused_rank(query: str, gold: str, top_k: int = 20):
    body = json.dumps({"kb_id": KB, "query": query, "top_k": top_k}).encode()
    req = urllib.request.Request(PROD_API, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        chunks = json.loads(r.read())["chunks"]
    return next((i for i, c in enumerate(chunks, 1) if c.get("doc_id") == gold), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from knc_engine import embedding, search  # 生产 env（knc_env.sh）

    docs = [json.loads(l) for l in
            open(os.path.join(DATA, "corpus_docs.jsonl"), encoding="utf-8") if l.strip()]
    extra_p = os.path.join(DATA, "corpus_docs_extra.jsonl")
    if os.path.exists(extra_p):
        docs += [json.loads(l) for l in open(extra_p, encoding="utf-8") if l.strip()]
    print(f"语料: {len(docs)} 篇（生产全库）", flush=True)

    known = {q["query"] for s in ("nea", "ner") for q in load_questions(s)}
    for f in ("stress_v1.jsonl", "stress_expand_v2.jsonl"):
        p = Path(REPO) / "eval/baselines/results/20260928_stressset" / f
        if p.exists():
            known |= {json.loads(l)["query"] for l in open(p, encoding="utf-8") if l.strip()}
    done_q = set()
    if OUT.exists():
        done_q = {json.loads(l)["query"] for l in open(OUT, encoding="utf-8") if l.strip()}

    rng = random.Random(args.seed)
    order = list(docs)
    rng.shuffle(order)

    accepted = len(done_q)
    out_f = open(OUT, "a", encoding="utf-8")
    for di, d in enumerate(order):
        if di >= args.drafts:
            break
        excerpt = _section_excerpt(d["text"], rng)
        q = llm(DRAFT_SYS, f"【文档内容】\n{excerpt}\n\n请写一条检索问题。")
        q = q.strip().strip('"').split("\n")[0][:120]
        if not q or q in known or q in done_q:
            continue
        if _gram_overlap(q, excerpt) > 0.15:
            continue
        # 通道可见（生产只读引擎直调；doc_id 即 KB 主键，无翻译）
        try:
            qvec = embedding.embed([q])[0]
            best = None
            for name, hits in (("lexical", search.lexical_search(KB, q, 100)),
                               ("dense", search.dense_search(KB, qvec, 100)),
                               ("synq", search.synq_search(KB, qvec, 100))):
                r = next((i for i, h in enumerate(hits, 1)
                          if h.get("doc_id") == d["doc_id"]), None)
                if r and (best is None or r < best[0]):
                    best = (r, name)
            rdv = next((i for i, h in enumerate(
                search.doc_vec_search(KB, qvec, 100), 1)
                if h.get("doc_id") == d["doc_id"]), None)
            if rdv and (best is None or rdv < best[0]):
                best = (rdv, "docvec")
        except Exception as exc:  # noqa: BLE001
            print(f"[probe-err] {exc}", flush=True)
            continue
        if not best:
            continue  # 通道不可见：不是稀释靶（属深失，另册）
        # 生产融合 top-10 缺席
        try:
            fr = fused_rank(q, d["doc_id"])
        except Exception as exc:  # noqa: BLE001
            print(f"[fused-err] {exc}", flush=True)
            continue
        if fr is not None and fr <= 10:
            continue
        out_f.write(json.dumps({
            "query": q, "gold_doc_ids": [d["doc_id"]],
            "source": "prod_funnel", "origin_title": d["title"][:60],
            "channel_best": {"rank": best[0], "channel": best[1]},
            "fused_first_gold_rank": fr,
        }, ensure_ascii=False) + "\n")
        out_f.flush()
        done_q.add(q)
        accepted += 1
        print(f"[{di+1}/{min(args.drafts, len(order))}] acc={accepted} "
              f"ch={best[0]}({best[1]}) fused={fr} | {q[:40]}", flush=True)
    out_f.close()
    print(f"完成：起草 {min(args.drafts, len(order))} → 接受 {accepted}（累计）→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
