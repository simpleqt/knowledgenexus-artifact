#!/usr/bin/env python3
"""深失压力题集扩题（2026-09-28，DATASET_PLAN §3）：corpus-first LLM 起草 + 双重验收。

- 母题池：stress_seed.jsonl 54 题（融合稀释/深失驱动）；扩题从 L 切片其余文档
  起草（避开种子金标文档），目标 +50 题 → 压力集 v1 ≈104 题
- 起草：LLM 读文档首段写一条真实问句式中文问题（该文档可答）
- 验收①（可答性）：LLM 判定"仅凭该文档能否回答"
- 验收②（难度画像）：进程内引擎直调四通道 top-100，记录金标最优通道名次
  （deep=四通道全无→深失生成器，保留并标注；dilute=有通道可见→稀释压力题）
- 产物：results/20260928_stressset/stress_expand.jsonl（含溯源与难度字段）
用法（注意：勿 source knc_env.sh——其 OPENSEARCH/MySQL 指向生产会覆盖测试栈
  env，导致通道画像全空、全部误标 deep；LLM/嵌入变量由 load_env_services 注入）：
  bash -c 'set -a; source deploy/.env.services; set +a; <测试栈 MYSQL/OS/Redis env> \
    .venv/bin/python stress_expand.py [--n 80]'
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

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, REPO, load_questions, load_slice_ids  # noqa: E402

KB = json.load(open(os.path.join(DATA, "teststack_ne", "kb.json")))["kb_id"]
OUT = os.path.join(REPO, "eval", "baselines", "results", "20260928_stressset",
                   "stress_expand.jsonl")
OUT_V2 = os.path.join(REPO, "eval", "baselines", "results", "20260928_stressset",
                      "stress_expand_v2.jsonl")

DRAFT_SYS = ("你是学科文献测评题的出题人。基于给定文档内容写一条中文检索问题："
             "问题应能由该文档回答，涉及具体内容（方法/参数/结论/比较），"
             "不要出现文档标题原文，不要用'有哪些研究'模板。只输出问题本身。")
DRAFT_SYS_V2 = ("你是学科文献测评题的出题人，专门构造高难度检索问题。基于给定文档"
                "片段写一条中文检索问题，要求：问题能由该片段回答；措辞必须与片段"
                "措辞脱离——不照抄原文连续 4 字以上的词串，用同义改写与不同句式，"
                "不得出现'该研究/本文/该文献'等指代；涉及具体内容（方法/参数/"
                "结论/比较）。只输出问题本身。")

def _section_excerpt(text: str, rng: random.Random) -> str:
    """v2：从文档后 2/3 随机窗口取片段，避开摘要头部（难度注入之一）。"""
    if len(text) <= 3600:
        return text[:3000]
    start = rng.randint(len(text) // 3, max(len(text) // 3 + 1, len(text) - 3000))
    return text[start:start + 3000]

def _gram_overlap(q: str, text: str, n: int = 4) -> float:
    """查询与原文的 n-gram 重合率（词法脱离验收，v2）。"""
    qg = {q[i:i + n] for i in range(len(q) - n + 1)}
    if not qg:
        return 1.0
    hit = sum(1 for g in qg if g in text)
    return hit / len(qg)
VERIFY_SYS = ("你是严谨的审题人。判断给定文档内容能否回答该问题。"
              '只输出 JSON：{"answerable": true/false, "reason": "一句话"}')


def load_env_services() -> None:
    for line in open(os.path.join(REPO, "deploy", ".env.services"), encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)


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


def _fused_first_rank(query: str, gtest: str, top_k: int = 20):
    """测试栈服务侧融合序的首金名次（真融合路径，非脚本复刻）。"""
    import urllib.request as _u
    body = json.dumps({"kb_id": KB, "query": query, "top_k": top_k}).encode()
    req = _u.Request("http://127.0.0.1:28121/api/retrieve", data=body,
                     headers={"Content-Type": "application/json"})
    try:
        with _u.urlopen(req, timeout=60) as r:  # noqa: S310
            chunks = json.loads(r.read())["chunks"]
    except Exception:  # noqa: BLE001
        return None
    return next((i for i, c in enumerate(chunks, 1)
                 if c.get("doc_id") == gtest), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="起草文档数")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--v2", action="store_true",
                    help="难度注入：非摘要片段起草+词法脱离+融合 top-10 缺席硬验收")
    args = ap.parse_args()
    load_env_services()

    from knc_engine import db, embedding, search  # noqa: E402  (测试栈 env 由外层注入)
    idmap = json.load(open(os.path.join(DATA, "teststack_ne", "idmap.json")))
    stable2test = idmap

    seed_queries = {json.loads(l)["query"] for l in
                    open(os.path.join(os.path.dirname(OUT), "stress_seed.jsonl"),
                         encoding="utf-8") if l.strip()}
    existing = set()
    if os.path.exists(OUT):  # 断点续跑：已验收题目不重起草
        existing = {json.loads(l)["query"] for l in open(OUT, encoding="utf-8") if l.strip()}
    seed_golds = {g for s in ("nea", "ner") for q in load_questions(s)
                  for g in q["gold_doc_ids"]}

    docs = [json.loads(l) for l in
            open(os.path.join(DATA, "corpus_docs.jsonl"), encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    pool = [d for d in docs if d["doc_id"] not in seed_golds]
    if args.v2:  # v2：种子金标文档正是稀释倾向人群，纳入出题池（v2 硬验收兜底难度）
        pool = list(docs)
    rng.shuffle(pool)

    done = 0
    v2 = args.v2
    rng = random.Random(args.seed + 7)
    out_f = open(OUT_V2 if v2 else OUT, "a", encoding="utf-8")
    for d in pool:
        if done >= args.n:
            break
        excerpt = _section_excerpt(d["text"], rng) if v2 else d["text"][:3000]
        sys_p = DRAFT_SYS_V2 if v2 else DRAFT_SYS
        drafts = []
        for _ in range(2 if v2 else 1):  # v2 验收严，每文档两次起草
            q = llm(sys_p, f"【文档内容】\n{excerpt}\n\n请写一条检索问题。")
            q = q.strip().strip('"').split("\n")[0][:120]
            if not q or q in seed_queries or q in existing:
                continue
            if v2:
                if _gram_overlap(q, excerpt, n=8) > 0.15:
                    continue  # 词法脱离：8-gram 成句照抄率 ≤15%（术语重合放行）
                drafts.append(q)
                continue
            verdict = llm(VERIFY_SYS, f"【问题】\n{q}\n\n【文档内容】\n{excerpt}")
            m = re.search(r"\{.*\}", verdict, re.S)
            if bool(m) and '"answerable":true' in m.group(0).replace(" ", "").lower():
                drafts.append(q)
        if not drafts:
            continue
        q = drafts[0]

        # 难度画像：四通道 top-100 内金标可见性（测试栈 doc_id 空间）
        gtest = stable2test.get(d["doc_id"]) or d["doc_id"]
        qvec = embedding.embed([q])[0]
        best = None
        for name, hits in (("lexical", search.lexical_search(KB, q, 100)),
                           ("dense", search.dense_search(KB, qvec, 100)),
                           ("synq", search.synq_search(KB, qvec, 100))):
            r = next((i for i, h in enumerate(hits, 1) if h.get("doc_id") == gtest), None)
            if r and (best is None or r < best[0]):
                best = (r, name)
        dv = search.doc_vec_search(KB, qvec, 100)
        rdv = next((i for i, h in enumerate(dv, 1) if h.get("doc_id") == gtest), None)
        if rdv and (best is None or rdv < best[0]):
            best = (rdv, "docvec")
        category = "dilute" if best else "deep"  # deep=四通道全无（深失生成器）
        # v2 硬验收（§5 稀释带约束）：通道可见 + 融合 top-10 缺席 = 按构造留 headroom
        fused_rank = None
        if v2 and best:
            fused_rank = _fused_first_rank(q, gtest)
            if fused_rank is not None and fused_rank <= 10:
                continue  # 融合已命中 → 无 headroom，弃
        out_f.write(json.dumps({
            "query": q, "gold_doc_ids": [d["doc_id"]],
            "source": "stress_expand_v2" if v2 else "stress_expand",
            "origin_doc": d["doc_id"], "origin_title": d["title"][:60],
            "channel_best": {"rank": best[0], "channel": best[1]} if best else None,
            "category": category,
            "fused_first_gold_rank": fused_rank,
        }, ensure_ascii=False) + "\n")
        out_f.flush()
        existing.add(q)
        done += 1
        print(f"[{done}/{args.n}] {'dilute' if best else 'deep'} "
              f"{(best[0] if best else '-'):>4} {q[:44]}", flush=True)
    out_f.close()
    print(f"完成：新增 {done} 题 → {OUT}")


if __name__ == "__main__":
    main()
