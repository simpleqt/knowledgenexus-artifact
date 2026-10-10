#!/usr/bin/env python3
"""自然措辞关联题集生成（2026-09-30）：K1/词典解锁价值的量尺。

背景：模板关联题（nea48）在部署配置下 hit@10=1.0000 饱和；K1 词典解锁把
realq 的 cooc 覆盖 6→9 但命中未动——natural 面需要专门题集。本脚本：
  ① 概念实体对挖掘（中 df 段，共现文档 2≤n≤8 为金标）；
  ② LLM 生成自然措辞问句（显式禁模板词：同时涉及/有哪些研究/关联检索/
     交叉研究/结合…与…），风格多样化；
  ③ 自检：生成问句含模板词即弃，重试一次仍含则弃对。
产物：eval/harness/ne_journal_natural_assoc.jsonl（query/gold_doc_ids/
pair 元数据），幂等可续跑。

用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  "$R/.venv/bin/python" "$R/eval/baselines/scripts/natural_assoc_gen.py" [--n 100]'
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
OUT = REPO / "eval/harness/ne_journal_natural_assoc.jsonl"
BIB = ("paper", "person", "organization")
TEMPLATE = re.compile(r"同时涉及|同时使用|有哪些研究|关联研究|相关研究|都涉及|"
                      r"共同涉及|交叉研究|关联检索|结合.{0,12}与")
STYLES = ["口语化提问", "学术检索式", "工程师找资料式", "简短关键词风",
          "问题描述式"]


def pairs_from_db(db, rng, want: int, used_pairs: set) -> list[dict]:
    """轻量挖掘：中 df 概念实体随机配对 + 便宜的共现文档查询（2≤n≤8 为金标）。
    首版 peer-discovery 多表 JOIN + ORDER BY RAND 在 137 万链接表上分钟级
    卡死（实测 0% CPU 等 MySQL），弃用。"""
    ents = [dict(r) for r in db.exec(
        f"SELECT entity_id e, canonical_name n, doc_df FROM canonical_entity"
        f" WHERE kb_id=UNHEX(%s) AND status='ACTIVE' AND entity_type NOT IN {BIB}"
        f" AND doc_df BETWEEN 3 AND 400", (KB,), fetch=True)]
    out = []
    tries = 0
    while len(out) < want and tries < want * 40:
        tries += 1
        e, p = rng.sample(ents, 2)
        key = frozenset((e["e"], p["e"]))
        if key in used_pairs:
            continue
        docs = [r["d"] for r in db.exec(
            "SELECT DISTINCT LOWER(HEX(m.doc_id)) d FROM entity_link_registry l"
            " JOIN raw_mention m ON m.mention_id=l.mention_id"
            " WHERE l.kb_id=UNHEX(%s) AND l.entity_id IN (%s,%s)"
            " GROUP BY m.doc_id HAVING COUNT(DISTINCT l.entity_id)=2",
            (KB, e["e"], p["e"]), fetch=True)]
        if not (2 <= len(docs) <= 8):
            continue
        used_pairs.add(key)
        out.append({"a": {"id": e["e"], "name": e["n"]},
                    "b": {"id": p["e"], "name": p["n"]},
                    "gold_doc_ids": docs})
    return out


def gen_q(llm, a: str, b: str, style: str) -> str | None:
    raw = llm.chat(
        f"两个检索概念：A={a}；B={b}。\n"
        f"用「{style}」写一条中文检索问句，意图=找到同时关于 A 和 B 的文献。"
        "硬性要求：不得出现这些模板词：同时涉及/同时使用/有哪些研究/关联研究/"
        "相关研究/都涉及/共同涉及/交叉研究/结合；不得逐字给出'A与B'这种并列"
        "骨架；只输出问句本身一行。",
        system="你是学术检索用户画像模拟器。", max_tokens=120, temperature=0.8,
        timeout=15).strip().strip('"“”').splitlines()[0] if True else None
    if not raw or TEMPLATE.search(raw):
        return None
    if a.lower() not in raw.lower() and b.lower() not in raw.lower():
        return None  # 至少一个概念面形要在句中（保证可检索性）
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    from knc_engine import db, llm
    done: list[dict] = []
    if OUT.exists():
        done = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    used = {frozenset((d["pair"]["a"]["id"], d["pair"]["b"]["id"])) for d in done}
    rng = random.Random(20260930 + len(done))
    print(f"已有 {len(done)} 题，目标 {args.n}")
    while len(done) < args.n:
        batch = pairs_from_db(db, rng, args.n - len(done), used)
        if not batch:
            print("[warn] 实体对挖掘空批，终止")
            break
        for p in batch:
            if len(done) >= args.n:
                break
            style = STYLES[rng.randrange(len(STYLES))]
            try:
                q = gen_q(llm, p["a"]["name"], p["b"]["name"], style)
                if q is None:
                    q = gen_q(llm, p["a"]["name"], p["b"]["name"],
                              STYLES[(STYLES.index(style) + 1) % len(STYLES)])
            except Exception:
                q = None
            if q:
                done.append({"query": q, "gold_doc_ids": p["gold_doc_ids"],
                             "pair": p and {"a": p["a"], "b": p["b"]}, "style": style})
                print(f"  [{len(done)}/{args.n}] {q[:46]}  (gold={len(p['gold_doc_ids'])})", flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            for d in done:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(OUT, "w", encoding="utf-8") as f:
        for d in done:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[written] {OUT}（{len(done)} 题）")


if __name__ == "__main__":
    main()
