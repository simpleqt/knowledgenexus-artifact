#!/usr/bin/env python3
"""关联题集生成器（corpus-first，参数化版——B78/B79 /tmp 工作流的正式入库）。

三段式：
  pairs   实体对挖掘：canonical_entity × entity_link 共现 → gold=同时提及双实体的文档
          （2..max-gold 篇，按覆盖面型过滤），seed 可复现
  phrase  LLM 措辞：实体对+gold 标题 → 中文关联问句（"有哪些研究同时涉及X与Y"）
  verify  校验：gold 非空、问句含双实体面形、去重；输出 harness jsonl

产物：{"query", "gold_doc_ids", "mode": "both"} jsonl，直接喂 retrieval_eval.py。
LLM 端点经环境变量注入（LLM_API_BASE/KEY/MODEL，与 run_services.sh 同款），不入库。

用法：
  bash -c 'source deploy/run_services.sh 仅导出环境' 后：
  python gen_assoc_set.py pairs  --kb $KB --n 200 --out pairs.json
  python gen_assoc_set.py phrase --pairs pairs.json --out draft.jsonl
  python gen_assoc_set.py verify --draft draft.jsonl --kb $KB --out final.jsonl
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

_SYSTEM = ("你是检索评测题集设计器。给定一对学科实体与若干相关文献标题，写一句中文"
           "关联检索问句（形如\"有哪些研究同时涉及X与Y…？\"），自然、像真实用户提问，"
           "不得照抄标题。只输出 JSON：{\"query\": \"...\"}。")

_TYPE_ZH = {"method": "方法", "detector": "探测器", "isotope": "核素", "term": "专业术语",
            "material": "材料", "device": "装置", "technique": "技术",
            "facility": "实验装置", "concept": "概念", "site": "场地", "process": "过程"}


def _db():
    from knc_engine import db  # PYTHONPATH 由运行环境注入
    return db


def _llm(prompt: str) -> str:
    base = os.environ["LLM_API_BASE"].rstrip("/")
    body = json.dumps({"model": os.environ["LLM_MODEL"], "temperature": 0.4,
                       "messages": [{"role": "system", "content": _SYSTEM},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        return json.loads(r.read())["choices"][0]["message"]["content"]


def cmd_pairs(args: argparse.Namespace) -> None:
    """实体对挖掘（两段式，生产友好）：随机抽实体 → 逐实体索引点查 doc 集 →
    Python 集合交取 |∩|∈[min_gold,max_gold] 的对。不走纯 SQL 自联接——
    177 万链接上按文档自联接是笛卡尔爆炸（2026-09-25 实测 500s 超时）。"""
    import random as _r
    db = _db()
    ents = db.exec(
        "SELECT entity_id eid, canonical_name nm, entity_type tp FROM canonical_entity"
        " WHERE kb_id=UNHEX(%s) AND status='ACTIVE' AND entity_type IN"
        " ('method','detector','isotope','term','material','device','technique',"
        "  'facility','concept','site','process')"
        " ORDER BY RAND(%s) LIMIT %s", [args.kb, args.seed, args.n * 8], fetch=True)
    rng = _r.Random(args.seed)
    docs_of = {}
    for e in ents:
        rows = db.exec(
            "SELECT DISTINCT LOWER(HEX(m.doc_id)) d FROM entity_link_registry l"
            " JOIN raw_mention m ON m.mention_id=l.mention_id"
            " WHERE l.kb_id=UNHEX(%s) AND l.entity_id=%s LIMIT 2000",
            [args.kb, e["eid"]], fetch=True)
        ds = {r["d"] for r in rows}
        if args.min_gold <= len(ds) <= 2000:
            docs_of[e["eid"]] = (e, ds)
    cand = list(docs_of.values())
    rng.shuffle(cand)
    out, tried = [], 0
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            if len(out) >= args.n or tried > args.n * 60:
                break
            (ea, da), (eb, db_) = cand[i], cand[j]  # 每项即 (实体, doc集)
            tried += 1
            inter = da & db_
            if args.min_gold <= len(inter) <= args.max_gold:
                out.append({"kb": args.kb, "x": ea["nm"], "tx": ea["tp"],
                            "y": eb["nm"], "ty": eb["tp"],
                            "gold": sorted(inter),
                            "titles": []})
            if len(out) >= args.n:
                break
        if len(out) >= args.n:
            break
    # 补 gold 标题（题录面，供 phrase 用）
    for p_ in out:
        if not p_["gold"]:
            continue
        ph = ",".join(["UNHEX(%s)"] * len(p_["gold"]))
        t = db.exec(f"SELECT doc_id_hex d, title FROM (SELECT HEX(doc_id) doc_id_hex,"
                    f" title FROM kb_documents WHERE doc_id IN ({ph})) t",
                    p_["gold"], fetch=True)
        p_["titles"] = [r["title"] or "" for r in t][:5]
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"pairs: {len(out)} 对（实体 {len(cand)}，尝试 {tried}）→ {args.out}")


def cmd_phrase(args: argparse.Namespace) -> None:
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    with open(args.out, "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            prompt = (f"实体A：{p['x']}（{_TYPE_ZH.get(p['tx'], p['tx'])}）\n"
                      f"实体B：{p['y']}（{_TYPE_ZH.get(p['ty'], p['ty'])}）\n"
                      f"相关文献标题：{json.dumps(p['titles'], ensure_ascii=False)}\n"
                      "写一句中文关联检索问句。")
            try:
                raw = _llm(prompt)
                q = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["query"]
            except Exception:
                continue
            f.write(json.dumps({"query": q, "gold_doc_ids": p["gold"],
                                "pair": [p["x"], p["y"]], "mode": "both"},
                               ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(pairs)}", flush=True)
            time.sleep(0.2)
    print(f"phrase 完成 → {args.out}")


def cmd_verify(args: argparse.Namespace) -> None:
    seen, kept = set(), 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for line in open(args.draft, encoding="utf-8"):
            r = json.loads(line)
            q, pair = r["query"], r.get("pair") or []
            if not q or q in seen or len(r["gold_doc_ids"]) < 1:
                continue
            if args.strict_pair and pair and not all(p.lower() in q.lower() for p in pair if p):
                continue  # --strict-pair：原词面形须在问句中（LLM 常译为中文，默认关）
            seen.add(q)
            fo.write(json.dumps({"query": q, "gold_doc_ids": r["gold_doc_ids"],
                                 "mode": "both"}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"verify: 保留 {kept} 题 → {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pairs")
    p1.add_argument("--kb", required=True)
    p1.add_argument("--n", type=int, default=200)
    p1.add_argument("--min-gold", type=int, default=2)
    p1.add_argument("--max-gold", type=int, default=6)
    p1.add_argument("--min-df", type=int, default=2)
    p1.add_argument("--max-df", type=int, default=400)
    p1.add_argument("--seed", type=int, default=20260925)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("phrase")
    p2.add_argument("--pairs", required=True)
    p2.add_argument("--out", required=True)
    p3 = sub.add_parser("verify")
    p3.add_argument("--draft", required=True)
    p3.add_argument("--out", required=True)
    p3.add_argument("--strict-pair", action="store_true",
                    help="要求原始实体面形出现在问句中（默认关：LLM 常将英文实体译为中文）")
    args = ap.parse_args()
    {"pairs": cmd_pairs, "phrase": cmd_phrase, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
