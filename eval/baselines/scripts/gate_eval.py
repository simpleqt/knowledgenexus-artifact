#!/usr/bin/env python3
"""O5 B83/B85 意图门控效度分析（混淆矩阵 + P/R/F1 + 误伤案例，2026-09-25）。

文献依据：Dissecting Agentic RAG（arXiv 2606.21553）诊断的路由失效标准模式——
命名实体启发式在非关联查询上过度路由；SelRoute（arXiv 2604.02431）证明正则级
查询类型分类可达 83% 有效路由精度。本脚本对 B83 双门（模板词 ∧ ≥2 实体面形）
与 B85 英文非关联门做量化效度检验：

  正类 = 关联题集（nea/csha，语料对子生成，应触发 synq 提权）
  负类 = 自然问句题集（ner/cshr，不应触发）
  B85  = 英文非关联查询（bench 三题集，应触发 lexical 降权）；中文集为负类

门谓词与服务端 main.py 逐字一致（模板正则/ASCII 词边界含复数/CJK 最长匹配）；
实体词典经 knc_engine.db 直连（只读 SELECT），栈由 MYSQL_* 环境变量决定。

用法：
  MYSQL_PORT=23306 MYSQL_USER=knc MYSQL_PASSWORD=knc_test \
  PYTHONPATH=libs/engine python3 gate_eval.py \
    --kb 0cf04665c3bd427281976c3502eb4d99 \
    --pos eval/harness/ne_journal_assoc_q48.jsonl \
    --neg eval/harness/ne_journal_realq.jsonl \
    --bench data/bench  # B85 英文触发率（可选）
产物：stdout 报告 + --out JSON（可入证据包）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import DATA, REPO  # noqa: E402

# —— 与 services/retrieval-api/app/main.py B83 块逐字一致 ——
ASSOC_RE = re.compile(
    r"同时涉及|关联检索|有哪些研究|交叉研究|结合.*与"
    r"|research.*both|studies.*involving|intersection of"
    r"|both.*and.*(in|from)|papers.*(on|about).*and", re.IGNORECASE)


def build_entity_patterns(kb_hex: str) -> tuple[object, object]:
    """复刻 main.py _entity_scan_build：canonical_entity+alias 全名 → ASCII/CJK 双正则。"""
    import knc_engine.db as db
    names = [r["canonical_name"] for r in db.exec(
        "SELECT canonical_name FROM canonical_entity"
        " WHERE kb_id=UNHEX(%s) AND status='ACTIVE'", (kb_hex,), fetch=True)]
    names += [r["a"] for r in db.exec(
        "SELECT a.alias_text a FROM entity_alias a JOIN canonical_entity c"
        " ON c.entity_id=a.entity_id WHERE a.kb_id=UNHEX(%s) AND c.status='ACTIVE'",
        (kb_hex,), fetch=True)]
    uniq = sorted({n.lower() for n in names if 1 < len(n) < 80}, key=len, reverse=True)
    a_pat = c_pat = None
    if uniq:
        a_terms = [t for t in uniq if t.isascii()]
        c_terms = [t for t in uniq if not t.isascii()]
        if a_terms:
            a_pat = re.compile(r"(?<![0-9A-Za-z])(?:" + "|".join(re.escape(t) for t in a_terms)
                               + r")s?(?![0-9A-Za-z])", re.IGNORECASE)
        if c_terms:
            c_pat = re.compile("|".join(re.escape(t) for t in c_terms))
    return a_pat, c_pat


def surface_hits(a_pat, c_pat, query: str) -> set[str]:
    """查询命中的实体面形集（与 _entity_scan_cached 的输出语义一致）。"""
    ql = query.lower()
    out: set[str] = set()
    for pat in (a_pat, c_pat):
        if pat is not None:
            out |= {m.group().lower() for m in pat.finditer(ql)}
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="KB hex（实体词典所在栈的库）")
    ap.add_argument("--pos", nargs="+", required=True, help="正类题集 jsonl（应触发）")
    ap.add_argument("--neg", nargs="+", required=True, help="负类题集 jsonl（不应触发）")
    ap.add_argument("--bench", default=None,
                    help="bench 数据目录（B85 英文触发率，data/bench）")
    ap.add_argument("--bench-n", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def load(path: str) -> list[str]:
        p = path if os.path.isabs(path) else next(
            (c for c in (os.path.join(DATA, path), os.path.join(REPO, path)) if os.path.exists(c)), path)
        return [json.loads(l)["query"] for l in open(p, encoding="utf-8") if l.strip()]

    a_pat, c_pat = build_entity_patterns(args.kb)
    print(f"实体词典正则：ascii={'有' if a_pat else '无'} cjk={'有' if c_pat else '无'}（kb={args.kb[:8]}…）")

    rows = []  # (query, label, tmpl, n_surf)
    for path in args.pos:
        for q in load(path):
            s = surface_hits(a_pat, c_pat, q)
            rows.append({"query": q, "label": 1, "tmpl": bool(ASSOC_RE.search(q)),
                         "n_surf": len(s), "surfs": sorted(s)[:6]})
    for path in args.neg:
        for q in load(path):
            s = surface_hits(a_pat, c_pat, q)
            rows.append({"query": q, "label": 0, "tmpl": bool(ASSOC_RE.search(q)),
                         "n_surf": len(s), "surfs": sorted(s)[:6]})

    def confusion(key) -> dict:
        tp = sum(1 for r in rows if r["label"] == 1 and key(r))
        fp = sum(1 for r in rows if r["label"] == 0 and key(r))
        fn = sum(1 for r in rows if r["label"] == 1 and not key(r))
        tn = sum(1 for r in rows if r["label"] == 0 and not key(r))
        p, f, f1 = prf(tp, fp, fn)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p,
                "recall": f, "f1": f1}

    gates = {
        "B83 双门（模板∧≥2面形）": lambda r: r["tmpl"] and r["n_surf"] >= 2,
        "仅模板词门": lambda r: r["tmpl"],
        "仅实体面形门（≥2）": lambda r: r["n_surf"] >= 2,
    }
    print(f"\n## B83 门控混淆（正类={sum(r['label'] for r in rows)} 负类={len(rows)-sum(r['label'] for r in rows)}）\n")
    print("| 门 | TP | FP | FN | TN | precision | recall | F1 |")
    print("|---|---|---|---|---|---|---|---|")
    conf = {}
    for name, key in gates.items():
        conf[name] = confusion(key)
        c = conf[name]
        print(f"| {name} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} "
              f"| {c['precision']} | {c['recall']} | {c['f1']} |")

    # 误伤/漏检案例（双门）
    fp_ex = [r for r in rows if r["label"] == 0 and r["tmpl"] and r["n_surf"] >= 2][:8]
    fn_ex = [r for r in rows if r["label"] == 1 and not (r["tmpl"] and r["n_surf"] >= 2)][:8]
    print("\n**双门误伤（负类被触发）案例：**")
    for r in fp_ex:
        print(f"- FP: {r['query'][:60]}  面形={r['surfs']}")
    print("\n**双门漏检（正类未触发）案例：**")
    for r in fn_ex:
        why = "缺模板词" if not r["tmpl"] else "面形不足"
        print(f"- FN: {r['query'][:60]}  原因={why} 面形={r['surfs']}")

    # B85：英文非关联门（query 首字符 [a-zA-Z] 且非关联模板）
    b85 = {"en_trig": 0, "en_n": 0, "zh_trig": 0, "zh_n": 0}
    ex85 = []
    if args.bench:
        for ds in sorted(os.listdir(args.bench)):
            qf = os.path.join(args.bench, ds, "questions.jsonl")
            if not os.path.exists(qf):
                continue
            qs = [json.loads(l)["query"] for l in open(qf, encoding="utf-8") if l.strip()]
            qs = qs[:args.bench_n]
            for q in qs:
                trig = bool(re.match(r"[a-zA-Z]", q)) and not bool(ASSOC_RE.search(q))
                b85["en_n"] += 1
                b85["en_trig"] += int(trig)
                if not trig and len(ex85) < 5:
                    ex85.append(q[:70])
    for r in rows:  # 中文集为 B85 负类
        trig = bool(re.match(r"[a-zA-Z]", r["query"])) and not r["tmpl"]
        b85["zh_n"] += 1
        b85["zh_trig"] += int(trig)
    print(f"\n## B85 英文非关联门触发率\n")
    print(f"- 英文 bench 查询：{b85['en_trig']}/{b85['en_n']}"
          f" = {round(b85['en_trig']/max(1,b85['en_n']), 4)}（应≈1）")
    print(f"- 中文题集查询：{b85['zh_trig']}/{b85['zh_n']}"
          f" = {round(b85['zh_trig']/max(1,b85['zh_n']), 4)}（应=0）")
    for q in ex85:
        print(f"- 未触发示例: {q}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"kb": args.kb, "confusion": conf, "rows": rows, "b85": b85,
                   "fp_examples": fp_ex, "fn_examples": fn_ex},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
