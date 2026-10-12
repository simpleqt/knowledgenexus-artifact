#!/usr/bin/env python3
"""P0 对照组三件（2026-10-12 审稿 P0 落地）：

1. CS 语言对齐 schema 臂：原始 schema 词典的英文规范化（去 CJK 别名、
   ASCII 别名小写化）——同冻结重放帧扫描，检验 CS 塌缩是否语言混杂
   （若对齐后仍 C10 高/召回 0 附近，则失效是类别词语义形态而非语言）。
2. 冻结样本 chunk→doc 去重统计（样本单位口径）。
3. 重叠抑制敏感性：replica 逐词扫描输出加生产式最长匹配/非重叠抑制，
   跨域五基线臂重扫对比结构指标变化。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python ... --part <aligned|units|suppress>'
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import precompile, scan  # noqa: E402
from lexicon_cross_domain import GENERIC_HEAD  # noqa: E402

XD = os.path.join(REPO, "eval", "baselines", "results",
                  "20261011_cross_domain")
OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261013_p0_controls")
sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402


def scan_metrics(rows_text, sample_ids, lex, suppress=False):
    pats = precompile(lex)
    links = collections.Counter()
    doc_ents = collections.defaultdict(set)
    for i in sample_ids:
        for surface, etype, s, e in scan(rows_text[i], pats):
            links[(surface.lower(), etype)] += 1
            doc_ents[i].add((surface.lower(), etype))
    if suppress:
        # 生产 dict-v3 语义：同 start 长词优先 + 跨 start 非重叠
        keep = collections.Counter()
        doc_ents = collections.defaultdict(set)
        for i in sample_ids:
            spans = [(s, e, sf.lower(), et) for sf, et, s, e
                     in scan(rows_text[i], pats)]
            spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
            last_end = -1
            for s, e, low, et in spans:
                if s >= last_end:
                    keep[(low, et)] += 1
                    doc_ents[i].add((low, et))
                    last_end = e
        links = keep
    total = sum(links.values())
    top10 = sorted(links.values(), reverse=True)[:10]
    per_doc = [len(v) for v in doc_ents.values()]
    head = sum(n for (f, t), n in links.items() if f in GENERIC_HEAD)
    return {"links": total, "entities": len(links),
            "c10": round(sum(top10) / total, 3) if total else 0.0,
            "conc_per_doc": round(statistics.mean(per_doc), 2)
            if per_doc else 0,
            "generic_head_share": round(head / total, 3) if total else 0.0,
            "docs": len(doc_ents)}


def load_cs_frame():
    rows = {r["doc_id"]: r["text"] for r in (
        json.loads(l) for l in open(
            os.path.join(REPO, "eval", "baselines", "data",
                         "public_corpora", "cs.jsonl"), encoding="utf-8"))}
    frame = json.load(open(os.path.join(XD, "cs", "frame.json")))
    return rows, frame


# ---------- 1. 语言对齐 schema 臂 ----------

def part_aligned():
    import yaml
    os.makedirs(OUT, exist_ok=True)
    doc = yaml.safe_load(open(os.path.join(REPO, "schemas",
                                           "computer_science.yaml"),
                              encoding="utf-8"))
    raw = {}
    aligned = {}
    for ty, spec in doc.get("entity_types", {}).items():
        for a in [spec.get("name")] + list(spec.get("aliases") or []):
            if not a:
                continue
            raw[a] = ty
            if a.isascii():
                low = a.lower()
                aligned[low] = ty
                if low.endswith("s") and len(low) > 3:
                    aligned.setdefault(low[:-1], ty)
    print(f"[aligned] raw={len(raw)} aligned(EN,lower)={len(aligned)}")
    rows, frame = load_cs_frame()
    res = {}
    for tag, lex in (("schema_raw", raw), ("schema_en_aligned", aligned)):
        per = [scan_metrics(rows, ids, lex) for ids in
               frame["sample_ids"]]
        agg = {k: {"mean": round(statistics.mean(p[k] for p in per), 3),
                   "sd": round(statistics.stdev(p[k] for p in per), 3)}
               for k in ("links", "entities", "c10", "conc_per_doc",
                         "generic_head_share", "docs")}
        res[tag] = agg
        print(f"[aligned] {tag}: {agg}")
    json.dump(res, open(os.path.join(OUT, "cs_aligned_schema.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    # 对齐臂词典样例
    json.dump(aligned, open(os.path.join(OUT, "cs_schema_en_aligned.json"),
                            "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)


# ---------- 2. chunk→doc 去重统计 ----------

def part_units():
    os.makedirs(OUT, exist_ok=True)
    ms = json.load(open(os.path.join(REPO, "eval", "baselines",
                                     "results",
                                     "20261010_lexicon_multisample",
                                     "multisample.json")))
    ids = ms["sample_chunk_ids"]["0"]
    c = M.search.client()
    rows = c.mget(index="knc-chunk",
                  body={"docs": [{"_id": x, "_source": ["doc_id"]}
                                 for x in ids]})
    docs = [h["_source"]["doc_id"] for h in rows["docs"] if h.get("found")]
    res = {"chunks": len(ids), "found": len(docs),
           "distinct_docs": len(set(docs)),
           "chunks_per_doc_mean": round(len(docs) / max(1, len(set(docs))), 2)}
    print("[units]", res)
    json.dump(res, open(os.path.join(OUT, "chunk_doc_dedup.json"), "w",
                        encoding="utf-8"), indent=1)


# ---------- 3. 重叠抑制敏感性 ----------

def part_suppress():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for dom in ("cs", "bio", "mat"):
        rows = {r["doc_id"]: r["text"] for r in (
            json.loads(l) for l in open(
                os.path.join(REPO, "eval", "baselines", "data",
                             "public_corpora", f"{dom}.jsonl"),
                encoding="utf-8"))}
        frame = json.load(open(os.path.join(XD, dom, "frame.json")))
        res[dom] = {}
        for arm in ("schema", "freq", "cvalue", "llmdirect", "threesource"):
            lex = json.load(open(os.path.join(XD, dom, "components",
                                              f"arm_{arm}.json"),
                                 encoding="utf-8"))
            base = [scan_metrics(rows, ids, lex) for ids in
                    frame["sample_ids"]]
            supp = [scan_metrics(rows, ids, lex, suppress=True)
                    for ids in frame["sample_ids"]]
            res[dom][arm] = {
                "base": {k: round(statistics.mean(p[k] for p in base), 3)
                         for k in ("links", "entities", "c10",
                                   "conc_per_doc", "docs")},
                "suppressed": {k: round(statistics.mean(
                    p[k] for p in supp), 3) for k in
                    ("links", "entities", "c10", "conc_per_doc", "docs")}}
            b, s = res[dom][arm]["base"], res[dom][arm]["suppressed"]
            print(f"[supp-{dom}] {arm:12s} links {b['links']}→{s['links']} "
                  f"ents {b['entities']}→{s['entities']} "
                  f"C10 {b['c10']}→{s['c10']} "
                  f"conc {b['conc_per_doc']}→{s['conc_per_doc']}")
    json.dump(res, open(os.path.join(OUT, "overlap_suppression.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["aligned", "units", "suppress"])
    args = ap.parse_args()
    {"aligned": part_aligned, "units": part_units,
     "suppress": part_suppress}[args.part]()


if __name__ == "__main__":
    main()
