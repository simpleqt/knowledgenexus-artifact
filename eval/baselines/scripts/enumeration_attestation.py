#!/usr/bin/env python3
"""P1-c 召回参照分层：为枚举术语计 construct 语料（3,000 篇建词典集）文档复现数，
按复现层（0 / 1-2 / 3-10 / >10 篇）分解五臂覆盖率——区分"有界词表可承载的
复现实词"与"论文自造名/单篇孤词"（后者属提及层职责，词表层无从覆盖）。

口径：词边界 + 大小写不敏感 + 空格/连字符互换等价（与扫描器宽松面形一致）。
输入：enumeration_recall.json（唯一术语集）+ 20261011_cross_domain frame.json
+ data/public_corpora/<dom>.jsonl（construct_ids 过滤）。
输出：enumeration_recall_strata.json（仅面形与计数）。
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ENUM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "results", "20261013_enumeration")
XD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "results", "20261011_cross_domain")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                    "public_corpora")
ARMS = ["schema", "freq", "cvalue", "llmdirect", "threesource"]
STRATA = [("df=0 (unseen in construct)", lambda d: d == 0),
          ("df=1-2 (rare)", lambda d: 1 <= d <= 2),
          ("df=3-10 (recurring)", lambda d: 3 <= d <= 10),
          ("df>10 (core)", lambda d: d > 10)]


def norm(s: str) -> str:
    s = s.strip().strip("\"'`.,;:()[]").casefold()
    return re.sub(r"\s+", " ", s)


def term_regex(t: str) -> re.Pattern:
    esc = re.escape(t.casefold())
    esc = esc.replace("\\ ", "[ -]").replace("\\-", "[ -]")
    return re.compile(r"(?<![a-z0-9])" + esc + r"(?![a-z0-9])")


def main() -> None:
    rec = json.load(open(os.path.join(ENUM_DIR, "enumeration_recall.json"),
                         encoding="utf-8"))
    out = {"per_domain": {}, "overall": {}, "df": {}}
    tot_terms = {s: 0 for s, _ in STRATA}
    tot_hits = {s: {a: 0 for a in ARMS} for s, _ in STRATA}

    for dom in ["cs", "bio", "mat"]:
        frame = json.load(open(os.path.join(XD_DIR, dom, "frame.json")))
        cids = set(frame["construct_ids"])
        texts = []
        for line in open(os.path.join(DATA, f"{dom}.jsonl"), encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["doc_id"] in cids:
                texts.append(r["text"].casefold())
        assert len(texts) == len(cids), (dom, len(texts), len(cids))

        arms_surf = {}
        for a in ARMS:
            d = json.load(open(os.path.join(XD_DIR, dom, "components",
                                            f"arm_{a}.json")))
            arms_surf[a] = {norm(s) for s in d}

        terms = rec["terms"][dom]["__enum__"]
        df = {}
        for t in terms:
            pat = term_regex(t)
            df[t] = sum(1 for tx in texts if pat.search(tx))
        out["df"][dom] = {t: df[t] for t in terms}

        dom_stats = {}
        for sname, pred in STRATA:
            sel = [t for t in terms if pred(df[t])]
            hits = {a: sum(1 for t in sel if norm(t) in arms_surf[a])
                    for a in ARMS}
            dom_stats[sname] = {"n": len(sel), "hits": hits,
                                "rate": {a: (round(hits[a] / len(sel), 3)
                                             if sel else None)
                                         for a in ARMS}}
            tot_terms[sname] += len(sel)
            for a in ARMS:
                tot_hits[sname][a] += hits[a]
        out["per_domain"][dom] = dom_stats
        print(f"[{dom}] construct docs={len(texts)} terms={len(terms)} "
              + " ".join(f"{s.split(' ')[0]}={dom_stats[s]['n']}"
                         for s, _ in STRATA))

    out["overall"] = {sname: {"n": tot_terms[sname],
                              "hits": tot_hits[sname],
                              "rate": {a: (round(tot_hits[sname][a] /
                                                 tot_terms[sname], 3)
                                           if tot_terms[sname] else None)
                                       for a in ARMS}}
                      for sname, _ in STRATA}
    dst = os.path.join(ENUM_DIR, "enumeration_recall_strata.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(json.dumps(out["overall"], ensure_ascii=False, indent=1))
    print("written:", dst)


if __name__ == "__main__":
    main()
