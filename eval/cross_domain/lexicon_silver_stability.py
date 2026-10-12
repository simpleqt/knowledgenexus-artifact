#!/usr/bin/env python3
"""silver 仪器敏感性（2026-10-12 复审 R3-P3-M2/R1-P3-M1/R2-P3-M2 落地）。

已有每域 30 篇跨家族（DeepSeek）复标 silver_C，在完全相同子集上以
A 与 C 两种参照重算五臂 occurrence 级 P/R——臂间相对排序是否随判官
替换而翻转即仪器敏感性。同时给判读有效率 Wilson 95% CI 与仲裁比例
（C-arbit + C-substitute 占总行数），匹配规则=norm(lowercase+朴素复数
剥除，与 stage_tables 同一双侧对称规则）。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/lexicon_silver_stability.py'
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import precompile, scan  # noqa: E402
from lexicon_cross_domain import _norm_term  # noqa: E402

ROOT = os.path.join(REPO, "eval", "baselines", "results", "20261011_cross_domain")
ARMS = ["schema", "freq", "cvalue", "llmdirect", "threesource"]
OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261012_revision")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def prf(test_docs, silver: dict, pats) -> dict:
    hit_n = sil_n = hit_in_sil = sil_hit = 0
    for d in test_docs:
        sil = {_norm_term(t) for t in silver.get(d["doc_id"], [])}
        matched = set()
        for surface, _et, _s, _e in scan(d["text"], pats):
            hit_n += 1
            if _norm_term(surface) in sil:
                hit_in_sil += 1
            matched.add(_norm_term(surface))
        sil_n += len(sil)
        sil_hit += len(sil & matched)
    p = hit_in_sil / hit_n if hit_n else 0.0
    r = sil_hit / sil_n if sil_n else 0.0
    return {"P": round(p, 3), "R": round(r, 3),
            "F1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "occurrences": hit_n}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    report = {"matching_rule": "norm = lowercase + naive plural strip "
                               "(symmetric on both sides)",
              "domains": {}}
    for dom in ("cs", "bio", "mat"):
        out = os.path.join(ROOT, dom)
        comps = os.path.join(out, "components")
        test = json.load(open(os.path.join(out, "test.json"),
                              encoding="utf-8"))
        silver_a = {r["doc_id"]: [t["surface"] for t in r.get("terms") or []]
                    for r in (json.loads(l) for l in
                              open(os.path.join(out, "silver_A.jsonl"),
                                   encoding="utf-8"))}
        silver_c = {r["doc_id"]: [t["surface"] for t in r.get("terms") or []]
                    for r in (json.loads(l) for l in
                              open(os.path.join(out, "silver_C.jsonl"),
                                   encoding="utf-8"))}
        subset_ids = set(silver_c) & set(silver_a)
        subset = [d for d in test if d["doc_id"] in subset_ids]

        cells = {}
        for arm in ARMS:
            lex = json.load(open(os.path.join(comps, f"arm_{arm}.json"),
                                 encoding="utf-8"))
            pats = precompile(lex)
            cells[arm] = {"vs_A": prf(subset, silver_a, pats),
                          "vs_C": prf(subset, silver_c, pats)}
            # Valid Wilson CI + 仲裁比例（arbit 终判文件）
            apath = os.path.join(out, f"arbit_{arm}.jsonl")
            if os.path.exists(apath):
                recs = [json.loads(l) for l in open(apath, encoding="utf-8")]
                final = [r["final"] for r in recs if r["final"] is not None]
                k = sum(final)
                n = len(final)
                c_judged = sum(1 for r in recs if r["mode"] in
                               ("C-arbit", "C-substitute"))
                cells[arm]["valid"] = {
                    "rate": round(k / n, 3) if n else None, "n": n,
                    "wilson95": wilson(k, n),
                    "arbit_fraction": round(c_judged / len(recs), 3)}

        rank_f1_a = sorted(ARMS, key=lambda a: -cells[a]["vs_A"]["F1"])
        rank_f1_c = sorted(ARMS, key=lambda a: -cells[a]["vs_C"]["F1"])
        rank_r_a = sorted(ARMS, key=lambda a: -cells[a]["vs_A"]["R"])
        rank_r_c = sorted(ARMS, key=lambda a: -cells[a]["vs_C"]["R"])
        report["domains"][dom] = {
            "n_docs": len(subset), "cells": cells,
            "rank_F1_vs_A": rank_f1_a, "rank_F1_vs_C": rank_f1_c,
            "rank_R_vs_A": rank_r_a, "rank_R_vs_C": rank_r_c,
            "rank_F1_stable": rank_f1_a == rank_f1_c,
            "rank_R_stable": rank_r_a == rank_r_c}
        print(f"[{dom}] n={len(subset)}  F1 rank A: {rank_f1_a}")
        print(f"[{dom}]          F1 rank C: {rank_f1_c}  "
              f"stable={rank_f1_a == rank_f1_c}  "
              f"R-rank stable={rank_r_a == rank_r_c}")

    json.dump(report, open(os.path.join(OUT, "silver_stability.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[saved]", os.path.join(OUT, "silver_stability.json"))


if __name__ == "__main__":
    main()
