#!/usr/bin/env python3
"""三源组件消融（2026-10-11 论文实验扩展 E4）。

复用 lexicon_cross_domain --stage build 的组件缓存（不重跑 LLM）：
  full        = seed ∪ classify ∪ expand（存在性校验后）
  -seed       = classify ∪ expand
  -classify   = seed ∪ expand（均经存在性校验）
  -expand     = seed ∪ classify
  -presence   = 原始件合并（seed_raw ∪ classify ∪ expand_raw，不做语料校验）
  seed-only / classify-only / expand-only
  size-matched freq = 与 full 等规模纯频率（规模控制，复用 arm_freq 构造）
评测 = 同域同冻结样本重放扫描（scan_one）→ 结构指标 mean±sd；
规模列与指标并列报告（规模混杂显式可见）。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/lexicon_ablate_sources.py --dom cs'
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import precompile, scan  # noqa: E402
from lexicon_cross_domain import DATA, SCHEMA, scan_one  # noqa: E402


def load_comps(comps):
    def j(name):
        p = os.path.join(comps, name)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return {n: j(f"{n}.json") for n in
            ("seed", "classify", "expand")} | \
           {"expand_raw": j("expand_raw.json"),
            "arm_freq": j("arm_freq.json"),
            "arm_threesource": j("arm_threesource.json")}


def union(*sources):
    lex = {}
    for src in sources:
        for tid, terms in src.items():
            for t in terms:
                lex.setdefault(t, tid)
    return lex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", required=True, choices=["cs", "mat", "bio"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-root", default=os.path.join(
        REPO, "eval", "baselines", "results", "20261011_cross_domain"))
    args = ap.parse_args()
    out = os.path.join(args.out_root, args.dom)
    comps = os.path.join(out, "components")
    C = load_comps(comps)
    seed, classify, expand = C["seed"], C["classify"], C["expand"]
    three = C["arm_threesource"]  # full（含类型主名与存在性校验）

    # 存在性闸：方法对各源实际施加的校验（与 arm_threesource 装配一致）
    construct = json.load(open(os.path.join(out, "construct.json"),
                               encoding="utf-8"))
    from lexicon_cross_domain import text_freq
    freq_set = set(text_freq(construct)) 
    freq_set |= {t.lower() for t in freq_set}
    seed_v = {tid: sorted({t for t in terms
                           if t in freq_set or t.lower() in freq_set})
              for tid, terms in seed.items()}
    arms = {
        "full": three,
        "-seed": union(classify, expand),
        "-classify": union(seed_v, expand),
        "-expand": union(seed_v, classify),
        "-presence": union(C["expand_raw"] or expand, seed, classify),
        "seed-only-v": union(seed_v),
        "seed-only-raw": union(seed),
        "classify-only": union(classify),
        "expand-only": union(expand),
        "freq-size-matched": C["arm_freq"],
    }
    os.makedirs(os.path.join(out, "ablation"), exist_ok=True)
    rows_txt = {r["doc_id"]: r["text"] for r in
                (json.loads(l) for l in open(os.path.join(DATA, f"{args.dom}.jsonl"),
                                             encoding="utf-8"))}
    frame = json.load(open(os.path.join(out, "frame.json")))
    samples = [[(i, rows_txt[i]) for i in ids] for ids in frame["sample_ids"]]
    tasks = [(name, lex, s) for name, lex in arms.items() for s in samples]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, m in pool.map(scan_one, tasks):
            results.setdefault(name, []).append(m)
    agg = {}
    print(f"arm | terms | links | ents | C10 | conc/doc | gen-head | docs")
    for name, lex in arms.items():
        rows_ = results[name]
        g = {k: round(statistics.mean(r[k] for r in rows_), 3) for k in
             ("links", "entities", "c10", "conc_per_doc",
              "generic_head_share", "docs")}
        agg[name] = {"terms": len(lex), **g}
        print(f"{name} | {len(lex)} | {g['links']} | {g['entities']} | "
              f"{g['c10']} | {g['conc_per_doc']} | {g['generic_head_share']} | {g['docs']}")
    json.dump(agg, open(os.path.join(out, "ablation", "ablation.json"), "w"),
              ensure_ascii=False, indent=1)
    print("[saved]", os.path.join(out, "ablation", "ablation.json"))


if __name__ == "__main__":
    main()
