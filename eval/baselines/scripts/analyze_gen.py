#!/usr/bin/env python3
"""生成质量消融的配对差分析：臂间逐题配对 bootstrap 95% CI（复用 score.py.paired_ci）。

用法：analyze_gen.py <results_dir> --ref full [--arms fullprune,fulldoc,fullboth]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from score import paired_ci  # noqa: E402

METRICS = ("faithfulness", "relevancy", "correctness")


def load_arm(d: str, arm: str) -> dict[str, dict]:
    rows = json.load(open(os.path.join(d, f"gen_{arm}.json"), encoding="utf-8"))
    return {r["query"]: r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--ref", default="full")
    ap.add_argument("--arms", default="fullprune,fulldoc,fullboth")
    args = ap.parse_args()

    arms = [args.ref] + [a for a in args.arms.split(",") if a]
    loaded: dict[str, dict[str, dict]] = {}
    for arm in arms:
        try:
            loaded[arm] = load_arm(args.results_dir, arm)
        except FileNotFoundError:
            print(f"[skip] {arm}: gen_{arm}.json 不存在")
    ref = loaded[args.ref]

    agg, ctx_note = {}, {}
    lines = [f"# 生成质量消融配对分析（{args.results_dir}）", "",
             f"参照臂：{args.ref}；指标为逐题配对 bootstrap 95% CI（iters=10000）。", "",
             "## 聚合（按两臂公共题对齐）", "",
             "| 臂 | n | 平均上下文块数 | faithfulness | relevancy | correctness |",
             "|---|---|---|---|---|---|"]
    for arm, cur in loaded.items():
        common = sorted(set(ref) & set(cur))
        agg[arm] = {m: round(sum(cur[q][m] for q in common) / len(common), 4)
                    for m in METRICS}
        agg[arm]["n"] = len(common)
        ctx_note[arm] = round(sum(cur[q]["n_ctx"] for q in common) / len(common), 1)
        lines.append(f"| {arm} | {agg[arm]['n']} | {ctx_note[arm]} | "
                     f"{agg[arm]['faithfulness']} | {agg[arm]['relevancy']} | "
                     f"{agg[arm]['correctness']} |")

    lines += ["", "## 配对差（臂 − 参照）", "",
              "| 臂 | metric | mean_diff | ci95 | 显著 |", "|---|---|---|---|---|"]
    for arm in arms:
        if arm == args.ref or arm not in loaded:
            continue
        cur = loaded[arm]
        common = sorted(set(ref) & set(cur))
        for m in METRICS:
            da = [dict(q=q, m=cur[q][m]) for q in common]
            db = [dict(q=q, m=ref[q][m]) for q in common]
            ci = paired_ci(da, db, "m")
            sig = "是" if ci["ci95"][0] * ci["ci95"][1] > 0 else "否"
            lines.append(f"| {arm} | {m} | {ci['mean_diff']} | {ci['ci95']} | {sig} |")

    text = "\n".join(lines)
    print(text)
    out = os.path.join(args.results_dir, "paired_analysis.md")
    open(out, "w", encoding="utf-8").write(text + "\n")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
