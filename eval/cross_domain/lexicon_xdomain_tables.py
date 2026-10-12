#!/usr/bin/env python3
"""跨域实验汇总（2026-10-11 论文实验扩展：三域 × 五臂 + 消融 → 论文表格）。

消费 20261011_cross_domain/{cs,bio,mat}/tables.json + ablation/ablation.json，
输出：
  results/xdomain_summary.md   —— 主表（结构指标 + P/R/F1 + 判读有效率）
  results/xdomain_ablation.md  —— 消融表
  figs 由 make_figs 扩展消费同目录 JSON。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402

ROOT = os.path.join(REPO, "eval", "baselines", "results", "20261011_cross_domain")
DOMS = {"cs": "Computer science", "bio": "Biomedicine", "mat": "Materials"}
ARMS = ["schema", "freq", "cvalue", "llmdirect", "threesource"]
ARM_LABEL = {"schema": "batch schema (pre)", "freq": "frequency top-K",
             "cvalue": "C-value", "llmdirect": "LLM direct",
             "threesource": "three-source (ours)"}


def main() -> None:
    lines = ["# 跨域泛化主表（10×690 冻结重放样本，mean；sd 见 scan_agg.json）", "",
             "| Domain | Arm | Terms | Entities | C10 | Conc/doc | "
             "P_silver | R_silver | F1 | judged-valid | A-B agree |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for dom in DOMS:
        t = os.path.join(ROOT, dom, "tables.json")
        if not os.path.exists(t):
            continue
        d = json.load(open(t, encoding="utf-8"))
        comp_sizes = {}
        comps = os.path.join(ROOT, dom, "components")
        for arm in ARMS:
            p = os.path.join(comps, f"arm_{arm}.json")
            if os.path.exists(p):
                comp_sizes[arm] = len(json.load(open(p, encoding="utf-8")))
        for arm in ARMS:
            sc = d.get("scan", {}).get(arm, {})
            prf = d.get("prf", {}).get(arm, {})
            vd = d.get("verdict", {}).get(arm, {})
            lines.append(
                f"| {DOMS[dom]} | {ARM_LABEL[arm]} | {comp_sizes.get(arm, '-')} | "
                f"{sc.get('entities', {}).get('mean', '-')} | "
                f"{sc.get('c10', {}).get('mean', '-')} | "
                f"{sc.get('conc_per_doc', {}).get('mean', '-')} | "
                f"{prf.get('P', '-')} | {prf.get('R', '-')} | {prf.get('F1', '-')} | "
                f"{vd.get('rate', '-')} | {vd.get('AB_agree', '-')} |")
    md = "\n".join(lines)
    print(md)
    open(os.path.join(ROOT, "xdomain_summary.md"), "w",
         encoding="utf-8").write(md + "\n")

    ab = ["# 三源消融（cs 域，同冻结样本；terms 列=规模混杂显式）", "",
          "| Arm | Terms | Entities | C10 | Conc/doc | Gen-head |", "|---|---|---|---|---|---|"]
    p = os.path.join(ROOT, "cs", "ablation", "ablation.json")
    if os.path.exists(p):
        for arm, g in json.load(open(p, encoding="utf-8")).items():
            ab.append(f"| {arm} | {g['terms']} | {g['entities']} | {g['c10']} | "
                      f"{g['conc_per_doc']} | {g['generic_head_share']} |")
    abmd = "\n".join(ab)
    print(abmd)
    open(os.path.join(ROOT, "xdomain_ablation.md"), "w",
         encoding="utf-8").write(abmd + "\n")


if __name__ == "__main__":
    main()
