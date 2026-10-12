#!/usr/bin/env python3
"""监控候选阈值留库（LOO）验证（2026-10-12 七项方案 §9 落地）。

13 库 (实体数, C10, 状态)：留一库，用其余 12 库冻结双带规则
（investigate = 与剩余库健康分布的分位；escalate = 更高分位），
预测留出库，对照部署状态标签计数 FP/FN。已知边界如实入文：
① 塌缩正例仅 1 个（备份行），LOO 的“漏报”格本质是小样本；
② N≤10 库 C10=1 是数学下界，按尺寸分层规则豁免（<100 实体不判）；
③ 阈值与标签均来自同一部署（事后），本验证只检查规则在库间的
   自洽泛化，不声称跨部署效度。
分位数取法（冻结规则）：investigate = 剩余健康库 C10 的 q90，
escalate = q97；健康分布不含已知塌缩行。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402

# (hex8, entities, docs, c10, status) —— 与 make_figs.py KBS 同源
KBS = [
    ("104112E6", 1871, 61137, 0.345, "repaired"),
    ("693372D5", 2726, 13408, 0.162, "repaired"),
    ("D2B4AD67", 629, 1324, 0.398, "repaired"),
    ("F54CE546", 153, 119, 0.768, "repaired"),
    ("E2ED00DD", 1239, 78, 0.184, "control"),
    ("21BCBED8", 2482, 3479, 0.287, "other"),
    ("6865B3A4", 1217, 354, 0.272, "other"),
    ("9FB0435E", 1024, 723, 0.259, "other"),
    ("AC6F2FDB", 874, 185, 0.286, "other"),
    ("D84F82AB", 424, 27, 0.443, "other"),
    ("37CE33A8", 385, 11, 0.407, "other"),
    ("343D0A4B", 118, 4, 0.395, "other"),
    ("316C267A", 5, 0, 1.000, "other"),
]
# 塌缩备份行（非现役库，仅作唯一已知正例参与）
COLLAPSED = ("BCKP-1006", 503, 61137, 0.419, "collapsed")

OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261012_revision")


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    i = q * (len(sorted_vals) - 1)
    lo = int(i)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = i - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> None:
    rows = KBS + [COLLAPSED]
    results = []
    for held in rows:
        rest = [r for r in rows if r[0] != held[0]]
        healthy = sorted(r[3] for r in rest
                         if r[4] != "collapsed" and r[1] >= 100)
        t_inv = round(quantile(healthy, 0.90), 3)
        t_esc = round(quantile(healthy, 0.97), 3)
        # 预测（尺寸分层：<100 实体豁免集中度判读）
        if held[1] < 100:
            pred = "exempt-small"
        elif held[3] > t_esc:
            pred = "escalate"
        elif held[3] > t_inv:
            pred = "investigate"
        else:
            pred = "quiet"
        # 状态标签：collapsed=应警（investigate 及以上）；其余=不应 escalate
        if held[4] == "collapsed":
            want_alarm = True
            correct = pred in ("investigate", "escalate")
            kind = "TP" if correct else "FN"
        else:
            want_alarm = False
            # escalate 视为误报；investigate 是人工排查触发，不算误报
            correct = pred != "escalate"
            kind = "TN" if correct else "FP"
        results.append({"kb": held[0], "entities": held[1], "docs": held[2],
                        "c10": held[3], "status": held[4],
                        "loo_t_investigate": t_inv, "loo_t_escalate": t_esc,
                        "pred": pred, "kind": kind})

    kinds = [r["kind"] for r in results]
    summary = {
        "n": len(results),
        "TP": kinds.count("TP"), "FN": kinds.count("FN"),
        "FP": kinds.count("FP"), "TN": kinds.count("TN"),
        "fp_list": [r["kb"] for r in results if r["kind"] == "FP"],
        "fn_list": [r["kb"] for r in results if r["kind"] == "FN"],
        "exempt": [r["kb"] for r in results if r["pred"] == "exempt-small"],
        "caveats": ["塌缩正例仅 1（备份行），FN 格为小样本",
                    "N≤10/<100 实体库豁免（C10 数学下界）",
                    "标签与阈值同源自单一部署，只证库间自洽"],
    }
    out = {"loo": results, "summary": summary}
    json.dump(out, open(os.path.join(OUT, "monitor_loo.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in results:
        print(f"  {r['kb']:10s} ents={r['entities']:5d} c10={r['c10']:.3f} "
              f"status={r['status']:9s} t=({r['loo_t_investigate']},"
              f"{r['loo_t_escalate']}) pred={r['pred']:12s} {r['kind']}")
    print("[saved]", os.path.join(OUT, "monitor_loo.json"))


if __name__ == "__main__":
    main()
