#!/usr/bin/env python3
"""组合监控规则评估（2026-10-12 审稿 P0-5 落地）。

C10 单指标已知 FP/FN 类（审稿问题 4）。本脚本在既有冻结退化存档上
评估四腿组合规则（任一腿触发=人工排查）：
  A 集中度：C10 > 0.35（库尺寸 <100 实体豁免）
  B 计数：links < 50% 健康基线
  C 泛词头：generic_head_share > 0.10（v2.0 塌缩 0.146 vs 修复 0.058 之间）
  D 基数：entities < 50% 健康基线
评估对象：替换轴 10-100%、收缩轴 10-90%（20261011_lexicon_degrade）
+ v2.0 塌缩臂（20261010 multisample）。无时间轴——不称检测延迟；
库间 LOO 已另行完成（monitor_loo），此处不重复。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402

DEG = os.path.join(REPO, "eval", "baselines", "results",
                   "20261011_lexicon_degrade", "degrade.json")
MS = os.path.join(REPO, "eval", "baselines", "results",
                  "20261010_lexicon_multisample", "multisample.json")
OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261013_p0_controls")

T_C10, T_LINKS, T_GENH, T_ENT = 0.35, 0.50, 0.10, 0.50


def legs(metrics, base):
    fired = []
    if metrics["c10"]["mean"] > T_C10:
        fired.append("A_conc")
    if metrics["links"]["mean"] < T_LINKS * base["links"]["mean"]:
        fired.append("B_count")
    if metrics["generic_head_share"]["mean"] > T_GENH:
        fired.append("C_genhead")
    if metrics["entities"]["mean"] < T_ENT * base["entities"]["mean"]:
        fired.append("D_card")
    return fired


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    deg = json.load(open(DEG))
    ms = json.load(open(MS))
    arms = dict(deg["arms"])
    # multisample 的 v2.0 塌缩臂（同冻结帧）并入评估
    for k in ("v2.0-collapse",):
        arms[k] = ms["aggregate"][k]
    base = arms["baseline-v2.1"]
    truth = {a: ("collapse" if ("subst-100" in a or a == "v2.0-collapse"
                                or a.startswith("shrink"))
                 else "healthy-ish") for a in arms}
    # 收缩轴不是"塌缩"，但 50%+ 删除应被计数腿抓住——按"应警"处理
    truth = {}
    for a in arms:
        if a.startswith("subst-") and int(a.split("-")[1].replace("pct", "")) >= 75:
            truth[a] = "should-alarm"
        elif a == "v2.0-collapse":
            truth[a] = "should-alarm"
        elif a.startswith("shrink-"):
            pct = int(a.split("-")[1].replace("pct", ""))
            truth[a] = "should-alarm" if pct >= 50 else "no-alarm-expected"
        else:
            truth[a] = "no-alarm-expected"

    rows = []
    for a, m in sorted(arms.items()):
        fired = legs(m, base)
        rows.append({"arm": a, "truth": truth[a], "fired": fired,
                     "c10": m["c10"]["mean"],
                     "links": m["links"]["mean"],
                     "genhead": m["generic_head_share"]["mean"],
                     "entities": m["entities"]["mean"]})
        print(f"{a:16s} truth={truth[a]:18s} fired={fired or '—'} "
              f"C10={m['c10']['mean']:.3f} links={m['links']['mean']:.0f} "
              f"genh={m['generic_head_share']['mean']:.3f} "
              f"ents={m['entities']['mean']:.0f}")
    miss = [r for r in rows if r["truth"] == "should-alarm"
            and not r["fired"]]
    false = [r for r in rows if r["truth"] == "no-alarm-expected"
             and r["fired"]]
    summary = {"legs": {"A_conc": f"C10>{T_C10}",
                        "B_count": f"links<{int(T_LINKS*100)}%baseline",
                        "C_genhead": f"genhead>{T_GENH}",
                        "D_card": f"entities<{int(T_ENT*100)}%baseline"},
               "missed": [r["arm"] for r in miss],
               "false_alarms": [r["arm"] for r in false],
               "rows": rows}
    json.dump(summary, open(os.path.join(OUT, "composite_monitor.json"),
                            "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print("missed:", [r["arm"] for r in miss],
          "false:", [r["arm"] for r in false])
    print("[saved]", os.path.join(OUT, "composite_monitor.json"))


if __name__ == "__main__":
    main()
