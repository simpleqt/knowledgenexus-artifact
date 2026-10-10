#!/usr/bin/env python3
"""C 线大修补图生成（2026-10-10）：全部出自真实生产数据。

fig_conc_scatter.png  — 13 库 (实体数, top-10 浓度) 散点 + 阈值线
fig_rebuild_telemetry.png — 10-08/09 链接层重建小时直方图（KB1/KB2 分色，
                          B93/B95 记录窗口阴影，时间戳勘误注记在 caption）
fig_audit_ci.png      — 链接审计三组 Wilson CI + 交叉裁判注记
fig_top_after.png     — KB1 修复后 top-30 实体横条（类别着色）
fig_top_before.png    — KB1 塌缩期 top-30（10-06 冷备，需临时库数据文件）
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- Fig 4: 13-KB scatter (entities, docs, concentration, status) ----
KBS = [  # hex8, entities, docs, conc, status
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
STYLE = {"repaired": ("#c0392b", "o", "repaired (this incident)"),
         "control": ("#27ae60", "*", "hand-curated control"),
         "other": ("#7f8c8d", "s", "other production KBs")}


def fig_scatter():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    seen = set()
    for hex8, ents, docs, conc, st in KBS:
        c, m, lab = STYLE[st]
        ax.scatter(ents, conc, c=c, marker=m, s=90 if m == "*" else 46,
                   label=lab if st not in seen else None, zorder=3,
                   edgecolors="white", linewidths=0.6)
        seen.add(st)
    OFFS = {"104112E6": (5, -11, "left"), "693372D5": (5, 6, "left"),
            "E2ED00DD": (-5, 5, "right"), "F54CE546": (5, 5, "left"),
            "D2B4AD67": (5, -11, "left")}
    for hex8, ents, docs, conc, st in KBS:
        if hex8 in OFFS:
            dx, dy, ha = OFFS[hex8]
            ax.annotate(hex8, (ents, conc), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8, color="#444",
                        ha=ha, annotation_clip=True)
    ax.axhline(0.35, ls="--", lw=1, color="#e67e22")
    ax.axhline(0.50, ls="--", lw=1, color="#c0392b")
    ax.text(6, 0.362, "investigate > 0.35", fontsize=8, color="#e67e22",
            ha="left", va="bottom")
    ax.text(6, 0.512, "escalate > 0.50", fontsize=8, color="#c0392b",
            ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("link-bearing canonical entities (log scale)")
    ax.set_ylabel("top-10 link concentration")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_conc_scatter.png"), dpi=300)
    plt.close(fig)


# ---- Fig 5: rebuild telemetry ----
HOURS = ["10-08 00", "10-08 01", "10-08 02", "10-08 03", "10-08 10", "10-08 11",
         "10-08 12", "10-08 13", "10-08 14", "10-08 15", "10-08 16", "10-08 17",
         "10-08 18", "10-08 19", "10-09 06", "10-09 10", "10-09 11"]
KB1 = [249337, 249379, 243870, 61249, 9231333, 3876799, 2353548, 1383603,
       1377972, 1363264, 994953, 1024594, 991747, 469428, 1292, 0, 0]
KB2 = [0, 0, 0, 0, 5009755, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 916413]
KB1 = [x + 1 for x in KB1]
KB2 = [x + 1 for x in KB2]


def fig_telemetry():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    xs = range(len(HOURS))
    w = 0.42
    ax.bar([i - w / 2 for i in xs], KB1, width=w, color="#c0392b",
           label="KB1 (nuclear electronics, 61,137 docs)")
    ax.bar([i + w / 2 for i in xs], KB2, width=w, color="#2980b9",
           label="KB2 (physics, 13,408 docs)")
    ax.set_yscale("log")
    ax.set_ylim(1, 3e7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([h.replace("10-0", "") for h in HOURS], rotation=45,
                       fontsize=7)
    ax.set_xlabel("hour of 2026-10-08 / 10-09 (link created_at)")
    ax.set_ylabel("link rows inserted per hour (log)")
    # 记录中的 B93/B95 窗口 11:41–18:35
    i0, i1 = HOURS.index("10-08 11") + 0.6, HOURS.index("10-08 18") + 1.5
    ax.axvspan(i0, i1, color="#f39c12", alpha=0.18, zorder=0)
    ax.text((i0 + i1) / 2, 2e6, "logged B95 window\n(11:41–18:35)",
            ha="center", fontsize=7.5, color="#b9770e")
    # 图例悬空于坐标区上方（视觉验收：upper right 框压住窗口标注首行）
    ax.legend(fontsize=8, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 1.02))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rebuild_telemetry.png"), dpi=300)
    plt.close(fig)


# ---- audit CI fig（2026-10-10 三轮审查后：300 分层为主，旧 100 均匀 + 交叉裁判为辅）----
def fig_audit():
    groups = ["overall (n=300)", "top-10 entities (46/50)",
              "medium rank (85/100)", "tail rank (91/100)",
              "short symbols (29/50)", "domain terms (142/149)",
              "generic words (109/151)", "cross-judge parseable (53/55)"]
    vals = [0.837, 0.920, 0.850, 0.910, 0.580, 0.953, 0.722, 0.964]
    lo = [0.791, 0.812, 0.767, 0.838, 0.442, 0.906, 0.646, 0.887]
    hi = [0.874, 0.968, 0.907, 0.952, 0.706, 0.977, 0.787, 0.994]
    colors = ["#2c3e50", "#2980b9", "#2980b9", "#2980b9", "#c0392b",
              "#27ae60", "#e67e22", "#8e44ad"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    y = range(len(groups))[::-1]
    for yi, v, l, h, c in zip(y, vals, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(v, yi, "o", color=c, ms=7)
        ax.text(h + 0.012, yi, f"{v:.3f} [{l:.2f},{h:.2f}]", va="center",
                fontsize=7.5, color="#333")
    ax.axvline(1.0, ls=":", lw=0.8, color="#999")
    ax.set_yticks(list(y))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlim(0.38, 1.35)
    ax.set_xlabel("link precision (Wilson 95% CI)")
    ax.set_title("Primary judge (Qwen3.6-35B): 300-link stratified audit\n"
                 "cross-judge bar = 55-link parseable subset (Qwen3.8-Flash)",
                 fontsize=7.5, color="#555")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_audit_ci.png"), dpi=300)
    plt.close(fig)


# ---- top-entity bars (after / before) ----
GEN = {"be", "alpha", "gamma", "beta", "method", "parameter", "term",
       "analysi", "theory", "temperature", "region", "mode", "range",
       "density", "error", "function", "event", "dataset", "new",
       "background", "paper", "mean", "simulation", "constant", "control",
       "material", "author", "effect", "field", "phase", "time", "model",
       "tau", "energy", "particle", "decay", "method", "measure"}
SHORT = {"be", "alpha", "gamma", "beta", "tau"}


def cls(name):
    n = name.lower()
    if n in SHORT:
        return "short ambiguous symbol"
    if n in GEN:
        return "generic academic word"
    return "domain term"


def fig_top(names_counts, title, fname, total):
    names = [n for n, _ in names_counts][::-1]
    vals = [c for _, c in names_counts][::-1]
    cols = []
    for n in names:
        c = cls(n)
        cols.append("#e67e22" if "generic" in c else
                    ("#c0392b" if "symbol" in c else "#27ae60"))
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.barh(range(len(names)), [v / total for v in vals], color=cols)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8, family="monospace")
    for i, v in enumerate(vals):
        ax.text(v / total + 0.002, i, f"{v/1e6:.2f}M" if v > 1e6 else f"{v/1e3:.0f}k",
                va="center", fontsize=6.5, color="#555")
    ax.set_xlabel("share of all entity links")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, max(vals) / total * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)


def _top_panel(ax, names_counts, total, title):
    names = [n for n, _ in names_counts][::-1]
    vals = [c for _, c in names_counts][::-1]
    cols = ["#e67e22" if "generic" in cls(n) else
            ("#c0392b" if "symbol" in cls(n) else "#27ae60") for n in names]
    ax.barh(range(len(names)), [v / total for v in vals], color=cols,
            height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7, family="monospace")
    for i, v in enumerate(vals):
        ax.text(v / total + 0.002, i,
                f"{v/1e6:.2f}M" if v > 1e6 else f"{v/1e3:.0f}k",
                va="center", fontsize=5.5, color="#555")
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(0, max(vals) / total * 1.18)
    ax.spines[["top", "right"]].set_visible(False)


def fig_top_pair(before, after, total_b, total_a, fname):
    """Before | After 并排（三轮审查 #11：横排更利比较阅读）。"""
    from matplotlib.patches import Patch
    fig, (axb, axa) = plt.subplots(1, 2, figsize=(10.6, 5.2))
    _top_panel(axb, before, total_b,
               f"Collapsed (2026-10-06 backup) — 4.81M links, $C_{{10}}$=0.419")
    _top_panel(axa, after, total_a,
               f"After repair — 25.10M links, $C_{{10}}$=0.345")
    axb.set_xlabel("share of all entity links", fontsize=8)
    axa.set_xlabel("share of all entity links", fontsize=8)
    handles = [Patch(color="#27ae60", label="domain term"),
               Patch(color="#e67e22", label="generic word"),
               Patch(color="#c0392b", label="short ambiguous symbol")]
    axa.legend(handles=handles, loc="lower right", fontsize=7,
               frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_scatter()
    fig_telemetry()
    fig_audit()
    after = [("be", 2264858), ("alpha", 1246756), ("gamma", 843979),
             ("beta", 828847), ("method", 727377), ("parameter", 709237),
             ("term", 669084), ("analysi", 467779), ("theory", 459229),
             ("temperature", 431613), ("region", 392961), ("particle", 369976),
             ("mode", 354183), ("range", 349490), ("density", 279773),
             ("decay", 276868), ("error", 274824), ("function", 272263),
             ("event", 260028), ("dataset", 258276), ("new", 255610),
             ("background", 247585), ("paper", 244466), ("mean", 244164),
             ("simulation", 243229), ("constant", 233017),
             ("control", 229190), ("material", 216285), ("author", 199820),
             ("effect", 197977)]
    fig_top(after, "KB1 after repair (top-30 by link count; 25.10M links, C10=0.345)",
            "fig_top_after.png", 25097704)
    # before panel: from cold snapshot file if present
    bf = os.path.join(OUT, "before_top30.json")
    if os.path.exists(bf):
        data = json.load(open(bf))
        before = [(d[0], d[1]) for d in data]
        fig_top(before,
                "KB1 collapsed state, 2026-10-06 cold backup (top-30; 4.81M links, C10=0.419)",
                "fig_top_before.png", 4808270)
        fig_top_pair(before, after, 4808270, 25097704,
                     "fig_top_pair.png")
    print("figs done")
