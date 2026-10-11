#!/usr/bin/env python3
"""分级词典退化 → 免金标监控告警验证（2026-10-11 复审 P3-R3 关键验证）。

问题：候选阈值（C10>0.35 调查 / >0.5 升级，13 库 post-hoc 读出）在受控
退化下何时告警？与基本计数监控（实体数/链接数跌幅）相比各自能看什么、
不能看什么？

设计（全部在 20261010_lexicon_multisample 的 10 个冻结样本上，无 LLM、
无金标，逐臂确定性扫描）：
  替换轴（mimic 事故）：删 m% 的 v2.1 词条 ∪ 全部 v2.0 类别词
    （m=10/25/50/75/100；m=100 = v2.0-only 塌缩态）
  收缩轴（词典衰减）：v2.1 纯随机删 k% 词条（k=10/25/50/75/90）
对照基线 = 未退化 v2.1（即 multisample 的 v2.1-repair 臂，重算复核）。
告警规则：C10 带（>0.35 / >0.5）；计数规则（实体或链接较基线跌 >50%）。

预测（写在前，结果见 summary）：C10 对替换轴敏感、对纯收缩不敏感
（随机删词不集中链接）；计数监控对收缩敏感、对替换轴钝感（m=100 时
链接总量仍高于收缩 90% 的情形）。→ 监控须成套部署。

用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/lexicon_degrade_monitor.py \
  --multisample eval/baselines/results/20261010_lexicon_multisample/multisample.json \
  --out eval/baselines/results/20261011_lexicon_degrade'
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import load_lexicon  # noqa: E402
from lexicon_multi_sample import one_task  # noqa: E402  (arm,si,lex,cs,sb,chunks)

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402  (OS client)


def fetch_chunks_by_ids(ids: list[str]) -> list[tuple[str, str]]:
    """mget 冻结样本块（chunk_id→(doc_id, text)），批 1000。"""
    c = M.search.client()
    out = []
    for i in range(0, len(ids), 1000):
        batch = ids[i:i + 1000]
        r = c.mget(index="knc-chunk", body={"ids": batch},
                   _source=["chunk_id", "doc_id", "content_text"])
        for d in r["docs"]:
            if d.get("found") and d["_source"].get("content_text"):
                out.append((d["_source"]["doc_id"], d["_source"]["content_text"]))
    return out


def build_arms(v21: dict, v20: dict) -> dict[str, dict]:
    """两条退化轴。

    注：v2.0 类别词 177/182 本就保留在 v2.1 里（泛词尾巴现象），所以
    「往 v2.1 里注入类别词」是无操作——事故机制是**替换**：类别词一直
    在，坏掉的是「其余词条都是类别词」。故：
    替换轴（mimic 事故）：删 m% 的 v2.1 词条（随机、定种子）∪ 全部
      v2.0 类别词（m=100 即 v2.0-only 塌缩臂）；
    收缩轴（词典衰减）：纯随机删 k% v2.1 词条（不注入任何词）。
    """
    arms: dict[str, dict] = {"baseline-v2.1": dict(v21)}
    terms21 = sorted(v21)
    for m in (10, 25, 50, 75, 100):
        rng = random.Random(f"20261011-subst-{m}")
        keep = set(rng.sample(terms21, round(len(v21) * (100 - m) / 100)))
        lex = {t: ty for t, ty in v21.items() if t in keep}
        for t, ty in v20.items():
            lex.setdefault(t, ty)
        arms[f"subst-{m}pct"] = lex
        print(f"[subst-{m}pct] keep {len(keep)} v2.1 terms ∪ "
              f"{len(v20)} v2.0 terms → {len(lex)} scan terms")
    for k in (10, 25, 50, 75, 90):
        rng = random.Random(f"20261011-shrink-{k}")
        drop = set(rng.sample(sorted(v21), round(len(v21) * k / 100)))
        arms[f"shrink-{k}pct"] = {t: ty for t, ty in v21.items() if t not in drop}
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="104112e63d9d4aa5a6c21e1c1549ef0f")
    ap.add_argument("--multisample", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ms = json.load(open(args.multisample, encoding="utf-8"))
    n_samples = ms["samples"]
    samples: list[list[tuple[str, str]]] = []
    for i in range(n_samples):
        ids = ms["sample_chunk_ids"][str(i)]
        chunks = fetch_chunks_by_ids(ids)
        assert len(chunks) >= len(ids) * 0.98, \
            f"sample {i}: fetched {len(chunks)}/{len(ids)}"
        samples.append(chunks)
        print(f"[sample {i}] {len(chunks)} chunks", flush=True)

    v21 = load_lexicon(os.path.join(REPO, "schemas", "nuclear_electronics.yaml"))
    v20 = load_lexicon("5ec5756:schemas/nuclear_electronics.yaml")
    arms = build_arms(v21, v20)

    tasks = [(name, si, lex, False, False, samples[si])
             for si in range(n_samples) for name, lex in arms.items()]
    results: dict[str, dict[int, dict]] = {a: {} for a in arms}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        done = 0
        for arm, si, m_ in pool.map(one_task, tasks):
            results[arm][si] = m_
            done += 1
            if done % 20 == 0:
                print(f"[progress] {done}/{len(tasks)}", flush=True)

    keys = ("links", "entities", "c10", "conc_per_doc",
            "generic_head_share", "docs")
    agg = {}
    for a in arms:
        rows = [results[a][s] for s in sorted(results[a])]
        agg[a] = {k: {"mean": round(statistics.mean(r[k] for r in rows), 3),
                      "sd": round(statistics.stdev(r[k] for r in rows), 3)}
                  for k in keys}

    base = agg["baseline-v2.1"]

    def c10_excl(a: str, lo: float) -> bool:
        return agg[a]["c10"]["mean"] > lo

    lines = ["# 分级退化 → 监控告警（10 冻结样本 mean±sd，候选带 C10>0.35/0.5，"
             "计数规则=实体或链接较基线跌>50%）", ""]
    lines.append(f"baseline v2.1: links {base['links']['mean']}±{base['links']['sd']}, "
                 f"ents {base['entities']['mean']}±{base['entities']['sd']}, "
                 f"C10 {base['c10']['mean']}±{base['c10']['sd']}")
    lines.append("")
    lines.append("arm | links | ents | C10 | C10>0.35 | C10>0.5 | ents跌>50% | links跌>50%")
    for a in arms:
        g = agg[a]
        ent_drop = 1 - g["entities"]["mean"] / base["entities"]["mean"]
        link_drop = 1 - g["links"]["mean"] / base["links"]["mean"]
        lines.append(
            f"{a} | {g['links']['mean']}±{g['links']['sd']} | "
            f"{g['entities']['mean']}±{g['entities']['sd']} | "
            f"{g['c10']['mean']}±{g['c10']['sd']} | "
            f"{'FIRE' if c10_excl(a, 0.35) else '-'} | "
            f"{'FIRE' if c10_excl(a, 0.5) else '-'} | "
            f"{'FIRE' if ent_drop > 0.5 else f'{ent_drop:+.0%}'} | "
            f"{'FIRE' if link_drop > 0.5 else f'{link_drop:+.0%}'}")
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(args.out, "summary_degrade.txt"), "w") as f:
        f.write(summary + "\n")
    json.dump({"arms": {a: {k: v for k, v in agg[a].items()} for a in arms},
               "per_run": {a: {str(s): results[a][s]
                               for s in sorted(results[a])} for a in arms},
               "alarm_rule": "C10>0.35 investigate, >0.5 escalate; "
                             "count: ents or links drop >50% vs baseline"},
              open(os.path.join(args.out, "degrade.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[saved]", os.path.join(args.out, "degrade.json"))


if __name__ == "__main__":
    main()
