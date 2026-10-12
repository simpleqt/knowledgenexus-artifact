#!/usr/bin/env python3
"""召回损失定位漏斗（2026-10-12 七项方案 §6 落地）。

R_link = P(C)·P(K|C)·P(D|K,C)·P(L|D,K,C) 的 occurrence 级分解，
参照=silver A（仪器局限照旧披露；本工具定位损失来源，不产出人工召回）：
  C  候选可表达 —— 该 silver 面形（norm 对称）出现在宽候选池
     （全量 n-gram 词频表 ∪ LLM 四源候选清单，无 freq≥5/top-4000 预过滤）
  K  词典保留   —— 面形在该臂词典键集（norm 对称）
  D  扫描识别   —— 该臂词典实际扫描在 occurrence 处命中（词边界/复数/
                   CJK 真扫描行为，非字符串包含）
  L  链接归一   —— 命中的归一面形 == silver 归一面形
1−C=候选缺口（选择目标救不了）；C→K=选择/闸门损失；K→D=形态/边界损失；
D→L=归一/链接损失。测试文档不参与任何建池。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/lexicon_recall_funnel.py'
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import precompile, scan  # noqa: E402
from lexicon_cross_domain import (_norm_term, text_freq,  # noqa: E402
                                  SCHEMA)

ROOT = os.path.join(REPO, "eval", "baselines", "results",
                    "20261011_cross_domain")
OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261012_revision")
ARMS = ("threesource", "llmdirect", "freq", "cvalue")


def variants(x: str) -> set[str]:
    """norm 对称的小形态变体集（双侧同规则，扫描层差异由 D 层真扫描捕）。"""
    n = _norm_term(x)
    out = {n}
    if len(n) > 3 and n.endswith("s") and not n.endswith("ss"):
        out.add(n[:-1])
    else:
        out.add(n + "s")
    return out


def main() -> None:
    import collections
    report = {"reference": "silver_A occurrence-level; instrument limits "
                          "disclosed in paper; funnel locates loss source",
              "domains": {}}
    for dom in ("cs", "bio", "mat"):
        out = os.path.join(ROOT, dom)
        comps = os.path.join(out, "components")
        test = json.load(open(os.path.join(out, "test.json"),
                              encoding="utf-8"))
        silver_a = {r["doc_id"]: r.get("terms") or []
                    for r in (json.loads(l) for l in
                              open(os.path.join(out, "silver_A.jsonl"),
                                   encoding="utf-8"))}
        construct = json.load(open(os.path.join(out, "construct.json"),
                                   encoding="utf-8"))
        freq = text_freq(construct)
        pool = set()
        for t in freq:
            pool.add(_norm_term(t))
        llm_lists = []
        for f in ("classify", "seed", "expand", "llmdirect"):
            p = os.path.join(comps, f + ".json")
            if os.path.exists(p):
                d = json.load(open(p, encoding="utf-8"))
                for terms in d.values():
                    llm_lists.extend(terms)
        n_gram_pool = len(pool)
        for t in llm_lists:
            pool |= variants(t)
        pool_llm = {_norm_term(t) for t in llm_lists}

        arms = {}
        for arm in ARMS:
            lex = json.load(open(os.path.join(comps, f"arm_{arm}.json"),
                                 encoding="utf-8"))
            keys = set()
            for t in lex:
                keys |= variants(t)
            pats = precompile(lex)
            hit_by_pos = collections.defaultdict(list)
            for d in test:
                for surface, _et, s, e in scan(d["text"], pats):
                    hit_by_pos[d["doc_id"]].append((s, e,
                                                    _norm_term(surface)))
            # 逐 occurrence 分桶
            n_total = n_C = n_K = n_D = n_L = 0
            miss_C = collections.Counter()
            miss_K = collections.Counter()
            miss_D = collections.Counter()
            miss_L = collections.Counter()
            for d in test:
                text = d["text"]
                for t in silver_a.get(d["doc_id"], []):
                    surf = t["surface"]
                    low = text.lower()
                    start = 0
                    while (idx := low.find(surf.lower(), start)) != -1:
                        start = idx + 1
                        n_total += 1
                        nv = variants(surf)
                        if nv & pool:
                            n_C += 1
                        else:
                            miss_C[_norm_term(surf)] += 1
                            continue
                        if nv & keys:
                            n_K += 1
                        else:
                            miss_K[_norm_term(surf)] += 1
                            continue
                        hit = next((h for h in hit_by_pos[d["doc_id"]]
                                    if h[0] <= idx and idx + len(surf)
                                    <= h[1]), None)
                        if hit is not None:
                            n_D += 1
                        else:
                            miss_D[_norm_term(surf)] += 1
                            continue
                        if hit[2] in nv:
                            n_L += 1
                        else:
                            miss_L[_norm_term(surf)] += 1
            arms[arm] = {
                "R_final": round(n_L / n_total, 4) if n_total else None,
                "P_C": round(n_C / n_total, 4),
                "P_K_given_C": round(n_K / n_C, 4) if n_C else None,
                "P_D_given_KC": round(n_D / n_K, 4) if n_K else None,
                "P_L_given_DKC": round(n_L / n_D, 4) if n_D else None,
                "loss_counts": {"candidate": n_total - n_C,
                                "selection": n_C - n_K,
                                "scan": n_K - n_D,
                                "link": n_D - n_L},
                "top_miss_candidate": miss_C.most_common(12),
                "top_miss_selection": miss_K.most_common(12),
                "top_miss_scan": miss_D.most_common(8),
            }
        report["domains"][dom] = {
            "pool_ngram_norms": n_gram_pool, "pool_llm_norms": len(pool_llm),
            "pool_union_norms": len(pool), "silver_occurrences": n_total,
            "arms": arms}
        print(f"[{dom}] occ={n_total} pool={len(pool)} "
              f"(ngram {n_gram_pool} + llm {len(pool_llm)})")
        for arm, v in arms.items():
            print(f"  {arm:12s} R={v['R_final']} P(C)={v['P_C']} "
                  f"P(K|C)={v['P_K_given_C']} P(D|KC)={v['P_D_given_KC']} "
                  f"P(L|DKC)={v['P_L_given_DKC']} losses={v['loss_counts']}")
    json.dump(report, open(os.path.join(OUT, "recall_funnel.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[saved]", os.path.join(OUT, "recall_funnel.json"))


if __name__ == "__main__":
    main()
