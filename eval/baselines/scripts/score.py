#!/usr/bin/env python3
"""统一评分器：消费 runner 预测产物，指标公式与 eval/harness/retrieval_eval.py 完全一致。

  Recall@k = 前 k 个命中（块序，允许重复文档占位）中含 gold 文档的题比例
  MRR@10   = 首个 gold 文档排名倒数均值
  nDCG@10  = 去重后文档位次折减 1/log2(rank+1)，二值相关理想 DCG=1
配对差 bootstrap CI（vs --ref 指定的参照系统）：同 retrieval_eval.paired_bootstrap_ci。
O2 分簇聚类稳健推断（2026-09-25，文献 Evidence-Unit Fairness / arXiv 2608.00183）：
  --cluster set 时对多题集合并的配对差报"簇级 bootstrap CI"（重采样簇=同学科题集，
  保留簇内相关）并与逐题 pooled CI 并列——同簇查询共享语料/生成器，逐题 bootstrap
  低估方差、高估显著性；两者并列即论文的统计口径证据。
  输入兼容 harness 原生存档（per_query 无 hits 但自带 recall@10 等指标，B77-B80）。
用法：python3 score.py runs/*.json --ref runs/knc_nea.json --out runs/summary.md
      python3 score.py eval/results/20260923_newsets_ablation/{nea,ner}_full.json \
        --ref ".../{nea,ner}_base.json" --cluster set \
        --cluster-map '{"nea":"ne","ner":"ne"}'
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random

METRICS = ("recall@5", "recall@10", "recall@20", "cov@10", "cov@20", "mrr@10", "ndcg@10")


def per_query_metrics(hits: list[dict], gold: set[str]) -> dict:
    docs = [h["doc_id"] for h in hits]
    rank = next((i + 1 for i, d in enumerate(docs[:10]) if d in gold), None)
    r5 = int(any(d in gold for d in docs[:5]))
    r10 = int(any(d in gold for d in docs[:10]))
    r20 = int(any(d in gold for d in docs[:20]))
    seen: set[str] = set()
    udocs = [d for d in docs if not (d in seen or seen.add(d))]
    dcg = sum(1.0 / math.log2(r + 2) for r, d in enumerate(udocs[:10]) if d in gold)
    # cov@k = 真集合召回 |gold∩topk 去重文档|/|gold|（nea 多 gold 关联题的本义指标；
    # recall@k 沿用 harness 的 any-hit 口径，两者并列报告）
    cov10 = round(len(set(udocs[:10]) & gold) / len(gold), 4) if gold else 0.0
    cov20 = round(len(set(udocs[:20]) & gold) / len(gold), 4) if gold else 0.0
    return {"recall@5": r5, "recall@10": r10, "recall@20": r20,
            "cov@10": cov10, "cov@20": cov20,
            "mrr@10": round(1.0 / rank, 4) if rank else 0.0,
            "ndcg@10": round(dcg, 4), "first_gold_rank": rank}


def aggregate(pred: dict) -> dict:
    pq, lat = [], []
    for row in pred["per_query"]:
        if "hits" in row:  # runner 原生产物（块级命中）
            m = per_query_metrics(row["hits"], set(row["gold_doc_ids"]))
        else:  # harness 原生存档（B77-B80）：per_query 已带指标
            m = {k: row[k] for k in row if k in METRICS or k == "first_gold_rank"}
        m["query"] = row["query"]
        pq.append(m)
        if row.get("latency_s") is not None:
            lat.append(row["latency_s"])
    n = len(pq)
    out = {"system": pred.get("system", "?"), "set": pred.get("set", "?"),
           "slice": pred.get("slice", "?"), "n": n,
           "corpus_docs": pred.get("corpus_docs"), "index_cost": pred.get("index_cost")}
    for k in METRICS:
        if all(k in q for q in pq):
            out[k] = round(sum(q[k] for q in pq) / n, 4)
    if lat:
        out["latency_p50_s"] = round(sorted(lat)[n // 2], 3)
        out["latency_p95_s"] = round(sorted(lat)[min(n - 1, int(n * 0.95))], 3)
    out["_pq"] = pq
    return out


def paired_ci(a: list[dict], b: list[dict], metric: str, iters: int = 10000,
              seed: int = 20260912) -> dict:
    rng = random.Random(seed)
    da = [q[metric] for q in a]
    db = [q[metric] for q in b]
    n = len(da)
    diffs = [x - y for x, y in zip(da, db)]
    means = []
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {"metric": metric, "n": n, "mean_diff": round(sum(diffs) / n, 4),
            "ci95": [round(means[int(iters * 0.025)], 4), round(means[int(iters * 0.975)], 4)]}


def _boot_ci(diffs: list[float], iters: int, rng: random.Random) -> list[float]:
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return [round(means[int(iters * 0.025)], 4), round(means[int(iters * 0.975)], 4)]


def cluster_paired_ci(pairs: list[tuple[str, str, list[dict], list[dict]]], metric: str,
                      cluster_of: dict[str, str], iters: int = 10000,
                      seed: int = 20260912) -> dict:
    """O2 分簇配对差：pairs = [(set, treat_pq, ref_pq 逐题对齐, …)]。

    返回逐题 pooled CI（旧口径，供对照）与簇级推断（新口径：重采样簇，
    簇内结构整体进出）。簇由 cluster_of[set] 给出（缺省 set 本身）。
    统计纪律：k<5 簇时 bootstrap 退化为 k^k 点离散分布（k=2 即 [簇1均值,簇2均值]，
    会把不显著效应判显著）——改报簇均值 t 区间（Cameron-Miller 小样本聚类稳健），
    并要求"各簇同号 ∧ t 区间不含 0"双条件才算簇级显著；k≥5 用 bootstrap。"""
    rng = random.Random(seed)
    by_cluster: dict[str, list[float]] = {}
    pooled: list[float] = []
    per_set = []
    for set_name, tpq, rpq in pairs:
        ds = [t[metric] - r[metric] for t, r in zip(tpq, rpq)
              if metric in t and metric in r]
        if not ds:
            continue
        c = cluster_of.get(set_name, set_name)
        by_cluster.setdefault(c, []).extend(ds)
        pooled.extend(ds)
        per_set.append({"set": set_name, "cluster": c, "n": len(ds),
                        "mean_diff": round(sum(ds) / len(ds), 4)})
    if not pooled:
        return {}
    naive = _boot_ci(pooled, iters, rng)
    names = sorted(by_cluster)
    sizes = {c: len(v) for c, v in by_cluster.items()}
    k = len(names)
    cmeans_raw = [sum(by_cluster[c]) / sizes[c] for c in names]
    point = sum(pooled) / len(pooled)
    if k >= 5:
        rng2 = random.Random(seed + 1)
        cmeans = []
        for _ in range(iters):  # 抽簇有放回 k 次，规模加权均值（等价于簇整体复制进样本）
            pick = [names[rng2.randrange(k)] for _ in range(k)]
            cmeans.append(sum(sum(by_cluster[c]) for c in pick)
                          / sum(sizes[c] for c in pick))
        cmeans.sort()
        clus = [round(cmeans[int(iters * 0.025)], 4), round(cmeans[int(iters * 0.975)], 4)]
        method = "cluster-bootstrap"
    else:  # 小 k：簇均值 t 区间（t_{0.975,k-1}，k=1 时退化为该簇均值的符号说明）
        m = sum(cmeans_raw) / k
        if k >= 2:
            var = sum((x - m) ** 2 for x in cmeans_raw) / (k - 1)
            tq = _T975.get(k, 2.0)
            clus = [round(m - tq * (var / k) ** 0.5, 4), round(m + tq * (var / k) ** 0.5, 4)]
        else:
            clus = [round(m, 4), round(m, 4)]
        method = f"cluster-t(k={k})"
    same_sign = all(x > 0 for x in cmeans_raw) or all(x < 0 for x in cmeans_raw)
    return {"metric": metric, "n": len(pooled), "n_clusters": k,
            "mean_diff": round(point, 4), "cluster_means":
            {c: round(mm, 4) for c, mm in zip(names, cmeans_raw)},
            "same_sign": same_sign,
            "naive_ci95": naive, "cluster_ci95": clus, "method": method,
            "per_set": per_set}


_T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
         9: 2.306, 10: 2.262}  # t_{0.975,dof=k-1}，k≥5 走 bootstrap 不用


def load_pairs(treat: list[dict], refs: list[dict]) -> list[tuple[str, str, list[dict], list[dict]]]:
    """treat/ref 按 (set, slice) 配对；逐题按序校验 query 一致。"""
    by_key = {(r["set"], r["slice"]): r for r in refs}
    pairs = []
    for t in treat:
        r = by_key.get((t["set"], t["slice"]))
        if r is None or r["n"] != t["n"]:
            continue
        if any(a["query"] != b["query"] for a, b in zip(t["_pq"], r["_pq"])):
            continue  # 题序不一致（不同题集），不构成配对
        pairs.append((t["set"], t["_pq"], r["_pq"]))
    return pairs


def load_run(f: str) -> dict:
    """读单个 pred 文件；harness 存档（B77-B80）无 system/set 字段时从
    文件名 {set}_{arm}.json 恢复，保证跨文件按 set 配对可用。"""
    r = aggregate(json.load(open(f, encoding="utf-8")))
    stem = os.path.basename(f).rsplit(".", 1)[0]
    if r["set"] == "?":
        r["set"], r["system"] = (stem.split("_", 1) + [stem])[:2]
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--ref", default=None,
                    help="参照系统 pred 文件（配对差 CI 用；可逗号分隔多文件按 set 配对）")
    ap.add_argument("--cluster", default=None, help="set=按题集字段分簇报聚类稳健 CI（O2）")
    ap.add_argument("--cluster-map", default=None,
                    help='簇映射 JSON（字符串或文件）：{"nea":"ne","ner":"ne"}——'
                         "set→学科簇；缺省簇=set 本身")
    ap.add_argument("--metrics", default="recall@10,cov@10,mrr@10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    files = sorted({f for pat in args.preds for f in glob.glob(pat)})
    rows = [load_run(f) for f in files]

    refs = []
    if args.ref:
        ref_files = sorted({f for pat in args.ref.split(",") for f in glob.glob(pat)})
        refs = [load_run(f) for f in ref_files]

    lines = ["| system | set | slice | n | R@10(any) | **cov@10** | cov@20 | MRR@10 | p50(s) | 建库 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r.get("index_cost") or {}
        wall = c.get("wall_s")
        wall_s = f"{wall:.0f}s" if isinstance(wall, (int, float)) else (c.get("note") or "-")[:14]
        lines.append(f"| {r['system']} | {r['set']} | {r['slice']} | {r['n']} | "
                     f"{r.get('recall@10', '-')} | **{r.get('cov@10', '-')}** | "
                     f"{r.get('cov@20', '-')} | {r.get('mrr@10', '-')} | "
                     f"{r.get('latency_p50_s', '-')} | {wall_s} |")

    metrics = [m for m in args.metrics.split(",") if m]
    if refs and not args.cluster:
        ref0 = refs[0]
        lines.append(f"\n**配对差（系统 − {ref0['system']}，{ref0['set']}，bootstrap 95% CI，n={ref0['n']}）**\n")
        lines.append("| system | metric | mean_diff | ci95 | 显著 |")
        lines.append("|---|---|---|---|---|")
        for r in rows:
            if r["set"] != ref0["set"] or r["n"] != ref0["n"]:
                continue
            if any(a["query"] != b["query"] for a, b in zip(r["_pq"], ref0["_pq"])):
                continue
            for m in metrics:
                if m not in r or m not in ref0 or not all(m in q for q in r["_pq"]):
                    continue
                ci = paired_ci(r["_pq"], ref0["_pq"], m)
                sig = "是" if ci["ci95"][0] * ci["ci95"][1] > 0 else "否"
                lines.append(f"| {r['system']} | {m} | {ci['mean_diff']} | {ci['ci95']} | {sig} |")

    if refs and args.cluster:
        cmap = {}
        if args.cluster_map:
            src = args.cluster_map
            if not src.startswith("{"):
                src = open(src, encoding="utf-8").read()
            cmap = json.loads(src)
        pairs_all = load_pairs(rows, refs)
        n_sets = len({p[0] for p in pairs_all})
        if pairs_all:
            lines.append(f"\n**分簇配对差（O2 聚类稳健推断：cluster={'set' if not cmap else 'map'}, "
                         f"题集={n_sets} 个；逐题 pooled CI vs 簇级 CI 并列；"
                         f"簇级显著=各簇同号∧簇级 CI 不含 0；k<5 簇用簇均值 t 区间）**\n")
            lines.append("| treat | metric | mean_diff | 逐题pooled ci95 | 显著 | 簇级 ci95（方法） | 显著 | 各簇均值 |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for tname in sorted({r["system"] for r in rows}):
                trows = [r for r in rows if r["system"] == tname]
                tpairs = load_pairs(trows, refs)
                if not tpairs:
                    continue
                for m in metrics:
                    ci = cluster_paired_ci(tpairs, m, cmap)
                    if not ci:
                        continue
                    sig_n = "是" if ci["naive_ci95"][0] * ci["naive_ci95"][1] > 0 else "否"
                    sig_c = "是" if (ci["same_sign"]
                                     and ci["cluster_ci95"][0] * ci["cluster_ci95"][1] > 0) else "否"
                    cmeans = "; ".join(f"{c}:{v:+.3f}" for c, v in ci["cluster_means"].items())
                    lines.append(f"| {tname} | {m} | {ci['mean_diff']} | {ci['naive_ci95']} "
                                 f"| {sig_n} | {ci['cluster_ci95']}（{ci['method']}） | {sig_c} "
                                 f"| {cmeans} |")
                det_sets = ", ".join(f"{s['set']}(簇{s['cluster']}) {s['mean_diff']:+.3f}"
                                     for s in ci["per_set"]) if metrics else ""
                lines.append(f"- {tname} 各题集分解：{det_sets}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        json.dump([{k: v for k, v in r.items() if k != "_pq"} for r in rows],
                  open(args.out.replace(".md", ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
