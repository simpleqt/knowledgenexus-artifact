#!/usr/bin/env python3
"""P1 审稿件：KB1 固定预算词典曲线 + 实体聚类 bootstrap CI + Expansion 判读
（三件共用 KB1 冻结帧/双标参照，一个脚本三 stage）。

budget  —— 冻结重放帧（multisample 10 样本）上 v2.1 子集（构造序前 K
           词条，v2.1 文件序=种子→分类→扩写的流水线序，非结果倒排）
           vs 等规模 freq-top-K vs cvalue-top-K，K∈{200,500,1000,2000,3733}；
           结构指标 + 80 双标单元 occurrence 召回。有效性判读另行
           （judge 阶段对 K=500/2000 两档三臂分层抽样）。
bootci  —— 实体聚类 bootstrap：300 链接审计（187 实体簇）+ 跨域
           threesource 有效率（按实体重采样 10,000 次，95% CI）。
verify3 —— 第三模型（网关 qwen3.8-flash-next，短任务可用）验证
           双标重叠 domain 项样本（每域+KB1 各 40）。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python ... --stage <budget|bootci|verify3>'
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import (precompile, scan,  # noqa: E402
                                      load_lexicon)
from lexicon_cross_domain import chat, _norm_term  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402

OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261013_p0_controls")
DA = os.path.join(REPO, "eval", "baselines", "results",
                  "20261012_dual_anno")
XD = os.path.join(REPO, "eval", "baselines", "results",
                  "20261011_cross_domain")
KB = "104112e63d9d4aa5a6c21e1c1549ef0f"
K_LIST = (200, 500, 1000, 2000, 3733)


def fetch_sample_chunks(n_samples=10, per=2000):
    ms = json.load(open(os.path.join(REPO, "eval", "baselines",
                                     "results",
                                     "20261010_lexicon_multisample",
                                     "multisample.json")))
    c = M.search.client()
    samples = []
    for si in range(n_samples):
        ids = ms["sample_chunk_ids"][str(si)][:per]
        rows = []
        for i in range(0, len(ids), 200):
            r = c.mget(index="knc-chunk",
                       body={"docs": [{"_id": x,
                                       "_source": ["doc_id",
                                                   "content_text"]}
                                      for x in ids[i:i + 200]]})
            rows += [(h["_source"]["doc_id"],
                      h["_source"]["content_text"])
                     for h in r["docs"] if h.get("found")]
        samples.append(rows)
    return samples


def metrics(chunks, lex):
    pats = precompile(lex)
    links = collections.Counter()
    doc_ents = collections.defaultdict(set)
    for did, text in chunks:
        for surface, etype, s, e in scan(text, pats):
            links[(surface.lower(), etype)] += 1
            doc_ents[did].add((surface.lower(), etype))
    total = sum(links.values())
    top10 = sorted(links.values(), reverse=True)[:10]
    return {"links": total, "entities": len(links),
            "c10": round(sum(top10) / total, 3) if total else 0.0,
            "docs": len(doc_ents),
            "conc_per_doc": round(sum(len(v) for v in doc_ents.values())
                                  / max(1, len(doc_ents)), 2)}


def kb1_recall(lex, units, adj):
    pats = precompile(lex)
    hit = ref = 0
    for u in units:
        r_all = {t["norm"] for t in adj.get(u["unit"], [])}
        matched = {_norm_term(s) for s, _e, _x, _y
                   in scan(u["text"], pats)}
        ref += len(r_all)
        hit += len(r_all & matched)
    return round(hit / ref, 3) if ref else None


def stage_budget():
    os.makedirs(OUT, exist_ok=True)
    v21 = load_lexicon(os.path.join(REPO, "schemas",
                                    "nuclear_electronics.yaml"))
    order = list(v21)                      # 流水线序（构造次序）
    units = json.load(open(os.path.join(DA, "units.json"),
                           encoding="utf-8"))
    kb_units = [u for u in units if u["unit"].startswith("kb1:")]
    adj = {r["unit"]: r["terms"] for r in (json.loads(l) for l in open(
        os.path.join(DA, "adjudicated.jsonl"), encoding="utf-8"))}
    # freq/cvalue 候选（构造侧 = KB1 全库文本太贵——用冻结样本文本聚合，
    # 与 v2.1 帧一致；如实记录候选来源=冻结重放帧非全库）
    samples = fetch_sample_chunks()
    import statistics
    from lexicon_cross_domain import text_freq, cvalue_scores
    # freq/cvalue 候选仅用样本 0 排序（2,000 块；全 20k 块 C-value 枚举
    # 不可承受）——候选来源与评估帧重叠如实披露
    docs = [{"text": t} for _d, t in samples[0]]
    freq = text_freq(docs)
    cv = {t: s for t, s in cvalue_scores(docs)}
    res = {}
    for K in K_LIST:
        arms = {}
        arms["v21-prefix"] = {t: v21[t] for t in order[:K]}
        arms["freq-top"] = {t: "term" for t, _ in
                            sorted(freq.items(), key=lambda kv: -kv[1])
                            [:K] if G_ok(t)}
        arms["cvalue-top"] = {t: "term" for t, _ in
                              sorted(cv.items(), key=lambda kv: -kv[1])
                              [:K] if G_ok(t)}
        row = {}
        for an, lex in arms.items():
            per = [metrics(ch, lex) for ch in samples]
            row[an] = {k: round(statistics.mean(p[k] for p in per), 2)
                       for k in ("links", "entities", "c10", "docs",
                                 "conc_per_doc")}
            row[an]["R_dual"] = kb1_recall(lex, kb_units, adj)
        res[f"K={K}"] = row
        print(f"[budget] K={K}: " + " | ".join(
            f"{an} ents={row[an]['entities']} C10={row[an]['c10']} "
            f"R={row[an]['R_dual']}" for an in row))
    json.dump(res, open(os.path.join(OUT, "kb1_budget_curve.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)


def G_ok(t):
    import gen_term_lexicons as G
    return G._ok_term(t)


def stage_bootci():
    """实体簇 bootstrap：审计 300 + 跨域 threesource 有效率。"""
    import statistics as st
    rng = random.Random(20261013)

    def boot(records, key_valid, key_cluster, n=10000):
        by_cluster = collections.defaultdict(list)
        for r in records:
            by_cluster[r[key_cluster]].append(bool(r[key_valid]))
        clusters = list(by_cluster)
        point = (sum(sum(v) for v in by_cluster.values())
                 / sum(len(v) for v in by_cluster.values()))
        stats = []
        for _ in range(n):
            pick = [rng.choice(clusters) for _ in clusters]
            num = den = 0
            for c in pick:
                num += sum(by_cluster[c])
                den += len(by_cluster[c])
            if den:
                stats.append(num / den)
        stats.sort()
        return round(point, 3), (round(stats[int(0.025 * n)], 3),
                                  round(stats[int(0.975 * n)], 3))

    out = {}
    # 300 链接审计：档案在哪？lexicon_semantic_verify 的分层审计记录
    cand = [os.path.join(REPO, "eval", "baselines", "results", d)
            for d in ("20261011_lexicon_semantic",)]
    for cd in cand:
        for f in os.listdir(cd):
            if f.endswith(".jsonl") and "strat" in f:
                rows = [json.loads(l) for l in open(
                    os.path.join(cd, f), encoding="utf-8")]
                if rows and "valid" in rows[0] and "entity" in rows[0]:
                    p, ci = boot(rows, "valid", "entity")
                    out[f"audit300:{f}"] = {"point": p,
                                            "cluster_boot95": ci,
                                            "n": len(rows)}
    # 跨域 threesource 仲裁记录按实体聚类
    for dom in ("cs", "bio", "mat"):
        p_ = os.path.join(XD, dom, "arbit_threesource.jsonl")
        rows = [json.loads(l) for l in open(p_, encoding="utf-8")]
        fin = [r for r in rows if r.get("final") is not None]
        p, ci = boot([{**r, "v": r["final"]} for r in fin],
                     "v", "entity")
        out[f"threesource_{dom}"] = {"point": p,
                                     "cluster_boot95": ci, "n": len(fin)}
        print(dom, p, ci)
    json.dump(out, open(os.path.join(OUT, "cluster_bootstrap.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:800])


def stage_verify3():
    """网关第三模型验证双标重叠 domain 项（短任务）。"""
    units = {u["unit"]: u for u in json.load(open(
        os.path.join(DA, "units.json"), encoding="utf-8"))}
    a1 = {r["unit"]: r["terms"] for r in (json.loads(l) for l in open(
        os.path.join(DA, "anno_p1_deepseek.jsonl"), encoding="utf-8"))}
    a2 = {r["unit"]: r["terms"] for r in (json.loads(l) for l in open(
        os.path.join(DA, "anno_p2_v4pro.jsonl"), encoding="utf-8"))}
    gw_key = os.environ["AIGATEWAY_API_KEY"]
    cfg = {"name": "qwen3.8-flash-next@gateway",
           "base": "https://aigateway.cdut.edu.cn/v1",
           "key": gw_key, "model": "qwen3.8-flash-next"}
    rng = random.Random("20261013-verify3")
    path = os.path.join(OUT, "verify3.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["unit"], r["norm"]))
    PROMPT = """Verify one terminology annotation decision.

Domain: {domain}
Term: {surface}
Context (term in ⟦ ⟧): {ctx}

Two annotators both marked this surface as a domain-specific technical term at this occurrence. Verify independently: is it genuinely a domain-specific technical term here (not a generic academic word, not another sense)? Output JSON only: {{"domain_term": true/false, "reason": "..."}}"""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    lock = threading.Lock()
    # 各域+KB1 抽 40 个双认 domain 项
    tasks = []
    for scope in ("kb1", "cs", "bio", "mat"):
        uids = [u for u in units if u.startswith(scope + (":" if
                 scope != "kb1" else ":"))]
        cand = []
        for uid in uids:
            s1 = {_norm_term(t["surface"]) for t in a1.get(uid, [])
                  if t.get("termness") == "domain"}
            s2 = {_norm_term(t["surface"]) for t in a2.get(uid, [])
                  if t.get("termness") == "domain"}
            for n in s1 & s2:
                cand.append((uid, n))
        tasks += rng.sample(cand, min(40, len(cand)))
    cnt = [0]

    def work(item):
        uid, norm = item
        if (uid, norm) in done:
            return
        u = units[uid]
        idx = u["text"].lower().find(norm)
        if idx < 0:
            return
        ctx = (u["text"][max(0, idx - 200):idx] + "⟦" +
               u["text"][idx:idx + len(norm)] + "⟧" +
               u["text"][idx + len(norm):idx + len(norm) + 200])
        raw = chat(cfg, PROMPT.format(domain=u["domain"],
                                      surface=norm, ctx=ctx),
                   max_tokens=6000)
        m = re.search(r'\{\s*"domain_term".*\}', raw, re.S)
        ok = None
        if m:
            try:
                ok = bool(json.loads(m.group())["domain_term"])
            except Exception:
                pass
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"unit": uid, "norm": norm,
                                    "verdict": ok},
                                   ensure_ascii=False) + "\n")
            cnt[0] += 1
            if cnt[0] % 20 == 0:
                print(f"[verify3] {cnt[0]}/{len(tasks)}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool_:
        list(pool_.map(work, tasks))
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ok = [r for r in rows if r["verdict"] is not None]
    by = collections.defaultdict(lambda: [0, 0])
    for r in ok:
        s = r["unit"].split(":")[0]
        by[s][1] += 1
        by[s][0] += int(r["verdict"])
    res = {k: {"verified": v[1], "confirmed": v[0],
               "rate": round(v[0] / v[1], 3)} for k, v in by.items()}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(OUT, "verify3_summary.json"), "w"),
              ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["budget", "bootci", "verify3"])
    args = ap.parse_args()
    {"budget": stage_budget, "bootci": stage_bootci,
     "verify3": stage_verify3}[args.stage]()


if __name__ == "__main__":
    main()
