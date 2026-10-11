#!/usr/bin/env python3
"""词典语义前后盲判 + 治理臂 removed/retained 验证 + 等规模频率对照
（2026-10-11 复审 P3-R4 两项关键验证）。

三问（全部同冻结文本 = multisample 10 样本，judge 盲，temp 0）：
  Q1 旧 vs 新词典命中语义：v2.0 / v2.1 各 entity-uniform 抽 150 命中，
     判「span 在此上下文是否为该术语的**术语性提及**」（面形=词名时
     「同概念」判同义反复，故换术语性判——泛词普通英语用法=塌缩机制本体）。
  Q2 治理臂 removed/retained：v2.1 短 ASCII 词（≤3 字符）命中中，
     caseShort 治理移除的（大小写不匹配面形）vs 保留的（精确大小写）各 60，
     判有效性——removed≪retained 即治理移除的主要是错误质量（补上
     「动机化但未证实」缺口）；反之则是召回损失。
  Q3 等规模频率对照：纯语料频率 top-K（无 LLM 分类/种子/扩写，K=v2.1
     扫描词条数，类型按 v2.1 类型清单随机指派）entity-uniform 抽 100，
     判有效性——同规模下「选词质量」的边际贡献。

工程：扫描段 ProcessPool 并行（命中不带全文，chunk_id+offset 记位）；
判读段 ThreadPool 4 并发 + 0.15s 节流（.32 router 单实例）。
产物：control_lexicon.json / spans.jsonl / annotations.jsonl /
summary_semantic.txt（Wilson CI 汇总）。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/lexicon_semantic_verify.py \
  --multisample eval/baselines/results/20261010_lexicon_multisample/multisample.json \
  --out eval/baselines/results/20261011_lexicon_semantic'
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import load_lexicon, precompile, scan  # noqa: E402
from audit_link_precision import wilson  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402  (OS client)

# 与 audit_link_precision 的差异：面形=canonical（词典自匹配），「同概念」
# 无信息量；判术语性提及（term-mention validity）+ 特异性。
PROMPT = """You are a strict terminology auditor for a scientific knowledge base (nuclear-electronics / physics journal corpus).

Dictionary term (canonical): {name}
Assigned type: {etype}
Matched surface in text: {surface}

Document context (the matched span is marked):
{ctx}

Answer TWO questions as a single JSON object only:
1. "match": is the marked span, **in this context**, a genuine terminological mention of the concept "{name}"? true only if the text is using it as a technical term for that concept. false if it is an ordinary-language use of the same letters (e.g. the word "method" as a common noun, "model" as an ordinary word, "be/si/or" as ordinary words), a different sense, a fragment of a different token, or boilerplate noise. If genuinely debatable, set "match": false and "borderline": true.
2. "specificity": "domain" if the term is a domain-specific technical concept (e.g. HPGe detector, Compton edge, neutron irradiation); "generic" if it is a generic academic/engineering word usable in any field (e.g. method, result, problem, term, parameter, function, experiment, error).

JSON keys: match (bool), borderline (bool), specificity ("domain"|"generic"), reason (one short sentence). JSON only, no other text."""


def llm_judge(base: str, key: str, model: str, name: str, etype: str,
              surface: str, ctx: str) -> dict:
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": 300,
        "messages": [{"role": "user", "content": PROMPT.format(
            name=name, etype=etype, surface=surface, ctx=ctx)}],
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"match": None, "borderline": True,
                        "specificity": "?", "reason": f"http error: {e}"}
            time.sleep(3)
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {"match": None, "borderline": True, "specificity": "?",
                "reason": f"unparseable: {txt[:80]}"}
    try:
        j = json.loads(m.group(0))
        return {"match": bool(j.get("match")),
                "borderline": bool(j.get("borderline")),
                "specificity": str(j.get("specificity", "?")),
                "reason": str(j.get("reason", ""))[:200]}
    except Exception:  # noqa: BLE001
        return {"match": None, "borderline": True, "specificity": "?",
                "reason": f"json error: {txt[:80]}"}


def fetch_sources_by_ids(index: str, ids: list[str],
                         fields: list[str]) -> dict[str, dict]:
    c = M.search.client()
    out = {}
    for i in range(0, len(ids), 1000):
        batch = ids[i:i + 1000]
        r = c.mget(index=index, body={"ids": batch}, _source=fields)
        for d in r["docs"]:
            if d.get("found"):
                out[d["_id"]] = d["_source"]
    return out


def scan_task(args):
    """worker：单臂单样本扫描，命中不带全文（chunk_id+offset 记位）。"""
    arm, lex, case_short, chunks = args  # chunks: [(chunk_id, doc_id, text)]
    pats = precompile(lex, case_short)
    hits: dict[tuple, list[dict]] = collections.defaultdict(list)
    for cid, did, text in chunks:
        for surface, etype, s, e in scan(text, pats):
            hits[(surface.lower(), etype)].append(
                {"chunk_id": cid, "doc_id": did, "surface": surface,
                 "etype": etype, "s": s, "e": e})
    return arm, hits


def build_freq_control(v21: dict, kb_hex: str, target: int) -> dict:
    """纯语料频率 top-target 对照词典（en+cn 两模式合并，freq≥5、_ok_term；
    类型 = v2.1 类型清单按词条数≥30 的集合随机指派，seed 20261011）。"""
    sys.path.insert(0, os.path.join(REPO, "corpus"))
    import gen_term_lexicons as G
    env = {"OPENSEARCH_ADMIN_PASSWORD": os.environ["OPENSEARCH_PASSWORD"]}
    freq = collections.Counter()
    for mode in ("en", "cn"):
        freq.update(G._corpus_freq(env, kb_hex, mode))
    cands = [t for t, c in freq.most_common(20000)
             if c >= 5 and G._ok_term(t)]
    type_count = collections.Counter(v21.values())
    types = sorted(t for t, n in type_count.items() if n >= 30)
    rng = random.Random("20261011-freq-control")
    lex: dict[str, str] = {}
    for t in cands:
        if len(lex) >= target:
            break
        if t not in lex:
            lex[t] = rng.choice(types)
    return lex


def draw_entity_uniform(hits: dict[tuple, list[dict]], n: int,
                        seed: str) -> list[dict]:
    """实体均匀抽（一实体一命中），与 300-link 审计同型设计。"""
    rng = random.Random(seed)
    ents = sorted(hits)
    picked = ents if len(ents) <= n else rng.sample(ents, n)
    rows = []
    for ent in picked:
        h = rng.choice(hits[ent])
        rows.append({"entity": ent[0], **{k: h[k] for k in
                                          ("etype", "surface", "doc_id",
                                           "chunk_id", "s", "e")}})
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="104112e63d9d4aa5a6c21e1c1549ef0f")
    ap.add_argument("--multisample", required=True)
    ap.add_argument("--n-old", type=int, default=150)
    ap.add_argument("--n-new", type=int, default=150)
    ap.add_argument("--n-ctrl", type=int, default=100)
    ap.add_argument("--n-rr", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ms = json.load(open(args.multisample, encoding="utf-8"))
    all_ids: list[str] = []
    for i in range(ms["samples"]):
        all_ids.extend(ms["sample_chunk_ids"][str(i)])
    print(f"[corpus] fetching {len(all_ids)} frozen chunks ...", flush=True)
    src = fetch_sources_by_ids("knc-chunk", all_ids,
                               ["chunk_id", "doc_id", "content_text"])
    samples = []
    for i in range(ms["samples"]):
        rows = [(cid, src[cid]["doc_id"], src[cid]["content_text"])
                for cid in ms["sample_chunk_ids"][str(i)] if cid in src]
        samples.append(rows)
    print(f"[corpus] {sum(len(x) for x in samples)} chunks fetched",
          flush=True)

    v21 = load_lexicon(os.path.join(REPO, "schemas", "nuclear_electronics.yaml"))
    v20 = load_lexicon("5ec5756:schemas/nuclear_electronics.yaml")
    ctrl = build_freq_control(v21, args.kb, target=len(v21))
    print(f"[lexica] v2.0={len(v20)} v2.1={len(v21)} control={len(ctrl)}",
          flush=True)
    json.dump({"v21_size": len(v21), "v20_size": len(v20),
               "control_size": len(ctrl),
               "control_sample_terms": sorted(ctrl)[:400]},
              open(os.path.join(args.out, "control_lexicon.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

    tasks = []
    for si, chunks in enumerate(samples):
        tasks.append(("old", v20, False, chunks))
        tasks.append(("new", v21, False, chunks))
        tasks.append(("ctrl", ctrl, False, chunks))
    merged: dict[str, dict[tuple, list[dict]]] = {"old": {}, "new": {},
                                                  "ctrl": {}}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for arm, hits in pool.map(scan_task, tasks):
            for ent, lst in hits.items():
                merged[arm].setdefault(ent, []).extend(lst)
        print(f"[hits] old ents={len(merged['old'])} "
              f"new ents={len(merged['new'])} "
              f"ctrl ents={len(merged['ctrl'])}", flush=True)

    rows: list[dict] = []
    for r in draw_entity_uniform(merged["old"], args.n_old, "20261011-old"):
        rows.append({"arm": "old-v2.0", **r})
    for r in draw_entity_uniform(merged["new"], args.n_new, "20261011-new"):
        rows.append({"arm": "new-v2.1", **r})
    for r in draw_entity_uniform(merged["ctrl"], args.n_ctrl, "20261011-ctrl"):
        rows.append({"arm": "freq-control", **r})

    # 治理臂 removed/retained：v2.1 短 ASCII 词（≤3 字符折叠形）命中按
    # 「surface 精确大小写是否为词典词条」分层 = caseShort 臂移除/保留
    short_ents = {ent: hs for ent, hs in merged["new"].items()
                  if ent[0].isascii() and len(ent[0]) <= 3}
    v21_terms_exact = set(v21)
    removed, retained = [], []
    for ent, hs in short_ents.items():
        for h in hs:
            rec = {"entity": ent[0], **{k: h[k] for k in
                                        ("etype", "surface", "doc_id",
                                         "chunk_id", "s", "e")}}
            (retained if h["surface"] in v21_terms_exact
             else removed).append(rec)
    rng = random.Random("20261011-gov")
    print(f"[gov] short-symbol hits: removed-pool={len(removed)} "
          f"retained-pool={len(retained)}", flush=True)
    for r in (rng.sample(removed, args.n_rr) if len(removed) > args.n_rr
              else removed):
        rows.append({"arm": "gov-removed", **r})
    for r in (rng.sample(retained, args.n_rr) if len(retained) > args.n_rr
              else retained):
        rows.append({"arm": "gov-retained", **r})
    rng.shuffle(rows)

    need_cids = sorted({r["chunk_id"] for r in rows})
    texts = fetch_sources_by_ids("knc-chunk", need_cids, ["content_text"])
    text_by_cid = {cid: s.get("content_text", "") for cid, s in texts.items()}

    with open(os.path.join(args.out, "spans.jsonl"), "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"idx": i, **r}, ensure_ascii=False) + "\n")

    base = os.environ["LLM_API_BASE"]
    key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL"]
    outp = open(os.path.join(args.out, "annotations.jsonl"), "w")
    done = [0]

    def work(i_r):
        i, r = i_r
        text = text_by_cid.get(r["chunk_id"], "")
        s, e = r["s"], r["e"]
        if text and text[s:e] == r["surface"]:
            lo, hi = max(0, s - 250), min(len(text), e + 250)
            ctx = text[lo:s] + "⟦" + text[s:e] + "⟧" + text[e:hi]
            span_ok = True
        elif text:
            j = text.find(r["surface"])
            if j >= 0:
                e2 = j + len(r["surface"])
                lo, hi = max(0, j - 250), min(len(text), e2 + 250)
                ctx = text[lo:j] + "⟦" + text[j:e2] + "⟧" + text[e2:hi]
            else:
                ctx = text[:500]
            span_ok = False
        else:
            ctx, span_ok = "(chunk text unavailable)", False
        v = llm_judge(base, key, model, r["entity"], r["etype"],
                      r["surface"], ctx)
        outp.write(json.dumps({"idx": i, "arm": r["arm"],
                               "span_ok": span_ok, **v},
                              ensure_ascii=False) + "\n")
        outp.flush()
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"[judge] {done[0]}/{len(rows)}", flush=True)
        time.sleep(0.15)

    with ThreadPoolExecutor(max_workers=args.conc) as pool:
        list(pool.map(work, list(enumerate(rows))))
    outp.close()

    ann = [json.loads(l) for l in
           open(os.path.join(args.out, "annotations.jsonl"))]
    lines = ["# 词典语义盲判（judge=qwen3.5-35b(=3.6-35B-A3B 权重) temp0，"
             "术语性提及判，entity-uniform，同冻结 10 样本）", ""]
    for arm in ("old-v2.0", "new-v2.1", "freq-control",
                "gov-removed", "gov-retained"):
        sub = [a for a in ann if a["arm"] == arm]
        n = len(sub)
        ok = sum(1 for a in sub if a["match"] is True)
        dom = sum(1 for a in sub
                  if a["match"] and a["specificity"] == "domain")
        gen = sum(1 for a in sub
                  if a["match"] and a["specificity"] == "generic")
        unp = sum(1 for a in sub if a["match"] is None)
        lo, hi = wilson(ok, n)
        span_ok = sum(1 for a in sub if a["span_ok"])
        lines.append(
            f"{arm}: n={n} term-mention-valid {ok}/{n} = {ok/n:.3f} "
            f"[{lo:.3f},{hi:.3f}] | valid&domain {dom} valid&generic {gen} "
            f"unparseable {unp} | span_ok {span_ok}/{n}")
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(args.out, "summary_semantic.txt"), "w") as f:
        f.write(summary + "\n")
    print("[saved]", args.out)


if __name__ == "__main__":
    main()
