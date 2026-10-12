#!/usr/bin/env python3
"""跨领域词典构建+复刻评测（2026-10-11 论文实验扩展 E1/E3 核心）。

三域（cs=arXiv cs.CL / mat=arXiv cond-mat.mtrl-sci / bio=PubMed），数据隔离：
construct 3,000 篇（建词典唯一数据源）/ 测试 100 篇（silver 标注，冻结）/
重放池 ~11.9k（10×2,000 不相交样本，扫描评测）——切分全定种子。

臂（同一扫描器/指标，词典为唯一变量）：
  schema     = 该域现行批次 Schema 词典（类别词形态，"pre/失效"臂）
  freq       = 纯频率 top-K（K=对齐 threesource 规模）
  cvalue     = C-value/NC-value 式术语性打分 top-K（同规模）
  llmdirect  = LLM 逐批直抽（300 篇 construct 子样本，不做语料校验）
  threesource= 三源适配版：LLM 领域知识种子（逐类型，语料存在性校验）
               ∪ 语料候选 LLM 分类 ∪ 逐类型扩写（存在性校验）
               ——跨域无近邻健康库可移植，源 1 以"LLM 知识种子+语料校验"
               替代移植，论文如实披露该适配。

stage（断点缓存 out/components/*.json）：
  frame   语料切分落盘
  build   组件构建（freq/classify/seed/expand/llmdirect）+ 五臂装配
  scan    10×2000 重放扫描 → 结构指标 mean±sd
  silver  测试 100 篇 × 双判官全术语标注（A=.32 3.6-35B，B=网关 3.8-flash）
  judge   分层抽样 × 双判官术语性盲判（head/medium/tail/短符号）
  arbit   分歧样本第三模型仲裁（--jc-base 本地 27B）
  tables  汇总表（P/R/F1 + 结构 + 一致性）
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/lexicon_cross_domain.py --stage build'
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import precompile, scan  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "corpus"))
import gen_term_lexicons as G  # noqa: E402  (_ok_term/STOP_EN/CATCHALL/_llm_chat/_json_obj)
sys.path.insert(0, os.path.join(REPO, "libs", "engine"))
from knc_engine import llm as KLLM  # noqa: E402

DATA = os.path.join(REPO, "eval", "baselines", "data", "public_corpora")
SCHEMA = {"cs": "computer_science.yaml", "mat": "materials_science.yaml",
          "bio": "medicine.yaml"}
GENERIC_HEAD = {"method", "energy", "parameter", "material", "model",
                "term", "time", "field", "phase", "temperature",
                "algorithm", "data", "system", "process", "structure",
                "disease", "drug", "protein", "cell"}
V20_HEAD = {"method", "energy", "parameter", "material", "model",
            "term", "time", "field", "phase", "temperature"}

# ---- 判官 ----
JUDGE_PROMPT = open(os.path.join(os.path.dirname(__file__),
                                 "lexicon_semantic_verify.py"), encoding="utf-8"
                    ).read().split('PROMPT = """', 1)[1].split('"""', 1)[0]

SILVER_PROMPT = """You are a senior domain terminologist building a reference term list.

Domain: {domain}
Schema entity types (id: description):
{types}

Abstract:
{ctx}

List EVERY domain-specific technical term that occurs in this abstract (terms of the listed types or clearly domain-technical multiword terms). For each: the exact surface form as it appears, and the best-matching type id. Include important acronyms only when defined or standard in the domain. Do NOT include generic academic words (method, result, problem, approach, study, analysis), organisms' common names, or verbs.
Output JSON only: {{"terms": [{{"surface": "...", "type": "type_id"}}]}}"""


def load_env_file(path):
    env = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
    return env


ENV_MAIN = load_env_file(os.path.join(REPO, "deploy", ".env"))
SVC = load_env_file(os.path.join(REPO, "deploy", ".env.services"))
JUDGES = {
    "A": {"name": "qwen3.6-35b@.32", "base": SVC["LLM_API_BASE"],
          "key": SVC["LLM_API_KEY"], "model": SVC["LLM_MODEL"]},
    "B": {"name": "qwen3.8-flash-next@gateway", "base": ENV_MAIN["AIGATEWAY_BASE_URL"],
          "key": ENV_MAIN["AIGATEWAY_API_KEY"], "model": "qwen3.8-flash-next"},
}


def chat(cfg, prompt, max_tokens=4000, temperature=0):
    import urllib.request
    req_body = {"model": cfg["model"], "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
    if "27b" in cfg.get("model", ""):  # Qwen3.5 思维链模型：关思考直出答案
        req_body["chat_template_kwargs"] = {"enable_thinking": False}
    if "deepseek" in cfg.get("model", ""):  # 关思考直出（官方文档原生参数）
        req_body["thinking"] = {"type": "disabled"}
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(
        cfg["base"].rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['key']}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=420) as r:
                d = json.load(r)
            msg = d["choices"][0]["message"]
            txt = msg.get("content") or ""
            if not txt.strip():
                txt = msg.get("reasoning_content") or ""
            return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                print(f"    [chat-fail] {cfg['model']}: {str(e)[:90]}", flush=True)
                return ""
            time.sleep(5 * (attempt + 1) ** 2)  # 5/20/45/80s 退避
    return ""


def domain_label(dom):
    return {"cs": "computer science (computation and language)",
            "mat": "materials science (condensed matter, materials)",
            "bio": "biomedicine / life sciences"}[dom]


# ---------- stage: frame ----------

def stage_frame(dom, out):
    rows = [json.loads(l) for l in open(os.path.join(DATA, f"{dom}.jsonl"),
                                        encoding="utf-8") if l.strip()]
    rng = random.Random(f"20261011-{dom}-split")
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    construct = [rows[i] for i in idx[:3000]]
    test = [rows[i] for i in idx[3000:3100]]
    pool = [rows[i] for i in idx[3100:3100 + 20000]]
    per = max(500, len(pool) // 10)
    samples = []
    used = set()
    for si in range(10):
        take = [r for r in pool if r["doc_id"] not in used][:per]
        used |= {r["doc_id"] for r in take}
        samples.append(take)
    assert all(len(s) == per for s in samples), "重放样本不足"
    json.dump({"construct_ids": [r["doc_id"] for r in construct],
               "test_ids": [r["doc_id"] for r in test],
               "sample_ids": [[r["doc_id"] for r in s] for s in samples]},
              open(os.path.join(out, "frame.json"), "w"))
    # 正文缓存（construct/test 全文；样本池从 jsonl 现取）
    json.dump(construct, open(os.path.join(out, "construct.json"), "w"),
              ensure_ascii=False)
    json.dump(test, open(os.path.join(out, "test.json"), "w"),
              ensure_ascii=False)
    print(f"[{dom}] frame: construct={len(construct)} test={len(test)} "
          f"samples={'x'.join(str(len(s)) for s in samples)}")


# ---------- 频率与 C-value ----------

EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}(?:\s+[A-Za-z][A-Za-z0-9\-]{2,}){0,3}")


def text_freq(docs):
    freq = collections.Counter()
    for d in docs:
        for m in EN_RE.finditer(d["text"]):
            t = m.group()
            if t.lower() not in G.STOP_EN:
                freq[t] += 1
    return freq


def cvalue_scores(docs, top_n=8000):
    """经典 C-value（Frantzi/Ananiadou 2000）：候选=1-4 词组，频率+嵌套修正。"""
    from collections import defaultdict
    freq = collections.Counter()
    nested = defaultdict(set)  # longer candidate -> containing shorter
    cands = []
    for d in docs:
        for m in EN_RE.finditer(d["text"]):
            cands.append(m.group())
    cand_set = set(cands)
    freq.update(cands)
    # 嵌套关系：更长候选包含的更短候选（枚举连续子序列，O(sum L^2) 而非成对比较）
    def _subsequences(phrase: str):
        toks = phrase.split()
        for i in range(len(toks)):
            for j in range(i + 2, len(toks) + 1):
                yield " ".join(toks[i:j])
    for longer in cand_set:
        if len(longer.split()) < 2:
            continue
        for sub in _subsequences(longer):
            if sub != longer and sub in cand_set:
                nested[longer].add(sub)
    scores = {}
    for c in cand_set:
        f = freq[c]
        w = len(c.split())
        if not nested.get(c):
            scores[c] = math.log(w, 2) * f
        else:
            ta = len(nested[c])
            scores[c] = math.log(w, 2) * (f - sum(freq[s] for s in nested[c]) / ta)
    return sorted(((t, s) for t, s in scores.items()
                   if s > 0 and G._ok_term(t) and freq[t] >= 3),
                  key=lambda kv: -kv[1])[:top_n]


# ---------- stage: build ----------

def llm_classify(cands, types_desc):
    out = collections.defaultdict(list)
    for i in range(0, len(cands), 300):
        batch = cands[i:i + 300]
        got = G._classify_batch(batch, types_desc)
        for tid, terms in got.items():
            out[tid] += terms
        print(f"    classify {min(i+300, len(cands))}/{len(cands)} "
              f"(+{sum(len(v) for v in got.values())})", flush=True)
        time.sleep(0.3)
    return {k: sorted(set(v)) for k, v in out.items()}


def seed_knowledge(types_desc, freq_set):
    got = {}
    for tid, spec in types_desc.items():
        desc = str(spec.get("description", ""))[:80]
        raw = G._llm_chat(
            f"领域：{domain_label(CTX['dom'])}。实体类型：{tid}（{desc}）。\n"
            f"列出 60 个该类型的真实领域术语（英文为主，具体到实例/专有名词，"
            f"如该域的器件/算法/材料/疾病等专名；禁止类别词如 method/result；"
            f"每条 ≤5 词）。\n输出 JSON：{{\"terms\": [\"术语\", ...]}}",
            max_tokens=6000)
        doc = G._json_obj(raw) or {}
        terms = [t for t in (doc.get("terms") or [])
                 if G._ok_term(t)]
        got[tid] = sorted(set(terms))
        print(f"    seed[{tid}] {len(got[tid])}", flush=True)
    return got


def expand_types(types_desc, seeds_by_type, freq_set, n=60):
    got = {}
    for tid, spec in types_desc.items():
        raw = G._expand_type(tid, str(spec.get("description", "")),
                             seeds_by_type.get(tid, []), n)
        keep = sorted({t for t in (raw or [])
                       if G._ok_term(t) and (t in freq_set or t.lower() in freq_set)})
        got[tid] = keep
        print(f"    expand[{tid}] +{len(keep)}", flush=True)
    return got


def llm_direct(construct, types_desc, n_docs=300):
    rng = random.Random("20261011-llmdirect")
    docs = rng.sample(construct, min(n_docs, len(construct)))
    got = collections.defaultdict(set)
    for i in range(0, len(docs), 10):
        batch = docs[i:i + 10]
        texts = "\n\n".join(f"[{j+1}] {d['text'][:1200]}" for j, d in enumerate(batch))
        type_desc = "\n".join(f"- {tid}: {str(s.get('description',''))[:60]}"
                              for tid, s in types_desc.items())
        raw = G._llm_chat(
            f"领域：{domain_label(CTX['dom'])}。从下列摘要中抽取全部领域专门术语"
            f"（器件/算法/材料/疾病等专名与标准多词术语；含常见缩写；"
            f"禁止泛学术词 method/result/approach）。\n类型清单：\n{type_desc}\n"
            f"\n摘要：\n{texts}\n\n"
            f'输出 JSON：{{"terms": [{{"surface":"...","type":"type_id"}}]}}',
            max_tokens=8000)
        doc = G._json_obj(raw) or {}
        for t in doc.get("terms") or []:
            s = (t.get("surface") or "").strip() if isinstance(t, dict) else str(t)
            ty = (t.get("type") if isinstance(t, dict) else None) or "term"
            if s and G._ok_term(s) and ty in types_desc:
                got[ty].add(s)
        print(f"    llmdirect {min(i+10, len(docs))}/{len(docs)} "
              f"(total {sum(len(v) for v in got.values())})", flush=True)
    return {k: sorted(v) for k, v in got.items()}


def assemble(types_desc, sources):
    """合并（特异性消解：同词多类型取词条少的类型）+ 类型主名入扫描面。"""
    by_term = {}
    for src in sources:
        for tid, terms in src.items():
            for t in terms:
                by_term.setdefault(t, []).append(tid)
    lex = {}
    for t, tids in by_term.items():
        tids = [x for x in tids if x in types_desc] or tids
        lex[t] = min(tids, key=lambda x: ((types_desc.get(x) or {}).get("_n", 9999), x))
    for tid in types_desc:
        lex.setdefault(tid, tid)
    return lex


CTX = {"dom": "?"}


def stage_build(dom, out, comps):
    import yaml
    CTX["dom"] = dom
    construct = json.load(open(os.path.join(out, "construct.json"), encoding="utf-8"))
    schema_doc = yaml.safe_load(
        open(os.path.join(REPO, "schemas", SCHEMA[dom]), encoding="utf-8"))
    ets = schema_doc.get("entity_types", {})
    types_desc = {t: s for t, s in ets.items()
                  if t not in ("person", "paper", "organization")
                  and t not in G.CATCHALL_TYPES}
    for t in types_desc:
        types_desc[t]["_n"] = len((ets[t].get("aliases") or []))

    freq = text_freq(construct)
    freq_set = {t for t in freq} | {t.lower() for t in freq}
    json.dump({"vocab": len(freq)}, open(os.path.join(comps, "freq.json"), "w"))

    def cached(name, fn, raw_var=None):
        p = os.path.join(comps, f"{name}.json")
        if os.path.exists(p):
            print(f"    [{name}] cached")
            return json.load(open(p, encoding="utf-8"))
        v = fn()
        json.dump(v, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        if raw_var is not None:
            json.dump(raw_var[0], open(p.replace(".json", "_raw.json"), "w",
                                       encoding="utf-8"), ensure_ascii=False)
        return v

    cands = [t for t, c in freq.most_common(4000) if c >= 5 and G._ok_term(t)]
    print(f"[{dom}] candidates={len(cands)}", flush=True)
    classify = cached("classify", lambda: llm_classify(cands, types_desc))
    seed = cached("seed", lambda: seed_knowledge(types_desc, freq_set))
    _raw_expand = {}

    def _expand_capture():
        got = {}
        for tid, spec in types_desc.items():
            raw = G._expand_type(tid, str(spec.get("description", "")),
                                 classify.get(tid, []), 60)
            keep = sorted({t for t in (raw or [])
                           if G._ok_term(t)})
            _raw_expand[tid] = keep
            got[tid] = sorted({t for t in keep
                               if t in freq_set or t.lower() in freq_set})
            print(f"    expand[{tid}] +{len(got[tid])} (raw {len(keep)})", flush=True)
        return got

    expand = cached("expand", _expand_capture, raw_var=(_raw_expand,))
    llmdir = cached("llmdirect", lambda: llm_direct(construct, types_desc))

    three = assemble(types_desc, [seed, classify, expand])
    three = {t: ty for t, ty in three.items()
             if t in freq_set or t in types_desc}  # 存在性校验（类型主名除外）
    k = len(three)
    freq_arm = {}
    rng = random.Random(f"20261011-{dom}-armtypes")
    tlist = sorted(types_desc)
    for t, c in freq.most_common(8000):
        if len(freq_arm) >= k:
            break
        if G._ok_term(t) and t not in freq_arm:
            freq_arm[t] = rng.choice(tlist)
    cv = cvalue_scores(construct)
    cv_arm = {}
    for t, s in cv:
        if len(cv_arm) >= k:
            break
        cv_arm[t] = rng.choice(tlist)
    lld = assemble(types_desc, [llmdir])

    schema_arm = {}
    for t, spec in ets.items():
        for alias in (spec.get("aliases") or []):
            a = alias.split(",")[0].strip()
            if a and a not in schema_arm:
                schema_arm[a] = t
    for t in ets:
        schema_arm.setdefault(t, t)

    arms = {"schema": schema_arm, "freq": freq_arm, "cvalue": cv_arm,
            "llmdirect": lld, "threesource": three}
    for name, lex in arms.items():
        json.dump(lex, open(os.path.join(comps, f"arm_{name}.json"), "w",
                            encoding="utf-8"), ensure_ascii=False)
        print(f"[{dom}] arm {name}: {len(lex)} terms")


# ---------- stage: scan ----------

def scan_one(args):
    arm, lex, chunks = args  # chunks: [(doc_id, text)]
    pats = precompile(lex)
    links = collections.Counter()
    doc_ents = collections.defaultdict(set)
    for did, text in chunks:
        for surface, etype, s, e in scan(text, pats):
            links[(surface.lower(), etype)] += 1
            doc_ents[did].add((surface.lower(), etype))
    total = sum(links.values())
    top10 = sorted(links.values(), reverse=True)[:10]
    per_doc = [len(v) for v in doc_ents.values()]
    head = sum(n for (f, t), n in links.items() if f in GENERIC_HEAD)
    return arm, {"links": total, "entities": len(links),
                 "c10": round(sum(top10) / total, 3) if total else 0.0,
                 "conc_per_doc": round(statistics.mean(per_doc), 2) if per_doc else 0,
                 "generic_head_share": round(head / total, 3) if total else 0.0,
                 "docs": len(doc_ents)}


def stage_scan(dom, out, comps, workers):
    from concurrent.futures import ProcessPoolExecutor
    rows = {r["doc_id"]: r["text"] for r in
            (json.loads(l) for l in open(os.path.join(DATA, f"{dom}.jsonl"),
                                         encoding="utf-8"))}
    frame = json.load(open(os.path.join(out, "frame.json")))
    samples = [[(i, rows[i]) for i in ids] for ids in frame["sample_ids"]]
    arms = ["schema", "freq", "cvalue", "llmdirect", "threesource"]
    results = {a: [] for a in arms}
    tasks = []
    for a in arms:
        lex = json.load(open(os.path.join(comps, f"arm_{a}.json"), encoding="utf-8"))
        for s in samples:
            tasks.append((a, lex, s))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for arm, m in pool.map(scan_one, tasks):
            results[arm].append(m)
    agg = {}
    for a in arms:
        rows_ = results[a]
        agg[a] = {k: {"mean": round(statistics.mean(r[k] for r in rows_), 3),
                      "sd": round(statistics.stdev(r[k] for r in rows_), 3)}
                  for k in ("links", "entities", "c10", "conc_per_doc",
                            "generic_head_share", "docs")}
        print(f"[{dom}] {a}: {agg[a]}")
    json.dump(agg, open(os.path.join(out, "scan_agg.json"), "w"), indent=1)


# ---------- stage: silver ----------

def stage_silver(dom, out, workers_conc=3):
    test = json.load(open(os.path.join(out, "test.json"), encoding="utf-8"))
    import yaml
    schema_doc = yaml.safe_load(
        open(os.path.join(REPO, "schemas", SCHEMA[dom]), encoding="utf-8"))
    type_desc = "\n".join(f"- {t}: {str(s.get('description',''))[:60]}"
                          for t, s in schema_doc.get("entity_types", {}).items())
    recs = []
    for j, cfg in (("A", JUDGES["A"]), ("B", JUDGES["B"])):
        path = os.path.join(out, f"silver_{j}.jsonl")
        if os.path.exists(path):
            recs += [json.loads(l) for l in open(path, encoding="utf-8")]
            print(f"[{dom}] silver {j} cached")
            continue
        from concurrent.futures import ThreadPoolExecutor
        done = [0]

        def work(item):
            i, d = item
            raw = chat(cfg, SILVER_PROMPT.format(
                domain=domain_label(dom), types=type_desc, ctx=d["text"][:2500]),
                max_tokens=14000 if j == "B" else 3000)
            doc = G._json_obj(raw) or {}
            terms = [t for t in (doc.get("terms") or [])
                     if isinstance(t, dict) and t.get("surface")]
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"doc_id": d["doc_id"], "judge": j,
                                    "terms": terms}, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"  [{dom}] silver {j} {done[0]}/{len(test)}", flush=True)
            time.sleep(0.3)

        with ThreadPoolExecutor(max_workers=workers_conc) as pool:
            list(pool.map(work, list(enumerate(test))))
        recs += [json.loads(l) for l in open(path, encoding="utf-8")]
    print(f"[{dom}] silver rows={len(recs)}")


# ---------- stage: judge ----------

def stratified_sample(hits_by_ent, n_per=20):
    """head/medium/tail（按实体链接数三分）+ 短符号，各 n_per（池小则尽取）。"""
    rng = random.Random("20261011-strat")
    ents = sorted(hits_by_ent, key=lambda e: -sum(1 for _ in hits_by_ent[e]))
    k = max(1, len(ents) // 3)
    tiers = {"head": ents[:k], "medium": ents[k:2 * k], "tail": ents[2 * k:]}
    rows = []
    for tier, pool in tiers.items():
        for e in (rng.sample(pool, min(n_per, len(pool))) if len(pool) > n_per else pool):
            h = rng.choice(hits_by_ent[e])
            rows.append({"stratum": tier, "entity": e[0], "etype": e[1],
                         "surface": h["surface"], "doc_id": h["doc_id"],
                         "s": h["s"], "e": h["e"]})
    shorts = [e for e in ents if e[0].isascii() and len(e[0]) <= 3]
    for e in (rng.sample(shorts, min(n_per, len(shorts))) if len(shorts) > n_per else shorts):
        h = rng.choice(hits_by_ent[e])
        rows.append({"stratum": "short", "entity": e[0], "etype": e[1],
                     "surface": h["surface"], "doc_id": h["doc_id"],
                     "s": h["s"], "e": h["e"]})
    return rows


def stage_judge(dom, out, comps, n_per=20, workers_conc=3):
    rows_txt = {r["doc_id"]: r["text"] for r in
                (json.loads(l) for l in open(os.path.join(DATA, f"{dom}.jsonl"),
                                             encoding="utf-8"))}
    frame = json.load(open(os.path.join(out, "frame.json")))
    all_ids = [i for ids in frame["sample_ids"] for i in ids]
    from concurrent.futures import ThreadPoolExecutor
    for arm in ("schema", "freq", "cvalue", "llmdirect", "threesource"):
        path = os.path.join(out, f"judge_{arm}.jsonl")
        if os.path.exists(path):
            print(f"[{dom}] judge {arm} cached")
            continue
        lex = json.load(open(os.path.join(comps, f"arm_{arm}.json"), encoding="utf-8"))
        pats = precompile(lex)
        hits_by_ent = collections.defaultdict(list)
        rng = random.Random(f"20261011-{dom}-judge")
        pool_ids = rng.sample(all_ids, 4000)
        for i in pool_ids:
            text = rows_txt[i]
            for surface, etype, s, e in scan(text, pats):
                hits_by_ent[(surface.lower(), etype)].append(
                    {"surface": surface, "doc_id": i, "s": s, "e": e})
        rows = stratified_sample(hits_by_ent, n_per)
        done = [0]

        def work(item):
            i, r = item
            text = rows_txt[r["doc_id"]]
            s, e = r["s"], r["e"]
            if text[s:e] != r["surface"]:
                j = text.find(r["surface"])
                if j < 0:
                    return
                s, e = j, j + len(r["surface"])
            ctx = text[max(0, s - 250):s] + "⟦" + text[s:e] + "⟧" + text[e:e + 250]
            rec = {"arm": arm, "ctx": ctx, **r}
            for j, cfg in (("A", JUDGES["A"]), ("B", JUDGES["B"])):
                raw = chat(cfg, JUDGE_PROMPT.format(
                    name=r["entity"], etype=r["etype"],
                    surface=r["surface"], ctx=ctx),
                    max_tokens=3000 if j == "B" else 400)
                d = G._json_obj(raw) or {}
                rec[f"j{j}"] = {"match": d.get("match"),
                                "specificity": d.get("specificity"),
                                "reason": str(d.get("reason", ""))[:150]}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"  [{dom}] judge {arm} {done[0]}/{len(rows)}", flush=True)
            time.sleep(0.2)

        with ThreadPoolExecutor(max_workers=workers_conc) as pool:
            list(pool.map(work, list(enumerate(rows))))
        print(f"[{dom}] judge {arm}: {len(rows)} sampled")


def stage_arbit(dom, out, jc_base):
    """分歧/缺判样本第三判官仲裁（judge_{arm}.jsonl 冻结，仲裁写 arbit_{arm}.jsonl）。

    终判规则：A/B 双判且一致 → 直接采用；任一缺失或分歧 → C（跨家族
    deepseek-flash，关思考）仲裁，C 失败回退 A（记 mode）。断点按
    (arm, entity, doc_id, s, e) 幂等重入。
    """
    jc = {"name": os.environ.get("KNC_JC_MODEL", "deepseek-flash"),
          "base": jc_base,
          "key": os.environ.get("DEEPSEEK_API_KEY", "none"),
          "model": os.environ.get("KNC_JC_MODEL", "deepseek-flash")}
    for arm in ("schema", "freq", "cvalue", "llmdirect", "threesource"):
        src = os.path.join(out, f"judge_{arm}.jsonl")
        dst = os.path.join(out, f"arbit_{arm}.jsonl")
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
        done = set()
        if os.path.exists(dst):
            for l in open(dst, encoding="utf-8"):
                r = json.loads(l)
                done.add((r["entity"], r["doc_id"], r["s"], r["e"]))
        n_c = 0
        with open(dst, "a", encoding="utf-8") as f:
            for r in rows:
                key = (r["entity"], r["doc_id"], r["s"], r["e"])
                if key in done:
                    continue
                a = (r.get("jA") or {}).get("match")
                b = (r.get("jB") or {}).get("match")
                if a is not None and a == b:
                    rec = {**r, "jC": None, "final": a, "mode": "AB-agree"}
                else:
                    n_c += 1
                    raw = chat(jc, JUDGE_PROMPT.format(
                        name=r["entity"], etype=r["etype"],
                        surface=r["surface"], ctx=r["ctx"]), max_tokens=500)
                    d = G._json_obj(raw) or {}
                    c = d.get("match")
                    if c is not None:
                        rec = {**r, "jC": {"match": c,
                                           "reason": str(d.get("reason", ""))[:150]},
                               "final": c,
                               "mode": "C-substitute" if (a is None or b is None)
                               else "C-arbit"}
                    else:
                        rec = {**r, "jC": None,
                               "final": a if a is not None else b,
                               "mode": "C-fail→A"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        modes = collections.Counter(
            json.loads(l)["mode"] for l in open(dst, encoding="utf-8"))
        print(f"[{dom}] arbit {arm}: +{len(rows) - len(done)} rows "
              f"(C judged {n_c}) modes={dict(modes)}", flush=True)


def _norm_term(x: str) -> str:
    """silver/命中面形归一：lowercase + 朴素复数剥除（对双侧对称施用）。"""
    x = x.lower().strip()
    if len(x) > 3 and x.endswith("s") and not x.endswith("ss"):
        x = x[:-1]
    return x


def stage_tables(dom, out, comps):
    """汇总 tables.json：结构指标（scan_agg）+ silver P/R/F1 + 判读有效率 +
    silver A-C 子集一致性（枚举工具可靠性上界，如实入文）。"""
    scan_agg = json.load(open(os.path.join(out, "scan_agg.json"),
                              encoding="utf-8"))
    test = json.load(open(os.path.join(out, "test.json"), encoding="utf-8"))
    silver_a = {r["doc_id"]: [t["surface"] for t in r.get("terms") or []]
                for r in (json.loads(l) for l in
                          open(os.path.join(out, "silver_A.jsonl"),
                               encoding="utf-8"))}

    # silver A-C 子集一致性（30 篇）——枚举工具可靠性
    sc_path = os.path.join(out, "silver_C.jsonl")
    ac = []
    if os.path.exists(sc_path):
        silver_c = {r["doc_id"]: [t["surface"] for t in r.get("terms") or []]
                    for r in (json.loads(l) for l in open(sc_path,
                                                         encoding="utf-8"))}
        for did, terms_c in silver_c.items():
            sa = {_norm_term(t) for t in silver_a.get(did, [])}
            scset = {_norm_term(t) for t in terms_c}
            u = len(sa | scset)
            ac.append(round(len(sa & scset) / u, 3) if u else 1.0)
    ac_mean = round(statistics.mean(ac), 3) if ac else None

    prf, verdict = {}, {}
    for arm in ("schema", "freq", "cvalue", "llmdirect", "threesource"):
        # ---- occurrence 级 P/R vs silver A（100 测试篇）----
        lex = json.load(open(os.path.join(comps, f"arm_{arm}.json"),
                             encoding="utf-8"))
        pats = precompile(lex)
        hit_n = sil_n = hit_in_sil = 0
        sil_hit = 0
        for d in test:
            sil = {_norm_term(t) for t in silver_a.get(d["doc_id"], [])}
            matched = set()
            for surface, _et, _s, _e in scan(d["text"], pats):
                hit_n += 1
                if _norm_term(surface) in sil:
                    hit_in_sil += 1
                matched.add(_norm_term(surface))
            sil_n += len(sil)
            sil_hit += len(sil & matched)
        p = round(hit_in_sil / hit_n, 3) if hit_n else 0.0
        r_ = round(sil_hit / sil_n, 3) if sil_n else 0.0
        f1 = round(2 * p * r_ / (p + r_), 3) if p + r_ else 0.0
        prf[arm] = {"P": p, "R": r_, "F1": f1,
                    "occurrences": hit_n, "silver_terms": sil_n}

        # ---- 判读有效率（仲裁终判）----
        apath = os.path.join(out, f"arbit_{arm}.jsonl")
        if os.path.exists(apath):
            recs = [json.loads(l) for l in open(apath, encoding="utf-8")]
            final = [r["final"] for r in recs if r["final"] is not None]
            both = [r for r in recs
                    if (r.get("jA") or {}).get("match") is not None
                    and (r.get("jB") or {}).get("match") is not None]
            verdict[arm] = {
                "rate": round(sum(final) / len(final), 3) if final else None,
                "n": len(final),
                "AB_agree": (round(sum(1 for r in both
                                       if r["jA"]["match"] == r["jB"]["match"])
                                   / len(both), 3) if both else None),
                "AB_n": len(both)}

    tables = {"scan": scan_agg, "prf": prf, "verdict": verdict,
              "silver_reliability": {
                  "AC_jaccard_mean": ac_mean, "AC_n_docs": len(ac),
                  "note": "B(gateway flash-next) 全摘要枚举不可用（强制思考"
                          ">180s 读超时），枚举可靠性以 A-C 子集一致性计量"}}
    json.dump(tables, open(os.path.join(out, "tables.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{dom}] tables.json written; silver AC-Jaccard={ac_mean}")
    for arm in prf:
        print(f"  {arm:12s} P={prf[arm]['P']} R={prf[arm]['R']} "
              f"F1={prf[arm]['F1']} verdict={verdict.get(arm, {}).get('rate')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", required=True, choices=["cs", "mat", "bio"])
    ap.add_argument("--stage", required=True,
                    choices=["frame", "build", "scan", "silver", "judge",
                             "arbit", "tables"])
    ap.add_argument("--out-root", default=os.path.join(
        REPO, "eval", "baselines", "results", "20261011_cross_domain"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jc-base", default=os.environ.get("KNC_JC_BASE", ""))
    args = ap.parse_args()
    out = os.path.join(args.out_root, args.dom)
    comps = os.path.join(out, "components")
    os.makedirs(comps, exist_ok=True)
    if args.stage == "frame":
        stage_frame(args.dom, out)
    elif args.stage == "build":
        stage_build(args.dom, out, comps)
    elif args.stage == "scan":
        stage_scan(args.dom, out, comps, args.workers)
    elif args.stage == "silver":
        stage_silver(args.dom, out)
    elif args.stage == "judge":
        stage_judge(args.dom, out, comps)
    elif args.stage == "arbit":
        assert args.jc_base, "--jc-base required (local 27B vLLM)"
        stage_arbit(args.dom, out, args.jc_base)
    elif args.stage == "tables":
        stage_tables(args.dom, out, comps)


if __name__ == "__main__":
    main()


# ---------- stage: arbit（第三模型仲裁 A/B 分歧） ----------

def stage_arbit(dom, out, jc_base):
    jc = {"name": "qwen3.5-27b@local", "base": jc_base,
          "key": "none", "model": os.environ.get("KNC_JC_MODEL", "qwen3.5-27b")}
    ARBIT_PROMPT = (
        "Two terminology auditors disagreed on the sample below.\n"
        "Auditor 1 verdict: {v1} ({r1})\nAuditor 2 verdict: {v2} ({r2})\n\n"
        + JUDGE_PROMPT)
    n_done = 0
    for arm in ("schema", "freq", "cvalue", "llmdirect", "threesource"):
        path = os.path.join(out, f"judge_{arm}.jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        outp = os.path.join(out, f"judge_{arm}_arbit.jsonl")
        done_idx = set()
        if os.path.exists(outp):
            done_idx = {r["idx"] for r in
                        (json.loads(l) for l in open(outp, encoding="utf-8"))}
        with open(outp, "a", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if i in done_idx:
                    continue
                a, b = r.get("jA"), r.get("jB")
                if a is None or b is None or \
                        (a.get("match") is not None and a.get("match") == b.get("match")):
                    continue  # 双 parseable 且一致：无需仲裁
                raw = chat(jc, ARBIT_PROMPT.format(
                    v1=str(a.get("match")), r1=str(a.get("reason"))[:100],
                    v2=str(b.get("match")), r2=str(b.get("reason"))[:100],
                    name=r["entity"], etype=r["etype"], surface=r["surface"],
                    ctx=r.get("ctx", "")), max_tokens=2000)
                d = G._json_obj(raw) or {}
                f.write(json.dumps({"idx": i, "arm": arm, "jc": {
                    "match": d.get("match"),
                    "specificity": d.get("specificity"),
                    "reason": str(d.get("reason", ""))[:150]}},
                    ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
        print(f"[{dom}] arbit {arm}: cumulative {n_done} arbitrations")
    print(f"[{dom}] arbitration done ({n_done} total)")


# ---------- stage: tables（P/R/F1 + 一致性 + 结构汇总） ----------

def _norm_term(t: str) -> str:
    t = t.strip().lower()
    if t.endswith("s") and len(t) > 3:
        t = t[:-1]
    return t


def stage_tables(dom, out, comps):
    test = json.load(open(os.path.join(out, "test.json"), encoding="utf-8"))
    silver = collections.defaultdict(dict)  # judge -> doc -> {norm_term: type}
    for j in ("A", "B"):
        p = os.path.join(out, f"silver_{j}.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            silver[j][r["doc_id"]] = {_norm_term(t["surface"]): t.get("type")
                                      for t in r.get("terms", [])}
    lines = [f"# {dom}: P/R/F1（silver=判官 A 全术语标注 100 篇；"
             f" occurrence 级，大小写+复数归一）", ""]
    prf = {}
    for arm in ("schema", "freq", "cvalue", "llmdirect", "threesource"):
        lex = json.load(open(os.path.join(comps, f"arm_{arm}.json"),
                             encoding="utf-8"))
        pats = precompile(lex)
        tp = fp = fn = 0
        type_ok = type_tot = 0
        for d in test:
            text = d["text"]
            got = {_norm_term(s) for s, _, _, _ in scan(text, pats)}
            gold_terms = silver.get("A", {}).get(d["doc_id"], {})
            for g in gold_terms:
                if g in got:
                    tp += 1
                    type_tot += 1
                    ty = lex.get(next(t for t in lex if _norm_term(t) == g), "?")
                    if ty == gold_terms[g]:
                        type_ok += 1
                else:
                    fn += 1
            for g in got:
                if g not in gold_terms:
                    fp += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        prf[arm] = {"P": round(prec, 3), "R": round(rec, 3), "F1": round(f1, 3),
                    "tp": tp, "fp": fp, "fn": fn,
                    "type_acc": round(type_ok / type_tot, 3) if type_tot else None}
        lines.append(f"{arm}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
                     f"(tp{tp}/fp{fp}/fn{fn}) type_acc={prf[arm]['type_acc']}")
    # 枚举 silver 可靠性：A vs C（27B 子集复标；B=网关 flash 枚举超时不可用）
    sc = {}
    pc = os.path.join(out, "silver_C.jsonl")
    if os.path.exists(pc):
        for l in open(pc, encoding="utf-8"):
            r = json.loads(l)
            sc[r["doc_id"]] = {_norm_term(t["surface"]) for t in r.get("terms", [])}
    jacc = []
    for did, terms_c in sc.items():
        sa = set(silver.get("A", {}).get(did, {}))
        if sa or terms_c:
            jacc.append(len(sa & terms_c) / len(sa | terms_c))
    if jacc:
        lines.append(f"silver A-C term-set Jaccard mean="
                     f"{statistics.mean(jacc):.3f} (n={len(jacc)}; "
                     f"B=flash 枚举任务推理超时，仅判读短任务可用)")
    # 判读一致性 + 终判
    lines.append("")
    lines.append("## 术语性判读（A/B 双盲 + C 仲裁）")
    verdict = {}
    for arm in ("schema", "freq", "cvalue", "llmdirect", "threesource"):
        p = os.path.join(out, f"judge_{arm}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        arb = {}
        pa = os.path.join(out, f"judge_{arm}_arbit.jsonl")
        if os.path.exists(pa):
            arb = {r["idx"]: r["jc"] for r in
                   (json.loads(l) for l in open(pa, encoding="utf-8"))}
        n = ok = both_parse = agree_n = agree_ok = 0
        by_stratum = collections.defaultdict(lambda: [0, 0])
        for i, r in enumerate(rows):
            a, b = r.get("jA"), r.get("jB")
            va = a.get("match") if a else None
            vb = b.get("match") if b else None
            if va is not None and vb is not None:
                both_parse += 1
                if va == vb:
                    agree_n += 1
                    if va:
                        agree_ok += 1
                    final = va
                else:
                    c = arb.get(i) or {}
                    final = c.get("match")
            else:
                final = va if va is not None else vb
            if final is None:
                continue
            n += 1
            if final:
                ok += 1
            by_stratum[r["stratum"]][1] += 1
            if final:
                by_stratum[r["stratum"]][0] += 1
        kappa = None
        if both_parse:
            # Cohen's kappa（A vs B，可解析子集）
            pass  # 简化为一致率（kappa 需边际，下版补）
        verdict[arm] = {"n": n, "valid": ok,
                        "rate": round(ok / n, 3) if n else None,
                        "AB_agree": round(agree_n / both_parse, 3) if both_parse else None,
                        "strata": {k: f"{v[0]}/{v[1]}" for k, v in by_stratum.items()}}
        lines.append(f"{arm}: valid {ok}/{n} = {verdict[arm]['rate']} | "
                     f"A-B agree {verdict[arm]['AB_agree']} | "
                     f"strata {verdict[arm]['strata']}")
    scan_agg = json.load(open(os.path.join(out, "scan_agg.json"))) \
        if os.path.exists(os.path.join(out, "scan_agg.json")) else {}
    out_doc = {"prf": prf, "verdict": verdict, "scan": scan_agg}
    json.dump(out_doc, open(os.path.join(out, "tables.json"), "w"),
              ensure_ascii=False, indent=1)
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "tables.md"), "w", encoding="utf-8").write(txt + "\n")
