#!/usr/bin/env python3
"""复审 P1-a/P1-b：统一类型基线 + 中文词条翻译对照（2026-10-13）。

背景：跨域等规模 freq/cvalue 臂的实体类型是 rng.choice 随机分配的，
判官盲判 prompt 里带 etype——类型分配质量与候选选择混杂。本脚本：

parts:
  xdomain   冻结 freq/cvalue 臂术语集不变，仅把随机类型换成与三源
            classify 阶段同一 LLM 分类器（G._classify_batch，同 prompt
            同模型）；重扫（预期扫描指标不变，实证）+ 双判官盲判 +
            C 仲裁 → 新有效率。
  kb1       KB1 预算扫描 freq-top/cvalue-top 臂的 "term" 退化类型换
            LLM 分类（nuclear_electronics 类型表），重跑五档扫描。
  translate CS schema 中文词条 LLM 译英并入 en_aligned（对照删除式
            对齐）→ 三臂扫描 + 逐词新增命中 + 双标参照召回。

断点：所有 LLM 分类结果落 components 缓存；judge/arbit jsonl 幂等追加。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "corpus"))
sys.path.insert(0, os.path.join(REPO, "eval", "baselines", "scripts"))
import gen_term_lexicons as G  # noqa: E402
from lexicon_isolated_compare import precompile, scan  # noqa: E402
import lexicon_cross_domain as XD  # noqa: E402
from lexicon_cross_domain import (chat, JUDGES, JUDGE_PROMPT,  # noqa: E402
                                  load_env_file, stratified_sample)

XD_OUT = os.path.join(REPO, "eval", "baselines", "results",
                      "20261011_cross_domain")
OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261013_unified_typing")
DATA = os.path.join(REPO, "eval", "baselines", "data", "public_corpora")
SCHEMA = {"cs": "computer_science.yaml", "bio": "medicine.yaml",
          "mat": "materials_science.yaml"}
DOMS = ("cs", "bio", "mat")


def dom_types(dom):
    import yaml
    doc = yaml.safe_load(open(os.path.join(REPO, "schemas", SCHEMA[dom]),
                              encoding="utf-8"))
    ets = doc.get("entity_types", {})
    return {t: s for t, s in ets.items()
            if t not in ("person", "paper", "organization")
            and t not in G.CATCHALL_TYPES}


def load_frame(dom):
    out = os.path.join(XD_OUT, dom)
    rows = {r["doc_id"]: r["text"] for r in
            (json.loads(l) for l in open(os.path.join(DATA, f"{dom}.jsonl"),
                                         encoding="utf-8"))}
    frame = json.load(open(os.path.join(out, "frame.json")))
    return out, rows, frame


def scan_agg(rows_text, sample_ids, lex):
    pats = precompile(lex)
    links = collections.Counter()
    doc_ents = collections.defaultdict(set)
    for i in sample_ids:
        for surface, etype, s, e in scan(rows_text[i], pats):
            links[(surface.lower(), etype)] += 1
            doc_ents[i].add((surface.lower(), etype))
    total = sum(links.values())
    top10 = sorted(links.values(), reverse=True)[:10]
    per_doc = [len(v) for v in doc_ents.values()]
    head = sum(n for (f, t), n in links.items() if f in XD.GENERIC_HEAD)
    return {"links": round(total, 1), "entities": len(links),
            "c10": round(sum(top10) / total, 4) if total else 0.0,
            "conc_per_doc": round(statistics.mean(per_doc), 2) if per_doc else 0,
            "generic_head_share": round(head / total, 4) if total else 0.0,
            "docs": len(doc_ents)}


def classify_terms(terms, types_desc, cache_path):
    """与三源 classify 同一分类器；结果缓存幂等。"""
    if os.path.exists(cache_path):
        return json.load(open(cache_path, encoding="utf-8"))
    mapping = {}
    terms = list(terms)
    for i in range(0, len(terms), 300):
        batch = terms[i:i + 300]
        got = G._classify_batch(batch, types_desc)
        for tid, ts in got.items():
            for t in ts:
                mapping[t] = tid
        print(f"  classify {min(i+300, len(terms))}/{len(terms)} "
              f"(typed {len(mapping)})", flush=True)
        time.sleep(0.3)
    json.dump(mapping, open(cache_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return mapping


def judge_arm(dom, out, rows_txt, arm_name, lex, all_ids, n_per=20):
    """分层抽样 × 双判官 × C 仲裁——复用跨域机器，rng 种子换 20261013。
    输出文件名带域前缀（judge_{dom}_{arm}.jsonl）——三域同名臂不得互叠。"""
    from concurrent.futures import ThreadPoolExecutor
    jpath = os.path.join(out, f"judge_{dom}_{arm_name}.jsonl")
    apath = os.path.join(out, f"arbit_{dom}_{arm_name}.jsonl")
    if not os.path.exists(jpath):
        pats = precompile(lex)
        hits_by_ent = collections.defaultdict(list)
        rng = random.Random(f"20261013-{dom}-judge")
        pool_ids = rng.sample(all_ids, 4000)
        for i in pool_ids:
            text = rows_txt[i]
            for surface, etype, s, e in scan(text, pats):
                hits_by_ent[(surface.lower(), etype)].append(
                    {"surface": surface, "doc_id": i, "s": s, "e": e})
        rows = stratified_sample(hits_by_ent, n_per)

        def work(item):
            _i, r = item
            text = rows_txt[r["doc_id"]]
            s, e = r["s"], r["e"]
            if text[s:e] != r["surface"]:
                j = text.find(r["surface"])
                if j < 0:
                    return
                s, e = j, j + len(r["surface"])
            ctx = text[max(0, s - 250):s] + "⟦" + text[s:e] + "⟧" + \
                text[e:e + 250]
            rec = {"arm": f"{dom}:{arm_name}", "ctx": ctx, **r}
            for j, cfg in (("A", JUDGES["A"]), ("B", JUDGES["B"])):
                raw = chat(cfg, JUDGE_PROMPT.format(
                    name=r["entity"], etype=r["etype"],
                    surface=r["surface"], ctx=ctx),
                    max_tokens=3000 if j == "B" else 400)
                d = G._json_obj(raw) or {}
                rec[f"j{j}"] = {"match": d.get("match"),
                                "specificity": d.get("specificity"),
                                "reason": str(d.get("reason", ""))[:150]}
            with open(jpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            time.sleep(0.2)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(work, list(enumerate(rows))))
    rows = [json.loads(l) for l in open(jpath, encoding="utf-8")]

    # 仲裁（与 stage_arbit 同规则）
    done = set()
    if os.path.exists(apath):
        for l in open(apath, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["entity"], r["doc_id"], r["s"], r["e"]))
    ENV_MAIN = load_env_file(os.path.join(REPO, "deploy", ".env"))
    jc = {"name": "deepseek-flash", "base": ENV_MAIN.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
          "key": os.environ.get("DEEPSEEK_API_KEY",
                                ENV_MAIN.get("DEEPSEEK_API_KEY", "none")),
          "model": "deepseek-flash"}
    with open(apath, "a", encoding="utf-8") as f:
        for r in rows:
            key = (r["entity"], r["doc_id"], r["s"], r["e"])
            if key in done:
                continue
            a = (r.get("jA") or {}).get("match")
            b = (r.get("jB") or {}).get("match")
            if a is not None and a == b:
                rec = {**r, "jC": None, "final": a, "mode": "AB-agree"}
            else:
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
    final = [json.loads(l)["final"] for l in open(apath, encoding="utf-8")]
    final = [x for x in final if x is not None]
    return {"rate": round(sum(final) / len(final), 3) if final else None,
            "n": len(final)}


def part_xdomain():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for dom in DOMS:
        out = os.path.join(XD_OUT, dom)
        comps = os.path.join(out, "components")
        rows_txt, frame = load_frame(dom)[1], load_frame(dom)[2]
        all_ids = [i for ids in frame["sample_ids"] for i in ids]
        types_desc = dom_types(dom)
        res[dom] = {}
        for base_arm in ("freq", "cvalue"):
            arm = json.load(open(os.path.join(comps, f"arm_{base_arm}.json"),
                                 encoding="utf-8"))
            cache = os.path.join(OUT, f"classify_{dom}_{base_arm}.json")
            mapping = classify_terms(sorted(arm), types_desc, cache)
            llm_arm = {t: mapping.get(t, "unclassified") for t in arm}
            typed = sum(1 for t in llm_arm if llm_arm[t] != "unclassified")
            dis = sum(1 for t in llm_arm
                      if llm_arm[t] != "unclassified" and arm[t] != llm_arm[t])
            name = f"{base_arm}LLM"
            scan_m = {"mean": scan_agg(rows_txt,
                                      frame["sample_ids"][0], llm_arm)}
            verdict = judge_arm(dom, OUT, rows_txt, name, llm_arm, all_ids)
            res[dom][name] = {
                "terms": len(llm_arm), "typed": typed,
                "unclassified": len(llm_arm) - typed,
                "rand_type_disagree_with_llm": dis,
                "disagree_of_typed": round(dis / max(1, typed), 3),
                "scan": scan_m, "verdict": verdict}
            json.dump(llm_arm, open(os.path.join(
                OUT, f"arm_{name}_{dom}.json"), "w", encoding="utf-8"),
                ensure_ascii=False, indent=1)
            print(f"[{dom}] {name}: typed {typed}/{len(llm_arm)} "
                  f"disagree {dis} verdict {verdict}", flush=True)
    json.dump(res, open(os.path.join(OUT, "unified_typing.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[xdomain] unified_typing.json written")


def part_kb1():
    from lexicon_isolated_compare import load_lexicon
    from lexicon_budget_curve import fetch_sample_chunks, metrics, kb1_recall
    K_LIST = [200, 500, 1000, 2000, 3733]
    DA = os.path.join(REPO, "eval", "baselines", "results",
                      "20261012_dual_anno")
    v21 = load_lexicon(os.path.join(REPO, "schemas",
                                    "nuclear_electronics.yaml"))
    types_desc = dom_types_kb1()
    samples = fetch_sample_chunks()
    from lexicon_cross_domain import text_freq, cvalue_scores
    docs = [{"text": t} for _d, t in samples[0]]
    freq = text_freq(docs)
    cv = {t: s for t, s in cvalue_scores(docs)}

    def G_ok(t):
        return G._ok_term(t)

    freq_order = [t for t, _ in sorted(freq.items(), key=lambda kv: -kv[1])
                  if G_ok(t)]
    cv_order = [t for t, _ in sorted(cv.items(), key=lambda kv: -kv[1])
                if G_ok(t)]
    terms = sorted(set(freq_order[:3733]) | set(cv_order[:554]))
    cache = os.path.join(OUT, "classify_kb1.json")
    mapping = classify_terms(terms, types_desc, cache)

    units = json.load(open(os.path.join(DA, "units.json"), encoding="utf-8"))
    kb_units = [u for u in units if u["unit"].startswith("kb1:")]
    adj = {r["unit"]: r["terms"] for r in (json.loads(l) for l in open(
        os.path.join(DA, "adjudicated.jsonl"), encoding="utf-8"))}

    res = {}
    for K in K_LIST:
        arms = {}
        arms["freq-top-LLM"] = {t: mapping.get(t, "unclassified")
                                for t in freq_order[:K]}
        arms["cvalue-top-LLM"] = {t: mapping.get(t, "unclassified")
                                  for t in cv_order[:K]}
        row = {}
        for an, lex in arms.items():
            per = [metrics(ch, lex) for ch in samples]
            row[an] = {k: round(statistics.mean(p[k] for p in per), 2)
                       for k in ("links", "entities", "c10", "docs",
                                 "conc_per_doc")}
            row[an]["R_dual"] = kb1_recall(lex, kb_units, adj)
        res[f"K={K}"] = row
        print(f"[kb1] K={K}: " + " | ".join(
            f"{an} ents={row[an]['entities']} C10={row[an]['c10']} "
            f"R={row[an]['R_dual']}" for an in row), flush=True)
    json.dump(res, open(os.path.join(OUT, "kb1_budget_curve_llmtypes.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def dom_types_kb1():
    import yaml
    doc = yaml.safe_load(open(os.path.join(REPO, "schemas",
                                           "nuclear_electronics.yaml"),
                              encoding="utf-8"))
    ets = doc.get("entity_types", {})
    return {t: s for t, s in ets.items()
            if t not in ("person", "paper", "organization")
            and t not in G.CATCHALL_TYPES}


def part_translate():
    import yaml
    # 本地复刻 lexicon_p0_controls.scan_metrics/load_cs_frame（避免 import
    # app.main 拉起 retrieval-api 应用对象）
    def scan_metrics(rows_text, sample_ids, lex):
        pats = precompile(lex)
        links = collections.Counter()
        doc_ents = collections.defaultdict(set)
        for i in sample_ids:
            for surface, etype, s, e in scan(rows_text[i], pats):
                links[(surface.lower(), etype)] += 1
                doc_ents[i].add((surface.lower(), etype))
        total = sum(links.values())
        top10 = sorted(links.values(), reverse=True)[:10]
        per_doc = [len(v) for v in doc_ents.values()]
        head = sum(n for (f, t), n in links.items() if f in XD.GENERIC_HEAD)
        return {"links": total, "entities": len(links),
                "c10": round(sum(top10) / total, 3) if total else 0.0,
                "conc_per_doc": round(statistics.mean(per_doc), 2)
                if per_doc else 0,
                "generic_head_share": round(head / total, 3) if total else 0.0,
                "docs": len(doc_ents)}

    XD_CS = os.path.join(XD_OUT, "cs")
    rows = {r["doc_id"]: r["text"] for r in
            (json.loads(l) for l in open(os.path.join(DATA, "cs.jsonl"),
                                         encoding="utf-8"))}
    frame = json.load(open(os.path.join(XD_CS, "frame.json")))
    doc = yaml.safe_load(open(os.path.join(REPO, "schemas",
                                           "computer_science.yaml"),
                              encoding="utf-8"))
    zh = []   # (alias, type)
    raw = {}
    aligned = {}
    for ty, spec in doc.get("entity_types", {}).items():
        for a in [spec.get("name")] + list(spec.get("aliases") or []):
            if not a:
                continue
            raw[a] = ty
            if a.isascii():
                low = a.lower()
                aligned[low] = ty
                if low.endswith("s") and len(low) > 3:
                    aligned.setdefault(low[:-1], ty)
            else:
                zh.append((a, ty))
    cache = os.path.join(OUT, "zh_translation.json")
    if os.path.exists(cache):
        mapping = json.load(open(cache, encoding="utf-8"))
    else:
        listing = "\n".join(f"{i}. {a}" for i, (a, _t) in enumerate(zh))
        raw_out = G._llm_chat(
            f"把下列计算机领域中文术语译成标准英文术语（一个即可，最通用的"
            f"英文说法；专有名词保持原文）。\n输出 JSON："
            f'{{"译名": ["英文术语", ...]}}，与输入序号一一对应。\n\n{listing}',
            max_tokens=6000)
        d = G._json_obj(raw_out) or {}
        vals = d.get("译名") or []
        mapping = {}
        for i, (a, ty) in enumerate(zh):
            if i < len(vals) and isinstance(vals[i], str) and vals[i].strip():
                en = vals[i].strip()
                if en.isascii():
                    mapping[a] = en
        json.dump(mapping, open(cache, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    translated = dict(aligned)
    for a, en in mapping.items():
        low = en.lower()
        translated.setdefault(low, dict(raw)[a])
    res = {}
    for tag, lex in (("schema_raw", raw), ("schema_en_aligned", aligned),
                     ("schema_en_translated", translated)):
        per = [scan_metrics(rows, ids, lex) for ids in frame["sample_ids"]]
        agg = {k: {"mean": round(statistics.mean(p[k] for p in per), 3),
                   "sd": round(statistics.stdev(p[k] for p in per), 3)}
               for k in ("links", "entities", "c10", "conc_per_doc",
                         "generic_head_share", "docs")}
        res[tag] = agg
        print(f"[translate] {tag}: {agg}", flush=True)
    # translated 独有 pattern（译文词条不在 aligned 内）的命中面
    only_tr = {t: ty for t, ty in translated.items() if t not in aligned}
    pats_only = precompile(only_tr)
    hit_counter = collections.Counter()
    doc_hit = collections.defaultdict(set)
    for i in frame["sample_ids"][0]:
        for surface, etype, s, e in scan(rows[i], pats_only):
            hit_counter[(surface.lower(), etype)] += 1
            doc_hit[i].add((surface.lower(), etype))
    res["translated_only_terms"] = {"n_terms": len(only_tr),
                                    "n_hit": len(hit_counter),
                                    "links": sum(hit_counter.values()),
                                    "docs": len(doc_hit),
                                    "top": [{"surface": k[0], "etype": k[1],
                                             "links": v}
                                            for k, v in hit_counter
                                            .most_common(15)]}
    json.dump(res, open(os.path.join(OUT, "cs_translation_control.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[translate] cs_translation_control.json written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["xdomain", "kb1", "translate"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"xdomain": part_xdomain, "kb1": part_kb1,
     "translate": part_translate}[a.part]()


if __name__ == "__main__":
    main()
