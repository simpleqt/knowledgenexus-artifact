#!/usr/bin/env python3
"""词典单变量离线对照（2026-10-10 大修 P0-1/P0-5 王牌实验）：

固定代码与文档语料，仅换词典——v2.0（塌缩期类别词典，139 词条）/
v2.5（09-20 中间态 2,022+）/v2.1（修复版 3,702）三臂 + v2.1 两个治理
变体（短符号大小写敏感 / 英文停用词阻断）。扫描器语义复刻 ingest-worker
的 _term_matches（CJK 子串直配；ASCII 词边界+可选复数 s+忽略大小写），
K2 折叠=（归一化面形, 类型）。指标：提及/实体/链接、top-10 浓度、
每文档 distinct 实体（均值±sd）、链接落在 v2.0 类别词上的质量占比、
LLM 抽样精度（每臂 40 链接，复用审计规则）。
语料：KB1 生产 chunk 确定性抽样 2,000 块（chunk_id 序每 N 取一）。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/lexicon_isolated_compare.py \
  --out eval/baselines/results/20261010_lexicon_iso'
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402  (OS client)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from audit_link_precision import llm_judge, wilson  # noqa: E402

ASCII_RE: dict[str, re.Pattern] = {}
STOP_EN = {"be", "am", "is", "are", "was", "were", "do", "does", "did",
           "if", "it", "an", "as", "at", "by", "or", "so", "no", "not"}


def load_lexicon(path_or_spec: str) -> dict[str, str]:
    if ":" in path_or_spec:
        rev, path = path_or_spec.split(":", 1)
        import subprocess
        raw = subprocess.run(["git", "-C", REPO, "show", f"{rev}:{path}"],
                             capture_output=True, text=True, check=True).stdout
    else:
        raw = open(path_or_spec, encoding="utf-8").read()
    doc = yaml.safe_load(raw)
    lex: dict[str, str] = {}
    for t, spec in doc.get("entity_types", {}).items():
        for alias in (spec.get("aliases") or []):
            a = alias.split(",")[0].strip()  # 'ADC, 模数转换器' 取首名
            if a and a not in lex:
                lex[a] = t
    # extends _base：主名本身也是扫描词（_mention_dict 语义：主名+别名）
    for t, spec in doc.get("entity_types", {}).items():
        lex.setdefault(t, t)
    return lex


def precompile(lex: dict[str, str], case_sensitive_short=False,
               stopword_block=False):
    """每臂预编译一次（首版 caseShort 臂逐 chunk 重复 re.compile 7.4M 次，
    py-spy 定位后重写）。"""
    pats = []
    for term, etype in lex.items():
        if not term.isascii():
            pats.append((term, etype, None))
            continue
        if stopword_block and term.lower() in STOP_EN:
            continue
        flags = 0 if (case_sensitive_short and len(term) <= 3) else re.IGNORECASE
        pats.append((term, etype, re.compile(
            rf"(?<![0-9A-Za-z]){re.escape(term)}s?(?![0-9A-Za-z])", flags)))
    return pats


def scan(text: str, pats) -> list[tuple[str, str, int, int]]:
    out = []
    for term, etype, pat in pats:
        if pat is None:
            start = 0
            while (idx := text.find(term, start)) != -1:
                out.append((term, etype, idx, idx + len(term)))
                start = idx + 1
            continue
        for m in pat.finditer(text):
            out.append((m.group(), etype, m.start(), m.end()))
    return out


def fetch_chunks(n: int, kb_hex: str) -> list[dict]:
    c = M.search.client()
    hits, pit = [], None
    while len(hits) < n:
        body = {"size": 1000, "query": {"bool": {"filter": [
            {"term": {"kb_id": kb_hex}}, {"term": {"visible": True}}]}},
            "_source": ["chunk_id", "doc_id", "content_text"],
            "sort": [{"chunk_id": "asc"}]}
        if pit:
            body["search_after"] = [hits[-1]["_source"]["chunk_id"]]
        r = c.search(index="knc-chunk", body=body)
        batch = r["hits"]["hits"]
        if not batch:
            break
        hits.extend(batch)
    step = max(1, len(hits) // n)
    return hits[::step][:n]


def run_arm(name: str, lex: dict, chunks: list, judge_cfg: dict,
            case_short=False, stop_block=False, n_judge=40) -> dict:
    links: dict[tuple, int] = {}
    doc_ents: dict[str, set] = {}
    v20_mass = 0
    pats = precompile(lex, case_short, stop_block)
    for ch in chunks:
        text = ch["_source"]["content_text"]
        did = ch["_source"]["doc_id"]
        for surface, etype, s, e in scan(text, pats):
            ent = (surface.lower(), etype)
            links[ent] = links.get(ent, 0) + 1
            doc_ents.setdefault(did, set()).add(ent)
    total = sum(links.values())
    top10 = sorted(links.values(), reverse=True)[:10]
    conc = sum(top10) / total if total else 0.0
    per_doc = [len(v) for v in doc_ents.values()]
    top20 = sorted(links.items(), key=lambda kv: -kv[1])[:20]
    # v2.0 类别词质量占比
    cat = {"method", "parameter", "material", "energy", "model", "term",
           "detector", "circuit", "isotope"}
    v20_mass = sum(n for (f, t), n in links.items() if f in
                   {"method", "energy", "parameter", "material", "model",
                    "term", "time", "field", "phase", "temperature"})
    # LLM 抽样精度
    rng = random.Random(20261010)
    sample = rng.sample(list(links.items()), min(n_judge, len(links)))
    ok = dom = domtot = gentot = genok = 0
    judged = []
    for (form, etype), _n in sample:
        v = llm_judge(judge_cfg["base"], judge_cfg["key"], judge_cfg["model"],
                      form, etype, form, f"(dictionary term: '{form}', type {etype})")
        judged.append({"form": form, "type": etype, **v})
        if v["match"]:
            ok += 1
        if v["specificity"] == "domain":
            domtot += 1
            dom += bool(v["match"])
        elif v["specificity"] == "generic":
            gentot += 1
            genok += bool(v["match"])
    lo, hi = wilson(ok, len(sample))
    res = {
        "arm": name, "terms": len(lex), "chunks": len(chunks),
        "mentions_links": total, "entities": len(links),
        "top10_concentration": round(conc, 3),
        "per_doc_distinct_entities_mean": round(statistics.mean(per_doc), 1) if per_doc else 0,
        "per_doc_distinct_entities_sd": round(statistics.pstdev(per_doc), 1) if len(per_doc) > 1 else 0,
        "generic_head_mass_share": round(v20_mass / total, 3) if total else 0,
        "top20": [[f"{f}:{t}", n] for (f, t), n in top20],
        "llm_precision": round(ok / len(sample), 3) if sample else None,
        "llm_precision_ci": [round(lo, 3), round(hi, 3)],
        "llm_domain_stratum": f"{dom}/{domtot}",
        "llm_generic_stratum": f"{genok}/{gentot}",
        "judged_sample": judged,
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="104112e63d9d4aa5a6c21e1c1549ef0f")
    ap.add_argument("--chunks", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    judge_cfg = {"base": os.environ["LLM_API_BASE"],
                 "key": os.environ["LLM_API_KEY"],
                 "model": os.environ["LLM_MODEL"]}
    chunks = fetch_chunks(args.chunks, args.kb)
    print(f"[corpus] {len(chunks)} chunks sampled")
    lexica = {
        "v2.0-collapse": load_lexicon("5ec5756:schemas/nuclear_electronics.yaml"),
        "v2.5-mid": load_lexicon("589af20:schemas/nuclear_electronics.yaml"),
        "v2.1-repair": load_lexicon(os.path.join(REPO, "schemas", "nuclear_electronics.yaml")),
    }
    results = []
    for name, lex in lexica.items():
        print(f"[arm] {name}: {len(lex)} terms")
        results.append(run_arm(name, lex, chunks, judge_cfg))
        print(json.dumps({k: v for k, v in results[-1].items() if k != "judged_sample"},
                         ensure_ascii=False)[:400])
    # 治理变体（v2.1 基础上）
    lex21 = lexica["v2.1-repair"]
    results.append(run_arm("v2.1+caseShort", lex21, chunks, judge_cfg,
                           case_short=True))
    results.append(run_arm("v2.1+stopwordBlock", lex21, chunks, judge_cfg,
                           stop_block=True))
    json.dump(results, open(os.path.join(args.out, "lexicon_iso.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[saved]", os.path.join(args.out, "lexicon_iso.json"))


if __name__ == "__main__":
    main()
