#!/usr/bin/env python3
"""replica 扫描器 vs 生产 K1 扫描器 parity 对照（2026-10-12 复审
R1-P3-M3 落地）：冻结 chunk 清单 + 同一生产词典，比较两扫描器的
命中面形/span/归一词/类型。

生产实现 = services/ingest-worker/app/main.py（dict-v3 合并扫描器：
ASCII 词边界+可选 s+IGNORECASE 长词优先 alternation；CJK 直配
alternation 非重叠）。replica = lexicon_isolated_compare（逐词条
独立匹配，str.find/正则可重叠）。预期分歧类=嵌套/重叠重复计数
（replica ⊇ 生产）；本脚本量化一致率并归因。
语料：20261010 multisample 冻结样本 0 的前 N 块（KB1，只读 OS）。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/lexicon_scanner_parity.py --n 200'
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import precompile, scan  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402  (OS client 只读)

KB = "104112e63d9d4aa5a6c21e1c1549ef0f"
OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261012_revision")


def load_prod_scanner():
    spec = importlib.util.spec_from_file_location(
        "ingest_main", os.path.join(REPO, "services", "ingest-worker",
                                    "app", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--kb", default=KB)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    prod = load_prod_scanner()
    lex = prod._mention_dict(args.kb)  # 生产合成词典（schema 三层 ∪ 种子）
    a_pat, c_pat, _ = prod._combined_scanner(args.kb)

    ms = json.load(open(os.path.join(REPO, "eval", "baselines", "results",
                                     "20261010_lexicon_multisample",
                                     "multisample.json")))
    ids = ms["sample_chunk_ids"]["0"][: args.n]
    c = M.search.client()
    rows = []
    for i in range(0, len(ids), 100):
        r = c.mget(index="knc-chunk",
                   body={"docs": [{"_id": i, "_source": ["content_text"]}
                                  for i in ids[i:i + 100]]})
        for h in r["docs"]:
            if h.get("found") and h["_source"].get("content_text"):
                rows.append((h["_id"], h["_source"]["content_text"]))
    print(f"[corpus] {len(rows)}/{len(ids)} frozen chunks fetched")

    def prod_hits(text):
        out = []
        for pat, ascii_mode in ((c_pat, False), (a_pat, True)):
            if pat is None:
                continue
            for m in pat.finditer(text):
                sf = m.group()
                term = sf[:-1] if (ascii_mode and sf.endswith("s")) else sf
                th = lex.get(term) or lex.get(sf)
                if th is None:
                    continue
                out.append((term.lower(), m.start(), m.end(), sf, th))
        return out

    pats = precompile(lex)
    rep_total = prod_total = both = 0
    span_cov = 0
    extra_nested = extra_crossing = keydiff_same_span = 0
    rep_only = collections.Counter()   # (term_lower) 计数
    examples = collections.defaultdict(list)
    for cid, text in rows:
        ph_list = prod_hits(text)
        # 逐行镜像 lexicon_isolated_compare.scan() 的循环（仅多携带 term
        # 身份；scan() 本身不返回词条，此处比较对象需要它）
        rh_list = []
        for term, etype, pat in pats:
            if pat is None:
                start = 0
                while (idx := text.find(term, start)) != -1:
                    rh_list.append((term.lower(), idx, idx + len(term),
                                    term, etype))
                    start = idx + 1
                continue
            for m in pat.finditer(text):
                rh_list.append((term.lower(), m.start(), m.end(),
                                m.group(), etype))
        ph = collections.Counter(ph_list)
        rh = collections.Counter(rh_list)
        rep_total += sum(rh.values())
        prod_total += sum(ph.values())
        both += sum((ph & rh).values())
        # span 级：生产 span 是否全被 replica 覆盖；replica 多出 span 的
        # 归因（嵌套于/跨越生产更长匹配 = 合并 alternation 设计性抑制）
        pspans = {(p[1], p[2]) for p in ph_list}
        rspans = {(r[1], r[2]) for r in rh_list}
        span_cov += len(pspans & rspans)
        for (s, e) in (rspans - pspans):
            if any(ps <= s and e <= pe for (ps, pe) in pspans):
                extra_nested += 1
            else:
                extra_crossing += 1
        # 同 span 不同键（归一词推导差：尾 s 剥除对本身以 s 结尾词条）
        pkeys = {((p[1], p[2]), p[3], p[4]) for p in ph_list}
        rkeys = {((r[1], r[2]), r[3], r[4]) for r in rh_list}
        keydiff_same_span += len({k[0] for k in (pkeys ^ rkeys)}
                                 & (pspans | rspans))
        for k, n in (rh - ph).items():
            rep_only[k[0]] += n
            if len(examples["rep_only"]) < 8:
                examples["rep_only"].append(
                    {"chunk": cid[:8], "surface": k[3], "term": k[0],
                     "span": [k[1], k[2]],
                     "context": text[max(0, k[1] - 40):k[2] + 40]})

    agree = both / max(rep_total, 1)
    report = {
        "lexicon_size": len(lex), "chunks": len(rows),
        "frozen_source": "20261010_lexicon_multisample sample 0 前 N 块",
        "prod_hits": prod_total, "replica_hits": rep_total,
        "exact_multiset_matches": both,
        "agreement_on_replica": round(agree, 4),
        "agreement_on_prod": round(both / max(prod_total, 1), 4),
        "span_level_prod_covered_by_replica": span_cov,
        "span_level_coverage": round(span_cov / max(prod_total, 1), 4),
        "replica_extra_nested_in_prod_span": extra_nested,
        "replica_extra_crossing_or_absent": extra_crossing,
        "same_span_normalized_key_diff": keydiff_same_span,
        "divergence_attribution": {
            "nested_crossing": "生产 dict-v3 合并 alternation 长词优先+非重叠"
                               "——嵌套/交叠短词设计性不重复计数；replica 逐词"
                               "扫描全部报出（span 级超集）",
            "normalized_key": "尾 s 剥除规则对本身以 s 结尾的词条（analysis→"
                              "analysi）——簿记差非匹配差"},
        "replica_only_top": rep_only.most_common(10),
        "examples": dict(examples)}
    json.dump(report, open(os.path.join(OUT, "scanner_parity.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("examples", "divergence_attribution")},
                     ensure_ascii=False, indent=1))
    print("[saved]", os.path.join(OUT, "scanner_parity.json"))


if __name__ == "__main__":
    main()
