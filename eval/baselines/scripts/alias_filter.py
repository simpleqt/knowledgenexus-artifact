#!/usr/bin/env python3
"""① 别名 manifest 质量过滤（202609-30）：LLM 产物的三层保守过滤。

规则（抽样实证驱动）：
  1) FULL 整类砍——同义改写漂移面最大（effective dose→effective dose
     equivalent 是不同物理量；compton suppression→...spectroscopy 是窄化）；
  2) ABBREV：len≥3 且语料实证（OpenSearch knc-chunk 至少 1 块含该词）——
     杜撰缩写（SFR/TPO/DE 类）在语料中不存在，客观可滤；
  3) TRANS：仅跨语系对（CJK↔ASCII）——同语系"改写"即漂移
     （无氧铜→电解铜 是不同材料）；查询侧中文面形是核心价值（语料为英文），
     故不做语料实证；
  4) 通用：砍纯复数/包含式变体（canonical 与 alias 互为子串）、≤3 条/实体。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  "$R/.venv/bin/python" "$R/eval/baselines/scripts/alias_filter.py"'
产物：results/20260930_aliasx/manifest_filtered.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
SRC = REPO / "eval/baselines/results/20260930_aliasx/manifest.json"
OUT = REPO / "eval/baselines/results/20260930_aliasx/manifest_filtered.json"
CJK = re.compile(r"[\u4e00-\u9fff]")


def corpus_has(term: str) -> bool:
    from knc_engine import search as S
    try:
        r = S.client().search(index=S._CHUNK_ALIAS, body={
            "size": 1, "query": {"match_phrase": {"content_text": {"query": term}}},
            "_source": False, "track_total_hits": True})
        return r["hits"]["total"]["value"] >= 1
    except Exception:
        return False


def main() -> None:
    m = json.load(open(SRC, encoding="utf-8"))
    out_items, kept, dropped = [], 0, 0
    for it in m["items"]:
        name = it["name"].lower()
        keep = []
        for a in it["aliases"]:
            al = a["alias"].strip()
            lo = al.lower()
            if lo in name or name in lo:          # 包含式/复数变体
                dropped += 1
                continue
            if a["type"] == "FULL":
                dropped += 1
                continue
            if a["type"] == "ABBREV":
                if len(al) < 3 or not corpus_has(al):
                    dropped += 1
                    continue
            if a["type"] == "TRANS":
                if bool(CJK.search(al)) == bool(CJK.search(it["name"])):
                    dropped += 1
                    continue
            keep.append(a)
        keep = keep[:3]
        kept += len(keep)
        if keep:
            out_items.append({**it, "aliases": keep})
    json.dump({"kb": m["kb"], "source": m["source"], "items": out_items},
              open(OUT, "w"), ensure_ascii=False, indent=1)
    import collections
    tc = collections.Counter(a["type"] for it in out_items for a in it["aliases"])
    print(f"过滤后 {len(out_items)} 实体 / {kept} 别名（弃 {dropped}） 类型={dict(tc)}")
    for it in out_items[:15]:
        print("  保留样例:", it["name"], "->", [a["alias"] for a in it["aliases"]])


if __name__ == "__main__":
    main()
