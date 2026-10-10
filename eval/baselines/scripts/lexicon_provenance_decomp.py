#!/usr/bin/env python3
"""P1-8 词法溯源分解：v2.1 隔离臂的链接按词条来源归因。

v2.1 YAML 不带逐词源标注；用词集代数近似：v2.1 命中词条 ∈ v2.0 词集 =
继承（RADAI 类别词），∈ v2.5−v2.0 = 中期批产，仅 ∈ v2.1 = 三源修复新增。
2026-10-10 三轮审查后：取样帧对齐 multisample（--from-multisample 读
random_score 样本的 chunk_id 清单，缺省样本 0）——不再用旧前缀帧
fetch_chunks（保留兼容回退）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lexicon_isolated_compare import fetch_chunks, precompile, scan, load_lexicon  # noqa
from common import REPO

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402  (OS client)

KB = "104112e63d9d4aa5a6c21e1c1549ef0f"


def chunks_from_multisample(path: str, si: int) -> list[dict]:
    d = json.load(open(path))
    ids = d["sample_chunk_ids"][str(si)]
    out = []
    for i in range(0, len(ids), 500):
        batch = ids[i:i + 500]
        for ch in M.search.chunks_by_ids(d["kb"], batch, len(batch)):
            out.append({"_source": ch})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-multisample",
                    default="eval/baselines/results/20261010_lexicon_multisample/multisample.json")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="eval/baselines/results/20261010_lexicon_multisample")
    args = ap.parse_args()
    v20 = load_lexicon("5ec5756:schemas/nuclear_electronics.yaml")
    v25 = load_lexicon("589af20:schemas/nuclear_electronics.yaml")
    v21 = load_lexicon(os.path.join(REPO, "schemas", "nuclear_electronics.yaml"))
    if args.from_multisample and os.path.exists(args.from_multisample):
        chunks = chunks_from_multisample(args.from_multisample, args.sample)
        frame = f"multisample random frame, sample {args.sample}"
    else:
        chunks = fetch_chunks(2000, KB)
        frame = "legacy ordered-prefix frame"
    print(f"[frame] {frame}: {len(chunks)} chunks")

    v20l = {x.lower() for x in v20}
    v25l = {x.lower() for x in v25}

    def bucket(term: str) -> str:
        t = term.lower()
        if t in v20l:
            return "v2.0-inherited"
        if t in v25l:
            return "v2.5-mid"
        return "v2.1-new"

    pats = precompile(v21)
    links: dict[tuple, int] = {}
    for ch in chunks:
        for surface, etype, _s, _e in scan(ch["_source"]["content_text"], pats):
            k = (surface.lower(), etype)
            links[k] = links.get(k, 0) + 1
    tot = sum(links.values())
    res = {"frame": frame, "chunks": len(chunks), "total_links": tot,
           "buckets": {}}
    for b in ("v2.0-inherited", "v2.5-mid", "v2.1-new"):
        L = sum(n for (f, _t), n in links.items() if bucket(f) == b)
        E = sum(1 for (f, _t) in links if bucket(f) == b)
        res["buckets"][b] = {"links": L, "link_share": round(L / tot, 3),
                             "entities": E}
    json.dump(res, open(os.path.join(args.out, "provenance.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
