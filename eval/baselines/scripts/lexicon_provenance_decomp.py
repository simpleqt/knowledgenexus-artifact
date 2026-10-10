#!/usr/bin/env python3
"""P1-8 词法溯源分解：v2.1 隔离臂的链接按词条来源归因。

v2.1 YAML 不带逐词源标注；用词集代数近似：v2.1 命中词条 ∈ v2.0 词集 =
继承（RADAI 类别词），∈ v2.5−v2.0 = 中期批产，仅 ∈ v2.1 = 三源修复新增。
对 lexicon_iso.json 的 v2.1-repair 臂 top20/全量链接无逐词计数——重扫一遍
v2.1（同 lexicon_isolated_compare 的确定性 chunk 采样）按来源桶计链接与实体。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lexicon_isolated_compare import fetch_chunks, precompile, scan, load_lexicon  # noqa
from common import REPO

OUT = "eval/baselines/results/20261010_lexicon_iso"


def main() -> None:
    v20 = load_lexicon("5ec5756:schemas/nuclear_electronics.yaml")
    v25 = load_lexicon("589af20:schemas/nuclear_electronics.yaml")
    v21 = load_lexicon(os.path.join(REPO, "schemas", "nuclear_electronics.yaml"))
    chunks = fetch_chunks(2000, "104112e63d9d4aa5a6c21e1c1549ef0f")

    def bucket(term: str) -> str:
        t = term.lower()
        if t in {x.lower() for x in v20}:
            return "v2.0-inherited"
        if t in {x.lower() for x in v25}:
            return "v2.5-mid"
        return "v2.1-new"

    pats = precompile(v21)
    links: dict[tuple, int] = {}
    for ch in chunks:
        for surface, etype, _s, _e in scan(ch["_source"]["content_text"], pats):
            k = (surface.lower(), etype)
            links[k] = links.get(k, 0) + 1
    tot = sum(links.values())
    res = {"total_links": tot, "buckets": {}}
    for b in ("v2.0-inherited", "v2.5-mid", "v2.1-new"):
        L = sum(n for (f, _t), n in links.items() if bucket(f) == b)
        E = sum(1 for (f, _t) in links if bucket(f) == b)
        res["buckets"][b] = {"links": L, "link_share": round(L / tot, 3),
                             "entities": E}
    json.dump(res, open(os.path.join(OUT, "provenance.json"), "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
