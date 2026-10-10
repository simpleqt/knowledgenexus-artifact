#!/usr/bin/env python3
"""难题集画像（2026-09-29）：生产只读 API 上 base 与 deep 双臂首金名次。

用法：.venv/bin/python hard_profile.py [--in hard_targets_prod.jsonl]
产物：同目录 profile.json + stdout 摘要。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
HERE = Path(__file__).resolve().parent.parent / "results" / "20260929_hardtargets"


def rank(url: str, q: str, gold: str, tk: int = 20):
    req = urllib.request.Request(url, data=json.dumps(
        {"kb_id": KB, "query": q, "top_k": tk}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
        chunks = json.loads(r.read())["chunks"]
    return next((i for i, c in enumerate(chunks, 1) if c.get("doc_id") == gold), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=str(HERE / "hard_targets_prod.jsonl"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.infile, encoding="utf-8") if l.strip()]
    out = []
    for i, r in enumerate(rows):
        gold = r["gold_doc_ids"][0]
        t0 = time.time()
        b = rank("http://127.0.0.1:18121/api/retrieve", r["query"], gold)
        tb = time.time() - t0
        t0 = time.time()
        d = rank("http://127.0.0.1:18121/api/retrieve/deep", r["query"], gold)
        td = time.time() - t0
        out.append({**r, "base_rank": b, "deep_rank": d,
                    "lat_base": round(tb, 1), "lat_deep": round(td, 1)})
        if (i + 1) % 5 == 0:
            print(f"{i+1}/{len(rows)}", flush=True)
    json.dump(out, open(HERE / "profile.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n = len(out)

    def h(k, t):
        return sum(1 for r in out if (r[k] or 99) <= t) / n

    rb = [r["base_rank"] for r in out if r["base_rank"]]
    rd = [r["deep_rank"] for r in out if r["deep_rank"]]
    print(f"n={n}")
    print(f"base: hit@10={h('base_rank',10):.3f} hit@20={h('base_rank',20):.3f} "
          f"20内可见={len(rb)} 中位={statistics.median(rb) if rb else None}")
    print(f"deep: hit@10={h('deep_rank',10):.3f} hit@20={h('deep_rank',20):.3f} "
          f"20内可见={len(rd)} 中位={statistics.median(rd) if rd else None}")
    print(f"lat p50: base={sorted(r['lat_base'] for r in out)[n//2]}s "
          f"deep={sorted(r['lat_deep'] for r in out)[n//2]}s")


if __name__ == "__main__":
    main()
