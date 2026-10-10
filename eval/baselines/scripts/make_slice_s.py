#!/usr/bin/env python3
"""派生 Slice-S（HippoRAG 等高建库成本基线用的小切片）。

构成：按文件序贪心选 nea 题，直到 gold 并集 ≥ N_GOLD；再加 N_FILL 篇补充文档。
产物：data/slice_s_doc_ids.txt、data/nea_S.jsonl（gold 完全被覆盖的题）。
用法：.venv/bin/python make_slice_s.py   （只读 data/ 下切片L产物，不访问 OS）
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, REPO, load_docs, load_questions

N_GOLD = 60
N_FILL = 20
SEED = 20260923


def main() -> None:
    qs = load_questions(os.path.join(REPO, "eval", "harness", "ne_journal_assoc_q48.jsonl"))
    docs = load_docs()
    all_ids = {d["doc_id"] for d in docs}
    chosen_q, gold_s = [], set()
    for q in qs:
        if len(gold_s) >= N_GOLD:
            break
        if set(q["gold_doc_ids"]) <= all_ids:
            chosen_q.append(q)
            gold_s.update(q["gold_doc_ids"])
    slice_l = [l.strip() for l in open(os.path.join(DATA, "slice_doc_ids.txt")) if l.strip()]
    rng = random.Random(SEED)
    pool = sorted(set(slice_l) - gold_s)
    fillers = rng.sample(pool, N_FILL)
    slice_ids = sorted(gold_s | set(fillers))

    with open(os.path.join(DATA, "slice_s_doc_ids.txt"), "w") as f:
        f.write("\n".join(slice_ids) + "\n")
    with open(os.path.join(DATA, "nea_S.jsonl"), "w", encoding="utf-8") as f:
        for q in chosen_q:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    kept = [d for d in docs if d["doc_id"] in set(slice_ids)]
    n_chunks = 0
    for line in open(os.path.join(DATA, "corpus_chunks.jsonl"), encoding="utf-8"):
        if json.loads(line)["doc_id"] in set(slice_ids):
            n_chunks += 1
    chars = sum(len(d["text"]) for d in kept)
    print(f"Slice-S: {len(slice_ids)} 篇（gold {len(gold_s)} + 补充 {len(fillers)}），"
          f"{n_chunks} 块，{chars/1e6:.1f}M 字符，题集 nea_S={len(chosen_q)} 题")


if __name__ == "__main__":
    main()
