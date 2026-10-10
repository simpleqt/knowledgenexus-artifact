#!/usr/bin/env python3
"""O-4 预计算：对三个题集全部查询离线跑一次 LLM 查询理解判定并冻结。

消融注入 req.deep_llm（离线冻结版），保证评测确定性（LLM 温度 0 仍有抖动，
冻结后两臂同判定可严格配对）。产物 verdicts.json {query: {intent, entities}}。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  "$R/.venv/bin/python" "$R/eval/baselines/scripts/deepq_verdicts.py"'
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "eval/baselines/results/20260930_deepq/verdicts.json"
SETS = {
    "hard93": REPO / "eval/baselines/results/20260929_hardtargets/hard_targets_prod.jsonl",
    "nea48": REPO / "eval/harness/ne_journal_assoc_q48.jsonl",
    "realq70": REPO / "eval/harness/ne_journal_realq.jsonl",
}
sys.path.insert(0, str(REPO / "services/retrieval-api"))


def main() -> None:
    queries: list[str] = []
    for p in SETS.values():
        for line in open(p, encoding="utf-8"):
            if line.strip():
                queries.append(json.loads(line)["query"])
    queries = sorted(set(queries))
    verdicts: dict = json.load(open(OUT)) if OUT.exists() else {}
    todo = [q for q in queries if q not in verdicts]
    print(f"待判定 {len(todo)}/{len(queries)}（已冻结 {len(verdicts)}）")
    from app.main import _deep_query_understand
    t0 = time.time()
    for i, q in enumerate(todo):
        verdicts[q] = _deep_query_understand(q)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(todo)} ({(time.time()-t0)/(i+1):.1f}s/q)", flush=True)
            json.dump(verdicts, open(OUT, "w"), ensure_ascii=False, indent=1)
    json.dump(verdicts, open(OUT, "w"), ensure_ascii=False, indent=1)
    n_assoc = sum(1 for v in verdicts.values() if v and v.get("intent") == "assoc")
    n_none = sum(1 for v in verdicts.values() if v is None)
    print(f"冻结 {len(verdicts)} 条：assoc={n_assoc} direct={len(verdicts)-n_assoc-n_none} 解析失败={n_none}")
    for name, p in SETS.items():
        qs = [json.loads(l)["query"] for l in open(p, encoding="utf-8") if l.strip()]
        na = sum(1 for q in qs if verdicts.get(q) and verdicts[q].get("intent") == "assoc")
        nn = sum(1 for q in qs if verdicts.get(q) is None)
        print(f"  {name}: LLM assoc {na}/{len(qs)}，失败 {nn}")


if __name__ == "__main__":
    main()
