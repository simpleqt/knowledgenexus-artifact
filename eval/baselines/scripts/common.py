#!/usr/bin/env python3
"""基线对比公共库：端点配置加载（不打印密钥）、语料/题集 IO、预测文件写出。"""
from __future__ import annotations

import json
import os
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASE = os.path.join(REPO, "eval", "baselines")
DATA = os.path.join(BASE, "data")
RUNS = os.path.join(BASE, "runs")
INDEXES = os.path.join(BASE, "indexes")
NE_JOURNAL_KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"


def load_services_env() -> dict[str, str]:
    """解析 deploy/.env.services（仅本机使用，值不打印）。"""
    env: dict[str, str] = {}
    for line in open(os.path.join(REPO, "deploy", ".env.services"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_chunks() -> list[dict]:
    return [json.loads(l) for l in open(os.path.join(DATA, "corpus_chunks.jsonl"), encoding="utf-8") if l.strip()]


def load_docs() -> list[dict]:
    return [json.loads(l) for l in open(os.path.join(DATA, "corpus_docs.jsonl"), encoding="utf-8") if l.strip()]


SHORT_SETS = {"nea": "ne_journal_assoc_q48.jsonl", "ner": "ne_journal_realq.jsonl",
              "nea_S": "nea_S.jsonl"}


def load_questions(set_name: str) -> list[dict]:
    name = SHORT_SETS.get(set_name, set_name)
    cands = [os.path.join(DATA, name), os.path.join(REPO, "eval", "harness", name)]
    p = name if os.path.isabs(name) else next((c for c in cands if os.path.exists(c)), None)
    if p is None or not os.path.exists(p):
        raise SystemExit(f"题集未找到: {name}（找过 data/ 与 eval/harness/）")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_slice_ids(name: str) -> list[str]:
    fn = "slice_doc_ids.txt" if name == "L" else f"slice_s_doc_ids.txt"
    return [l.strip() for l in open(os.path.join(DATA, fn)) if l.strip()]


class CostTimer:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.notes: dict = {}

    def done(self, **notes) -> dict:
        return {"wall_s": round(time.time() - self.t0, 1), **notes}


def write_predictions(system: str, slice_name: str, set_name: str, cost: dict,
                      per_query: list[dict], n_docs: int) -> str:
    """统一预测产物：runs/<system>_<set>_<slice>.json"""
    os.makedirs(RUNS, exist_ok=True)
    out = os.path.join(RUNS, f"{system}_{set_name}_{slice_name}.json")
    doc = {"system": system, "slice": slice_name, "set": set_name,
           "corpus_docs": n_docs, "index_cost": cost, "per_query": per_query}
    json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[written] {out}  ({len(per_query)} queries)")
    return out
