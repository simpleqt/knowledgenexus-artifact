#!/usr/bin/env python3
"""cooc 通道活性强制触发检查（2026-10-10 审查 N2 补证）：

"bitwise identical" 消融的替代解释是"通道没接线/门控永不放行"。本脚本在
自然措辞关联题集（ne_journal_natural_assoc，99 题，模板词禁用——自适应门
历史上面向该分布打开）上双臂对照：KNC_COOC_OFF 未设 vs 设 1，逐题记录
cooc 是否触发（resp.channels 含 cooc）与 top-10 差异，并按题汇总 gold
命中变化。输出 JSON 到 --out。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/cooc_liveness_check.py \
  --out eval/baselines/results/20261009_chab/cooc_liveness.json'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import NE_JOURNAL_KB, REPO  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402

M.cache.admit = lambda *a, **k: (True, "")
M.cache.session_admit = lambda *a, **k: (True, 0)
M.cache.singleflight_lock = lambda *a, **k: True
M.cache.singleflight_release = lambda *a, **k: None
M.cache.result_get = lambda *a, **k: None
M.cache.record_latency_ms = lambda *a, **k: None
M.cache.bump_metric = lambda *a, **k: None
M.cache.acl_version = lambda *a, **k: 0


def retrieve(query: str):
    req = M.RetrieveRequest.model_construct(
        kb_id=NE_JOURNAL_KB, kb_ids=None, query=query, top_k=20,
        channels=None, doc_ids=None, strategy="A", session_id=None,
        tenant_id="public", acl_scopes=["public"])
    deadline = time.monotonic() + 120.0
    resp = M._retrieve(req, [NE_JOURNAL_KB], deadline,
                       lambda: time.monotonic() < deadline, "knc-cooclive")
    return [c["doc_id"] for c in resp.chunks[:10]], "cooc" in (resp.channels or [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(
        REPO, "eval", "harness", "ne_journal_natural_assoc.jsonl"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    qs = [json.loads(l) for l in open(args.set, encoding="utf-8") if l.strip()]

    os.environ.pop("KNC_COOC_OFF", None)
    arm_on = {}
    fired_on = []
    for q in qs:
        top, fired = retrieve(q["query"])
        arm_on[q["query"]] = top
        if fired:
            fired_on.append(q["query"])
    os.environ["KNC_COOC_OFF"] = "1"
    diffs = 0
    win = loss = tie = 0
    for q in qs:
        gold = set(q["gold_doc_ids"])
        top_off, _ = retrieve(q["query"])
        if q["query"] in fired_on:
            on = set(arm_on[q["query"]])
            off = set(top_off)
            if on == off:
                tie += 1
            else:
                diffs += 1
                ho, hf = len(gold & on), len(gold & off)
                if hf > ho:
                    win += 1
                elif hf < ho:
                    loss += 1
                else:
                    tie += 1
    res = {
        "n": len(qs), "cooc_fired_on_arm": len(fired_on),
        "fired_queries_changed_top10": diffs,
        "off_vs_on_gold_hits_win_loss_tie": [win, loss, tie],
        "note": "win=off-arm better (cooc hurt), loss=off worse (cooc helped)",
        "fired_sample": fired_on[:5],
    }
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
