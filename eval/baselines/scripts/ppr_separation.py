#!/usr/bin/env python3
"""hard37 深轨组件分离（2026-09-29）：PPR 从未触发的 bug 量化 + 图谱贡献测量。

背景：PPR 分支条件 `"ppr" in want`，deep 端点只注入 "rerank"（B86 同类 bug），
KNC_PPR_FORCE 为死代码——深轨 PPR 自 channels 化以来从未运行。
本脚本进程内直调生产 _retrieve（只读），三臂分离 62.2% 恢复率的构成：

  now     现网 deep 实际行为（profile + rerank，无 ppr）
  ppr     now + "ppr"（修复后预期行为）
  norr    profile + "ppr" 无 rerank（PPR 单独贡献）

用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python ppr_separation.py'
产物：results/20260929_hardtargets/ppr_separation.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402

KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
PROFILE = ["lexical", "dense", "graph", "synq", "docvec", "cooc"]  # 生产 profile + 门控通道
OUT = Path(REPO) / "eval/baselines/results/20260929_hardtargets/ppr_separation.json"
ARMS = {"now": PROFILE + ["rerank"],
        "ppr": PROFILE + ["rerank", "ppr"],
        "norr": PROFILE + ["ppr"]}


def main() -> None:
    rows = [json.loads(l) for l in
            open(Path(REPO) / "eval/baselines/results/20260929_hardtargets/"
                 "hard_targets_prod.jsonl", encoding="utf-8") if l.strip()]
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
    os.environ["RERANK_DISABLED"] = ""  # deep 语义：强制重排可用
    from app.main import DEFAULT_CHANNELS  # noqa: F401  （只 import 侧无效果）

    results = {}
    for arm, channels in ARMS.items():
        out_rows = []
        t_all = time.time()
        for qi, r in enumerate(rows):
            gold = set(r["gold_doc_ids"])
            req = M.RetrieveRequest.model_construct(
                kb_id=KB, kb_ids=None, query=r["query"], top_k=50,
                channels=channels, doc_ids=None, strategy="A", session_id=None,
                tenant_id="public", acl_scopes=["public"])
            deadline = time.monotonic() + 150.0
            t0 = time.time()
            resp = M._retrieve(req, [KB], deadline,
                               lambda: time.monotonic() < deadline, f"pprsep-{arm}")
            dt = time.time() - t0
            docs = [c.get("doc_id") for c in resp.chunks]
            fr = next((i for i, d in enumerate(docs, 1) if d in gold), None)
            out_rows.append({"query": r["query"], "rank": fr,
                             "latency_s": round(dt, 2),
                             "channels": list(resp.channels)})
            if (qi + 1) % 10 == 0:
                print(f"[{arm} {qi+1}/{len(rows)}]", flush=True)
        n = len(out_rows)
        ch_ppr = sum(1 for x in out_rows if "ppr" in x["channels"])
        lat = sorted(x["latency_s"] for x in out_rows)
        results[arm] = {
            "hit@10": round(sum(1 for x in out_rows if (x["rank"] or 99) <= 10) / n, 4),
            "hit@20": round(sum(1 for x in out_rows if (x["rank"] or 99) <= 20) / n, 4),
            "mrr@10": round(sum(1.0 / x["rank"] for x in out_rows
                                if x["rank"] and x["rank"] <= 10) / n, 4),
            "ppr_channel_fired": ch_ppr, "n": n,
            "p50_s": lat[n // 2], "wall_s": round(time.time() - t_all, 1),
            "rows": out_rows,
        }
        print(f"[{arm}] hit@10={results[arm]['hit@10']} hit@20={results[arm]['hit@20']} "
              f"mrr={results[arm]['mrr@10']} ppr_fired={ch_ppr} p50={lat[n//2]}s", flush=True)
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
