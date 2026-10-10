#!/usr/bin/env python3
"""文档级聚合变体消融（2026-09-28）：O10 doc-fuse 的门控扩展测试。

归因（20260928_stressset/autopsy_*.json）：采纳配置下漏检金标的 56~64% 是
"融合稀释"——票散在同一文档不同块上，且该模式在真实问句（ner 64.3%）比关联题
（56.4%）更重，而 O10 门控只覆盖关联题。本消融在测试栈进程内直调 _retrieve，
以 env 切换 doc 聚合臂（零服务重启、Redis 旁路无缓存混杂）：

  base     采纳配置（仅关联意图 λ=2+stride，真实问句 chunk 模式）
  pool1    全局 λ=1（MaxP）+stride
  pool2    全局 λ=2+stride（门控取消，覆盖 ner）
  pool3    全局 λ=3+stride

用法（测试栈存储 env 见 knc_env 注释）：
  .venv/bin/python docpool_ablation.py --sets nea,ner
产物：runs/dp-{arm}_{set}_teststack.json + score.py 配对。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO, load_questions, write_predictions  # noqa: E402

KB_FILE = os.path.join(REPO, "eval", "baselines", "data", "teststack_ne", "kb.json")
ARMS = {"base": ("0", None), "pool1": ("1", None), "pool2": ("2", None), "pool3": ("3", None)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="nea,ner")
    args = ap.parse_args()

    kb = json.load(open(KB_FILE))["kb_id"]
    # 引擎直调返回测试栈 doc_id；金标是 stable id——须经 idmap rev 翻译再比名次
    # （miss_autopsy/openbench 同款教训：全零先查 id 空间）
    idmap = json.load(open(os.path.join(REPO, "eval", "baselines", "data",
                                        "teststack_ne", "idmap.json")))
    rev = {v: k for k, v in idmap.items()}
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

    os.environ["RRF_DOC_FUSE_EMIT"] = "stride"  # 全臂统一 stride 发射
    questions: list[tuple[str, list[dict]]] = []
    for s in args.sets.split(","):
        if s:
            questions.append((s, load_questions(s)))

    for arm, (lam, _) in ARMS.items():
        os.environ["RRF_DOC_FUSE"] = lam
        for set_name, qs in questions:
            per_query = []
            for qi, q in enumerate(qs):
                req = M.RetrieveRequest.model_construct(
                    kb_id=kb, kb_ids=None, query=q["query"], top_k=20, channels=None,
                    doc_ids=None, strategy="A", session_id=None,
                    tenant_id="public", acl_scopes=["public"])
                deadline = time.monotonic() + 120.0
                t0 = time.time()
                resp = M._retrieve(req, [kb], deadline,
                                   lambda: time.monotonic() < deadline, f"dp-{arm}")
                dt = time.time() - t0
                per_query.append({
                    "query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                    "hits": [{"rank": i + 1, "doc_id": rev.get(c.get("doc_id"),
                                                                  c.get("doc_id")),
                              "chunk_id": c.get("chunk_id"), "score": c.get("_score")}
                             for i, c in enumerate(resp.chunks)],
                    "latency_s": round(dt, 3)})
                if (qi + 1) % 20 == 0:
                    print(f"[{arm}/{set_name} {qi+1}/{len(qs)}] p50="
                          f"{sorted(x['latency_s'] for x in per_query)[len(per_query)//2]:.2f}s",
                          flush=True)
            write_predictions(f"dp-{arm}", "teststack", set_name,
                              {"wall_s": 0,
                               "note": f"进程内直调；RRF_DOC_FUSE={lam}(全局)+stride；"
                                       f"ASSOC 门控仍设（base 臂即采纳配置）"},
                              per_query, 300)


if __name__ == "__main__":
    main()
