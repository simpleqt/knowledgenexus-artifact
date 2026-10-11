#!/usr/bin/env python3
"""F0–F3 冻结候选得分重放（2026-10-11 复审 §5 第 4 步 Paper1 关键验证）。

问题（P1-R1/R2）：doc-fusion 的 λ 聚合与 pair/stride 发射序是两个独立决策，
但历史测量里 λ 扫描在 pair 下做、生产采纳是 stride 全局——两者之间配置
（候选取深/自适应 cooc 等）也在变，−2.7pt cov@10 不能读作纯发射序效应。
复审的可检验推论：冻结候选得分与文档得分后，pair 与 stride 去重后的
top-10 文档集相同 → 文档预算 cov@10 应当相等。

设计：
  捕获段：进程内 scoped 检索（与 run_knc_scoped 同管线），monkeypatch
    search.rrf_fuse 为 tee——每请求记录最后一次调用的原始通道列表+票权
    （融合前的输入，即"冻结候选得分集"的生产同源物）。
  重放段：对每查询用**同一个 rrf_fuse 本体**重放 7 臂：
    F0 = doc_lam=0（chunk-RRF 基线）
    F1λ/F2λ = doc_lam∈{1,2,3} × 发射{pair,stride}（env RRF_DOC_FUSE_EMIT
    在调用点即时读，逐臂设置）
    产物 = 标准 run 文件（top_k=60 引擎输出 → 切片过滤取前 20，与
    scoped 驱动同规约），score.py 算指标 + 配对 bootstrap CI。
零复刻漂移：聚合/发射代码 = 生产本体；唯一实验变量 = doc_lam/emit。

用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/frozen_replay_f0f3.py'
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import NE_JOURNAL_KB, REPO, load_questions, load_slice_ids, write_predictions  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402
from knc_engine import search as KSEARCH  # noqa: E402

# Redis 旁路 stub（与 run_knc_scoped 同款——准入/缓存与排序无关）
M.cache.admit = lambda *a, **k: (True, "")
M.cache.session_admit = lambda *a, **k: (True, 0)
M.cache.singleflight_lock = lambda *a, **k: True
M.cache.singleflight_release = lambda *a, **k: None
M.cache.result_get = lambda *a, **k: None
M.cache.record_latency_ms = lambda *a, **k: None
M.cache.bump_metric = lambda *a, **k: None
M.cache.acl_version = lambda *a, **k: 0

_ORIG_FUSE = KSEARCH.rrf_fuse
TEE: list[dict] = []          # 每查询最后一次 rrf_fuse 调用的输入
_CALLS: list[dict] = []       # 逐调用记录（诊断用）


def _tee_fuse(result_lists, k=60, top_k=10, weights=None, doc_lam=None):
    rec = {"lists": [[dict(i) for i in lst[:120]] for lst in result_lists],
           "weights": list(weights) if weights else None,
           "k": k, "top_k": top_k, "doc_lam": doc_lam}
    _CALLS.append(rec)
    return _ORIG_FUSE(result_lists, k=k, top_k=top_k, weights=weights,
                      doc_lam=doc_lam)


KSEARCH.rrf_fuse = _tee_fuse


def scoped(query: str, doc_ids: list[str]):
    req = M.RetrieveRequest.model_construct(
        kb_id=NE_JOURNAL_KB, kb_ids=None, query=query, top_k=60, channels=None,
        doc_ids=list(doc_ids), strategy="A", session_id=None,
        tenant_id="public", acl_scopes=["public"])
    deadline = time.monotonic() + 120.0
    resp = M._retrieve(req, [NE_JOURNAL_KB], deadline,
                       lambda: time.monotonic() < deadline, "knc-frz-eval")
    return resp


def main() -> None:
    slice_ids = load_slice_ids("L")
    sliceset = set(slice_ids)
    outdir = os.path.join(REPO, "eval", "baselines", "results",
                          "20261011_frozen_replay")
    os.makedirs(outdir, exist_ok=True)
    arms = [("F0", 0, "pair")] + [
        (f"F{lam}{'p' if emit == 'pair' else 's'}", lam, emit)
        for lam in (1, 2, 3) for emit in ("pair", "stride")]

    for set_name in ("nea", "ner"):
        questions = load_questions(set_name)
        captured = []
        for i, q in enumerate(questions):
            _CALLS.clear()
            resp = scoped(q["query"], slice_ids)
            rec = dict(_CALLS[-1])  # 最后一次调用 = 输出发射用的融合
            rec["query"] = q["query"]
            rec["gold"] = list(q["gold_doc_ids"] if isinstance(
                q.get("gold_doc_ids"), list) else q.get("gold", []))
            captured.append(rec)
            if (i + 1) % 10 == 0:
                print(f"[{set_name}] {i+1}/{len(questions)} captured "
                      f"({len(rec['lists'])} lists)", flush=True)
        json.dump(captured, open(os.path.join(
            outdir, f"captured_{set_name}.json"), "w"), ensure_ascii=False)

        for arm, lam, emit in arms:
            os.environ["RRF_DOC_FUSE_EMIT"] = emit
            per_query = []
            for rec in captured:
                fused = _ORIG_FUSE(rec["lists"], k=rec["k"],
                                   top_k=rec["top_k"],
                                   weights=rec["weights"], doc_lam=lam)
                in_slice = [c for c in fused if c.get("doc_id") in sliceset][:20]
                per_query.append({
                    "query": rec["query"], "gold_doc_ids": rec["gold"],
                    "hits": [{"rank": i + 1, "doc_id": c.get("doc_id"),
                              "chunk_id": c.get("chunk_id"),
                              "score": c.get("score")}
                             for i, c in enumerate(in_slice)]})
            write_predictions(f"frp_{arm}", "L", set_name,
                              {"capture": "frozen-replay 20261011"},
                              per_query, len(slice_ids))
        print(f"[{set_name}] 7 arm files written", flush=True)
    os.environ.pop("RRF_DOC_FUSE_EMIT", None)
    print("[done] score with:")
    print("  score.py runs/frp_F0_nea_L.json runs/frp_F2p_nea_L.json ... "
          "--ref runs/frp_F0_nea_L.json")


if __name__ == "__main__":
    main()
