#!/usr/bin/env python3
"""票权网格调参（O3，文献调研 §2 P0 第 3 项）：train-then-freeze，门控参数网格。

网格对象是 **B83/B84 门控 env**（RRF_SYNQ_BOOST_ASSOC × RRF_DOCVEC_ASSOC_W），
不是 RRF_CHANNEL_WEIGHTS 静态权重——静态全局提权已被 B83 证伪（ner -7.7pt，
票权增益查询类型依赖），网格静态权重等于重跑已否决的设计。纪律照
arXiv 2609.22770：调参集上搜索 → 冻结 → 验证集上确认不过拟合。

用法（测试栈，约 1h）：
  .venv/bin/python grid_weights.py                       # 默认网格+nea调参/ner验证
  .venv/bin/python grid_weights.py --dry                 # 只列网格与产物路径
产物：runs/knc-gridG<i>_<set>_teststack.json + stdout 报告（建议 env 行）。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402

SCRIPTS = os.path.dirname(__file__)
RESTART = "/tmp/knc-restart-retrieval.sh"
ENV_FILE = "/tmp/knc-rrf-env"
RUNS = os.path.join(REPO, "eval/baselines/runs")

# 部署基线（当前测试栈定稿）：网格点必须包含它，作为"调参 vs 已采用"对照
ADOPTED = {"synq_boost": 1.4, "docvec_w": 0.6}


def flush_cache() -> None:
    subprocess.run(
        ["docker", "exec", "knc-test-redis", "sh", "-c",
         'redis-cli -a knc_test --scan --pattern "knc:rc:*" 2>/dev/null'
         ' | xargs -r redis-cli -a knc_test DEL'],
        capture_output=True, timeout=60)


def restart_with(synq_boost: float, docvec_w: float) -> None:
    """写 env 文件（保留 B83 双门 + docvec 开；只变两个门控参数）→ 重启 → 清缓存。

    重启脚本必须文件重定向调用——stdout 为管道时其内联后台子壳会挂起
    （O3 首跑实测：管道 120s 超时、文件重定向链式 3/3 正常）；重启后 pkill
    锚定 `knc-restart-retrieval.sh$` 清掉滞留的孤儿脚本壳（外层 bash -c 命令行
    以 exit 0 结尾，不会自匹配）。"""
    with open(ENV_FILE, "w") as f:
        f.write(f"RRF_SYNQ_BOOST_ASSOC={synq_boost}\n"
                f"RRF_DOCVEC_ASSOC_W={docvec_w}\n"
                "KNC_DOCVEC_ON=1\n"
                "RRF_DOCVEC_PRIOR=\n")
    subprocess.run(
        ["bash", "-c",
         f"timeout -k 5 90 bash {RESTART} > /tmp/o3-restart.log 2>&1; "
         "pkill -f 'knc-restart-retrieval\\.sh$' 2>/dev/null; exit 0"],
        capture_output=True, timeout=150)
    flush_cache()


def eval_set(set_name: str, tag: str) -> str:
    """单集评测，返回 pred 文件路径。"""
    subprocess.run(
        [f"{REPO}/.venv/bin/python", f"{SCRIPTS}/teststack_ne.py", "eval",
         "--tag", tag, "--sets", set_name],
        capture_output=True, timeout=900, cwd=REPO)
    return os.path.join(RUNS, f"knc{tag}_{set_name}_teststack.json")


def score(pred_file: str) -> dict:
    d = json.load(open(pred_file))
    rows = d["per_query"]
    n = len(rows)
    r10 = cov10 = mrr = 0.0
    for p in rows:
        gold = set(p["gold_doc_ids"])
        docs = [h["doc_id"] for h in p["hits"]]
        r10 += any(x in gold for x in docs[:10])
        seen: set[str] = set()
        udocs = [x for x in docs if not (x in seen or seen.add(x))]
        cov10 += len(set(udocs[:10]) & gold) / len(gold)
        rank = next((i + 1 for i, x in enumerate(docs[:10]) if x in gold), 0)
        mrr += 1.0 / rank if rank else 0.0
    return {"R@10": round(r10 / n, 4), "cov@10": round(cov10 / n, 4),
            "MRR@10": round(mrr / n, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="nea", help="调参集")
    ap.add_argument("--valid", default="ner", help="验证集（防过拟合）")
    ap.add_argument("--synq", default="1.2,1.4,1.6,1.8", help="RRF_SYNQ_BOOST_ASSOC 候选")
    ap.add_argument("--docvec", default="0.4,0.6,0.8,1.0", help="RRF_DOCVEC_ASSOC_W 候选")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    synq_grid = [float(x) for x in args.synq.split(",")]
    dv_grid = [float(x) for x in args.docvec.split(",")]
    combos = list(itertools.product(synq_grid, dv_grid))
    print(f"网格 {len(combos)} 组（synq {synq_grid} × docvec {dv_grid}），"
          f"train={args.train} valid={args.valid}，基线=采用配置 {ADOPTED}")
    if args.dry:
        return

    results = []
    for i, (sb, dw) in enumerate(combos, start=1):
        tag = f"-gridG{i}"
        print(f"[{i}/{len(combos)}] synq_boost={sb} docvec_w={dw} ... ",
              end="", flush=True)
        restart_with(sb, dw)
        pred = eval_set(args.train, tag)
        s = score(pred)
        results.append({"synq_boost": sb, "docvec_w": dw, "train": s})
        mark = " ←基线" if (sb, dw) == (ADOPTED["synq_boost"], ADOPTED["docvec_w"]) else ""
        print(f"{s}{mark}", flush=True)

    # 目标：train 集 R@10 + cov@10 + MRR@10 之和（关联集 cov 是本义指标）
    results.sort(key=lambda r: -(r["train"]["R@10"] + r["train"]["cov@10"]
                                 + r["train"]["MRR@10"]))
    print(f"\n=== Top-5（train={args.train}，R+cov+MRR 之和排序）===")
    for r in results[:5]:
        print(f"  synq {r['synq_boost']} / docvec {r['docvec_w']} → {r['train']}")

    # train 冠军 + 基线 → 验证集（过拟合检查：冠军在 valid 上不得输基线）
    print(f"\n=== 验证（valid={args.valid}）===")
    valid_scores = {}
    for label, cand in (("best", results[0]),
                        ("adopted", next(r for r in results
                                         if (r["synq_boost"], r["docvec_w"])
                                         == (ADOPTED["synq_boost"], ADOPTED["docvec_w"])))):
        restart_with(cand["synq_boost"], cand["docvec_w"])
        pred = eval_set(args.valid, f"-gridV{label}")
        valid_scores[label] = score(pred)
        print(f"  {label}（synq {cand['synq_boost']} / docvec {cand['docvec_w']}"
              f"）→ {valid_scores[label]}", flush=True)

    best, adopted = valid_scores["best"], valid_scores["adopted"]
    d_r = round(best["R@10"] - adopted["R@10"], 4)
    d_c = round(best["cov@10"] - adopted["cov@10"], 4)
    print(f"\n=== 裁决 ===\nbest − adopted（valid）：R@10 {d_r:+} / cov@10 {d_c:+}")
    if d_r > 0 and d_c >= 0:
        print(f"建议冻结：RRF_SYNQ_BOOST_ASSOC={results[0]['synq_boost']} "
              f"RRF_DOCVEC_ASSOC_W={results[0]['docvec_w']}")
    else:
        print(f"维持采用配置（train 增益未跨到验证集 = 过拟合信号）："
              f"RRF_SYNQ_BOOST_ASSOC={ADOPTED['synq_boost']} "
              f"RRF_DOCVEC_ASSOC_W={ADOPTED['docvec_w']}")

    # 收尾：恢复定稿配置
    restart_with(ADOPTED["synq_boost"], ADOPTED["docvec_w"])
    print("已恢复定稿配置。")


if __name__ == "__main__":
    main()
