#!/usr/bin/env python3
"""未命中归因解剖（2026-09-25）：漏检 gold 文档在检索管线哪一层丢失。

对预测文件（采用配置）中 R@10/cov@10 的漏检 gold 文档，做引擎级单通道隔离
（直接调 knc_engine 四通道，不经服务层 want 混合），定位丢失层：

  fused@20   —— gold 在融合序 11-20 名（近失排序，融合票差一点）
  ch@100     —— gold 在某单通道 top-100 内但融合序 >20/缺席（融合稀释：票被摊薄）
  deep-miss  —— 四通道 top-100 均无 gold（召回根本性缺失）
               细分归因：词典覆盖（gold 文档是否链接了问题实体对）/
               synq 覆盖 / docvec 题录相似度

关联题（nea）另查：问题实体对是否在词典、gold 文档是否双向链接（cooc 通道本可投票）。

用法（测试栈，离线直连引擎）：
  MYSQL_PORT=23306 MYSQL_PASSWORD=knc_test PYTHONPATH=libs/engine \
  EMBEDDING_*（deploy/.env.services） .venv/bin/python miss_autopsy.py \
    --pred eval/baselines/runs/knc-o8dv_nea_teststack.json --set nea \
    --kb 0cf04665c3bd427281976c3502eb4d99
产物：stdout 报告 + --out JSON（可入证据包）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, REPO  # noqa: E402

K = 100  # 单通道探查深度


def load_env_services() -> None:
    for line in open(os.path.join(REPO, "deploy", ".env.services"), encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)


def pair_of(q: str) -> tuple[str, str] | None:
    m = re.match(r"有哪些研究同时涉及(.+?)与(.+?)？（关联检索）", q)
    return (m.group(1).strip().lower(), m.group(2).strip().lower()) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--kb", default="0cf04665c3bd427281976c3502eb4d99")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    load_env_services()

    from knc_engine import db, embedding, search

    pred = json.load(open(args.pred, encoding="utf-8"))
    kb = args.kb
    # 预测文件 gold/hits 是 stable id 空间；引擎直调返回测试栈 doc_id 空间——
    # 须经 idmap 翻译后再比名次（首版漏此步，通道名次全 None 的构造性伪影）。
    idmap = json.load(open(os.path.join(DATA, "teststack_ne", "idmap.json")))
    stable2test = idmap  # stable_id -> test doc_id

    # 词典与链接信息（一次性拉取）
    ent_names: dict[str, str] = {}          # lower name -> entity_id
    for r in db.exec("SELECT entity_id e, canonical_name n FROM canonical_entity"
                     " WHERE kb_id=UNHEX(%s) AND status='ACTIVE'", (kb,), fetch=True):
        ent_names[r["n"].lower()] = r["e"]
    for r in db.exec("SELECT a.entity_id e, a.alias_text t FROM entity_alias a"
                     " JOIN canonical_entity c ON c.entity_id=a.entity_id"
                     " WHERE a.kb_id=UNHEX(%s) AND c.status='ACTIVE'", (kb,), fetch=True):
        ent_names.setdefault(r["t"].lower(), r["e"])

    def doc_entity_sets(doc_hex: str) -> set[str]:
        """文档链接到的实体 id 集（links 经 raw_mention 关联到文档）。"""
        return {r["e"] for r in db.exec(
            "SELECT DISTINCT l.entity_id e FROM entity_link_registry l"
            " JOIN raw_mention m ON m.mention_id=l.mention_id"
            " WHERE l.kb_id=UNHEX(%s) AND m.doc_id=UNHEX(%s)",
            (kb, doc_hex), fetch=True)}

    def rank_of(hits: list[dict], gold: str) -> int | None:
        for i, h in enumerate(hits, start=1):
            if h.get("doc_id") == gold:
                return i
        return None

    records = []
    cat = {"fused@20 近失": 0, "ch@100 融合稀释": 0, "deep-miss 深召回缺失": 0}
    deep_detail = {"无词典实体对": 0, "gold 未链接实体对": 0, "gold 双向链接但无通道命中": 0}
    n_miss_docs = 0
    for row in pred["per_query"]:
        gold_set = set(row["gold_doc_ids"])
        fused_docs = [h["doc_id"] for h in row["hits"]]
        missed = [g for g in gold_set if g not in fused_docs[:10]]
        if not missed:
            continue
        q = row["query"]
        qvec = embedding.embed([q])[0]
        # 单通道隔离（引擎直调，top-100）
        ch_hits = {
            "lexical": search.lexical_search(kb, q, K),
            "dense": search.dense_search(kb, qvec, K),
            "synq": search.synq_search(kb, qvec, K),
        }
        dv = search.doc_vec_search(kb, qvec, K)  # doc 级命中
        pair = pair_of(q)
        pair_eids = None
        if pair:
            pair_eids = {ent_names.get(s) for s in pair}
            pair_eids.discard(None)
        for g in missed:
            n_miss_docs += 1
            g_test = stable2test.get(g) or g  # 引擎/DB 查询一律用测试栈 doc_id
            ranks = {c: rank_of(h, g_test) for c, h in ch_hits.items()}
            ranks["docvec"] = next((i for i, d in enumerate(dv, start=1)
                                    if d["doc_id"] == g_test), None)
            fused_rank = next((i for i, d in enumerate(fused_docs, start=1) if d == g), None)
            any_ch = any(v is not None for v in ranks.values())
            if fused_rank is not None and fused_rank <= 20:
                layer = "fused@20 近失"
            elif any_ch:
                layer = "ch@100 融合稀释"
            else:
                layer = "deep-miss 深召回缺失"
                if pair_eids is not None:
                    de = doc_entity_sets(g_test)
                    if len(pair_eids) < 2:
                        deep_detail["无词典实体对"] += 1
                    elif not pair_eids <= de:
                        deep_detail["gold 未链接实体对"] += 1
                    else:
                        deep_detail["gold 双向链接但无通道命中"] += 1
            cat[layer] += 1
            records.append({
                "query": q[:60], "gold_doc": g, "layer": layer,
                "fused_rank": fused_rank,
                "ch_ranks": {c: ranks[c] for c in ranks},
                "pair_in_dict": (len(pair_eids) >= 2) if pair_eids is not None else None,
            })

    print(f"## 未命中归因（{args.set}，n_missed_gold_docs={n_miss_docs}，探深 {K}）\n")
    print("| 丢失层 | 数量 | 占比 |")
    print("|---|---|---|")
    for k, v in cat.items():
        print(f"| {k} | {v} | {v / max(1, n_miss_docs):.1%} |")
    if any(deep_detail.values()):
        print("\n**deep-miss 细分（关联题）**：")
        for k, v in deep_detail.items():
            print(f"- {k}: {v}")
    print("\n**典型样本**：")
    for r in records[:10]:
        print(f"- [{r['layer']}] {r['query']} gold={r['gold_doc']} "
              f"ch={r['ch_ranks']} fused_rank={r['fused_rank']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"set": args.set, "pred": os.path.basename(args.pred),
                   "n_missed_docs": n_miss_docs, "layers": cat,
                   "deep_detail": deep_detail, "records": records},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
