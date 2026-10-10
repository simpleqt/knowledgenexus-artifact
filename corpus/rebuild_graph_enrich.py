#!/usr/bin/env python3
"""B92 存量图谱层重建驱动（实体塌缩事故，2026-10-07）。

前置（缺一不可）：
  1. 新术语词典 Schema 已落盘并通过 validator（corpus/gen_term_lexicons.py 产物，
     changelog 含 B92）——词典不换直接重跑 = 再塌一次；
  2. 富化消费者已全部停止（本脚本清库期间不可有消费者），换词典后重启
     （进程内 _MENTION_CACHE 按 schema_id 缓存，必须重启才装新词典）。

动作（逐 KB）：
  ① Nebula：收集 canonical_entity 全部 vid（小写 hex），DELETE VERTEX ... WITH EDGE
  ② MySQL：分块清 raw_mention / entity_link_registry / relation_registry /
     canonical_entity（kb_id 过滤）
  ③ OpenSearch：knc-entity-v2 该库实体向量投影 delete_by_query
  ④ 重入队：mentions/links/relations/graph 四阶段 × 全部 ACTIVE 文档

幂等：重复执行安全（清库=0 行、入队前先删该库四阶段旧任务）。
用法：.venv/bin/python corpus/rebuild_graph_enrich.py --kb <hex> [<hex>...] [--execute]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "engine"))

STAGES = ("mentions", "links", "relations", "graph")
CHUNK = 20000


def _env() -> dict:
    env = {}
    for line in open(os.path.join(os.path.dirname(__file__), "..", "deploy", ".env"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def _chunked_delete(cur, table: str, kb: str) -> int:
    total = 0
    while True:
        cur.execute(f"DELETE FROM {table} WHERE kb_id=UNHEX(%s) LIMIT %s", (kb, CHUNK))
        total += cur.rowcount
        cur.connection.commit()
        if cur.rowcount < CHUNK:
            return total
        time.sleep(0.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", nargs="+", required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    import pymysql
    env = _env()
    conn = pymysql.connect(host="127.0.0.1", port=13306, user="knc", password=env["MYSQL_PASSWORD"],
                           database="knc", charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor)
    cur = conn.cursor()

    for kb in args.kb:
        kb = kb.lower()
        cur.execute("SELECT name FROM kb_registry WHERE kb_id=UNHEX(%s)", (kb,))
        nm = (cur.fetchone() or {}).get("name", "?")
        cur.execute("SELECT COUNT(*) n FROM kb_documents WHERE kb_id=UNHEX(%s) AND state='ACTIVE'", (kb,))
        docs = cur.fetchone()["n"]
        cur.execute("SELECT entity_id FROM canonical_entity WHERE kb_id=UNHEX(%s)", (kb,))
        vids = [r["entity_id"].lower() for r in cur.fetchall()]
        for t in ("raw_mention", "entity_link_registry", "relation_registry", "canonical_entity"):
            cur.execute(f"SELECT COUNT(*) n FROM {t} WHERE kb_id=UNHEX(%s)", (kb,))
            print(f"[{kb[:8]}] {nm} docs={docs} {t}={cur.fetchone()['n']}  vid={len(vids)}")
        if not args.execute:
            print(f"[{kb[:8]}] [dry-run] 以上为将清理量")
            continue

        # ① Nebula 清投影（分批 WITH EDGE）
        from knc_engine import graph
        for i in range(0, len(vids), 200):
            batch = vids[i:i + 200]
            graph.execute("DELETE VERTEX " + ",".join(graph.q(v) for v in batch) + " WITH EDGE")
        print(f"[{kb[:8]}] nebula 删除 {len(vids)} 顶点（含边）", flush=True)
        # ② MySQL 四表分块清
        for t in ("raw_mention", "entity_link_registry", "relation_registry", "canonical_entity"):
            n = _chunked_delete(cur, t, kb)
            print(f"[{kb[:8]}] {t} 清 {n}", flush=True)
        # ③ OS 实体投影清
        from opensearchpy import OpenSearch
        osc = OpenSearch(hosts=[{"host": "127.0.0.1", "port": 19200, "scheme": "https"}],
                         http_auth=("admin", env["OPENSEARCH_ADMIN_PASSWORD"]),
                         verify_certs=False, ssl_show_warn=False)
        osc.delete_by_query(index="knc-entity-v2", body={"query": {"term": {"kb_id": kb}}},
                            params={"conflicts": "proceed", "refresh": "true"})
        print(f"[{kb[:8]}] OS knc-entity-v2 投影清", flush=True)
        # ④ 四阶段重入队（先清旧任务行防重复）
        cur.execute("DELETE FROM enrich_job WHERE kb_id=UNHEX(%s) AND stage IN (%s)", (kb, ",".join(f"'{s}'" for s in STAGES)))
        for s in STAGES:
            cur.execute("INSERT INTO enrich_job (kb_id, doc_id, stage)"
                        " SELECT kb_id, doc_id, %s FROM kb_documents"
                        " WHERE kb_id=UNHEX(%s) AND state='ACTIVE'", (s, kb))
        conn.commit()
        cur.execute("SELECT COUNT(*) n FROM enrich_job WHERE kb_id=UNHEX(%s)", (kb,))
        print(f"[{kb[:8]}] 重入队 {cur.fetchone()['n']} 单（{docs} 文档 × {len(STAGES)} 阶段）", flush=True)
    conn.close()
    print("REBUILD QUEUED" if args.execute else "DRY-RUN DONE")


if __name__ == "__main__":
    main()
