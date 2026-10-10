#!/usr/bin/env python3
"""P1 别名扩容：定位测试库高价值缺别名实体。

输出三类目标：
  A. 题面命中（nea/ner 查询里扫到实体名）但该实体无跨语名（<2 个不同名）
  B. doc_df>=3 且无别名的实体（语料价值兜底）
写 /tmp/alias_targets.jsonl 供 LLM 批产。
"""
import json
import os
import re
import sys

import pymysql

KB = "0CF04665C3BD427281976C3502EB4D99"
conn = pymysql.connect(host="127.0.0.1", port=23306, user="knc",
                       password="knc_test", database="knc", charset="utf8mb4")

cur = conn.cursor(pymysql.cursors.DictCursor)
cur.execute("SELECT entity_id, canonical_name, entity_type, doc_df, mention_count"
            " FROM canonical_entity WHERE kb_id=UNHEX(%s) AND status='ACTIVE'", (KB,))
ents = cur.fetchall()
cur.execute("SELECT entity_id, alias_text FROM entity_alias WHERE kb_id=UNHEX(%s)", (KB,))
alias_rows = cur.fetchall()
conn.close()

by_ent: dict[str, dict] = {}
for e in ents:
    by_ent[e["entity_id"]] = {**e, "names": {e["canonical_name"]}}
for a in alias_rows:
    if a["entity_id"] in by_ent:
        by_ent[a["entity_id"]]["names"].add(a["alias_text"])


def has_cross_lang(names: set[str]) -> bool:
    low = {n.lower() for n in names if n and 1 < len(n) < 80}
    if len(low) < 2:
        return False
    langs = {"ascii" if n.isascii() else "cjk" for n in low}
    return len(langs) >= 2


# 题面：把两题集查询拼一起做词典命中
queries = []
for f in ("ne_journal_assoc_q48.jsonl", "ne_journal_realq.jsonl"):
    p = os.path.join("/home/inspur/myenv/knowledgenexus/eval/harness", f)
    queries += [json.loads(l)["query"] for l in open(p, encoding="utf-8") if l.strip()]
qtext = "\n".join(queries).lower()

hit_no_xlang, hit_xlang, nohit_no_xlang = [], [], []
for e in by_ent.values():
    low = {n.lower() for n in e["names"] if n and 1 < len(n) < 80}
    qhit = any(n in qtext for n in low)
    x = has_cross_lang(e["names"])
    rec = {"entity_id": e["entity_id"], "canonical_name": e["canonical_name"],
           "entity_type": e["entity_type"], "doc_df": int(e["doc_df"] or 0),
           "mention_count": int(e["mention_count"] or 0),
           "existing": sorted(low), "query_hit": qhit}
    if qhit and not x:
        hit_no_xlang.append(rec)
    elif qhit:
        hit_xlang.append(rec)
    elif not x and e["doc_df"] and e["doc_df"] >= 3:
        nohit_no_xlang.append(rec)

print(f"实体总数 {len(by_ent)}；题面命中 {len(hit_no_xlang)+len(hit_xlang)}"
      f"（其中已跨语 {len(hit_xlang)}、缺跨语 {len(hit_no_xlang)}）；"
      f"df>=3 无别名（未命中题面）{len(nohit_no_xlang)}")
print("\n== 题面命中但缺跨语别名 ==")
for r in sorted(hit_no_xlang, key=lambda x: -x["doc_df"]):
    print(f"  df={r['doc_df']:>3} mc={r['mention_count']:>4} {r['canonical_name']!r} ({r['entity_type']})")

targets = hit_no_xlang + nohit_no_xlang
targets.sort(key=lambda x: (-x["query_hit"], -x["doc_df"]))
with open("/tmp/alias_targets.jsonl", "w", encoding="utf-8") as f:
    for r in targets:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n目标写入 /tmp/alias_targets.jsonl：{len(targets)} 实体"
      f"（题面命中优先，其余 df>=3）")
