#!/usr/bin/env python3
"""P1 别名扩容：对测试库缺跨语别名的实体 LLM 批产双语别名并入库。

前置：alias_gap 分析（/tmp/alias_targets.jsonl，题面命中优先 + df>=3 兜底）。
流程：硬停用词（抽取伪实体通用词）→ LLM 批产（JSON in/out，宁缺毋滥）→
  INSERT IGNORE entity_alias（uk=entity+normalized_alias_hash，幂等可重跑）。
用法：
  gen_alias_expand.py --dry          # 只生成 /tmp/alias_gen_out.jsonl 不入库
  gen_alias_expand.py                # 生成 + 入库（source='llm-p1-v1'）
env：LLM_API_BASE / LLM_MODEL / LLM_API_KEY（deploy/.env.services 同款）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request

KB_HEX = "0CF04665C3BD427281976C3502EB4D99"
KB_HEX_PROD = "21BCBED8EA6B4A5BBF55F3E3E9988C1F"
TARGETS = "/tmp/alias_targets.jsonl"
OUT = "/tmp/alias_gen_out.jsonl"

# 抽取伪实体（无学科专指的通用词）：扩了只加 BM25 噪声
STOP = {"文献", "paper", "article", "方法", "method", "参数", "parameter", "材料",
        "material", "term", "数据集", "dataset", "事件", "event", "模块", "module",
        "ct", "te", "fib", "sca", "lead"}  # lead 双关（动词）风险大于收益

_SYSTEM = (
    "你是核科学技术（核电子学/探测器/辐射测量/核物理）术语专家。"
    "给实体补跨语言别名：中文↔英文标准译名、缩写与全称互扩。"
    "输出必须是严格的 JSON 数组，不含任何其他文字。"
)

_USER_TMPL = (
    "实体列表（i=编号, name=名称, type=类型）。为每个实体给出 aliases："
    "该实体其他语言的标准名称、常用缩写或全称。\n"
    "规则：\n"
    "1. 只收学科术语级名称（如 量能器↔calorimeter、准直器↔collimator、"
    "SiPM↔silicon photomultiplier、同步辐射↔synchrotron radiation）\n"
    "2. 名称是无学科专指性的通用词时给空数组 []\n"
    "3. 不确定时宁缺毋滥给 []\n"
    "4. 别名不得与 name 相同、不得是解释性长短语（≤6 词）\n"
    "输出：一个 JSON 数组，每元素形如 {\"i\":编号,\"aliases\":[\"...\"]}"
    "，每个输入实体一条，缺一不可。\n\n"
)


def llm_batch(env: dict, items: list[dict]) -> dict[int, list[str]]:
    body = json.dumps({
        "model": env["LLM_MODEL"], "temperature": 0,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER_TMPL + json.dumps(
                [{"i": k, "name": it["canonical_name"], "type": it["entity_type"]}
                 for k, it in enumerate(items)], ensure_ascii=False)}],
    }).encode()
    req = urllib.request.Request(
        f"{env['LLM_API_BASE'].rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {env['LLM_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        raise ValueError(f"无 JSON 数组: {content[:200]}")
    out: dict[int, list[str]] = {}
    for row in json.loads(m.group(0)):
        if not isinstance(row, dict) or not isinstance(row.get("aliases"), list):
            continue
        # i 可能回成字符串；缺 i 时按 name 匹配（qwen3.5 实测两种都出现）
        ri = row.get("i")
        if isinstance(ri, str) and ri.isdigit():
            ri = int(ri)
        if isinstance(ri, int):
            out[ri] = [str(a).strip() for a in row["aliases"] if str(a).strip()]
        elif isinstance(row.get("name"), str):
            for k, it in enumerate(items):
                if it["canonical_name"].lower() == row["name"].lower():
                    out[k] = [str(a).strip() for a in row["aliases"] if str(a).strip()]
                    break
    return out


def alias_type(canonical: str, alias: str) -> str:
    if alias.isascii() and len(alias) <= 8 and alias.upper() == alias \
            and any(c.isalpha() for c in alias):
        return "ABBREV"
    if alias.isascii() != canonical.isascii():
        return "TRANS"
    return "FULL"


def insert_curated(path: str) -> None:
    """curated 臂：{canonical_name: [aliases]} 直接入库（.32 LLM 端点不可用时的
    等价手段——别名是实体真属性，策划/LLM 仅来源不同）。source=curated-p1-v1。"""
    import pymysql
    data = json.load(open(path, encoding="utf-8"))
    pairs: dict[str, list[str]] = data["aliases"]
    conn = pymysql.connect(host="127.0.0.1", port=23306, user="knc",
                           password=os.environ.get("MYSQL_PASSWORD", "knc_test"),
                           database="knc", charset="utf8mb4")
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT entity_id, canonical_name FROM canonical_entity"
                " WHERE kb_id=UNHEX(%s) AND status='ACTIVE'", (KB_HEX,))
    by_name = {r["canonical_name"].lower(): r["entity_id"] for r in cur.fetchall()}
    cur.execute("SELECT entity_id, LOWER(alias_text) t FROM entity_alias"
                " WHERE kb_id=UNHEX(%s)", (KB_HEX,))
    have: dict[str, set[str]] = {}
    for r in cur.fetchall():
        have.setdefault(r["entity_id"], set()).add(r["t"])
    ins = skip_name = skip_dup = 0
    for name, aliases in pairs.items():
        eid = by_name.get(name.lower())
        if not eid:
            print(f"  [miss] {name!r} 不在测试库实体表中")
            skip_name += 1
            continue
        for a in aliases:
            if not 1 < len(a) < 80:
                continue
            h = hashlib.sha256(unicodedata.normalize("NFKC", a).lower()
                               .encode()).hexdigest()
            if a.lower() in have.get(eid, set()) or a.lower() == name.lower():
                skip_dup += 1
                continue
            cur.execute(
                "INSERT IGNORE INTO entity_alias (entity_id, kb_id, alias_text,"
                " normalized_alias_hash, alias_type, source)"
                " VALUES (%s,UNHEX(%s),%s,%s,%s,'curated-p1-v1')",
                (eid, KB_HEX, a, h, alias_type(name, a)))
            ins += cur.rowcount
    conn.commit()
    print(f"curated 入库：新增 {ins} 条（未命中实体 {skip_name}、重复跳过 {skip_dup}）")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--curated", help="curated 别名 JSON 直接入库（不经 LLM）")
    ap.add_argument("--prod", action="store_true",
                    help="生产库模式：目标 /tmp/alias_targets_prod.jsonl，"
                         "出 /tmp/alias_gen_out_prod.jsonl，写 13306 生产 entity_alias")
    args = ap.parse_args()

    if args.curated:
        insert_curated(args.curated)
        print("提醒：重启 retrieval-api（/tmp/knc-restart-retrieval.sh）+ 清 knc:rc:* 后再评测")
        return

    if args.prod:
        global TARGETS, OUT, KB_HEX
        TARGETS, OUT = "/tmp/alias_targets_prod.jsonl", "/tmp/alias_gen_out_prod.jsonl"
        KB_HEX = KB_HEX_PROD

    env = {k: os.environ[k] for k in ("LLM_API_BASE", "LLM_MODEL", "LLM_API_KEY")}
    targets = [json.loads(l) for l in open(TARGETS, encoding="utf-8") if l.strip()]
    targets = [t for t in targets if t["canonical_name"].lower() not in STOP]
    done: dict[str, dict] = {}
    if os.path.exists(OUT):  # 断点续跑
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["entity_id"]] = r
    todo = [t for t in targets if t["entity_id"] not in done]
    print(f"目标 {len(targets)}（停用词滤后），已完成 {len(done)}，本轮 {len(todo)}")

    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        for attempt in range(3):
            try:
                got = llm_batch(env, chunk)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [retry {attempt}] {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            got = {}
        with open(OUT, "a", encoding="utf-8") as f:
            for k, it in enumerate(chunk):
                aliases = got.get(k, [])
                valid = sorted({a for a in aliases
                                if 1 < len(a) < 80
                                and a.lower() not in {x.lower() for x in it["existing"]}})
                f.write(json.dumps({"entity_id": it["entity_id"],
                                    "canonical_name": it["canonical_name"],
                                    "doc_df": it["doc_df"], "query_hit": it["query_hit"],
                                    "aliases": valid}, ensure_ascii=False) + "\n")
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)}"
              f"（本批别名 {sum(len(got.get(k, [])) for k in range(len(chunk)))} 条）",
              flush=True)

    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    n_ent = sum(1 for r in rows if r["aliases"])
    n_alias = sum(len(r["aliases"]) for r in rows)
    print(f"生成完成：{n_ent}/{len(rows)} 实体有别名，共 {n_alias} 条 → {OUT}")
    if args.dry:
        return

    import pymysql
    if args.prod:
        _pw = open(os.path.join(os.path.dirname(__file__), "../../../deploy/.env"),
                   encoding="utf-8").read().split("MYSQL_PASSWORD=")[1].splitlines()[0].strip()
        conn = pymysql.connect(host="127.0.0.1", port=13306, user="knc",
                               password=_pw, database="knc", charset="utf8mb4")
    else:
        conn = pymysql.connect(host="127.0.0.1", port=23306, user="knc",
                               password=os.environ.get("MYSQL_PASSWORD", "knc_test"),
                               database="knc", charset="utf8mb4")
    cur = conn.cursor()
    ins = 0
    for r in rows:
        for a in r["aliases"]:
            h = hashlib.sha256(unicodedata.normalize("NFKC", a).lower()
                               .encode()).hexdigest()
            cur.execute(
                "INSERT IGNORE INTO entity_alias (entity_id, kb_id, alias_text,"
                " normalized_alias_hash, alias_type, source)"
                " VALUES (%s,UNHEX(%s),%s,%s,%s,'llm-p1-v1')",
                (r["entity_id"], KB_HEX, a, h, alias_type(r["canonical_name"], a)))
            ins += cur.rowcount
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM entity_alias WHERE kb_id=UNHEX(%s)", (KB_HEX,))
    print(f"入库新增 {ins} 条（冲突跳过不计）；当前别名总数 {cur.fetchone()[0]}")
    conn.close()
    print("提醒：重启 retrieval-api（/tmp/knc-restart-retrieval.sh）+ 清 knc:rc:* 后再评测")


if __name__ == "__main__":
    sys.exit(main())
