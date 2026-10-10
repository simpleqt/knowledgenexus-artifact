#!/usr/bin/env python3
"""分层 300 链接审计采样器（2026-10-10 三轮审查补 #2：100→300 分层）。

四层（实体频次排名 = canonical_entity.mention_count per-KB 已维护统计列，
10-09 重算归位——不做跨库/全表 GROUP BY）：
  top10   50 条  排名 1-10 实体（含 be/alpha/gamma 等修复后头部）
  medium 100 条  排名 11-100
  tail   100 条  排名 101+（KB1 修复后共 1,871 活跃实体）
  short   50 条  canonical_name REGEXP '^[A-Za-z]{1,4}$'（短符号，任意排名，
                  与 top10 层重叠如实保留——be 既在头部又是头号短符号）

两级均匀采样（避免 ORDER BY RAND 在 25M 行 JOIN 上全量物化）：
层内实体均匀抽 → 实体内按 link_id 序偏移均匀抽一条；Python 定种子 20261010
全程可重放。实体 mention_count 与实际链接数不一致时懒加载 COUNT 校正。
产物：<out>/stratified_sample.tsv（audit_link_precision.py 输入格式）+
manifest.json（link_id→层/排名；judge 结果按 entity|s|e|surface 回连出报告：
--report <dir> 模式）。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/stratified_link_sample.py \
  --out eval/baselines/results/20261010_stratified_audit [--report <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys

import pymysql

sys.path.insert(0, os.path.dirname(__file__))

STRATA = [("top10", 1, 10, 50), ("medium", 11, 100, 100),
          ("tail", 101, None, 100), ("short", None, None, 50)]
SEED = 20261010
SHORT_RE = "^[A-Za-z]{1,4}$"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def db() -> pymysql.Connection:
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"], port=int(os.environ["MYSQL_PORT"]),
        user=os.environ["MYSQL_USER"], password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DB"], charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor, read_timeout=300)


def rank_entities(conn, kb_hex: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT entity_id, canonical_name, entity_type, mention_count "
            "FROM canonical_entity WHERE kb_id=UNHEX(%s) AND status='ACTIVE' "
            "ORDER BY mention_count DESC, entity_id", (kb_hex,))
        rows = cur.fetchall()
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def link_count(conn, kb_hex: str, entity_id: str, cache: dict) -> int:
    # entity_id 为全局唯一内容哈希（canonical_entity PK 前导列），不再带
    # kb_id 过滤——idx_entity(entity_id) 不含 kb_id，带上过滤每条索引都回表
    if entity_id not in cache:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) c FROM entity_link_registry "
                "WHERE entity_id=%s", (entity_id,))
            cache[entity_id] = cur.fetchone()["c"]
    return cache[entity_id]


def draw_link(conn, kb_hex: str, ent: dict, rng: random.Random,
              link_cache: dict) -> dict | None:
    """实体内按 link_id 序偏移均匀抽一条。

    三步式（首版单查询 ORDER BY l.link_id 带双 JOIN 无法走索引——planner
    全表 PK 扫描过滤 25M 行，单查分钟级实测卡死）：
      ① FORCE INDEX(idx_entity)：二级索引 (entity_id, PK) 同实体内天然按
         link_id 有序，ORDER BY link_id 即索引序，offset 走覆盖索引跳行；
      ② link_id PK 点查取 mention_id；
      ③ mention_id 主键前导列点查水合（实体名/类型用 python 侧排名缓存）。
    """
    for _ in range(6):
        c = link_count(conn, kb_hex, ent["entity_id"], link_cache)
        if c == 0:
            return None
        r = rng.randrange(c)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT l.link_id FROM entity_link_registry l "
                "FORCE INDEX (idx_entity) "
                "WHERE l.entity_id=%s "
                "ORDER BY l.link_id LIMIT %s, 1", (ent["entity_id"], r))
            lid = cur.fetchone()
            if not lid:
                # mention_count 虚高（部分提及无链接）：按实际计数收缩后重抽
                link_cache[ent["entity_id"]] = max(1, r)
                continue
            lid = lid["link_id"]
            cur.execute(
                "SELECT mention_id FROM entity_link_registry "
                "WHERE link_id=%s", (lid,))
            mid = cur.fetchone()["mention_id"]
            cur.execute(
                "SELECT chunk_id, surface_form, span_start, span_end "
                "FROM raw_mention WHERE mention_id=%s LIMIT 1", (mid,))
            m = cur.fetchone()
        if m:
            return {"link_id": lid, "entity_id": ent["entity_id"],
                    "canonical_name": ent["canonical_name"],
                    "entity_type": ent["entity_type"],
                    "surface_form": m["surface_form"],
                    "span_start": m["span_start"], "span_end": m["span_end"],
                    "chunk_id": m["chunk_id"]}
    return None


def sample(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    conn = db()
    kb_hex = args.kb.lower()
    ranked = rank_entities(conn, kb_hex)
    print(f"[entities] {len(ranked)} active")
    rng = random.Random(SEED)
    link_cache: dict[str, int] = {}
    taken: set[str] = set()
    manifest: dict[str, dict] = {}
    tsv_rows: list[list] = []
    for name, lo, hi, n in STRATA:
        if name == "short":
            frame = [e for e in ranked
                     if re.match(SHORT_RE, e["canonical_name"])]
        elif hi:
            frame = [e for e in ranked if lo <= e["rank"] <= hi]
        else:
            frame = [e for e in ranked if e["rank"] >= lo]
        print(f"[stratum {name}] frame={len(frame)} draw={n}")
        got = 0
        while got < n:
            ent = frame[rng.randrange(len(frame))]
            row = draw_link(conn, kb_hex, ent, rng, link_cache)
            if not row or row["link_id"] in taken:
                continue
            taken.add(row["link_id"])
            manifest[row["link_id"]] = {
                "stratum": name, "entity": row["canonical_name"],
                "etype": row["entity_type"], "rank": ent["rank"],
                "mention_count": ent["mention_count"], "offset_draw": True}
            tsv_rows.append([
                row["entity_id"], row["canonical_name"], row["entity_type"],
                row["surface_form"], str(row["span_start"]), str(row["span_end"]),
                row["chunk_id"]])
            got += 1
            if got % 25 == 0:
                print(f"  {got}/{n}")
    with open(os.path.join(args.out, "stratified_sample.tsv"), "w") as f:
        for r in tsv_rows:
            f.write("\t".join(r) + "\n")
    json.dump({"seed": SEED, "kb": args.kb, "strata_def": STRATA,
               "n": len(tsv_rows), "manifest": manifest},
              open(os.path.join(args.out, "manifest.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[saved] {len(tsv_rows)} rows -> {args.out}")


def report(args) -> None:
    man = json.load(open(os.path.join(args.dir, "manifest.json")))
    # TSV 与 manifest 同序生成——直接按行序对齐
    tsv = [ln.rstrip("\n").split("\t") for ln in
           open(os.path.join(args.dir, "stratified_sample.tsv"))]
    strata_seq = [m["stratum"] for m in man["manifest"].values()]
    if len(tsv) != len(strata_seq):
        raise SystemExit(f"tsv/manifest 长度不一致 {len(tsv)} vs {len(strata_seq)}")
    anns = [json.loads(ln) for ln in
            open(os.path.join(args.dir, "annotations.jsonl"))]
    if len(anns) != len(tsv):
        raise SystemExit(f"annotations 不完整 {len(anns)}/{len(tsv)}")
    per: dict[str, dict] = {}
    overall = [0, 0]
    span_ok = 0
    for ann, st in zip(anns, strata_seq):
        d = per.setdefault(st, {"ok": 0, "n": 0, "domain": [0, 0],
                                "generic": [0, 0], "borderline": 0})
        valid = bool(ann.get("match"))
        d["n"] += 1
        d["ok"] += valid
        overall[0] += valid
        overall[1] += 1
        span_ok += bool(ann.get("span_ok"))
        if ann.get("borderline"):
            d["borderline"] += 1
        sp = ann.get("specificity")
        if sp in ("domain", "generic"):
            d[sp][1] += 1
            d[sp][0] += valid
    lines = [f"n={overall[1]} valid={overall[0]} "
             f"precision={overall[0]/overall[1]:.3f} "
             f"wilson95=[{wilson(*overall)[0]:.3f},{wilson(*overall)[1]:.3f}]",
             f"span_integrity={span_ok}/{overall[1]}"]
    for st in ("top10", "medium", "tail", "short"):
        d = per.get(st)
        if not d:
            continue
        lo, hi = wilson(d["ok"], d["n"])
        lines.append(
            f"{st}: {d['ok']}/{d['n']} = {d['ok']/d['n']:.3f} [{lo:.3f},{hi:.3f}]"
            f"  domain {d['domain'][0]}/{d['domain'][1]}"
            f"  generic {d['generic'][0]}/{d['generic'][1]}"
            f"  borderline {d['borderline']}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.dir, "summary_strata.txt"), "w") as f:
        f.write("KB=" + args.kb + "\n" + txt + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="104112E63D9D4AA5A6C21E1C1549EF0F")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", metavar="DIR",
                    help="对已完成审计的目录出分层报告（忽略 --out）")
    args = ap.parse_args()
    if args.report:
        args.dir = args.report
        report(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()
