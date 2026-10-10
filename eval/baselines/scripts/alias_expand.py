#!/usr/bin/env python3
"""① 概念实体别名扩产（2026-09-30）：LLM 批量生成缩写/双语/同义别名。

背景（20260930 诊断）：概念实体 457 个零别名、有余者平均仅 1.5 条；缩写
（SiPM/ICF 类查询真实写法）是最大空档。别名流入三处已验证机制：扫描词典
（cooc/graph/PPR 种子面形）、O11 词法双语扩展（+2.2pt 定稿）、⑦ gctx 间接
（链接不受影响，扫描面形受益）。

流程：generate（LLM 温度 0，冻结到 manifest）→ 人工抽样 → apply（INSERT
IGNORE + source='aliasx-20260930'，manifest 可整体回滚）。

校验（生成侧硬规则）：2≤len≤80；≠自身 canonical 名；不与全库任何既有
canonical/alias 名碰撞（扫描映射完整性——碰撞别名会让面组解析歧义）；
(entity_id, NFKC lower) 去重；每实体 ≤4 条。

用法：
  bash -c 'source eval/baselines/scripts/knc_env.sh && \
    "$R/.venv/bin/python" "$R/eval/baselines/scripts/alias_expand.py" generate'
  ... 抽样检查 manifest 后：
  ... alias_expand.py apply
产物：eval/baselines/results/20260930_aliasx/{manifest.json,applied.json}
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.path.dirname(__file__))
from common import REPO as _R  # noqa: F401,E402  （目录锚定）

KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
BIB = ("paper", "person", "organization")
SOURCE = "aliasx-20260930"
OUT = Path(REPO) / "eval/baselines/results/20260930_aliasx"


def ahash(text: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFKC", text).lower().encode()).hexdigest()


def load_targets(db) -> list[dict]:
    return [dict(r) for r in db.exec(
        "SELECT entity_id e, canonical_name n, entity_type t, doc_df"
        " FROM canonical_entity WHERE kb_id=UNHEX(%s) AND status='ACTIVE'"
        f" AND entity_type NOT IN {BIB} ORDER BY doc_df DESC", (KB,), fetch=True)]


def existing_names(db) -> set[str]:
    names = {r["n"].lower() for r in db.exec(
        "SELECT canonical_name n FROM canonical_entity WHERE kb_id=UNHEX(%s)",
        (KB,), fetch=True)}
    names |= {r["a"].lower() for r in db.exec(
        "SELECT alias_text a FROM entity_alias WHERE kb_id=UNHEX(%s)", (KB,), fetch=True)}
    return names


def existing_alias_keys(db) -> set[tuple[str, str]]:
    return {(r["e"], r["h"]) for r in db.exec(
        "SELECT entity_id e, normalized_alias_hash h FROM entity_alias"
        " WHERE kb_id=UNHEX(%s)", (KB,), fetch=True)}


def gen_one(llm, name: str, etype: str) -> list[dict]:
    raw = llm.chat(
        f'实体名：{name}\n类型：{etype}\n'
        "为该核科技/学术领域的实体列出检索用别名：常用缩写（如 silicon "
        "photomultiplier→SiPM）、跨语言对应（中↔英）、常见同义改写。"
        '只输出 JSON 数组，元素 {"alias": "...", "type": "ABBREV|FULL|TRANS"}，'
        "最多 4 个；不要输出实体名本身；没有把握的不要编。[] 表示无。",
        system="你是核科技领域术语专家，只输出 JSON。",
        max_tokens=300, temperature=0.0, timeout=15)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    arr = json.loads(raw)
    return [{"alias": str(x["alias"]), "type": x["type"]}
            for x in arr if isinstance(x, dict) and x.get("alias")
            and x.get("type") in ("ABBREV", "FULL", "TRANS")][:4]


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    from knc_engine import db, llm
    OUT.mkdir(parents=True, exist_ok=True)
    targets = load_targets(db)
    taken = existing_names(db)
    have_keys = existing_alias_keys(db)
    print(f"目标概念实体 {len(targets)} 个；全库既有名 {len(taken)} 个")

    if cmd == "generate":
        manifest = {"kb": KB, "source": SOURCE, "items": []}
        mf = OUT / "manifest.json"
        done_ids = set()
        if mf.exists():  # 断点续跑
            old = json.load(open(mf, encoding="utf-8"))
            manifest = old
            done_ids = {it["entity_id"] for it in old["items"]}
            print(f"续跑：已完成 {len(done_ids)} 实体")
        t0 = time.time()
        n_new = 0
        for i, ent in enumerate(targets):
            if ent["e"] in done_ids:
                continue
            aliases: list[dict] = []
            try:
                cands = gen_one(llm, ent["n"], ent["t"])
            except Exception:
                cands = []
            for c in cands:
                a = c["alias"].strip()
                lo = a.lower()
                if not (2 <= len(a) <= 80) or lo == ent["n"].lower() or lo in taken:
                    continue
                if (ent["e"], ahash(a)) in have_keys:
                    continue
                aliases.append({"alias": a, "type": c["type"], "hash": ahash(a)})
                taken.add(lo)
            if aliases:
                manifest["items"].append({"entity_id": ent["e"], "name": ent["n"],
                                          "type": ent["t"], "aliases": aliases})
                n_new += len(aliases)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(targets)} 新别名 {n_new} ({(time.time()-t0)/(i+1):.2f}s/ent)",
                      flush=True)
                json.dump(manifest, open(mf, "w"), ensure_ascii=False, indent=1)
        json.dump(manifest, open(mf, "w"), ensure_ascii=False, indent=1)
        print(f"[manifest] {len(manifest['items'])} 实体 / {n_new} 新别名 -> {mf}")
        for it in manifest["items"][:20]:
            print("  样例:", it["name"], "->", [a["alias"] for a in it["aliases"]])

    elif cmd == "apply":
        manifest = json.load(open(OUT / "manifest.json", encoding="utf-8"))
        n = 0
        for it in manifest["items"]:
            for a in it["aliases"]:
                db.exec("INSERT IGNORE INTO entity_alias (entity_id, kb_id, alias_text,"
                        " normalized_alias_hash, alias_type, source)"
                        " VALUES (%s,UNHEX(%s),%s,%s,%s,%s)",
                        (it["entity_id"], KB, a["alias"], a["hash"], a["type"], SOURCE))
                n += 1
        json.dump({"applied_at": time.strftime("%F %T"), "rows_attempted": n,
                   "entities": len(manifest["items"]), "source": SOURCE},
                  open(OUT / "applied.json", "w"), ensure_ascii=False, indent=1)
        cnt = db.exec("SELECT COUNT(*) n FROM entity_alias WHERE kb_id=UNHEX(%s)"
                      " AND source=%s", (KB, SOURCE), fetch=True)[0]["n"]
        print(f"[applied] 尝试 {n} 行，落库 {cnt} 行（INSERT IGNORE 幂等）")

    elif cmd == "rollback":
        db.exec("DELETE FROM entity_alias WHERE kb_id=UNHEX(%s) AND source=%s",
                (KB, SOURCE))
        print("[rollback] 已删除 source=aliasx-20260930 的全部别名")

    else:
        raise SystemExit("用法: alias_expand.py generate|apply|rollback")


if __name__ == "__main__":
    main()
