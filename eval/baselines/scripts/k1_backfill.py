#!/usr/bin/env python3
"""K1 查询侧词典缺口回填（2026-09-30）：缺典概念 → 语料验证 → 治理式建实体+EXACT 链接。

背景（20260930_aliasx 裁决）：realq 5 题卡 cooc 门的概念不在 canonical_entity
——别名救不了不存在的实体。本脚本以 O-4 冻结的 LLM 查询实体判定为发现面
（verdicts.json），对缺典概念做语料实证后治理式入库：

  dryrun   收集缺典概念 → LLM 映射英文检索词 → OS df 验证 → 候选清单（不写库）
  apply    按 manifest 建实体（canonical=英文语料面形，中文查询词作 TRANS 别名）
           + raw_mention/entity_link EXACT 链接（span 实算，≤200 块/实体）
  rollback 按 source/detector_version 标记整体回滚

判据：df≥2 文档；entity_type 由 LLM 从既有类型词表选；manifest 全程可回滚。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  "$R/.venv/bin/python" "$R/eval/baselines/scripts/k1_backfill.py" dryrun|apply|rollback'
产物：results/20260930_k1gap/{candidates.json,manifest.json,applied.json}
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
KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
VERDICTS = REPO / "eval/baselines/results/20260930_deepq/verdicts.json"
OUT = Path(REPO) / "eval/baselines/results/20260930_k1gap"
MARK = "k1gap-20260930"
TYPES = ("parameter", "method", "circuit", "material", "particle", "detector",
         "experimental_facility", "scintillator", "isotope", "radiation_measurement",
         "nuclear_reaction", "energy_spectrum", "photodetector", "instrument",
         "technology", "concept", "term", "process", "facility", "phenomenon")


def nhash(text: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFKC", text).lower().encode()).hexdigest()


def dict_names(db) -> set[str]:
    names = {r["n"].lower() for r in db.exec(
        "SELECT canonical_name n FROM canonical_entity WHERE kb_id=UNHEX(%s)",
        (KB,), fetch=True)}
    names |= {r["a"].lower() for r in db.exec(
        "SELECT alias_text a FROM entity_alias WHERE kb_id=UNHEX(%s)", (KB,), fetch=True)}
    return names


def os_search(term: str, size: int = 200) -> list[dict]:
    from knc_engine import search as S
    r = S.client().search(index=S._CHUNK_ALIAS, body={
        "size": size, "track_total_hits": True,
        "query": {"bool": {"filter": [{"term": {"kb_id": KB}}, {"term": {"visible": True}}],
                            "must": [{"match_phrase": {"content_text": term}}]}},
        "_source": ["chunk_id", "doc_id", "chunk_generation", "content_text"]})
    return [h["_source"] for h in r["hits"]["hits"]]


def llm_map(llm, concept: str) -> dict | None:
    raw = llm.chat(
        f'查询概念：{concept}\n'
        "这是核科技学术文献库（英文为主）的检索词典补全任务。判断该概念是否为"
        "领域实体（非人名/机构/泛日常词），并给出英文语料面形："
        '只输出 JSON {"ok": true/false, "name": "英文规范名(小写)", '
        '"type": "类型", "variants": ["英文变体≤2"]}。type 从这些里选：'
        + "/".join(TYPES) + "。没把握 ok=false。",
        system="你是核科技领域术语专家，只输出 JSON。",
        max_tokens=200, temperature=0.0, timeout=15)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    d = json.loads(raw)
    if d.get("ok") and d.get("name") and d.get("type") in TYPES:
        return {"name": str(d["name"]).lower()[:200], "type": d["type"],
                "variants": [str(v).lower()[:200] for v in (d.get("variants") or [])][:2]}
    return None


def collect_missing(db) -> list[str]:
    verdicts = json.load(open(VERDICTS, encoding="utf-8"))
    have = dict_names(db)
    missing: dict[str, int] = {}
    for v in verdicts.values():
        for e in (v or {}).get("entities") or []:
            if e and e.lower() not in have:
                missing[e.lower()] = missing.get(e.lower(), 0) + 1
    return sorted(missing, key=lambda k: -missing[k])


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dryrun"
    from knc_engine import db, llm
    OUT.mkdir(parents=True, exist_ok=True)

    if cmd == "dryrun":
        missing = collect_missing(db)
        print(f"缺典概念 {len(missing)} 个（verdicts 实体去重后）")
        cands = []
        mf = OUT / "candidates.json"
        done = set()
        if mf.exists():
            cands = json.load(open(mf, encoding="utf-8"))
            done = {c["concept"] for c in cands if c.get("verdict") in ("accept", "reject")}
        t0 = time.time()
        for i, c in enumerate(missing):
            if c in done:
                continue
            verdict = "reject"
            item = {"concept": c}
            m = None
            try:
                m = llm_map(llm, c)
            except Exception:
                m = None
            if m:
                terms = [m["name"]] + [v for v in m["variants"] if v]
                best = None
                for t in dict.fromkeys(terms):
                    hits = os_search(t)
                    docs = {h["doc_id"] for h in hits}
                    if len(docs) >= 2 and (best is None or len(docs) > best["df"]):
                        best = {"term": t, "df": len(docs), "chunks": len(hits)}
                if best:
                    verdict = "accept"
                    item.update({**m, **best})
            item["verdict"] = verdict
            cands.append(item)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(missing)} accept={sum(1 for x in cands if x['verdict']=='accept')}"
                      f" ({(time.time()-t0)/(i+1):.1f}s)", flush=True)
                json.dump(cands, open(mf, "w"), ensure_ascii=False, indent=1)
        json.dump(cands, open(mf, "w"), ensure_ascii=False, indent=1)
        acc = [x for x in cands if x["verdict"] == "accept"]
        print(f"[candidates] accept {len(acc)} / reject {len(cands)-len(acc)}")
        for x in acc:
            print(f"   {x['concept']} -> {x['name']}({x['type']}) df={x['df']}")

    elif cmd == "apply":
        cands = [x for x in json.load(open(OUT / "candidates.json", encoding="utf-8"))
                 if x["verdict"] == "accept"]
        manifest = []
        for x in cands:
            hits = os_search(x["term"])
            by_chunk = {h["chunk_id"]: h for h in hits}
            docs = {h["doc_id"] for h in hits}
            eid = hashlib.sha256(f"{KB}|k1gap|{x['name']}".encode()).hexdigest()
            db.exec("INSERT IGNORE INTO canonical_entity (entity_id, kb_id, sense_id,"
                    " canonical_name, normalized_name_hash, entity_type,"
                    " ownership_partition_id, creation_blocking_key_hash,"
                    " mention_count, doc_df, status)"
                    " VALUES (%s,UNHEX(%s),1,%s,%s,%s,1,%s,%s,%s,'ACTIVE')",
                    (eid, KB, x["name"], nhash(x["name"]), x["type"],
                     hashlib.sha256(f"{KB}|{x['name']}|{x['type']}".encode()).hexdigest(),
                     len(hits), len(docs)))
            if x["concept"].lower() != x["name"].lower():
                db.exec("INSERT IGNORE INTO entity_alias (entity_id, kb_id, alias_text,"
                        " normalized_alias_hash, alias_type, source)"
                        " VALUES (%s,UNHEX(%s),%s,%s,'TRANS',%s)",
                        (eid, KB, x["concept"], nhash(x["concept"]), MARK))
            n_links = 0
            for h in hits[:200]:
                text = h.get("content_text") or ""
                pos = text.lower().find(x["term"].lower())
                if pos < 0:
                    continue
                mid = hashlib.sha256(
                    f"{KB}|{h['chunk_id']}|{eid}".encode()).hexdigest()[:64]
                db.exec("INSERT IGNORE INTO raw_mention (mention_id, kb_id, doc_id,"
                        " chunk_id, chunk_generation, surface_form, normalized_form,"
                        " span_start, span_end, confidence, detector_version)"
                        " VALUES (%s,UNHEX(%s),UNHEX(%s),%s,%s,%s,%s,%s,%s,0.95,%s)",
                        (mid, KB, h["doc_id"], h["chunk_id"], h["chunk_generation"],
                         x["term"], x["term"].lower(), pos, pos + len(x["term"]), MARK))
                db.exec("INSERT IGNORE INTO entity_link_registry (kb_id, mention_id,"
                        " entity_id, confidence, method, adjudicated_by, chunk_generation)"
                        " VALUES (UNHEX(%s),%s,%s,0.95,'MANUAL','RULE',%s)",
                        (KB, mid, eid, h["chunk_generation"]))
                n_links += 1
            manifest.append({"entity_id": eid, "concept": x["concept"], "name": x["name"],
                             "type": x["type"], "df": len(docs), "links": n_links})
            print(f"  + {x['name']} ({x['concept']}) df={len(docs)} links={n_links}")
        json.dump({"kb": KB, "mark": MARK, "entities": manifest},
                  open(OUT / "manifest.json", "w"), ensure_ascii=False, indent=1)
        json.dump({"applied_at": time.strftime("%F %T"),
                   "entities": len(manifest)}, open(OUT / "applied.json", "w"))
        print(f"[applied] {len(manifest)} 实体（detector_version={MARK} 可回滚）")

    elif cmd == "rollback":
        db.exec("DELETE FROM entity_link_registry WHERE kb_id=UNHEX(%s) AND chunk_generation=%s",
                (KB, MARK))
        db.exec("DELETE FROM raw_mention WHERE kb_id=UNHEX(%s) AND detector_version=%s",
                (KB, MARK))
        db.exec("DELETE FROM entity_alias WHERE kb_id=UNHEX(%s) AND source=%s", (KB, MARK))
        mf = OUT / "manifest.json"
        if mf.exists():
            for e in json.load(open(mf, encoding="utf-8"))["entities"]:
                db.exec("DELETE FROM canonical_entity WHERE entity_id=%s", (e["entity_id"],))
        print("[rollback] 完成")
    else:
        raise SystemExit("用法: k1_backfill.py dryrun|apply|rollback")


if __name__ == "__main__":
    main()
