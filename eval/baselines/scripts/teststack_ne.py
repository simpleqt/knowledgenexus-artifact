#!/usr/bin/env python3
"""ne 切片（300 篇）导入隔离测试栈并评测 KNC 通道消融臂。

用法：
  ingest: .venv/bin/python teststack_ne.py ingest          # 300 篇 → 28111，文件名=原 doc_id
  eval:   .venv/bin/python teststack_ne.py eval [channels]  # 缺省=融合；传 lexical / dense
产物：data/teststack_ne/{kb.json,idmap.json}、runs/knc[-arm]_<set>_teststack.json
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import BASE, DATA, REPO, load_docs, load_questions, write_predictions

ADMIN = "http://127.0.0.1:28100"
INGEST = "http://127.0.0.1:28111"
API = os.environ.get("TESTSTACK_API", "http://127.0.0.1:28121/api/retrieve")
OUT = os.path.join(DATA, "teststack_ne")


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        return json.loads(r.read())


def multipart(files: list[tuple[str, str]], kb_hex: str, url: str) -> dict:
    boundary = "----kncne" + str(time.time_ns())
    body = io.BytesIO()
    for name, text in files:
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
                   f"filename=\"{name}\"\r\nContent-Type: text/markdown\r\n\r\n".encode())
        body.write(text.encode())
        body.write(b"\r\n")
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"kb_hex\"\r\n\r\n"
               f"{kb_hex}\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(url, data=body.getvalue(),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=1800) as r:  # noqa: S310
        return json.loads(r.read())


def do_ingest() -> None:
    os.makedirs(OUT, exist_ok=True)
    docs = load_docs()                                  # 300 篇切片全文
    kb_file = os.path.join(OUT, "kb.json")
    if os.path.exists(kb_file):
        kb_hex = json.load(open(kb_file))["kb_id"]
        print(f"[skip] 测试库已存在 {kb_hex}")
    else:
        kb_hex = post_json(f"{ADMIN}/api/kbs", {
            "kb_code": "ne-slice-test", "name": "ne-slice-test",
            "subject_domain": "nuclear-electronics", "schema_id": "generic"})["kb_id"].lower()
        json.dump({"kb_id": kb_hex}, open(kb_file, "w"))
        print(f"[created] {kb_hex}")
    map_file = os.path.join(OUT, "idmap.json")
    idmap: dict[str, str] = json.load(open(map_file)) if os.path.exists(map_file) else {}
    todo = [d for d in docs if d["doc_id"] not in idmap]
    print(f"待入库 {len(todo)}/{len(docs)} 篇")
    t0 = time.time()
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        resp = multipart([(d["doc_id"] + ".md", d["title"] + "\n\n" + d["text"]) for d in batch],
                         kb_hex, f"{INGEST}/api/ingest/batch")
        for r in resp["results"]:
            sid = r["file"].rsplit(".", 1)[0]
            if r["ok"]:
                idmap[sid] = r["doc_id"]
            else:
                print(f"[fail] {sid}: {str(r.get('error'))[:100]}")
        done = min(i + 50, len(todo))
        print(f"  {done}/{len(todo)}  ({done/max(1,time.time()-t0):.1f} 篇/s)", flush=True)
        json.dump(idmap, open(map_file, "w"))
    print(f"完成：{len(idmap)}/{len(docs)} 映射")


def do_eval(channels: str | None, tag: str = "", sets: tuple[str, ...] = ("nea", "ner"),
            deep: bool = False) -> None:
    kb_hex = json.load(open(os.path.join(OUT, "kb.json")))["kb_id"]
    idmap = json.load(open(os.path.join(OUT, "idmap.json")))
    rev = {v: k for k, v in idmap.items()}
    url = API + "/deep" if deep else API
    for set_name in sets:
        questions = load_questions(set_name)
        per_query = []
        for q in questions:
            body = {"kb_id": kb_hex, "query": q["query"], "top_k": 20}
            if channels:
                body["channels"] = channels.split(",")
            t0 = time.time()
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=int(os.environ.get('TESTSTACK_TIMEOUT', '300'))) as r:  # noqa: S310
                chunks = json.loads(r.read())["chunks"]
            dt = time.time() - t0
            hits, rank = [], 0
            for c in chunks:
                oid = rev.get(c["doc_id"])
                if oid is None:
                    continue
                rank += 1
                hits.append({"rank": rank, "doc_id": oid, "chunk_id": c.get("chunk_id"),
                             "score": c.get("_score")})
            per_query.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                              "hits": hits, "latency_s": round(dt, 3)})
        arm = f"-{channels.replace(',', '+')}" if channels else ""
        write_predictions(f"knc{arm}{tag}", "teststack", set_name,
                          {"wall_s": 0, "note": f"隔离测试栈（300 篇 ne 切片，lexical+dense）；"
                           f"channels={channels or '默认'}；top_k=20"},
                          per_query, len(idmap))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ingest":
        do_ingest()
    elif cmd == "eval":
        sets_arg = (sys.argv[sys.argv.index("--sets") + 1]
                    if "--sets" in sys.argv else "nea,ner")
        tag = (sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv
               else next((a[6:] for a in sys.argv if a.startswith("--tag=")), ""))
        do_eval(sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else None,
                tag=tag,
                sets=tuple(s for s in sets_arg.split(",") if s),
                deep="--deep" in sys.argv)
    else:
        raise SystemExit("用法: teststack_ne.py ingest | eval [lexical|dense|...] [--tag T] [--sets nea,ner]")
