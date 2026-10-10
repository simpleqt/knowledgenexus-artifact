#!/usr/bin/env python3
"""基准语料入库：建 KB（幂等：按名查现库）→ /api/ingest/batch 分批直传 → stable_id↔doc_id 映射。

语料单元：每 passage 一个 markdown 文件（文件名 = stable_id.md，标题作首行）。
嵌入同步完成，入库即可检索（lexical+dense）；graph/synq 需富化，本轮不开。
用法：.venv/bin/python bench_ingest.py --dataset hotpotqa [--batch 100]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import BASE

ADMIN = "http://127.0.0.1:18100"
INGEST = "http://127.0.0.1:18111"


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        return json.loads(r.read())


def multipart(files: list[tuple[str, str]], kb_hex: str, url: str) -> dict:
    boundary = "----kncbench" + str(time.time_ns())
    body = io.BytesIO()
    for name, text in files:
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
                   f"filename=\"{name}\"\r\nContent-Type: text/markdown\r\n\r\n".encode())
        body.write(text.encode())
        body.write(b"\r\n")
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"kb_hex\"\r\n\r\n"
               f"{kb_hex}\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{url}/api/ingest/batch", data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=1800) as r:  # noqa: S310
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--admin", default=ADMIN, help="admin-api 基址（测试栈 http://127.0.0.1:28100）")
    ap.add_argument("--ingest", default=INGEST, help="ingest 基址（测试栈 http://127.0.0.1:28111）")
    args = ap.parse_args()
    d = os.path.join(BASE, "data", "bench", args.dataset)
    passages = [json.loads(l) for l in open(os.path.join(d, "passages.jsonl"), encoding="utf-8") if l.strip()]

    kb_file = os.path.join(d, "kb.json")
    if os.path.exists(kb_file):
        kb_hex = json.load(open(kb_file))["kb_id"]
        print(f"[skip] KB 已存在 {kb_hex}")
    else:
        kb_code = f"bench-{args.dataset}"
        with urllib.request.urlopen(f"{args.admin}/api/kbs", timeout=30) as r:  # noqa: S310
            kbs = json.loads(r.read())
        hit = next((k for k in (kbs if isinstance(kbs, list) else kbs.get("items", kbs.get("kbs", [])))
                    if k.get("kb_code") == kb_code or k.get("name") == kb_code), None)
        if hit:
            kb_hex = hit.get("kb_id") or hit.get("kb_hex") or hit.get("id")
            print(f"[reuse] 现有库 {kb_hex}")
        else:
            kb_hex = post_json(f"{args.admin}/api/kbs", {
                "kb_code": kb_code, "name": kb_code,
                "subject_domain": "multihop-qa", "schema_id": "generic"})["kb_id"]
            print(f"[created] {kb_hex}")
        json.dump({"kb_id": kb_hex}, open(kb_file, "w"))

    map_file = os.path.join(d, "idmap.json")
    idmap: dict[str, str] = json.load(open(map_file)) if os.path.exists(map_file) else {}
    todo = [p for p in passages if p["stable_id"] not in idmap]
    print(f"待入库 {len(todo)}/{len(passages)} 篇")

    t0 = time.time()
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        resp = multipart([(f"{p['stable_id']}.md", f"{p['title']}\n\n{p['text']}") for p in chunk],
                         kb_hex, args.ingest)
        for r in resp["results"]:
            sid = r["file"].rsplit(".", 1)[0]
            if r["ok"]:
                idmap[sid] = r["doc_id"]
            else:
                print(f"[fail] {sid}: {str(r.get('error'))[:120]}")
        done = min(i + args.batch, len(todo))
        rate = done / max(1, time.time() - t0)
        print(f"  {done}/{len(todo)}  ({rate:.1f} 篇/s, ETA {(len(todo)-done)/max(rate,0.01)/60:.0f}min)")
        json.dump(idmap, open(map_file, "w"))
    print(f"完成：{len(idmap)}/{len(passages)} 映射落盘 idmap.json")


if __name__ == "__main__":
    main()
