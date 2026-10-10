#!/usr/bin/env python3
"""链接精度抽样审计（2026-10-09 审查补实验 B，C线#10）：

  100 条 entity_link（KB1 ORDER BY RAND(20261009) 定种子可复现）逐条：
  ① OS 取提及 chunk 全文，程序校验 span 完整性（text[s:e]==surface）；
  ② 上下文窗 ±250 字符，⟦⟧ 标记提及位，LLM（router qwen3.5-35b，temp 0）
     双问严格判：
       match       —— span 是否确指实体 canonical 同一概念（异义/异概念/非
                     术语提及即 false；可争议标 borderline）
       specificity —— domain（领域术语）/ generic（泛学术词 method/result/…）
  ③ 汇总：总体精度 + Wilson 95% CI，generic/domain 分层，span 完整率。
  审查回应点：词典质量长尾（泛词实体）到底把链接精度压到多少——
  "链接正确但信息量为零"与"真错链"分开计数。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/audit_link_precision.py \
  --sample /tmp/link_sample.tsv --out eval/baselines/results/20261009_linkaudit'
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402  (OS client 经 search 模块复用)

PROMPT = """You are a strict information-extraction auditor for a scientific knowledge base (nuclear-electronics / physics journal corpus).

Entity canonical name: {name}
Entity type: {etype}
Extracted mention surface: {surface}

Document context (the extracted span is marked):
{ctx}

Answer TWO questions as a single JSON object only:
1. "match": does the marked span in context denote the SAME concept as the entity canonical name "{name}"? true only if clearly yes. false if the span is a different sense (e.g. verb "lead" vs metal lead), a different concept, or not a term mention at all. If genuinely debatable, set "match": false and "borderline": true.
2. "specificity": "domain" if the canonical name is a domain-specific technical concept (e.g. HPGe detector, Compton edge, neutron irradiation, silicon drift detector); "generic" if it is a generic academic/engineering word usable in any field (e.g. method, result, problem, term, parameter, function, experiment, error).

JSON keys: match (bool), borderline (bool), specificity ("domain"|"generic"), reason (one short sentence). JSON only, no other text."""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def llm_judge(base: str, key: str, model: str, name: str, etype: str,
              surface: str, ctx: str) -> dict:
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": 300,
        "messages": [{"role": "user", "content": PROMPT.format(
            name=name, etype=etype, surface=surface, ctx=ctx)}],
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        txt = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {"match": None, "borderline": True, "specificity": "?",
                "reason": f"unparseable: {txt[:80]}"}
    try:
        j = json.loads(m.group(0))
        return {"match": bool(j.get("match")),
                "borderline": bool(j.get("borderline")),
                "specificity": str(j.get("specificity", "?")),
                "reason": str(j.get("reason", ""))[:200]}
    except Exception:
        return {"match": None, "borderline": True, "specificity": "?",
                "reason": f"json error: {txt[:80]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/tmp/link_sample.tsv")
    ap.add_argument("--kb", required=True,
                    help="采样链接所属库 hex（首版误用评测切片库过滤，"
                         "100 条全部取空文本——judge 无上下文，结果作废）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    base = os.environ["LLM_API_BASE"]
    key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL"]

    rows = []
    with open(args.sample) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) == 7:
                rows.append(dict(zip(
                    ("entity_id", "name", "etype", "surface",
                     "s", "e", "chunk_id"), p)))
    print(f"[sample] {len(rows)} links")

    outp = open(os.path.join(args.out, "annotations.jsonl"), "w")
    n = ok = span_ok = 0
    by_class = {"domain": [0, 0], "generic": [0, 0]}  # [valid, total]
    for i, r in enumerate(rows):
        s, e = int(r["s"]), int(r["e"])
        try:
            ch = M.search.chunks_by_ids(args.kb, [r["chunk_id"]], 1)
            text = ch[0]["content_text"] if ch else ""
        except Exception as exc:
            text = ""
            print(f"[warn] OS fetch {r['chunk_id'][:8]}: {exc}")
        span_hit = bool(text) and text[s:e].lower() == r["surface"].lower()
        span_ok += span_hit
        if span_hit:
            lo, hi = max(0, s - 250), min(len(text), e + 250)
            ctx = text[lo:s] + "⟦" + text[s:e] + "⟧" + text[e:hi]
        elif text:
            # span 漂移：以最近一次 surface 出现为中心兜底（并如实记录）
            j = text.lower().find(r["surface"].lower(), max(0, s - 80))
            j = j if j >= 0 else text.lower().find(r["surface"].lower())
            if j >= 0:
                e2 = j + len(r["surface"])
                lo, hi = max(0, j - 250), min(len(text), e2 + 250)
                ctx = text[lo:j] + "⟦" + text[j:e2] + "⟧" + text[e2:hi]
            else:
                ctx = text[:500]
        else:
            ctx = "(chunk text unavailable)"
        v = llm_judge(base, key, model, r["name"], r["etype"], r["surface"], ctx)
        rec = {**r, "span_ok": span_hit, **v}
        outp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        outp.flush()
        n += 1
        if v["match"]:
            ok += 1
            by_class.setdefault(v["specificity"], [0, 0])[0] += 1
        by_class.setdefault(v["specificity"], [0, 0])[1] += 1
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(rows)}] running precision {ok}/{n}")
        time.sleep(0.2)
    outp.close()

    lo, hi = wilson(ok, n)
    lines = [
        f"n={n} valid={ok} precision={ok/n:.3f} wilson95=[{lo:.3f},{hi:.3f}]",
        f"span_integrity={span_ok}/{n} ({span_ok/n:.3f})",
    ]
    for c, (k, t) in sorted(by_class.items()):
        if t:
            cl, ch_ = wilson(k, t)
            lines.append(f"{c}: valid {k}/{t} = {k/t:.3f} [{cl:.3f},{ch_:.3f}]")
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("KB=" + args.kb + "\n" + summary + "\n")


if __name__ == "__main__":
    main()
