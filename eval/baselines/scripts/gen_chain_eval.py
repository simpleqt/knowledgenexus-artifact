#!/usr/bin/env python3
"""⑧ 链证据进生成：knowledge_chains 注入生成上下文的生产侧评测（2026-09-30）。

背景：knowledge_chains 是图谱通道的纯输出面（不投票），生成侧从未消费过
图谱结构证据。本脚本在生产深轨（含 graph 通道）上对比两臂：

  base   深轨 chunks（定稿配置）构建上下文
  chain  同 chunks + knowledge_chains 渲染为「图谱证据链」块注入上下文

裁判三件套（faithfulness/relevancy/correctness）+ 参考答案（金标文档证据块
→ LLM 撰写），n=nea48（关联题是链的主场）。题集为生产 id 空间，直接调
生产 API（两级口径：全量走生产只读）。

用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  "$R/.venv/bin/python" "$R/eval/baselines/scripts/gen_chain_eval.py" [--limit N]'
产物：results/20260930_genchain/{references.json,gen_base.json,gen_chain.json,summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KB = "21bcbed8ea6b4a5bbf55f3e3e9988c1f"
OUT = REPO / "eval/baselines/results/20260930_genchain"
API = "http://127.0.0.1:18121/api/retrieve/deep"

GEN_SYSTEM = ("你是核电子学领域的研究助理，仅基于给定检索上下文回答，"
              "上下文不足以回答的部分明确说明。")


def load_env() -> None:
    for line in open(REPO / "deploy" / ".env.services", encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)


def llm(system: str, user: str, max_tokens: int = 900) -> str:
    from knc_engine import llm as L
    return L.chat(user, system=system, max_tokens=max_tokens, temperature=0.2, timeout=60)


def jscore(text: str) -> dict:
    try:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        return json.loads(raw)
    except Exception:
        return {}


def retrieve_deep(query: str) -> dict:
    req = urllib.request.Request(
        API, data=json.dumps({"kb_id": KB, "query": query, "top_k": 10}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=150) as r:  # noqa: S310
        return json.loads(r.read())


def render_chains(kcs: list[dict]) -> str:
    lines = []
    for kc in (kcs or [])[:8]:
        src = kc.get("src_name") or kc.get("src") or "?"
        dst = kc.get("dst_name") or kc.get("dst") or "?"
        rel = "/".join(kc.get("relation_labels") or kc.get("types") or []) or "相关"
        lines.append(f"- {src} —[{rel}]→ {dst}")
    return "\n".join(lines)


def build_ctx(chunks: list[dict], kcs: list[dict], with_chain: bool) -> str:
    parts = [f"[{i+1}] {c.get('title_text') or ''}\n{c['content_text'][:1200]}"
             for i, c in enumerate(chunks)]
    if with_chain:
        ch = render_chains(kcs)
        if ch:
            parts.append("【图谱证据链】（实体关系证据，供交叉参考）\n" + ch)
    return "\n\n---\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    qs = [json.loads(l) for l in
          open(REPO / "eval/harness/ne_journal_assoc_q48.jsonl", encoding="utf-8")
          if l.strip()]
    if args.limit:
        qs = qs[:args.limit]

    # ① 参考答案（深轨检索 → 金标块 → LLM 撰写）
    ref_file = OUT / "references.json"
    refs: dict[str, str] = json.load(open(ref_file)) if ref_file.exists() else {}
    for q in qs:
        if q["query"] in refs:
            continue
        resp = retrieve_deep(q["query"])
        gold = set(q["gold_doc_ids"])
        gctx = [c for c in resp["chunks"] if c.get("doc_id") in gold][:6]
        if not gctx:
            refs[q["query"]] = "（深轨 top-10 未含金标，参考缺失）"
        else:
            ctx = "\n\n---\n\n".join(f"[{i+1}] {c['content_text'][:1200]}"
                                     for i, c in enumerate(gctx))
            refs[q["query"]] = llm(
                "你是核电子学专家，基于给定证据撰写参考答案，只覆盖证据支持的内容。",
                f"【证据】\n{ctx}\n\n【问题】\n{q['query']}\n\n请写参考答案。")
        json.dump(refs, open(ref_file, "w"), ensure_ascii=False, indent=1)
        print(f"[ref {len(refs)}/{len(qs)}]", flush=True)

    # ② 两臂生成 + 裁判（同一次检索，两臂共享 chunks/kcs —— 纯上下文差）
    for arm in ("base", "chain"):
        out_file = OUT / f"gen_{arm}.json"
        rows: list[dict] = json.load(open(out_file)) if out_file.exists() else []
        done = {r["query"] for r in rows}
        for qi, q in enumerate(qs):
            if q["query"] in done:
                continue
            resp = retrieve_deep(q["query"])
            ctx = build_ctx(resp["chunks"], resp.get("knowledge_chains"), arm == "chain")
            answer = llm(GEN_SYSTEM, f"【检索上下文】\n{ctx}\n\n【问题】\n{q['query']}")
            faith = jscore(llm(
                "你是严谨的评审。判断回答中的论断被上下文支持的程度。",
                f"【上下文】\n{ctx}\n\n【回答】\n{answer}\n\n"
                '请把回答拆成独立论断，逐条判断是否被上下文支持。只输出 JSON：'
                '{"claims": 总论断数, "supported": 被支持数}'))
            rel = jscore(llm(
                "你是评审。给回答对问题的相关性打分（1-5 整数）。只输出 JSON："
                '{"relevancy": 分}', f"【问题】\n{q['query']}\n\n【回答】\n{answer}"))
            corr = jscore(llm(
                "你是评审。对比参考答案给正确性打分（1-5 整数）。只输出 JSON："
                '{"correctness": 分}',
                f"【参考答案】\n{refs.get(q['query'], '')}\n\n【回答】\n{answer}"))
            rows.append({"query": q["query"], "answer": answer,
                         "n_chains": len(resp.get("knowledge_chains") or []),
                         "faith": faith, "rel": rel, "corr": corr})
            if (qi + 1) % 5 == 0:
                print(f"[{arm} {qi+1}/{len(qs)}]", flush=True)
            json.dump(rows, open(out_file, "w"), ensure_ascii=False, indent=1)
        f = [r["faith"].get("supported", 0) / max(1, r["faith"].get("claims", 1)) for r in rows]
        print(f"[{arm}] n={len(rows)} faith={sum(f)/max(1,len(f)):.3f} "
              f"rel={sum(r['rel'].get('relevancy',0) for r in rows)/max(1,len(rows)):.2f} "
              f"corr={sum(r['corr'].get('correctness',0) for r in rows)/max(1,len(rows)):.2f} "
              f"chains题均={sum(r['n_chains'] for r in rows)/max(1,len(rows)):.1f}", flush=True)


if __name__ == "__main__":
    main()
