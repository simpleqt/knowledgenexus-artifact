#!/usr/bin/env python3
"""端到端生成质量消融（2026-09-27 起）：检索走当前 API，生成与裁判走 LLM API。

臂：
  full（默认七通道）/ denseonly / lexonly —— 生成质量差异反映检索质量差异；
  fullprune  句级上下文裁剪：查询-句子嵌入相似度做查询感知选择（LongLLMLingua/
             LooComp 式 selective context），预算 6500 字符（约全量 55%）；
  fulldoc    文档卡片：每块前拼《文档标题》/章节头（消费侧文档级组织，
             B81 hier_summary"消费另议"的落点）；
  fullboth   裁剪 + 文档卡片。
题集：ne_journal_realq 39 题（真实问句，金标文档已知）。
指标（裁判 LLM 逐题 JSON 打分，0~1）：Faithfulness（论断被上下文支持比例）、
Answer Relevancy（聚焦度）、Answer Correctness（对照参考答案）。
参考答案：由金标文档的证据分块经 LLM 生成一次并缓存（无人工参考答案时的
标准做法，生成与裁判同模型——与论文口径一致并如实报告）。
用法：gen_eval.py [--arms full,denseonly,lexonly,fullprune,fulldoc,fullboth]
                 [--limit N] [--out 20260927_gen_quality]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from common import BASE, DATA, load_questions  # noqa: E402

API = "http://127.0.0.1:28121/api/retrieve"
KB_FILE = os.path.join(DATA, "teststack_ne", "kb.json")
IDMAP = os.path.join(DATA, "teststack_ne", "idmap.json")

ARM_CHANNELS = {"full": None, "denseonly": ["dense"], "lexonly": ["lexical"],
                "fullprune": None, "fulldoc": None, "fullboth": None}
PRUNE_BUDGET_CHARS = 6500

GEN_SYSTEM = (
    "你是核电子学与核探测技术专家。只根据给定检索上下文回答；"
    "上下文不足时明确指出知识库中未检索到充分依据，不得编造。"
    "关键结论后以 [n] 标注来源片段编号；回答要完整覆盖问题问点，"
    "参数与数值须与上下文逐项核对，不要引入上下文之外的工程数值。")


def llm(system: str, user: str, max_tokens: int = 1024) -> str:
    base = os.environ["LLM_API_BASE"].rstrip("/")
    body = json.dumps({
        "model": os.environ["LLM_MODEL"], "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(8 * (attempt + 1))
            req.data = body  # 重试同请求
    return ""


def retrieve(kb: str, query: str, channels: list[str] | None, k: int = 10) -> list[dict]:
    body: dict = {"kb_id": kb, "query": query, "top_k": k}
    if channels:
        body["channels"] = channels
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310
        return json.loads(r.read())["chunks"]


def jscore(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量嵌入（OpenAI 兼容 /embeddings；每批 ≤48 条防超限）。"""
    base = os.environ["EMBEDDING_API_BASE"].rstrip("/")
    out: list[list[float]] = []
    for i in range(0, len(texts), 48):
        batch = [t[:2000] for t in texts[i:i + 48]]
        body = json.dumps({"model": os.environ["EMBEDDING_MODEL"],
                           "input": batch}).encode()
        req = urllib.request.Request(f"{base}/embeddings", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {os.environ['EMBEDDING_API_KEY']}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                    data = json.loads(r.read())["data"]
                out.extend(d["embedding"] for d in sorted(data, key=lambda d: d["index"]))
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
    return out


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > 300:  # 超长句按逗号硬切
            cut = max(p.rfind("，", 200, 300), p.rfind(",", 200, 300), 200)
            out.append(p[:cut + 1])
            p = p[cut + 1:]
        out.append(p)
    return out


def prune_context(chunks: list[dict], query: str, budget: int) -> list[dict]:
    """查询感知句级裁剪：嵌入相似度选句，预算内保留高分句，按原序重组为
    {doc_id, content_text} 列表（块编号与文档卡片逻辑保持兼容）。"""
    units: list[tuple[int, str]] = []  # (chunk_idx, sentence)
    for ci, c in enumerate(chunks):
        for s in split_sentences(c.get("content_text", "")):
            units.append((ci, s))
    if not units:
        return chunks
    qemb = embed_texts([query])[0]
    semb = embed_texts([s for _, s in units])
    order = sorted(range(len(units)), key=lambda j: -_cos(qemb, semb[j]))
    chosen: set[int] = set()
    used = 0
    for j in order:
        s = units[j][1]
        if used + len(s) > budget:
            continue
        chosen.add(j)
        used += len(s)
        if used >= budget:
            break
    merged: list[dict] = []
    for ci, c in enumerate(chunks):
        text = "".join(s for j, (cj, s) in enumerate(units)
                       if cj == ci and j in chosen)
        if text.strip():
            merged.append({"doc_id": c.get("doc_id"), "content_text": text,
                           "chunk_id": c.get("chunk_id"), "_score": c.get("_score")})
    return merged or chunks


def _doc_title(c: dict) -> str:
    src = c.get("source")
    if isinstance(src, str):
        try:
            src = ast.literal_eval(src)
        except Exception:  # noqa: BLE001
            src = None
    if isinstance(src, dict) and src.get("title"):
        return str(src["title"])
    return ""


def build_block(c: dict, idx: int, doccard: bool) -> str:
    head = f"[{idx}] "
    if doccard:
        title = _doc_title(c)
        section = c.get("title_text") or ""
        head += "《" + (title or "未知文档") + "》"
        if section and section != title:
            head += f"/{section}"
        head += "\n"
    return head + c.get("content_text", "")[:1200]


def assemble_ctx(chunks: list[dict], arm: str) -> str:
    doccard = arm in ("fulldoc", "fullboth")
    return "\n\n---\n\n".join(build_block(c, i + 1, doccard)
                              for i, c in enumerate(chunks))


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="full,denseonly,lexonly")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="20260927_gen_quality")
    ap.add_argument("--sets", default="ner",
                    help="逗号分隔题集短码（ner/nea 等），多集合并扩统计功效")
    args = ap.parse_args()
    OUT_DIR = Path(BASE) / "results" / args.out
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kb = json.load(open(KB_FILE))["kb_id"]
    idmap = json.load(open(IDMAP))
    rev = {v: k for k, v in idmap.items()}
    qs: list[dict] = []
    for s in args.sets.split(","):
        if s:
            qs.extend(load_questions(s))
    if args.limit:
        qs = qs[: args.limit]

    # ① 参考答案：金标文档证据块（稠密 top-30 过滤金标，至多 6 块）→ LLM 写参考
    ref_file = OUT_DIR / "references.json"
    refs: dict[str, str] = json.load(open(ref_file)) if ref_file.exists() else {}
    for q in qs:
        if q["query"] in refs:
            continue
        gold = set(q["gold_doc_ids"])
        chunks = retrieve(kb, q["query"], ["dense"], k=30)
        gold_ctx = [c for c in chunks if rev.get(c["doc_id"]) in gold][:6]
        if not gold_ctx:  # 稠密未命中金标：直接取金标库内检索（词法兜底）
            chunks = retrieve(kb, q["query"], None, k=30)
            gold_ctx = [c for c in chunks if rev.get(c["doc_id"]) in gold][:6]
        ctx = "\n\n---\n\n".join(f"[{i+1}] {c['content_text'][:1200]}"
                                 for i, c in enumerate(gold_ctx))
        refs[q["query"]] = llm(
            "你是核电子学专家，基于给定证据撰写参考答案，只覆盖证据支持的内容。",
            f"【证据】\n{ctx}\n\n【问题】\n{q['query']}\n\n请写参考答案（若无充分证据，"
            "写明该问题在证据下无法完整回答）。", 900)
        json.dump(refs, open(ref_file, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[ref {len(refs)}/{len(qs)}]", flush=True)

    # ② 生成 + 裁判
    for arm in [a for a in args.arms.split(",") if a]:
        out_file = OUT_DIR / f"gen_{arm}.json"
        rows: list[dict] = json.load(open(out_file)) if out_file.exists() else []
        done = {r["query"] for r in rows}
        for qi, q in enumerate(qs):
            if q["query"] in done:
                continue
            chunks = retrieve(kb, q["query"], ARM_CHANNELS[arm], k=10)
            if arm in ("fullprune", "fullboth"):
                chunks = prune_context(chunks, q["query"], PRUNE_BUDGET_CHARS)
            ctx = assemble_ctx(chunks, arm)
            answer = llm(GEN_SYSTEM, f"【检索上下文】\n{ctx}\n\n【问题】\n{q['query']}", 900)
            faith = jscore(llm(
                "你是严谨的评审。判断回答中的论断被上下文支持的程度。",
                f"【上下文】\n{ctx}\n\n【回答】\n{answer}\n\n"
                '请把回答拆成独立论断，逐条判断是否被上下文支持。只输出 JSON：'
                '{"claims": 总论断数, "supported": 被支持数}')) 
            rel = jscore(llm(
                "你是严谨的评审。评估回答对问题的聚焦与相关程度。",
                f"【问题】\n{q['query']}\n\n【回答】\n{answer}\n\n"
                '只输出 JSON：{"relevancy": 0到1的分数, "reason": 一句话}'))
            corr = jscore(llm(
                "你是严谨的评审。对照参考答案评估回答的正确性与完整性。",
                f"【问题】\n{q['query']}\n\n【参考答案】\n{refs[q['query']]}\n\n"
                f'【回答】\n{answer}\n\n'
                '只输出 JSON：{"correctness": 0到1的分数, "missing": 缺失要点一句话, "wrong": 错误点一句话或空}'))
            fa = (faith.get("supported", 0) / faith["claims"]) if faith.get("claims") else 0.0
            rows.append({"query": q["query"], "gold_doc_ids": q["gold_doc_ids"],
                         "n_ctx": len(chunks),
                         "gold_in_ctx": sum(1 for c in chunks if rev.get(c["doc_id"]) in set(q["gold_doc_ids"])),
                         "faithfulness": round(fa, 4),
                         "relevancy": float(rel.get("relevancy", 0)),
                         "correctness": float(corr.get("correctness", 0)),
                         "answer": answer[:600]})
            json.dump(rows, open(out_file, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"[{arm} {len(rows)}/{len(qs)}] f={fa:.2f} r={rel.get('relevancy')} c={corr.get('correctness')}", flush=True)

    # ③ 汇总
    summary = {}
    for arm in args.arms.split(","):
        if not arm:
            continue
        rows = json.load(open(OUT_DIR / f"gen_{arm}.json"))
        n = len(rows)
        if not n:
            continue
        summary[arm] = {
            "n": n,
            "faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 4),
            "relevancy": round(sum(r["relevancy"] for r in rows) / n, 4),
            "correctness": round(sum(r["correctness"] for r in rows) / n, 4),
            "gold_in_ctx": round(sum(r["gold_in_ctx"] > 0 for r in rows) / n, 4),
        }
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
