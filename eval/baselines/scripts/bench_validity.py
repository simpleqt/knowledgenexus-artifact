#!/usr/bin/env python3
"""O6 题集效度检验（2026-09-25，文献调研 §3 O6：闭卷对照 + unjudged 补判）。

  closed-book：闭卷 LLM 直答对照（HybridRAG-Bench，arXiv 2602.10210）——不给检索
    结果直接问 LLM 哪些论文回答该问题，统计其答案中点名 gold 文档标题的比例。
    ≈0 即证明题集测的是检索而非模型记忆（gold 是库内领域文献，预训练语料未见）。
  unjudged：cov@k 低估量化（Topic-Specific Classifiers，arXiv 2510.04633）——对
    预测文件 top-10 中未判（非 gold）文档抽样 LLM 复核相关性，报
    cov@10_corrected 上界：写成"保守下界，修正 ≤x%"。

LLM 端点读 deploy/.env.services（LLM_API_*，qwen3.5-35b，temperature 0）。
用法：
  .venv/bin/python bench_validity.py closed-book --sets nea,ner
  .venv/bin/python bench_validity.py unjudged --pred runs/knc-o8dv_nea_teststack.json \
      --set nea --per-query 5
产物：eval/results/20260925_o6_bench_validity/（json + stdout 报告）。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, REPO  # noqa: E402

OUT_DIR = os.path.join(REPO, "eval", "results", "20260925_o6_bench_validity")
ENVF = os.path.join(REPO, "deploy", ".env.services")


def llm_cfg() -> tuple[str, str, str]:
    cfg = {}
    for line in open(ENVF, encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            cfg[k] = v
    return cfg["LLM_API_BASE"], cfg["LLM_API_KEY"], cfg["LLM_MODEL"]


def chat(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    base, key, model = llm_cfg()
    body = json.dumps({"model": model, "temperature": temperature,
                       "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        return json.loads(r.read())["choices"][0]["message"]["content"]


def title_map() -> dict[str, str]:
    """stable_id → 真实标题（corpus_docs.jsonl）。"""
    out = {}
    p = os.path.join(DATA, "corpus_docs.jsonl")
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        out[d["doc_id"]] = d["title"]
    return out


def doc_text() -> dict[str, str]:
    out = {}
    p = os.path.join(DATA, "corpus_docs.jsonl")
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        out[d["doc_id"]] = d["text"]
    return out


def load_set(name: str) -> list[dict]:
    from common import load_questions
    return load_questions(name)


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def cmd_closed_book(sets: list[str], out: str) -> None:
    tm = title_map()
    gold_titles = {norm(t) for t in tm.values()}
    rows = []
    for set_name in sets:
        qs = load_set(set_name)
        def one(q):
            try:
                ans = chat(
                    "你是一名核电子学领域的文献专家。请**只凭你自己的知识**（不允许检索、"
                    "不允许猜测不存在的文献）回答下面的问题；如果问题指向具体论文，请列出"
                    "你确信存在的论文标题（中文或英文原名）；如果你不知道，直接回答"
                    "\"不知道\"。\n\n问题：" + q["query"], max_tokens=300)
            except Exception as e:  # noqa: BLE001
                return {"query": q["query"], "answer": f"[ERROR {e}]", "hit": 0}
            a = norm(ans)
            # gold 命中 = 答案中出现任一该题 gold 文档标题（归一化包含）
            hits = [norm(tm[g]) for g in q["gold_doc_ids"]
                    if g in tm and norm(tm[g]) and norm(tm[g]) in a]
            return {"query": q["query"], "answer": ans[:200],
                    "hit": int(bool(hits)), "hit_titles": hits}
        with cf.ThreadPoolExecutor(max_workers=20) as ex:
            res = list(ex.map(one, qs))
        rate = sum(r["hit"] for r in res) / len(res)
        print(f"[{set_name}] n={len(res)} 闭卷 gold 点名率 = {rate:.4f}（应≈0）")
        rows.append({"set": set_name, "n": len(res), "closed_book_hit_rate":
                     round(rate, 4), "per_query": res})
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def entity_names_map(kb_hex: str) -> dict[str, set[str]]:
    """实体任一已知名（canonical+alias，lower）→ 该实体全部名称集。

    供严格判定：nea 问题是英文面形、文档是中文——判定器必须拿到实体的
    双语名称全集才能判断"是否双方都在文档中出现"。"""
    import knc_engine.db as db
    ent_names: dict[str, set[str]] = {}
    for r in db.exec(
            "SELECT entity_id e, canonical_name n FROM canonical_entity"
            " WHERE kb_id=UNHEX(%s) AND status='ACTIVE'", (kb_hex,), fetch=True):
        ent_names[r["e"].lower()] = {r["n"]}
    for r in db.exec(
            "SELECT a.entity_id e, a.alias_text t FROM entity_alias a"
            " JOIN canonical_entity c ON c.entity_id=a.entity_id"
            " WHERE a.kb_id=UNHEX(%s) AND c.status='ACTIVE'", (kb_hex,), fetch=True):
        ent_names.setdefault(r["e"].lower(), set()).add(r["t"])
    out: dict[str, set[str]] = {}
    for names in ent_names.values():
        for n in names:
            out[n.lower()] = names
    return out


def query_pair(q: str) -> tuple[str, str] | None:
    m = re.match(r"有哪些研究同时涉及(.+?)与(.+?)？（关联检索）", q)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def cmd_unjudged(pred_file: str, set_name: str, per_query: int, out: str,
                 strict: bool = True, kb: str = "0cf04665c3bd427281976c3502eb4d99") -> None:
    tm, txt = title_map(), doc_text()
    pred = json.load(open(pred_file, encoding="utf-8"))
    qs = {q["query"]: q for q in load_set(set_name)}

    tasks = []  # (query, doc_id, gold_hit)
    for row in pred["per_query"]:
        gold = set(row["gold_doc_ids"])
        unj = [h["doc_id"] for h in row["hits"][:10] if h["doc_id"] not in gold]
        for d in unj[:per_query]:
            tasks.append((row["query"], d))
    print(f"未判文档抽样：{len(tasks)} 条（{set_name}，每题至多 {per_query}，"
          f"{'严格证据引用' if strict else '宽松一词判定'}）")

    names_map = entity_names_map(kb) if strict else {}

    def one(t):
        query, did = t
        text = (txt.get(did) or "")[:2500]
        if strict:
            pair = query_pair(query)
            hint = ""
            if pair:
                hints = []
                for s in pair:
                    nm = names_map.get(s.lower())
                    hints.append(f"实体「{s}」的已知名称：{' / '.join(sorted(nm))}" if nm
                                 else f"实体「{s}」（词典无此面形，用同义/译名判断）")
                hint = "\n".join(hints)
            try:
                ans = chat(
                    "判断文档是否**同时实质性出现**下列问题涉及的两个实体（含译名/同义表述；"
                    "泛主题相近不算）。必须先分别引用两处原文证据片段（每处≤30字），"
                    "再给结论。找不到任一侧证据就判否。按此格式回答：\n"
                    "证据1：<片段>\n证据2：<片段>\n结论：是/否\n\n"
                    f"问题：{query}\n{hint}\n\n文档（前 2500 字）：\n{text}", max_tokens=200)
            except Exception as e:  # noqa: BLE001
                return {"query": query, "doc_id": did, "judge": f"[ERROR {e}]"}
            m = re.search(r"结论[:：]\s*(是|否)", ans)
            verdict = m.group(1) if m else "否"
            q1 = (re.search(r"证据1[:：]\s*(.+)", ans) or [None, ""])[1]
            q2 = (re.search(r"证据2[:：]\s*(.+)", ans) or [None, ""])[1]
            # 程序化复核：判"是"必须双侧证据片段非空且互不相同（防幻觉结论）
            ok = verdict == "是" and q1.strip() and q2.strip() and q1.strip() != q2.strip()
            return {"query": query, "doc_id": did, "judge": ans[:160],
                    "verdict": "是" if ok else "否", "e1": q1.strip()[:40], "e2": q2.strip()[:40]}
        try:
            ans = chat(
                "判断下面这篇文档是否实质性回答/涉及该检索问题。只回答一个词："
                "是（明确涉及问题主题或实体对双方）/部分（仅涉及一方或弱相关）/否（无关）。\n\n"
                f"问题：{query}\n\n文档（前 2000 字）：\n{text}", max_tokens=8)
        except Exception as e:  # noqa: BLE001
            return {"query": query, "doc_id": did, "judge": f"[ERROR {e}]"}
        return {"query": query, "doc_id": did,
                "judge": norm(ans).replace("。", "")[:4]}

    with cf.ThreadPoolExecutor(max_workers=20) as ex:
        judged = list(ex.map(one, tasks))

    if strict:
        cnt = {"是": sum(1 for r in judged if r["verdict"] == "是"),
               "否": sum(1 for r in judged if r["verdict"] != "是")}
        for r in judged:
            r["label"] = "rel" if r["verdict"] == "是" else "no"
    else:
        def lab(j):
            if j.startswith("是"):
                return "rel"
            if j.startswith("部分"):
                return "part"
            return "no"
        for r in judged:
            r["label"] = lab(r["judge"])
        cnt = {"rel": 0, "part": 0, "no": 0}
        for r in judged:
            cnt[r["label"]] += 1
    n = len(judged)
    # cov@10 修正上界：题级 relevant-unjudged 数并入分子（保守口径只计"是"）
    per_q_rel: dict[str, int] = {}
    for r in judged:
        if r["label"] == "rel":
            per_q_rel[r["query"]] = per_q_rel.get(r["query"], 0) + 1
    # 原始 cov@10（从 pred 重算）
    cov10 = {}
    for row in pred["per_query"]:
        gold = set(row["gold_doc_ids"])
        docs = [h["doc_id"] for h in row["hits"]]
        seen: set[str] = set()
        udocs = [d for d in docs if not (d in seen or seen.add(d))]
        cov10[row["query"]] = len(set(udocs[:10]) & gold) / len(gold) if gold else 0
    corrected = []
    for row in pred["per_query"]:
        gold = set(row["gold_doc_ids"])
        base = cov10[row["query"]]
        extra = min(per_q_rel.get(row["query"], 0),
                    # 只补到 gold 规模封顶：relevant-unjudged 不可能超过缺席 gold 数
                    max(0, len(gold) - round(base * len(gold))))
        corrected.append(min(1.0, base + extra / len(gold)) if gold else 0)
    m_base = sum(cov10.values()) / len(cov10)
    m_corr = sum(corrected) / len(corrected)
    if strict:
        print(f"判定分布：是(双侧证据齐) {cnt['是']} / 否 {cnt['否']}（n={n}，"
              f"程序化复核=判是须双侧证据片段非空且互异）")
    else:
        print(f"判定分布：rel {cnt['rel']} / part {cnt['part']} / no {cnt['no']}（n={n}）")
    print(f"cov@10 = {m_base:.4f}（保守下界）→ 修正上界 {m_corr:.4f}"
          f"（修正 ≤ +{(m_corr-m_base)*100:.1f}pt）")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump({"set": set_name, "pred": os.path.basename(pred_file), "n_judged": n,
                   "counts": cnt, "cov10_base": round(m_base, 4),
                   "cov10_corrected_ub": round(m_corr, 4),
                   "judged": judged}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cb = sub.add_parser("closed-book")
    cb.add_argument("--sets", default="nea,ner")
    cb.add_argument("--out", default=os.path.join(OUT_DIR, "closed_book.json"))
    uj = sub.add_parser("unjudged")
    uj.add_argument("--pred", required=True)
    uj.add_argument("--set", required=True)
    uj.add_argument("--per-query", type=int, default=5)
    uj.add_argument("--lenient", action="store_true",
                    help="宽松一词判定（首版协议，判定器拿不到实体双语名——已证过宽，"
                         "仅留作对照；缺省严格证据引用式）")
    uj.add_argument("--kb", default="0cf04665c3bd427281976c3502eb4d99")
    uj.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    if args.cmd == "closed-book":
        cmd_closed_book([s for s in args.sets.split(",") if s], args.out)
    else:
        suffix = "_lenient" if args.lenient else "_strict"
        out = args.out or os.path.join(OUT_DIR, f"unjudged_{args.set}{suffix}.json")
        cmd_unjudged(args.pred, args.set, args.per_query, out,
                     strict=not args.lenient, kb=args.kb)
    print(f"（{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
