#!/usr/bin/env python3
"""短符号歧义四臂对照（2026-10-12 七项方案 §5.2 落地）。

总体 = KB1 冻结重放块上现行 v2.1 词典的短 ASCII 符号（≤3 字符）命中
（含 be/alpha/gamma 类）。四臂（同词典、同文本、同判读口径）：
  original   全部链接（现行生产行为）
  caseShort  仅面形精确大小写命中词典条目才链接（既有治理臂）
  context    精确大小写命中，或（大小写失配但上下文证据：±150 字符窗内
             ≥2 个其他 v2.1 长词命中，或元素符号+材料/核词邻接）
  ctxNIL     context 臂 + 反向拒绝：精确大小写命中但无任何上下文证据的
             提及判 NIL（测“上下文判别”而非“大小写代理”）
判读：每条提及独立判一次（judge A 400tok + 跨家族 C 同题），臂无关——
用作“真短符号提及”参照；各臂指标：
  链接精度 = retained 中判真比例（judged-valid）
  真符号召回 = retained∩true / true（true=judges 一致判真）
  拒绝比例 与 NIL 精度 = rejected 中确非真比例
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  KNC_JC_BASE=https://api.deepseek.com .venv/bin/python ... [--judge|--tables]'
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import precompile, scan  # noqa: E402
from lexicon_cross_domain import chat, JUDGE_PROMPT  # noqa: E402
import gen_term_lexicons as G  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402  (OS client 只读)

KB = "104112e63d9d4aa5a6c21e1c1549ef0f"
LEX = "schemas/nuclear_electronics.yaml"
OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261012_revision", "ambiguity")
N_SAMPLE = 150
import yaml  # noqa: E402


def short_terms():
    doc = yaml.safe_load(open(os.path.join(REPO, LEX), encoding="utf-8"))
    out = {}
    for ty, spec in doc.get("entity_types", {}).items():
        for a in [spec.get("name")] + list(spec.get("aliases") or []):
            if a and a.isascii() and len(a) <= 3 and re.fullmatch(
                    r"[A-Za-z]+", a):
                out[a] = ty
    return out


def fetch_chunks():
    ms = json.load(open(os.path.join(REPO, "eval", "baselines", "results",
                                     "20261010_lexicon_multisample",
                                     "multisample.json")))
    ids = [i for s in ms["sample_chunk_ids"].values() for i in s][:4000]
    c = M.search.client()
    rows = []
    for i in range(0, len(ids), 200):
        r = c.mget(index="knc-chunk",
                   body={"docs": [{"_id": x, "_source": ["content_text"]}
                                  for x in ids[i:i + 200]]})
        rows += [(h["_id"], h["_source"]["content_text"])
                 for h in r["docs"] if h.get("found")
                 and h["_source"].get("content_text")]
    return rows


def sample_mentions():
    """分层：be 类（表面折叠=be）60 + 其他短符号 90。"""
    st = short_terms()
    short_lex = dict(st)
    pats = precompile(short_lex)
    long_lex = {}
    doc = yaml.safe_load(open(os.path.join(REPO, LEX), encoding="utf-8"))
    for ty, spec in doc.get("entity_types", {}).items():
        for a in [spec.get("name")] + list(spec.get("aliases") or []):
            if a and (len(a) > 3 or not a.isascii()):
                long_lex[a] = ty
    lpats = precompile(long_lex)
    rows = fetch_chunks()
    rng = random.Random("20261012-amb")
    be_pool, other_pool = [], []
    for cid, text in rows:
        long_hits = [(m_s, m_e) for _sf, _ty, m_s, m_e in scan(text, lpats)]
        for surface, ty, s, e in scan(text, pats):
            rec = {"chunk": cid, "surface": surface, "term_ty": ty,
                   "s": s, "e": e,
                   "ctx": text[max(0, s - 250):s] + "⟦" + surface +
                          "⟧" + text[e:e + 250],
                   "n_long_ctx": sum(
                       1 for ls, le in long_hits
                       if abs(ls - s) < 150 or abs(le - e) < 150)}
            if surface.lower() == "be":
                be_pool.append(rec)
            else:
                other_pool.append(rec)
    sel = (rng.sample(be_pool, min(60, len(be_pool)))
           + rng.sample(other_pool, min(N_SAMPLE - 60, len(other_pool))))
    for r in sel:
        r["entity"] = r["surface"].lower()
        r["etype"] = r["term_ty"]
    return sel


def stage_judge():
    os.makedirs(OUT, exist_ok=True)
    rows = sample_mentions()
    json.dump(rows, open(os.path.join(OUT, "mentions.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[amb] {len(rows)} mentions sampled "
          f"(be: {sum(1 for r in rows if r['surface'].lower()=='be')})")
    svc = {}
    for line in open(os.path.join(REPO, "deploy", ".env.services"),
                     encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            svc[k.strip()] = v.strip()
    ja = {"name": "qwen3.6-35b@.32", "base": svc["LLM_API_BASE"],
          "key": svc["LLM_API_KEY"], "model": svc["LLM_MODEL"]}
    jc = {"name": os.environ.get("KNC_JC_MODEL", "deepseek-flash"),
          "base": os.environ["KNC_JC_BASE"],
          "key": os.environ.get("DEEPSEEK_API_KEY", "none"),
          "model": os.environ.get("KNC_JC_MODEL", "deepseek-flash")}
    p = os.path.join(OUT, "judged.jsonl")
    done = set()
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["chunk"], r["s"], r["e"]))
    from concurrent.futures import ThreadPoolExecutor
    import threading
    lock = threading.Lock()
    cnt = [0]

    def work(rec):
        key = (rec["chunk"], rec["s"], rec["e"])
        if key in done:
            return
        out = {**rec}
        for tag, cfg in (("A", ja), ("C", jc)):
            raw = chat(cfg, JUDGE_PROMPT.format(
                name=rec["entity"], etype=rec["etype"],
                surface=rec["surface"], ctx=rec["ctx"]),
                max_tokens=400)
            d = G._json_obj(raw) or {}
            out[f"j{tag}"] = {"match": d.get("match"),
                              "reason": str(d.get("reason", ""))[:120]}
        with lock:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            cnt[0] += 1
            if cnt[0] % 25 == 0:
                print(f"[amb] judged {cnt[0]}/{len(rows)}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool_:
        list(pool_.map(work, rows))
    print(f"[amb] judging done (+{cnt[0]})")


def _arm_retains(rec, st):
    exact = rec["surface"] in st or rec["surface"] in \
        {t.lower() for t in st} and rec["surface"] == rec["surface"].upper()
    exact_case = any(rec["surface"] == t for t in st)
    ctx_ev = rec["n_long_ctx"] >= 2
    return {
        "original": True,
        "caseShort": exact_case,
        "context": exact_case or (rec["surface"] == rec["surface"].upper()
                                  and ctx_ev),
        "ctxNIL": (exact_case and ctx_ev) or
                  (not exact_case and rec["surface"] ==
                   rec["surface"].upper() and ctx_ev),
    }


def stage_tables():
    st = short_terms()
    rows = [json.loads(l) for l in open(os.path.join(OUT, "judged.jsonl"),
                                        encoding="utf-8")]
    # 判读终判：A/C 一致取之，分歧记 disagree（保守=按 invalid 计入 recall
    # 分母、不计入 true）
    for r in rows:
        a = (r.get("jA") or {}).get("match")
        c = (r.get("jC") or {}).get("match")
        r["true"] = bool(a) if a == c else None
        r["agree"] = a == c
    n_dis = sum(1 for r in rows if not r["agree"])
    res = {"n": len(rows), "judge_disagree": n_dis}
    for arm in ("original", "caseShort", "context", "ctxNIL"):
        ret = [r for r in rows if _arm_retains(r, st)[arm]]
        rej = [r for r in rows if not _arm_retains(r, st)[arm]]
        true_all = [r for r in rows if r["true"] is True]
        valid_ret = [r for r in ret if r["true"] is True]
        res[arm] = {
            "retained": len(ret),
            "link_precision": round(len(valid_ret) / len(ret), 3)
            if ret else None,
            "true_symbol_recall": round(len(valid_ret) / len(true_all), 3)
            if true_all else None,
            "reject_rate": round(len(rej) / len(rows), 3),
            "nil_precision": round(
                sum(1 for r in rej if r["true"] is not True) / len(rej), 3)
            if rej else None,
        }
    # be 子类
    be = [r for r in rows if r["surface"].lower() == "be"]
    be_true = [r for r in be if r["true"] is True]
    res["be_subset"] = {"n": len(be), "n_true": len(be_true)}
    for arm in ("original", "caseShort", "context", "ctxNIL"):
        ret = [r for r in be if _arm_retains(r, st)[arm]]
        res["be_subset"][arm] = {
            "retained": len(ret),
            "precision": round(sum(1 for r in ret if r["true"] is True)
                               / len(ret), 3) if ret else None,
            "recall": round(sum(1 for r in ret if r["true"] is True)
                            / len(be_true), 3) if be_true else None}
    json.dump(res, open(os.path.join(OUT, "ambiguity_arms.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["judge", "tables"], default="judge")
    args = ap.parse_args()
    (stage_judge if args.stage == "judge" else stage_tables)()


if __name__ == "__main__":
    main()
