#!/usr/bin/env python3
"""双模型交叉全文档标注（2026-10-12 七项方案 §7.1 的模型化替代）。

两个在线判官独立对同一完整文本单元做全量术语标注（非审批算法候选）：
  P1 = deepseek-flash（官方 API，thinking disabled）
  P2 = qwen3.8-flash-next（CDUT 网关新 key，强制思考——大 max_tokens
       + 420s 读超时下可完成，内容层 JSON 完整）
标注 schema（方案 §4 三层质量）：每个术语面形 + termness
（domain/generic/ambiguous）+ specificity（specific/category）+ 类型 +
面形歧义标记。分歧由 judge A（.32 35B，Qwen 家族——与 P2 同家族，
如实披露）仲裁。产出：
  ① 双标一致性（面形 Jaccard / termness 一致 / 类型一致）
  ② 裁定参考集（双认 domain∩specific ∪ 仲裁纳入）——五基线臂+三选择臂
    的 occurrence 召回 vs 该参考（cs/bio/mat）
  ③ KB1 80 冻结块上 v2.0/v2.1 词典召回（生产域，语言分布如实记录）
样本：80 KB1 冻结块（10 样本×8，定种子）+ cs/bio/mat 测试集各 40。
测试集内文档此前仅做过枚举型 silver——本轮是超集（span/三层/NIL）。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && \
  .venv/bin/python ... --stage <sample|annotate|adjudicate|eval>'
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
from lexicon_isolated_compare import precompile, scan, load_lexicon  # noqa: E402
from lexicon_cross_domain import chat, _norm_term, domain_label  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402  (OS 只读)

XD = os.path.join(REPO, "eval", "baselines", "results",
                  "20261011_cross_domain")
OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261012_dual_anno")
KB = "104112e63d9d4aa5a6c21e1c1549ef0f"
ARMS = ("schema", "freq", "cvalue", "llmdirect", "threesource",
        "learnedrank", "cover", "coverRisk")

PROMPT = """You are a terminologist building a span-level reference annotation. Read the COMPLETE text below and annotate every technical term mention.

Domain: {domain}

Text:
{ctx}

For EVERY domain-relevant term that occurs, output one entry with:
- "surface": the exact surface form as it appears (copy precisely)
- "termness": "domain" (domain-specific technical term) / "generic" (generic academic word like method, result, approach) / "ambiguous" (short symbol or polysemous surface whose identity needs context, e.g. be, Al, Ga)
- "specificity": "specific" (names one specific concept/model/material/dataset) / "category" (names a whole category, e.g. neural network, algorithm)
- "type": one-word type guess (e.g. model, method, material, dataset, disease, concept)
- "nil_ambiguous": true if this surface could denote different entities out of context

Annotate what the text actually contains; do not add terms not present. Generic words only when used as the named entity of the sentence. Output JSON only: {{"terms":[{{"surface":"...","termness":"domain","specificity":"specific","type":"...","nil_ambiguous":false}}]}}"""


def models():
    svc = {}
    for line in open(os.path.join(REPO, "deploy", ".env.services"),
                     encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            svc[k.strip()] = v.strip()
    gw_key = os.environ.get("AIGATEWAY_API_KEY") or "sk-W5HfKEH_09sFLQRsfaRIcA"
    p1 = {"name": "deepseek-flash", "base": "https://api.deepseek.com",
          "key": os.environ["DEEPSEEK_API_KEY"],
          "model": "deepseek-flash"}
    # P2 原定网关 qwen3.8-flash-next：强制思考在全文标注任务上无界
    # （1300 字符输入 336s/67k 思考字符 finish=length；32k 触发网关 504）
    # ——用户指示回退 deepseek。P2=deepseek-v4-pro（同厂商不同模型，
    # 独立标注者；仲裁=Qwen 跨家族，家族关系如实披露）。
    p2 = {"name": "deepseek-v4-pro", "base": "https://api.deepseek.com",
          "key": os.environ["DEEPSEEK_API_KEY"],
          "model": "deepseek-v4-pro"}
    arb = {"name": "qwen3.6-35b@.32", "base": svc["LLM_API_BASE"],
           "key": svc["LLM_API_KEY"], "model": svc["LLM_MODEL"]}
    return p1, p2, arb


# ---------- stage: sample ----------

def stage_sample():
    os.makedirs(OUT, exist_ok=True)
    ms = json.load(open(os.path.join(REPO, "eval", "baselines", "results",
                                     "20261010_lexicon_multisample",
                                     "multisample.json")))
    rng = random.Random("20261012-dualanno")
    ids = []
    for s in ms["sample_chunk_ids"].values():     # 10 样本 × 8
        ids += rng.sample(s, 8)
    c = M.search.client()
    kb_rows = []
    for i in range(0, len(ids), 40):
        r = c.mget(index="knc-chunk",
                   body={"docs": [{"_id": x, "_source": ["content_text"]}
                                  for x in ids[i:i + 40]]})
        kb_rows += [(h["_id"][:16], h["_source"]["content_text"][:2600])
                    for h in r["docs"] if h.get("found")
                    and h["_source"].get("content_text")]
    units = [{"unit": f"kb1:{cid}", "domain": "nuclear-electronics",
              "text": t, "lang": "zh" if re.search(
                  r"[\u4e00-\u9fff]", t) else "en"}
             for cid, t in kb_rows]
    for dom in ("cs", "bio", "mat"):
        test = json.load(open(os.path.join(XD, dom, "test.json"),
                              encoding="utf-8"))
        for d in rng.sample(test, 40):
            units.append({"unit": f"{dom}:{d['doc_id']}",
                          "domain": domain_label(dom),
                          "text": d["text"][:2600], "lang": "en"})
    json.dump(units, open(os.path.join(OUT, "units.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[dual] {len(units)} units "
          f"(kb1 {sum(1 for u in units if u['unit'].startswith('kb1'))}, "
          f"zh {sum(1 for u in units if u['lang']=='zh')})")


# ---------- stage: annotate（双模型独立） ----------

def stage_annotate():
    units = json.load(open(os.path.join(OUT, "units.json"),
                           encoding="utf-8"))
    p1, p2, _ = models()
    from concurrent.futures import ThreadPoolExecutor
    import threading
    lock = threading.Lock()

    def run(tag, cfg):
        path = os.path.join(OUT, f"anno_{tag}.jsonl")
        done = set()
        if os.path.exists(path):
            for l in open(path, encoding="utf-8"):
                r = json.loads(l)
                if r.get("terms"):
                    done.add(r["unit"])
        cnt = [0]

        def work(u):
            if u["unit"] in done:
                return
            # 网关强制思考：思考+答案共享 max_tokens 预算，8k 下长文
            # 的 JSON 会被截断（raw_failures 实证）——网关判官给 16k
            mt = 16000 if "flash-next" in cfg.get("model", "") else 8000
            raw = chat(cfg, PROMPT.format(domain=u["domain"],
                                          ctx=u["text"]),
                       max_tokens=mt)
            # v4-pro 输出美化缩进 JSON（{\n  "terms":）——紧凑式正则会
            # 全部解析失败（首跑 198/200 空行根因）；剥代码围栏+容空白匹配
            txt = raw.strip()
            if txt.startswith("```"):
                txt = txt.strip("`").strip()
                if txt[:4].lower() == "json":
                    txt = txt[4:].strip()
            m = re.search(r'\{\s*"terms".*\}', txt, re.S)
            terms = []
            if m:
                try:
                    terms = [t for t in json.loads(m.group())["terms"]
                             if isinstance(t, dict) and t.get("surface")]
                except Exception:
                    pass
            if not terms and raw:
                dbg = os.path.join(OUT, "raw_failures.jsonl")
                with open(dbg, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"unit": u["unit"], "tag": tag,
                                        "raw_head": raw[:300],
                                        "raw_len": len(raw)},
                                       ensure_ascii=False) + "\n")
            with lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"unit": u["unit"], "terms": terms},
                                       ensure_ascii=False) + "\n")
                cnt[0] += 1
                if cnt[0] % 10 == 0:
                    print(f"[dual-{tag}] {cnt[0]}/{len(units)}", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool_:
            list(pool_.map(work, units))
        print(f"[dual-{tag}] done (+{cnt[0]})")

    run("p1_deepseek", p1)
    run("p2_v4pro", p2)


# ---------- stage: adjudicate ----------

ARB_PROMPT = """Two terminologists disagree about one term in a domain text. You are the adjudicator.

Domain: {domain}
Term surface: {surface}
Context (term in ⟦ ⟧): {ctx}

Annotator 1 (family A) says: {v1}
Annotator 2 (family B) says: {v2}

Decide: is this surface, AT THIS OCCURRENCE, a domain-specific technical term (not a generic academic word, not an ambiguous symbol used in another sense)? Output JSON only: {{"domain_term": true/false, "specific": true/false, "reason": "..."}}"""


def stage_adjudicate():
    units = {u["unit"]: u for u in json.load(open(
        os.path.join(OUT, "units.json"), encoding="utf-8"))}
    a1 = {r["unit"]: r["terms"] for r in (json.loads(l) for l in open(
        os.path.join(OUT, "anno_p1_deepseek.jsonl"), encoding="utf-8"))}
    a2 = {r["unit"]: r["terms"] for r in (json.loads(l) for l in open(
        os.path.join(OUT, "anno_p2_v4pro.jsonl"), encoding="utf-8"))}
    _, _, arb = models()
    path = os.path.join(OUT, "adjudicated.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if r.get("terms") is not None:
                done.add(r["unit"])
    from concurrent.futures import ThreadPoolExecutor
    import threading
    lock = threading.Lock()
    cnt = [0]

    def work(uid):
        if uid in done:
            return
        u = units[uid]
        t1 = {(_norm_term(t["surface"])): t for t in a1.get(uid, [])
              if t.get("termness") == "domain"}
        t2 = {(_norm_term(t["surface"])): t for t in a2.get(uid, [])
              if t.get("termness") == "domain"}
        both = sorted(set(t1) & set(t2))
        out_terms = [{"norm": n,
                      "specific": bool(t1[n].get("specificity") == "specific"
                                       and t2[n].get("specificity")
                                       == "specific"),
                      "type": t1[n].get("type") or t2[n].get("type"),
                      "origin": "both"}
                     for n in both]
        for n in sorted(set(t1) ^ set(t2)):
            t = (t1.get(n) or t2.get(n))
            src = "p1" if n in t1 else "p2"
            # 单方标 domain：仲裁（judge A，Qwen 家族与 P2 重叠——披露）
            idx = u["text"].lower().find(n)
            if idx < 0:
                continue    # 单方面形在文中找不到（幻觉面形）→ 弃
            ctx = (u["text"][max(0, idx - 200):idx] + "⟦" +
                   u["text"][idx:idx + len(n)] + "⟧" +
                   u["text"][idx + len(n):idx + len(n) + 200])
            other = "not marked" if src == "p1" else \
                f'{{"termness": "{(t2.get(n) or {}).get("termness")}}}'
            this = json.dumps({k: t.get(k) for k in
                               ("termness", "specificity")},
                              ensure_ascii=False)
            v1 = this if src == "p1" else other
            v2 = other if src == "p1" else this
            raw = chat(arb, ARB_PROMPT.format(
                domain=u["domain"], surface=n, ctx=ctx, v1=v1, v2=v2),
                max_tokens=400)
            d = {}
            try:
                mm = re.search(r'\{.*\}', raw, re.S)
                d = json.loads(mm.group()) if mm else {}
            except Exception:
                pass
            if d.get("domain_term"):
                out_terms.append({"norm": n,
                                  "specific": bool(d.get("specific")),
                                  "type": t.get("type"),
                                  "origin": f"arb+{src}"})
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"unit": uid, "terms": out_terms},
                                   ensure_ascii=False) + "\n")
            cnt[0] += 1
            if cnt[0] % 10 == 0:
                print(f"[dual-adjud] {cnt[0]}/{len(units)}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool_:
        list(pool_.map(work, list(units)))
    print(f"[dual-adjud] done (+{cnt[0]})")


# ---------- stage: eval ----------

def stage_eval():
    units = json.load(open(os.path.join(OUT, "units.json"),
                           encoding="utf-8"))
    a1 = {r["unit"]: r["terms"] for r in (json.loads(l) for l in open(
        os.path.join(OUT, "anno_p1_deepseek.jsonl"), encoding="utf-8"))}
    a2 = {r["unit"]: r["terms"] for r in (json.loads(l) for l in open(
        os.path.join(OUT, "anno_p2_v4pro.jsonl"), encoding="utf-8"))}
    adj = {r["unit"]: r["terms"] for r in (json.loads(l) for l in open(
        os.path.join(OUT, "adjudicated.jsonl"), encoding="utf-8"))}

    rep = {"agreement": {}, "arm_recall": {}, "kb1": {}, "notes": [
        "双模型交叉标注替代双人标注（方案 §7.1 的模型化形态）。P1=deepseek-",
        "flash、P2=deepseek-v4-pro——同厂商不同模型（家族重叠披露）；仲裁判官",
        "A=Qwen3.6-35B 跨家族。参考集=双认 domain 项 ∪ 仲裁纳入（单方面形且",
        "在文中可定位）。仪器仍是 LLM，非人类金标。网关 flash-next 全文标注",
        "不可用（强制思考无界：1300 字符 336s/67k 字符 finish=length）。"]}

    # ① 一致性
    for scope in ("kb1", "cs", "bio", "mat"):
        uids = [u["unit"] for u in units if u["unit"].startswith(scope)]
        jac, t_agree, t_n = [], [], 0
        for uid in uids:
            s1 = {_norm_term(t["surface"]) for t in a1.get(uid, [])
                  if t.get("termness") == "domain"}
            s2 = {_norm_term(t["surface"]) for t in a2.get(uid, [])
                  if t.get("termness") == "domain"}
            u = len(s1 | s2)
            if u:
                jac.append(len(s1 & s2) / u)
            m1 = {_norm_term(t["surface"]): t.get("termness")
                  for t in a1.get(uid, [])}
            m2 = {_norm_term(t["surface"]): t.get("termness")
                  for t in a2.get(uid, [])}
            for k in set(m1) & set(m2):
                t_n += 1
                if m1[k] == m2[k]:
                    t_agree.append(1)
        rep["agreement"][scope] = {
            "n_units": len(uids),
            "domain_set_jaccard_mean": round(sum(jac) / len(jac), 3)
            if jac else None,
            "termness_agreement_on_overlap": round(
                sum(t_agree) / t_n, 3) if t_n else None,
            "termness_overlap_n": t_n}

    # ② 五+三臂召回 vs 裁定参考（cs/bio/mat 测试 40 篇）
    for dom in ("cs", "bio", "mat"):
        uids = [u["unit"] for u in units if u["unit"].startswith(dom + ":")]
        texts = {u["unit"]: u["text"] for u in units
                 if u["unit"].startswith(dom + ":")}
        ref = {}
        for uid in uids:
            ref[uid] = {t["norm"] for t in adj.get(uid, [])
                        if t.get("specific")} | \
                       {t["norm"] for t in adj.get(uid, [])}
        # 用全部裁定项（specific ∪ category）作召回参照，specific 另报
        rows = {}
        for arm in ARMS:
            p = os.path.join(XD, dom, "components", f"arm_{arm}.json")
            if not os.path.exists(p):
                continue
            lex = json.load(open(p, encoding="utf-8"))
            pats = precompile(lex)
            hit_n = ref_n = hit = ref_sp_n = hit_sp = 0
            for uid in uids:
                r_all = {t["norm"] for t in adj.get(uid, [])}
                r_spec = {t["norm"] for t in adj.get(uid, [])
                          if t.get("specific")}
                matched = set()
                for surface, _e, _s, _x in scan(texts[uid], pats):
                    matched.add(_norm_term(surface))
                ref_n += len(r_all)
                hit += len(r_all & matched)
                ref_sp_n += len(r_spec)
                hit_sp += len(r_spec & matched)
            rows[arm] = {"R_all": round(hit / ref_n, 3) if ref_n else None,
                         "R_specific": round(hit_sp / ref_sp_n, 3)
                         if ref_sp_n else None}
        rep["arm_recall"][dom] = rows

    # ③ KB1 v2.0/v2.1 召回
    kb_uids = [u["unit"] for u in units if u["unit"].startswith("kb1:")]
    texts = {u["unit"]: u["text"] for u in units
             if u["unit"].startswith("kb1:")}
    for tag, ref_path in (("v2.0", "5ec5756:schemas/nuclear_electronics.yaml"),
                          ("v2.1", os.path.join(REPO, "schemas",
                                                "nuclear_electronics.yaml"))):
        lex = load_lexicon(ref_path)
        pats = precompile(lex)
        ref_n = hit = 0
        for uid in kb_uids:
            r_all = {t["norm"] for t in adj.get(uid, [])}
            matched = {_norm_term(s) for s, _e, _x, _y
                       in scan(texts[uid], pats)}
            ref_n += len(r_all)
            hit += len(r_all & matched)
        rep["kb1"][tag] = {"R_all": round(hit / ref_n, 3) if ref_n else None,
                           "n_units": len(kb_uids), "ref_terms": ref_n}
    rep["kb1"]["lang_split"] = collections.Counter(
        u["lang"] for u in units if u["unit"].startswith("kb1:"))

    json.dump(rep, open(os.path.join(OUT, "dual_anno_eval.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["sample", "annotate", "adjudicate", "eval"])
    args = ap.parse_args()
    {"sample": stage_sample, "annotate": stage_annotate,
     "adjudicate": stage_adjudicate, "eval": stage_eval}[args.stage]()


if __name__ == "__main__":
    main()
