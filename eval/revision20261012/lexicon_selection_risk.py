#!/usr/bin/env python3
"""风险约束的概念覆盖词典选择（2026-10-12 七项方案 §5 落地）。

在同一宽候选池、同一预算 K=|threesource| 下比较：
  freq / cvalue / threesource（既有三臂，基线）
  learnedrank —— 线性特征评分 s_θ(t)=θ·φ(t) 取 top-K（构造侧 silver
                开发标签训练，纯 Python logistic+L2，无新依赖）
  cover       —— 方案 §5.3 集合目标（覆盖+提及效用−误链惩罚），贪心
                初始化+局部交换搜索，q/p 为显式声明的代理分数
  coverRisk   —— cover + ε 期望误链比例约束（低 p 提及拒绝）
小池参考解：top-300 切片的 beam search（宽度 32）报与局部搜索差距。
特征 φ(t)：域对比(log tf/df)、C-value、词组结构、缩写证据、边界、
歧义（短符号/泛词表）。测试文档不参与任何建池/训练（构造侧
construct-silver 由跨家族 deepseek-flash 枚举，60 篇/域定种子）。
用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/lexicon_selection_risk.py --stage <pool|dev|arms|eval>'
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_isolated_compare import precompile, scan  # noqa: E402
from lexicon_cross_domain import (_norm_term, text_freq,  # noqa: E402
                                  cvalue_scores, EN_RE, chat, SCHEMA,
                                  domain_label, GENERIC_HEAD)
import gen_term_lexicons as G  # noqa: E402

ROOT = os.path.join(REPO, "eval", "baselines", "results",
                    "20261011_cross_domain")
OUT = os.path.join(REPO, "eval", "baselines", "results",
                   "20261012_selection")
NEW_ARMS = ("learnedrank", "cover", "coverRisk")
K_CURVE = (0.5, 1.0, 2.0)   # 预算曲线（×|threesource|）

SILVER_PROMPT_CON = """You are a senior domain terminologist building a reference term list.

Domain: {domain}
Abstract:
{ctx}

List EVERY domain-specific technical term that occurs in this abstract. For each: the exact surface form as it appears, and a one-word type guess. Do NOT include generic academic words (method, result, problem, approach, study, analysis) or verbs.
Output JSON only: {{"terms": [{{"surface": "...", "type": "..."}}]}}"""


def dom_out(dom):
    return os.path.join(ROOT, dom)


def load_all(dom):
    out = dom_out(dom)
    construct = json.load(open(os.path.join(out, "construct.json"),
                               encoding="utf-8"))
    test = json.load(open(os.path.join(out, "test.json"),
                          encoding="utf-8"))
    silver_a = {r["doc_id"]: [t["surface"] for t in r.get("terms") or []]
                for r in (json.loads(l) for l in
                          open(os.path.join(out, "silver_A.jsonl"),
                               encoding="utf-8"))}
    silver_c = {r["doc_id"]: [t["surface"] for t in r.get("terms") or []]
                for r in (json.loads(l) for l in
                          open(os.path.join(out, "silver_C.jsonl"),
                               encoding="utf-8"))}
    three = json.load(open(os.path.join(out, "components",
                                        "arm_threesource.json"),
                           encoding="utf-8"))
    return construct, test, silver_a, silver_c, three


# ---------- stage: pool + features ----------

def stage_pool(dom):
    out = dom_out(dom)
    construct, *_ , three, = (*load_all(dom),)
    freq = text_freq(construct)
    tf = {t: c for t, c in freq.items()}
    df = collections.Counter()
    for d in construct:
        seen = {m.group() for m in EN_RE.finditer(d["text"])}
        for s in seen:
            df[s] += 1
    # 缩写定义证据："long form (ABBR)"
    abbr_def = collections.Counter()
    abbr_re = re.compile(
        r"\((([A-Z][A-Za-z0-9\-]{0,11}))\)")
    for d in construct:
        for m in abbr_re.finditer(d["text"]):
            a = m.group(1)
            pre = d["text"][max(0, m.start() - 80):m.start()]
            ini = "".join(w[0] for w in pre.replace("-", " ").split()
                          [-len(a):]) if len(a) >= 2 else ""
            if ini and ini.lower() == a.lower():
                abbr_def[a] += 1
    # LLM 候选清单
    llm_terms = {}
    for f in ("classify", "seed", "expand", "llmdirect"):
        p = os.path.join(out, "components", f + ".json")
        if os.path.exists(p):
            for ty, terms in json.load(open(p, encoding="utf-8")).items():
                for t in terms:
                    llm_terms.setdefault(t, ty)
    # 宽池：n-gram(freq≥2, _ok_term) ∪ 2 字符已定义缩写 ∪ LLM 清单
    pool = {}
    for t, c in freq.items():
        if c >= 2 and G._ok_term(t):
            pool[t] = {"tf": c, "df": df.get(t, 0), "src": "ngram"}
    for a, n in abbr_def.items():
        if len(a) >= 2 and a not in pool:
            pool[a] = {"tf": n, "df": df.get(a, 0), "src": "abbr"}
    for t, ty in llm_terms.items():
        if t in pool:
            pool[t]["src"] += "+llm"
            pool[t]["type"] = ty
        else:
            pool[t] = {"tf": tf.get(t, 0), "df": df.get(t, 0),
                       "src": "llm", "type": ty}
    # 特征 + C-value（全池；cvalue_scores 返回 (term, score) 序列）
    cv = {t: s for t, s in cvalue_scores(construct)}
    for t, meta in pool.items():
        toks = t.split()
        meta.update({
            "cvalue": round(cv.get(t, 0.0), 3),
            "log_tf": round(math.log1p(meta["tf"]), 3),
            "log_df": round(math.log1p(meta["df"]), 3),
            "ntok": len(toks),
            "is_acronym": int(bool(re.fullmatch(r"[A-Z0-9\-]{2,8}", t))),
            "has_hyphen": int("-" in t),
            "is_short": int(len(t) <= 3),
            "in_generic": int(t.lower() in GENERIC_HEAD
                              or t.lower() in G.STOP_EN),
            "abbr_def": abbr_def.get(t, 0),
            "norm": _norm_term(t),
        })
    json.dump({"pool": pool, "n_construct": len(construct)},
              open(os.path.join(OUT, dom, "pool_features.json"), "w",
                   encoding="utf-8"), ensure_ascii=False)
    print(f"[{dom}] pool={len(pool)} (abbr {sum(1 for m in pool.values() if m['src']=='abbr')}, "
          f"llm-only {sum(1 for m in pool.values() if m['src']=='llm')})")


# ---------- stage: dev（构造侧 silver，跨家族 deepseek） ----------

def stage_dev(dom):
    construct, *_ = load_all(dom)
    rng = random.Random(f"20261012-{dom}-dev")
    docs = rng.sample(construct, 60)
    jc = {"name": os.environ.get("KNC_JC_MODEL", "deepseek-flash"),
          "base": os.environ["KNC_JC_BASE"],
          "key": os.environ.get("DEEPSEEK_API_KEY", "none"),
          "model": os.environ.get("KNC_JC_MODEL", "deepseek-flash")}
    p = os.path.join(OUT, dom, "dev_silver.jsonl")
    done = set()
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("terms"):
                done.add(r["doc_id"])
    from concurrent.futures import ThreadPoolExecutor
    lock = __import__("threading").Lock()
    cnt = [0]

    def work(d):
        if d["doc_id"] in done:
            return
        raw = chat(jc, SILVER_PROMPT_CON.format(
            domain=domain_label(dom), ctx=d["text"][:2500]),
            max_tokens=6000)
        m = re.search(r'\{"terms".*\}', raw, re.S)
        terms = []
        if m:
            try:
                terms = [t["surface"] for t in json.loads(m.group())["terms"]
                         if isinstance(t, dict) and t.get("surface")]
            except Exception:
                pass
        with lock:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps({"doc_id": d["doc_id"], "terms": terms},
                                   ensure_ascii=False) + "\n")
            cnt[0] += 1
            if cnt[0] % 15 == 0:
                print(f"[{dom}] dev silver {cnt[0]}/60", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool_:
        list(pool_.map(work, docs))
    print(f"[{dom}] dev silver done (+{cnt[0]})")


# ---------- 训练 + 三新臂 ----------

FEATS = ("log_tf", "log_df", "ntok", "is_acronym", "has_hyphen",
         "is_short", "in_generic", "abbr_def", "cvalue")


def _feat_row(meta):
    return [float(meta.get(f, 0)) for f in FEATS]


def train_scorer(pool, dev_norms):
    """logistic + L2，纯 Python；正例=池内构造 silver 命中，负例=高置信
    非术语（泛词表/停用词/2-3 字符短符号且无缩写定义证据）。"""
    xs, ys = [], []
    for t, meta in pool.items():
        n = meta["norm"]
        if n in dev_norms:
            xs.append(_feat_row(meta)); ys.append(1.0)
        elif (meta["in_generic"] or (meta["is_short"] and not meta["abbr_def"])
              or (meta["ntok"] == 1 and meta["log_df"] > 4
                  and meta["log_tf"] < 5 and t.lower() in G.STOP_EN)):
            xs.append(_feat_row(meta)); ys.append(0.0)
    if sum(ys) < 20 or (len(ys) - sum(ys)) < 20:
        return None
    # 标准化
    k = len(FEATS)
    mu = [sum(r[i] for r in xs) / len(xs) for i in range(k)]
    sd = [max(1e-6, (sum((r[i] - mu[i]) ** 2 for r in xs) / len(xs)) ** .5)
          for i in range(k)]
    xs = [[(r[i] - mu[i]) / sd[i] for i in range(k)] for r in xs]
    theta = [0.0] * k
    lr, l2, steps = 0.1, 1e-3, 400
    for _ in range(steps):
        grad = [l2 * w for w in theta]
        for x, y in zip(xs, ys):
            z = sum(w * v for w, v in zip(theta, x))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = p - y
            for i in range(k):
                grad[i] += e * x[i] / len(xs)
        theta = [w - lr * g for w, g in zip(theta, grad)]
    return theta, mu, sd


def _score(t_meta, model):
    if model is None:
        return t_meta["cvalue"]
    theta, mu, sd = model
    x = [(v - m) / s for v, m, s in zip(_feat_row(t_meta), mu, sd)]
    return sum(w * v for w, v in zip(theta, x))


def _mentions_construct(dom, lex):
    """构造语料上的提及索引（提及→文档），用于覆盖目标。"""
    construct, *_ = load_all(dom)
    pats = precompile(lex)
    occ = collections.defaultdict(list)   # norm-term -> [doc_id...]
    for d in construct:
        for surface, _et, s, e in scan(d["text"], pats):
            occ[_norm_term(surface)].append(d["doc_id"])
    return occ


def select_cover(dom, pool, K, eta=0.05, lam=0.5, eps=0.15,
                 with_risk=False, seed=0, beam_ref=False,
                 occ=None, model=None, scored=None):
    """§5.3 集合选择。单元 g=构造 silver 归一词（w_g=1）；q/p 代理显式：
    q = min(1, df/5)·spec，p = 1−ambig。贪心初始化（learnedrank 序）
    + 增/换局部搜索；每步目标精确重估（固定 x 下 z/a 最优分配可分解）。"""
    dev = [json.loads(l) for l in open(os.path.join(OUT, dom, "dev_silver.jsonl"),
                                       encoding="utf-8") if l.strip()]
    dev_norms = {_norm_term(t) for r in dev if r.get("terms")
                 for t in r["terms"]}
    if model is None:
        model = train_scorer(pool, dev_norms)
    if scored is None:
        scored = sorted(pool.items(), key=lambda kv: -_score(kv[1], model))
    M = min(len(scored), 1500)
    cand = [t for t, _ in scored[:M]]
    if occ is None:
        lex_tmp = {t: (pool[t].get("type") or "term") for t in cand}
        occ = _mentions_construct(dom, lex_tmp)

    def qp(meta):
        spec = 0.3 if (meta["in_generic"] or meta["is_short"]) else 1.0
        q = min(1.0, meta["df"] / 5.0) * spec
        amb = 0.6 if meta["is_short"] else (0.4 if meta["in_generic"] else 0.05)
        p = 1.0 - amb
        return q, p

    def objective(sel):
        """固定 x=sel：每提及取最优链接（q·1 − λ(1−p)≥0 才接受 z=1），
        每单元至多一次最佳支持；返回 (obj, risk_ratio, covered_units)。"""
        sel_set = set(sel)
        link_util = 0.0
        risk_num = risk_den = 0.0
        per_unit = collections.defaultdict(float)
        linked_docs = collections.Counter()
        for t in sel:
            meta = pool[t]
            q, p = qp(meta)
            w_m = 1.0 / max(1, len(occ.get(meta["norm"], [])) // 50)
            util = q + eta * q - lam * (1 - p)
            if util < 0:
                continue
            if with_risk:
                # ε 约束下的保守接受：加入后 ratio 超 eps 则跳过该提及
                if (risk_num + (1 - p)) > eps * (risk_den + 1) + 1e-9 \
                        and risk_den > 0:
                    continue
            link_util += w_m * util
            risk_num += (1 - p); risk_den += 1
            if meta["norm"] in dev_norms:
                per_unit[meta["norm"]] = max(per_unit[meta["norm"]], q)
        cover = sum(per_unit.values())
        return cover + link_util, (risk_num / risk_den if risk_den else 0), len(per_unit)

    sel = cand[:K]
    best = objective(sel)[0]
    rng = random.Random(seed)
    improved = True
    it = 0
    while improved and it < 8:
        improved = False
        it += 1
        sel_set = set(sel)
        outside = [t for t in cand if t not in sel_set]
        rng.shuffle(outside)
        for out_t in outside[:400]:
            for in_i in rng.sample(range(len(sel)), 6):
                trial = sel.copy()
                trial[in_i] = out_t
                v = objective(trial)[0]
                if v > best + 1e-6:
                    sel, best, improved = trial, v, True
                    break
            if improved:
                break
    obj, risk, units = objective(sel)
    res = {"sel": sorted(sel), "objective": round(obj, 3),
           "risk_ratio": round(risk, 3), "dev_units_covered": units,
           "model": None if model is None else
           {"theta": [round(w, 4) for w in model[0]]}}
    if beam_ref:
        ref = beam_reference(cand, K, objective, width=24)
        res["beam_reference_obj"] = ref
    return res


def beam_reference(cand, K, objective, width=24):
    """top-300 池小参考：beam 维护 width 个部分解，按目标延展。"""
    small = cand[:300]
    beams = [([], 0.0)]
    for t in small:
        nxt = []
        for sel, v in beams:
            nxt.append((sel, v))
            if len(sel) < K:
                s2 = sel + [t]
                nxt.append((s2, objective(s2)[0]))
        nxt.sort(key=lambda x: -x[1])
        beams = nxt[:width]
    return round(max(objective(b[0])[0] for b in beams if len(b[0]) <= K), 3)


def stage_arms(dom):
    pool_doc = json.load(open(os.path.join(OUT, dom, "pool_features.json"),
                              encoding="utf-8"))
    pool = pool_doc["pool"]
    _, _, _, _, three = load_all(dom)
    K = len(three)
    dev_norms = {_norm_term(t) for r in (json.loads(l) for l in
                  open(os.path.join(OUT, dom, "dev_silver.jsonl"),
                       encoding="utf-8")) if r.get("terms")
                  for t in r["terms"]}
    model = train_scorer(pool, dev_norms)
    arms = {}
    # learnedrank：全池 top-K
    ranked = sorted(pool.items(), key=lambda kv: -_score(kv[1], model))
    arms["learnedrank"] = {t: (m.get("type") or "term")
                           for t, m in ranked[:K]}
    # 共享：模型/排序/构造提及只算一次（此前逐调用重扫 3000 篇是热点）
    scored = ranked
    cand = [t for t, _ in scored[:1500]]
    occ = _mentions_construct(
        dom, {t: (pool[t].get("type") or "term") for t in cand})
    print(f"[{dom}] shared mentions built ({len(occ)} norms)", flush=True)
    for arm in ("cover", "coverRisk"):
        r = select_cover(dom, pool, K, with_risk=(arm == "coverRisk"),
                         beam_ref=(arm == "cover" and dom == "cs"),
                         occ=occ, model=model, scored=scored)
        arms[arm] = {t: (pool[t].get("type") or "term") for t in r["sel"]}
        json.dump(r, open(os.path.join(OUT, dom, f"sel_{arm}.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{dom}] {arm}: obj={r['objective']} risk={r['risk_ratio']} "
              f"units={r['dev_units_covered']}"
              + (f" beamRef={r.get('beam_reference_obj')}" if
                 "beam_reference_obj" in r else ""))
    # 预算曲线（learnedrank/cover/coverRisk × K/2, K, 2K）——cover 用
    # select_cover 重跑不同 K
    curve = {}
    for arm in NEW_ARMS:
        for kf in K_CURVE:
            kk = max(10, int(K * kf))
            if arm == "learnedrank":
                lex = {t: (m.get("type") or "term")
                       for t, m in ranked[:kk]}
            else:
                r = select_cover(dom, pool, kk,
                                 with_risk=(arm == "coverRisk"), seed=1,
                                 occ=occ, model=model, scored=scored)
                lex = {t: (pool[t].get("type") or "term") for t in r["sel"]}
            curve[f"{arm}@{kk}"] = lex
    json.dump({"arms": arms, "curve": curve, "K": K,
               "model": None if model is None else
               {"theta": [round(w, 4) for w in model[0]],
                "feats": FEATS}},
              open(os.path.join(OUT, dom, "new_arms.json"), "w",
                   encoding="utf-8"), ensure_ascii=False)
    # 写入 components/ 供 judge/tables 复用
    comps = os.path.join(dom_out(dom), "components")
    for arm, lex in arms.items():
        json.dump(lex, open(os.path.join(comps, f"arm_{arm}.json"), "w",
                            encoding="utf-8"), ensure_ascii=False)
    print(f"[{dom}] new arms written: {list(arms)} K={K}")


# ---------- stage: eval ----------

def prf(test, silver, lex):
    pats = precompile(lex)
    hit_n = sil_n = hit_in = sil_hit = 0
    for d in test:
        sil = {_norm_term(t) for t in silver.get(d["doc_id"], [])}
        matched = set()
        for surface, _et, _s, _e in scan(d["text"], pats):
            hit_n += 1
            if _norm_term(surface) in sil:
                hit_in += 1
            matched.add(_norm_term(surface))
        sil_n += len(sil)
        sil_hit += len(sil & matched)
    p = hit_in / hit_n if hit_n else 0.0
    r = sil_hit / sil_n if sil_n else 0.0
    return {"P": round(p, 3), "R": round(r, 3),
            "F1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def stage_eval(dom):
    construct, test, silver_a, silver_c, three = load_all(dom)
    new = json.load(open(os.path.join(OUT, dom, "new_arms.json"),
                         encoding="utf-8"))
    comps = os.path.join(dom_out(dom), "components")
    rows = {}
    for arm in ("freq", "cvalue", "threesource", *NEW_ARMS):
        lex = json.load(open(os.path.join(comps, f"arm_{arm}.json"),
                             encoding="utf-8"))
        rows[arm] = {"vs_A": prf(test, silver_a, lex),
                     "vs_C": prf([d for d in test if d["doc_id"] in silver_c],
                                 silver_c, lex)}
    curve = {k: prf(test, silver_a, lex) for k, lex in new["curve"].items()}
    json.dump({"main": rows, "curve": curve, "K": new["K"]},
              open(os.path.join(OUT, dom, "eval.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    for arm, v in rows.items():
        print(f"[{dom}] {arm:12s} A:P{v['vs_A']['P']}/R{v['vs_A']['R']} "
              f"C:R{v['vs_C']['R']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["pool", "dev", "arms", "eval"])
    ap.add_argument("--dom", required=True, choices=["cs", "bio", "mat"])
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, args.dom), exist_ok=True)
    {"pool": stage_pool, "dev": stage_dev,
     "arms": stage_arms, "eval": stage_eval}[args.stage](args.dom)


if __name__ == "__main__":
    main()
