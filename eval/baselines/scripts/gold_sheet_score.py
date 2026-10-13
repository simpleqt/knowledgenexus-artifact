#!/usr/bin/env python3
"""人工金标评分（第四轮评审必改-1(a)，填表后运行）。

输入：gold_sheet_builder.py 产出的已填写 sheet + 密钥 key。
输出：gold_agreement.json —— **verdicts-only**（不含任何语料正文/上下文，
可入 git）；控制台打印一致性摘要。

口径：
  Part A（KB1 生产链接）：人-模一致性（人 match vs 存档 model_match，
  borderline 单列不并入 valid），按 泛名/领域名/短符号 分层；
  Part B（跨域盲判）：人 vs 仲裁 final（jA/jB/jC 的 final）；
  双标注者：A/B 间一致性（Cohen's kappa + raw agreement），
  单人填写时只算 A。

用法：python3 eval/baselines/scripts/gold_sheet_score.py \
  --sheet eval/baselines/results/20261013_gold_sheet/gold_annotation_sheet.md \
  --key   eval/baselines/results/20261013_gold_sheet/gold_sheet_key.json \
  --out   eval/baselines/results/20261013_gold_sheet
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

FILL = re.compile(
    r"标注人(?P<who>[AB])\s*[：:]\s*match\s*=\s*(?P<m>valid|invalid|borderline)"
    r"[，,\s]*specificity\s*=\s*(?P<sp>domain|generic)",
    re.I)
ITEM = re.compile(r"^### (?P<id>[AB]-\d{3})")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def kappa(a: list, b: list) -> float:
    if not a:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / len(a)
    cnt_a = {l: a.count(l) / len(a) for l in labels}
    cnt_b = {l: b.count(l) / len(b) for l in labels}
    pe = sum(cnt_a.get(l, 0) * cnt_b.get(l, 0) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def parse_sheet(path: str) -> dict[str, dict]:
    """item id -> {"A": (match, spec), "B": (match, spec)}（未填的栏缺位）。"""
    out: dict[str, dict] = {}
    cur = None
    for ln in open(path):
        m = ITEM.match(ln)
        if m:
            cur = m.group("id")
            out.setdefault(cur, {})
            continue
        f = FILL.search(ln)
        if f and cur:
            out[cur][f.group("who").upper()] = (
                f.group("m").lower(), f.group("sp").lower())
    return {k: v for k, v in out.items() if v}


def agree(human: str, model: bool | None) -> bool | None:
    """模型 bool → 词面；borderline 视为独立类，不折算。"""
    if model is None:
        return None
    return human == ("valid" if model else "invalid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    key = json.load(open(args.key))
    filled = parse_sheet(args.sheet)

    res = {"partA": [], "partB": [], "summary": {}}
    for part, items in (("partA", key["partA"]), ("partB", key["partB"])):
        for it in items:
            f = filled.get(it["id"], {})
            row = {"id": it["id"]}
            for who in ("A", "B"):
                if who in f:
                    row[f"human_{who}"] = f[who][0]
                    row[f"spec_{who}"] = f[who][1]
            if part == "partA":
                row["source"] = it["source"]
                row["name_class"] = ("short" if len(it["name"]) <= 4 else
                                     ("generic" if it["model_specificity"] == "generic"
                                      else "domain"))
                row["model_match"] = it["model_match"]
                row["model_specificity"] = it["model_specificity"]
            else:
                row["domain"] = it["domain"]
                row["arm"] = it["arm"]
                row["model_final"] = it["model_final"]
            res[part].append(row)

    def summarize(part: str, model_key: str) -> dict:
        s: dict = {}
        rows = res[part]
        for who in ("A", "B"):
            got = [r for r in rows if f"human_{who}" in r]
            if not got:
                continue
            valid = [r for r in got if r[f"human_{who}"] == "valid"]
            lo, hi = wilson(len(valid), len(got))
            ag = [agree(r[f"human_{who}"], r[model_key]) for r in got]
            ag = [a for a in ag if a is not None]
            s[f"annot{who}"] = {
                "n": len(got),
                "valid": len(valid), "valid_rate": round(len(valid) / len(got), 3),
                "wilson95": [round(lo, 3), round(hi, 3)],
                "agree_model": sum(ag), "agree_n": len(ag),
                "agree_rate": round(sum(ag) / len(ag), 3) if ag else None,
                "verdicts": {v: sum(1 for r in got if r[f"human_{who}"] == v)
                             for v in ("valid", "invalid", "borderline")},
            }
        a = [r["human_A"] for r in rows if "human_A" in r and "human_B" in r]
        b = [r["human_B"] for r in rows if "human_A" in r and "human_B" in r]
        if a:
            s["inter_annotator"] = {
                "n": len(a), "raw": round(
                    sum(x == y for x, y in zip(a, b)) / len(a), 3),
                "kappa": round(kappa(a, b), 3)}
        return s

    res["summary"]["partA"] = summarize("partA", "model_match")
    res["summary"]["partB"] = summarize("partB", "model_final")
    # Part A 分层（以标注人 A 为主，B 有则并）
    strat: dict = {}
    for r in res["partA"]:
        cls = r["name_class"]
        for who in ("A", "B"):
            h = r.get(f"human_{who}")
            if h:
                c = strat.setdefault(cls, {"n": 0, "valid": 0, "agree": 0, "agree_n": 0})
                c["n"] += 1
                c["valid"] += h == "valid"
                a_ = agree(h, r["model_match"])
                if a_ is not None:
                    c["agree_n"] += 1
                    c["agree"] += a_
    for c in strat.values():
        if c["n"]:
            c["valid_rate"] = round(c["valid"] / c["n"], 3)
            if c["agree_n"]:
                c["agree_rate"] = round(c["agree"] / c["agree_n"], 3)
    res["summary"]["partA_by_class"] = strat
    # Part B 分臂
    by_arm: dict = {}
    for r in res["partB"]:
        for who in ("A", "B"):
            h = r.get(f"human_{who}")
            if h:
                c = by_arm.setdefault(r["arm"], {"n": 0, "valid": 0})
                c["n"] += 1
                c["valid"] += h == "valid"
    for c in by_arm.values():
        if c["n"]:
            c["valid_rate"] = round(c["valid"] / c["n"], 3)
    res["summary"]["partB_by_arm"] = by_arm

    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, "gold_agreement.json")
    with open(outp, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    print(f"[ok] verdicts-only -> {outp}")


if __name__ == "__main__":
    main()
