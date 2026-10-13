#!/usr/bin/env python3
"""P1-c 召回参照：解析人工从零枚举表（20261013_enumeration/enumeration_sheet.md），
与跨域五臂选定词表（20261011_cross_domain/<dom>/components/arm_*.json）做面形对照。

口径：
- 枚举表由用户从零填写（无候选提示），面形照抄原文；
- 匹配 = casefold + 空白折叠 + 去外围标点后的精确面形相等（不做词干/复数归一）；
- 每域统计唯一术语集（跨单元去重），逐臂计覆盖率；
- 输出仅含面形与计数（不含摘要原文），可入库；枚举表本身 gitignore。

用法：python3 enumeration_recall.py   （结果写 20261013_enumeration/enumeration_recall.json）
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "results")
ENUM_DIR = os.path.join(ROOT, "20261013_enumeration")
XD_DIR = os.path.join(ROOT, "20261011_cross_domain")
ARMS = ["schema", "freq", "cvalue", "llmdirect", "threesource"]

UNIT_RE = re.compile(r"^## 单元 (\w+):(\d+)（(\w+)）")
TERM_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*([^|]*?)\s*\|")


def norm(s: str) -> str:
    s = s.strip().strip("\"'`.,;:()[]").casefold()
    return re.sub(r"\s+", " ", s)


def parse_sheet(path: str) -> dict[str, list[str]]:
    """unit_terms[dom] = 按单元分组的原始填写术语列表。"""
    unit_terms: dict[str, list[str]] = defaultdict(list)
    cur_terms: list[str] | None = None
    for line in open(path, encoding="utf-8"):
        m = UNIT_RE.match(line)
        if m:
            cur_terms = []
            unit_terms[m.group(3)].append((f"{m.group(1)}:{m.group(2)}",
                                           cur_terms))
            continue
        if cur_terms is not None and line.startswith("|"):
            rm = TERM_ROW_RE.match(line)
            if rm and rm.group(1).strip():
                cur_terms.append(rm.group(1).strip())
            elif rm and not rm.group(1).strip():
                pass  # 空行模板
    # 重组：{dom: [(unit_id, [terms])]}
    by_dom: dict[str, list] = defaultdict(list)
    for dom, units in unit_terms.items():
        by_dom[dom] = units
    return by_dom


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    sheet = os.path.join(ENUM_DIR, "enumeration_sheet.md")
    by_dom = parse_sheet(sheet)
    assert set(by_dom) == {"cs", "bio", "mat"}, set(by_dom)

    out = {"per_domain": {}, "overall": {}, "terms": {}}
    total_enum = 0
    arm_hits = {a: 0 for a in ARMS}
    for dom in ["cs", "bio", "mat"]:
        units = by_dom[dom]
        n_units = len(units)
        # 唯一术语集（跨单元去重，保留首个面形）
        uniq: dict[str, str] = {}
        for uid, terms in units:
            for t in terms:
                uniq.setdefault(norm(t), t)
        terms_n = sorted(uniq.values(), key=lambda x: norm(x))
        n_terms = len(uniq)
        total_enum += n_terms

        arms_surf: dict[str, set[str]] = {}
        arm_sizes: dict[str, int] = {}
        for arm in ARMS:
            d = json.load(open(os.path.join(XD_DIR, dom, "components",
                                            f"arm_{arm}.json")))
            arms_surf[arm] = {norm(s) for s in d}
            arm_sizes[arm] = len(d)

        matched = {a: sorted([uniq[n] for n in uniq if n in arms_surf[a]],
                             key=str.casefold)
                   for a in ARMS}
        for a in ARMS:
            arm_hits[a] += len(matched[a])
        out["per_domain"][dom] = {
            "units": n_units,
            "terms_filled_raw": sum(len(t) for _, t in units),
            "terms_unique": n_terms,
            "arm_sizes": arm_sizes,
            "coverage": {a: {"hits": len(matched[a]), "n": n_terms,
                             "rate": round(len(matched[a]) / n_terms, 3)}
                         for a in ARMS},
        }
        out["terms"][dom] = {a: matched[a] for a in ARMS}
        out["terms"][dom]["__enum__"] = terms_n

    n_all = total_enum
    out["overall"] = {
        "terms_unique_total": n_all,
        "coverage": {a: {"hits": arm_hits[a], "n": n_all,
                         "rate": round(arm_hits[a] / n_all, 3),
                         "wilson95": [round(v, 3) for v in
                                      wilson(arm_hits[a], n_all)]}
                     for a in ARMS},
        "union_any_arm": {"hits": -1},  # 填充见下
    }
    # 任意臂并集覆盖
    union_hits = 0
    for dom in ["cs", "bio", "mat"]:
        enumset = {norm(t) for t in out["terms"][dom]["__enum__"]}
        union_surf = set()
        for a in ARMS:
            d = json.load(open(os.path.join(XD_DIR, dom, "components",
                                            f"arm_{a}.json")))
            union_surf |= {norm(s) for s in d}
        union_hits += len(enumset & union_surf)
    out["overall"]["union_any_arm"] = {
        "hits": union_hits, "n": n_all,
        "rate": round(union_hits / n_all, 3)}

    dst = os.path.join(ENUM_DIR, "enumeration_recall.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(json.dumps({k: out[k] for k in ["per_domain", "overall"]},
                     ensure_ascii=False, indent=1))
    print("written:", dst)


if __name__ == "__main__":
    main()
