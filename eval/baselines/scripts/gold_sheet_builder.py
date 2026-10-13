#!/usr/bin/env python3
"""人工金标标注表生成器（第四轮评审必改-1(a)，2026-10-13）。

目的：为论文 judged-valid 系列数字（0.84 生产链接审计 / 跨域盲判
0.838-0.912）提供独立人工金标与一致性——审稿人要求"独立双标注者
金标 + 一致性"，本表即标注底稿。**盲标**：模型判定不进表。

Part A  50 条 KB1 生产链接（模型判定已知，评分时对人-模一致性）：
        20 条取自 20260909 均匀审计存档（100 条，head 自然偏置）；
        30 条取自 20261010 分层审计存档（300 条）——原表无行级 stratum
        标签，按 canonical 名长度近似风险分层：≤4 字母短符号 12、
        其余按模型 specificity 分 泛名 9 / 领域名 9。
        上下文按 chunk_id 从 OS 现取（±250 字符，⟦⟧ 标记 span 位）。
Part B  50 条跨域判定项（仲裁存档 /data1/knc_eval_backup/xdomain_judge_ctx，
        生产机本地，不入 git）：3 域 × 5 臂——schema/freq/cvalue/llmdirect
        各 3 条 + threesource 4 条，CS/bio 的 threesource 各 +1 条补足 50。
        臂名与模型判定都不进表（盲）；域信息保留（上下文语言域）。

输出（均含授权语料正文 / 公开语料摘要——git 必须忽略）：
  <out>/gold_annotation_sheet.md   标注底稿（双标注者 A/B 两栏）
  <out>/gold_sheet_key.json        模型判定 + 坐标密钥（评分用）

评分：填写后运行 gold_sheet_score.py（输出 verdicts-only JSON 可入库）。

用法：bash -c 'source eval/baselines/scripts/knc_env.sh && cd <repo> && \
  .venv/bin/python eval/baselines/scripts/gold_sheet_builder.py \
  --out eval/baselines/results/20261013_gold_sheet'
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "services", "retrieval-api"))
import app.main as M  # noqa: E402  (OS client 经 search 模块复用，只读)

KB1 = "104112e63d9d4aa5a6c21e1c1549ef0f"
UNIFORM = "eval/baselines/results/20261009_linkaudit/annotations.jsonl"
STRAT = "eval/baselines/results/20261010_stratified_audit/annotations.jsonl"
XDOM = "/data1/knc_eval_backup/xdomain_judge_ctx"
DOMS = ("cs", "bio", "mat")
DOM_CN = {"cs": "CS (arXiv cs.CL)", "bio": "生物医学 (PubMed)", "mat": "材料 (cond-mat)"}
ARMS = ("schema", "freq", "cvalue", "llmdirect", "threesource")

RUBRIC_A = """**问题 1（match）**：⟦ ⟧ 内的提取位在此上下文中，是否确指该实体
canonical 名的**同一概念**？
- `valid`：明确同一概念（如 ⟦germanium detectors⟧ → germanium detector）；
- `invalid`：异义（如动词 lead vs 金属铅）、异概念、或根本不是术语性提及
  （普通语言用法）；
- `borderline`：确实两可——拿不准就标这个，不要猜。

**问题 2（specificity）**：该实体 canonical 名本身是
- `domain`：领域专用技术概念（如 hpge detector、Compton edge）；
- `generic`：任何学科都能用的泛学术/工程词（如 method、result、parameter）。"""

RUBRIC_B = """**问题 1（match）**：⟦ ⟧ 内的提取位在此上下文中，是否是该术语的一次
**真实术语性提及**（即以该术语的技术含义被使用）？
- `valid`：是（术语性用法，含义与给定类型相符）；
- `invalid`：否（普通语言用法 / 异义 / 断章碎片）；
- `borderline`：两可。

**问题 2（specificity）**：该术语是 `domain`（领域术语）还是 `generic`
（泛学术词）。"""


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def fetch_ctx(chunk_id: str, surface: str, s: int, e: int) -> tuple[str, bool]:
    """OS 取 chunk 全文，±250 字符窗 + ⟦⟧ 标记；span 漂移时兜底寻位。"""
    try:
        ch = M.search.chunks_by_ids(KB1, [chunk_id], 1)
        text = ch[0]["content_text"] if ch else ""
    except Exception:
        text = ""
    if not text:
        return "(chunk text unavailable)", False
    if text[s:e].lower() == surface.lower():
        lo, hi = max(0, s - 250), min(len(text), e + 250)
        return text[lo:s] + "⟦" + text[s:e] + "⟧" + text[e:hi], True
    j = text.lower().find(surface.lower(), max(0, s - 80))
    if j < 0:
        j = text.lower().find(surface.lower())
    if j < 0:
        return text[:500], False
    e2 = j + len(surface)
    lo, hi = max(0, j - 250), min(len(text), e2 + 250)
    return text[lo:j] + "⟦" + text[j:e2] + "⟧" + text[e2:hi], False


def sample_part_a(rng: random.Random) -> list[dict]:
    uniform = load_jsonl(os.path.join(REPO, UNIFORM))
    strat = load_jsonl(os.path.join(REPO, STRAT))
    a = rng.sample(uniform, 20)
    short = [r for r in strat if len(r["name"]) <= 4]
    rest = [r for r in strat if len(r["name"]) > 4]
    generic = [r for r in rest if r.get("specificity") == "generic"]
    domain = [r for r in rest if r.get("specificity") == "domain"]
    a += rng.sample(short, 12) + rng.sample(generic, 9) + rng.sample(domain, 9)
    rng.shuffle(a)
    return a


def sample_part_b(rng: random.Random) -> list[dict]:
    out = []
    for d in DOMS:
        for arm in ARMS:
            recs = load_jsonl(os.path.join(XDOM, f"{d}_arbit_{arm}.jsonl"))
            k = 4 if arm == "threesource" else 3
            out += [{"domain": d, "arm": arm, **r} for r in rng.sample(recs, k)]
    for d in ("cs", "bio"):  # 补足 50：threesource 是论文主张核心
        recs = load_jsonl(os.path.join(XDOM, f"{d}_arbit_threesource.jsonl"))
        out.append({"domain": d, "arm": "threesource", **rng.choice(recs)})
    rng.shuffle(out)
    return out


def md_escape(t: str) -> str:
    return t.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--out", default="eval/baselines/results/20261013_gold_sheet")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    part_a = sample_part_a(rng)
    part_b = sample_part_b(rng)

    key = {"seed": args.seed, "kb1": KB1, "partA": [], "partB": []}
    lines = [
        "# 人工金标标注表（论文 When Schema Lexicons Fail，第四轮评审必改-1a）",
        "",
        f"生成：gold_sheet_builder.py --seed {args.seed}（{len(part_a)}+{len(part_b)} 条，"
        "采样确定性可复现）。本表**盲标**：不含任何模型判定。",
        "",
        "**填写约定**：每条在『标注人A』行填两个词（match 三选一：`valid` / "
        "`invalid` / `borderline`；specificity 二选一：`domain` / `generic`），"
        "理由可选。两位标注者各填一行；只有一人时只填 A 行。",
        "",
        "---",
        "",
        "# Part A：KB1 生产链接（50 条）",
        "",
        "语料为核电子学期刊论文（授权语料，表不入 git）。每条给出实体 "
        "canonical 名与类型、提取面形、以及提取位前后 ±250 字符的原文"
        "（⟦ ⟧ 内为提取位）。",
        "",
        RUBRIC_A,
        "",
    ]
    drop = []
    for i, r in enumerate(part_a):
        ctx, span_ok = fetch_ctx(r["chunk_id"], r["surface"], int(r["s"]), int(r["e"]))
        if "(chunk text unavailable)" in ctx:
            drop.append(i)
            continue
        iid = f"A-{i+1:03d}"
        lines += [
            f"### {iid}",
            "",
            f"- 实体：**{md_escape(r['name'])}**（类型 {md_escape(r['etype'])}）"
            + ("　<span style=\"color:#a00\">span 漂移已就近寻位</span>" if not span_ok else ""),
            f"- 提取面形：{md_escape(r['surface'])}",
            "- 上下文：",
            f"  > {md_escape(ctx)}",
            "",
            f"- 标注人A：match = ____，specificity = ____，理由 =",
            f"- 标注人B：match = ____，specificity = ____",
            "",
        ]
        key["partA"].append({
            "id": iid, "source": "uniform" if r in part_a else "stratified",
            "entity_id": r["entity_id"], "name": r["name"], "etype": r["etype"],
            "surface": r["surface"], "s": int(r["s"]), "e": int(r["e"]),
            "chunk_id": r["chunk_id"], "span_ok": span_ok,
            "model_match": r.get("match"), "model_borderline": r.get("borderline"),
            "model_specificity": r.get("specificity"),
        })
    # 修正 source 标记（上面 in 判定在 shuffle 后不可靠，按归属重算）
    uni_ids = {r["entity_id"] + r["chunk_id"] + str(r["s"]) for r in load_jsonl(os.path.join(REPO, UNIFORM))}
    for k in key["partA"]:
        k["source"] = "uniform" if k["entity_id"] + k["chunk_id"] + str(k["s"]) in uni_ids else "stratified"

    lines += [
        "---",
        "",
        "# Part B：跨域术语提及判定（50 条）",
        "",
        "语料为公开摘要（arXiv / PubMed）。每条给出术语（canonical + 类型）、"
        "面形、以及提取位上下文（⟦ ⟧ 内为提取位）。臂来源信息**不提供**"
        "（保持盲态）。",
        "",
        RUBRIC_B,
        "",
    ]
    for i, r in enumerate(part_b):
        iid = f"B-{i+1:03d}"
        lines += [
            f"### {iid}（{DOM_CN[r['domain']]}）",
            "",
            f"- 术语：**{md_escape(r['entity'])}**（类型 {md_escape(r['etype'])}）"
            f"｜面形：{md_escape(r['surface'])}",
            "- 上下文：",
            f"  > {md_escape(r.get('ctx', ''))}",
            "",
            f"- 标注人A：match = ____，specificity = ____，理由 =",
            f"- 标注人B：match = ____，specificity = ____",
            "",
        ]
        key["partB"].append({
            "id": iid, "domain": r["domain"], "arm": r["arm"],
            "stratum": r.get("stratum"), "entity": r["entity"],
            "etype": r["etype"], "surface": r["surface"],
            "doc_id": r.get("doc_id"), "s": r.get("s"), "e": r.get("e"),
            "model_final": r.get("final"), "jA": r.get("jA"),
            "jB": r.get("jB"), "jC": r.get("jC"), "mode": r.get("mode"),
        })

    lines += [
        "---",
        "",
        f"共 Part A {len(key['partA'])} 条 + Part B {len(key['partB'])} 条"
        f"（{len(drop)} 条因 chunk 文本缺失被剔除）。填完运行：",
        "`python3 eval/baselines/scripts/gold_sheet_score.py"
        f" --sheet {args.out}/gold_annotation_sheet.md"
        f" --key {args.out}/gold_sheet_key.json --out {args.out}`",
        "",
    ]
    with open(os.path.join(args.out, "gold_annotation_sheet.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(args.out, "gold_sheet_key.json"), "w") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    print(f"[ok] sheet={len(key['partA'])}A+{len(key['partB'])}B "
          f"(dropped {len(drop)}) -> {args.out}")


if __name__ == "__main__":
    main()
