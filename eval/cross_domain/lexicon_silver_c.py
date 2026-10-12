#!/usr/bin/env python3
"""silver 第三判官（27B 本地）子集复标——枚举型 silver 的可靠性估计。

背景：网关 flash-next 在全摘要术语枚举上推理时长 >180s（读超时全军覆没，
仅判读短任务可用），silver 双标注降级为 A(=.32 35B) 主标 + C(=本地 27B)
30 篇/域子集复标；A-C 一致率即枚举工具的可靠性上界，如实入文。
用法：KNC_JC_BASE=http://127.0.0.1:18360/v1 .venv/bin/python ... --dom cs
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO  # noqa: E402
from lexicon_cross_domain import (SILVER_PROMPT, SCHEMA, chat,  # noqa: E402
                                  domain_label)
import yaml  # noqa: E402


def main():
    dom = sys.argv[sys.argv.index("--dom") + 1]
    root = os.path.join(REPO, "eval", "baselines", "results",
                        "20261011_cross_domain")
    out = os.path.join(root, dom)
    test = json.load(open(os.path.join(out, "test.json"), encoding="utf-8"))
    rng = random.Random(f"20261011-{dom}-silverC")
    subset = rng.sample(test, 30)
    schema_doc = yaml.safe_load(
        open(os.path.join(REPO, "schemas", SCHEMA[dom]), encoding="utf-8"))
    type_desc = "\n".join(f"- {t}: {str(s.get('description', ''))[:60]}"
                          for t, s in schema_doc["entity_types"].items())
    jc = {"name": os.environ.get("KNC_JC_MODEL", "deepseek-flash"),
          "base": os.environ["KNC_JC_BASE"],
          "key": os.environ.get("DEEPSEEK_API_KEY", "none"),
          "model": os.environ.get("KNC_JC_MODEL", "deepseek-flash")}
    p = os.path.join(out, "silver_C.jsonl")
    done = set()
    if os.path.exists(p):
        keep = []
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("terms"):  # 空行=超时/解析失败 → 视为未完成，重试
                keep.append(l)
                done.add(r["doc_id"])
        open(p, "w", encoding="utf-8").writelines(keep)
    with open(p, "a", encoding="utf-8") as f:
        for i, d in enumerate(subset):
            if d["doc_id"] in done:
                continue
            raw = chat(jc, SILVER_PROMPT.format(domain=domain_label(dom),
                                                types=type_desc,
                                                ctx=d["text"][:2500]),
                       max_tokens=6000)
            m = re.search(r'\{"terms".*\}', raw, re.S)
            terms = []
            if m:
                try:
                    terms = [t for t in json.loads(m.group())["terms"]
                             if isinstance(t, dict) and t.get("surface")]
                except Exception:  # noqa: BLE001
                    pass
            f.write(json.dumps({"doc_id": d["doc_id"], "judge": "C",
                                "terms": terms}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{dom}] silverC {i + 1}/30 (terms {len(terms)})",
                  flush=True)


if __name__ == "__main__":
    main()
