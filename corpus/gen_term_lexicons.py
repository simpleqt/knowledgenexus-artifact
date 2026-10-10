#!/usr/bin/env python3
"""B92 学科 Schema 术语词典批产（2026-10-07 实体塌缩事故的根治件）。

背景：K1 提及抽取=纯词典扫描，词典=Schema entity_types 的主名+别名。18 学科层
批产（gen_subject_schemas.py）产出的是"类型分类学"（每类 4-8 个类别词别名），
hydrogeology 手工演进到 3,742 条真术语所以健康；physics/nuclear_electronics
停留在类别词 → 大库扫描全命中"方法/参数/energy" → 61k 文档塌缩成 575 个类型级
实体（19M 链接挂 575 顶点）。本工具给既有 Schema 扩真术语别名：

  种子三源：
    1. 健康库实体移植（nuclear_electronics ← 21BCBED8 RADAI 真术语，带类型）
    2. 语料候选分类（两库抽样块文本的高频技术词 → LLM 归类/丢弃）
    3. LLM 逐类型扩写（.32 抽取集群，类型描述+种子 → 补充真术语）
  质量闸：LLM 产物必须命中语料词频集或种子集（防幻觉）；全局跨类型去重
  （"不同类型 aliases 不得重复"，种子类型优先）；长度/停用词过滤。
  落盘：原 YAML 保留类型/关系体系，仅扩 aliases，version bump + changelog 注释，
  loader.validator 校验通过才写。

幂等：重跑只补缺失词条（按已存在词条集合差集），已演进版本不回退。
用法：.venv/bin/python corpus/gen_term_lexicons.py --schema nuclear_electronics \
        [--seed-kb 21bcbed8...] [--corpus-kbs 104112e6,693372d5] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "schema"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "engine"))
os.environ.setdefault("SCHEMA_DIR", os.path.join(os.path.dirname(__file__), "..", "schemas"))

CATCHALL_TYPES = {"term", "term_cn", "model_or_code", "concept", "question"}
# 学术版面/话语词（Title 词形单词条拒收——语料分类的版面噪声）
BOILER = set("""According Accepted Acknowledgements Application Approach Assembly Author
Authors Content Contribution Data Declaration Discussed During Each Effects Else
Engineers Figure Figures Funding However Information Introduction Its Journal
Letter Letters Materials Methods Note Notes Online Paper Papers Part Proceedings
Published Publisher Received Research Results Section See Selected Shows Study
Studies Supported Table Tables Using Values Vol Volume Work article authors based
between both during each from have into more most only other over per several
shown shows some such supported than that the their then these this those through
under using various via when where which with work""".split())
STOP_EN = set("""the and for with this that from are was were has have which using based between
these such can also our its their than then when where while both each other more most over
into only used use may one two three fig table figure section abstract introduction conclusion
references keywords results methods discussion study paper research work proposed show shown
respectively therefore however thus furthermore moreover addition various different several
some all not but per via due to""".split())

def _llm_chat(prompt: str, max_tokens: int = 4000) -> str:
    from knc_engine import llm
    last = ""
    for _ in range(3):  # 偶发空响应/无 JSON 重试（.32 router 观测到）
        r = llm.chat(prompt, system="你是领域术语编典器，只输出 JSON。",
                     max_tokens=max_tokens, temperature=0.2)
        r = re.sub(r"<think>.*?</think>", "", r or "", flags=re.S).strip()
        if r:
            return r
        last = r or ""
        time.sleep(2)
    return last


def _json_obj(raw: str) -> dict | None:
    """从 LLM 输出稳健提取 JSON 对象（截断/包裹文本容忍）。"""
    try:
        return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return None
        return None


def _load_env() -> dict:
    env = {}
    for line in open(os.path.join(os.path.dirname(__file__), "..", "deploy", ".env"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def _seed_from_kb(env: dict, kb_hex: str, bib_types: set[str]) -> dict[str, set]:
    import pymysql
    conn = pymysql.connect(host="127.0.0.1", port=13306, user="knc", password=env["MYSQL_PASSWORD"],
                           database="knc", charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor)
    cur = conn.cursor()
    cur.execute("SELECT canonical_name, entity_type FROM canonical_entity"
                " WHERE kb_id=UNHEX(%s) AND status='ACTIVE'", (kb_hex,))
    seed: dict[str, set] = collections.defaultdict(set)
    for r in cur.fetchall():
        if r["entity_type"] in bib_types or not r["canonical_name"]:
            continue
        n = r["canonical_name"].strip()
        if 1 < len(n) <= 40:
            seed[r["entity_type"]].add(n)
    conn.close()
    return seed


def _corpus_freq(env: dict, kb_hex: str, mode: str, sample: int = 3000) -> collections.Counter:
    """语料词频（候选池 + 幻觉校验集共用）。"""
    from opensearchpy import OpenSearch
    c = OpenSearch(hosts=[{"host": "127.0.0.1", "port": 19200, "scheme": "https"}],
                   http_auth=("admin", env["OPENSEARCH_ADMIN_PASSWORD"]),
                   verify_certs=False, ssl_show_warn=False)
    r = c.search(index="knc-chunk-v2", body={"size": sample, "query": {"bool": {
        "filter": [{"term": {"kb_id": kb_hex}}, {"term": {"visible": True}}]}},
        "_source": ["content_text"]})
    text = "\n".join(h["_source"]["content_text"] for h in r["hits"]["hits"])
    freq = collections.Counter()
    if mode == "en":
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9\-]{2,}(?:\s+[A-Za-z][A-Za-z0-9\-]{2,}){0,3}", text):
            t = m.group()
            if t.lower() not in STOP_EN:
                freq[t] += 1
    else:
        import jieba
        for w in jieba.cut(text):
            if 2 <= len(w) <= 10 and not w.isspace():
                freq[w] += 1
    return freq


def _ok_term(t: str) -> bool:
    t = t.strip()
    if "\n" in t or "\t" in t:
        return False
    if not (1 < len(t) <= 40) or t.lower() in STOP_EN or t.isdigit():
        return False
    if len(t.split()) > 5:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", t):
        return False
    # 单个 Title 词形 ASCII 词且为版面/话语词 → 版面噪声
    if t.isascii() and len(t.split()) == 1 and t[0].isupper() and not t.isupper():
        if t in BOILER:
            return False
    return True


def _classify_batch(terms: list[str], types: dict[str, str]) -> dict[str, list[str]]:
    """语料候选 → LLM 归类到 schema 类型或丢弃。"""
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(terms))
    type_desc = "\n".join(f"- {tid}: {spec.get('description', '')[:60]}" for tid, spec in types.items())
    fmt = '{"归类": {"类型id": [术语编号列表]}, "丢弃": [编号列表]}'
    raw = _llm_chat(
        f"把下列术语归类到最合适的实体类型，或丢弃（非领域实体术语/泛词/停用词）。\n"
        f"类型清单：\n{type_desc}\n\n术语：\n{listing}\n\n"
        f"输出 JSON：{fmt}", max_tokens=6000)
    out = collections.defaultdict(list)
    doc = _json_obj(raw) or {}
    try:
        for tid, idxs in (doc.get("归类") or {}).items():
            if tid in types:
                for i in idxs if isinstance(idxs, list) else []:
                    if isinstance(i, int) and 0 <= i < len(terms):
                        out[tid].append(terms[i])
    except Exception:
        pass
    return dict(out)


def _expand_type(tid: str, desc: str, seeds: list[str], n: int) -> list[str]:
    seed_s = "\n".join(seeds[:150]) or "（无）"
    raw = _llm_chat(
        f"实体类型：{tid}（{desc[:80]}）\n已有种子术语：\n{seed_s}\n\n"
        f"补充 {n} 个该类型的真实领域术语（中英混合，具体到实例/专有名词，"
        f"如 HPGe、康普顿散射、Geant4；禁止类别词如 方法/能量/参数；每条 ≤5 词）。"
        f'输出 JSON：{{"terms": ["术语", ...]}}', max_tokens=6000)
    doc = _json_obj(raw) or {}
    return doc.get("terms") or []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--seed-kb", default="")
    ap.add_argument("--corpus-kbs", default="",
                    help="kb_hex:mode[,kb_hex:mode] mode∈en|cn")
    ap.add_argument("--expand-per-type", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from knc_schema import loader
    import yaml

    path = os.path.join(os.environ["SCHEMA_DIR"], f"{args.schema}.yaml")
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    types = doc.get("entity_types") or {}
    existing: dict[str, set] = {tid: {(a or "").strip() for a in (spec.get("aliases") or [])}
                                for tid, spec in types.items()}
    # 类型特异性序（子类型先于父类型——跨类型冲突消解与遗留修复共用）
    PRIORITY = ["semiconductor_detector", "photodetector", "scintillator", "isotope",
                "nuclear_reaction", "energy_spectrum", "radiation_measurement",
                "data_acquisition", "particle", "material", "method", "parameter",
                "circuit", "detector", "experimental_facility", "technology",
                "instrument", "process", "phenomenon", "theory", "model", "effect",
                "energy", "field", "law", "equation", "physical_constant",
                "physical_quantity", "state_of_matter", "interaction", "event",
                "location", "operation", "prediction", "disturbance", "wave",
                "facility", "geobody", "contaminant", "dataset", "experiment"]

    def _prio(tid):
        return PRIORITY.index(tid) if tid in PRIORITY else 50 + len(existing.get(tid, ()))

    # 遗留跨类型别名冲突修复（2026-10-08：nuclear v1.0 原文件即含 223 处，当日
    # validator 规则晚于文件落盘——按特异性保留一处，父类型让位子类型）
    _by_alias: dict[str, list] = {}
    for tid, spec in types.items():
        for a in spec.get("aliases") or []:
            _by_alias.setdefault(str(a).strip().lower(), []).append(tid)
    _fixed = 0
    for alias, tids in _by_alias.items():
        if len(tids) > 1:
            keep = min(tids, key=_prio)
            for tid in tids:
                if tid != keep:
                    before = len(types[tid].get("aliases") or [])
                    types[tid]["aliases"] = [a for a in (types[tid].get("aliases") or [])
                                             if str(a).strip().lower() != alias]
                    _fixed += before - len(types[tid]["aliases"])
    existing = {tid: {(a or "").strip() for a in (spec.get("aliases") or [])}
                for tid, spec in types.items()}
    if _fixed:
        print(f"遗留冲突修复：按特异性移除 {_fixed} 处重复别名")
    print(f"{args.schema}: {len(types)} 类型，现有词条 {sum(len(v) for v in existing.values())}")

    env = _load_env()
    freq = collections.Counter()
    for part in filter(None, args.corpus_kbs.split(",")):
        kb, mode = part.split(":")
        f = _corpus_freq(env, kb, mode)
        freq.update(f)
        print(f"  语料 {kb[:8]}({mode}) 词条 {len(f)}")
    freq_set = set(freq)

    got: dict[str, set] = collections.defaultdict(set)
    # 源1：健康库种子
    if args.seed_kb:
        for tid, terms in _seed_from_kb(env, args.seed_kb, {"person", "paper"} | CATCHALL_TYPES).items():
            tgt = tid if tid in types else None
            if not tgt:  # RADAI 类型不在 schema → 就近映射留人工（打印）
                print(f"  [seed] 类型 {tid} 不在 schema（{len(terms)} 词条待映射），跳过")
                continue
            got[tid] |= {t for t in terms if _ok_term(t)}
        print(f"  种子入库：{sum(len(v) for v in got.values())}")
    # 源2：语料候选分类（频率≥5 的前 8000，300/批）
    cands = [t for t, c in freq.most_common(8000) if c >= 5 and _ok_term(t)]
    for i in range(0, len(cands), 300):
        batch = cands[i:i + 300]
        for tid, terms in _classify_batch(batch, types).items():
            if tid in CATCHALL_TYPES:
                continue
            got[tid] |= {t for t in terms if _ok_term(t)}
        print(f"  分类 {min(i + 300, len(cands))}/{len(cands)}，累计 {sum(len(v) for v in got.values())}", flush=True)
        time.sleep(0.5)
    # 源3：逐类型 LLM 扩写（产物须命中语料或种子，防幻觉）
    seeds_all = {t for v in got.values() for t in v}
    for tid, spec in types.items():
        if tid in ("person", "paper", "organization") or tid in CATCHALL_TYPES:
            continue
        for t in _expand_type(tid, str(spec.get("description", "")), sorted(got.get(tid, [])), args.expand_per_type):
            if _ok_term(t) and (t in freq_set or t.lower() in freq_set or t in seeds_all):
                got[tid].add(t)
        print(f"  扩写 {tid}: +{len(got.get(tid, set()))}", flush=True)

    # 合并落盘：全局去重 + 特异性优先（子类型先于父类型拿词）
    global_seen: set = set()
    for tid in sorted(existing, key=lambda t: -len(existing[t])):
        global_seen |= {a.lower() for a in existing[tid]}
    added_total = 0
    for tid, terms in sorted(got.items(), key=lambda x: _prio(x[0])):
        add = []
        for t in sorted(terms):
            tl = t.lower()
            if tl in global_seen:
                continue
            global_seen.add(tl)
            add.append(t)
        if add:
            types[tid].setdefault("aliases", [])
            types[tid]["aliases"] = sorted(set(types[tid]["aliases"]) | set(add), key=str.lower)
            added_total += len(add)
            print(f"  [merge] {tid} +{len(add)}（样例：{add[:6]}）")
    ver = str(doc.get("version", "1.0"))
    try:
        doc["version"] = f"{int(float(ver)) + 0.1:.1f}"
    except ValueError:
        doc["version"] = "1.1"
    doc.setdefault("changelog", []).append(
        f"v{doc['version']} 术语词典批产（B92）：+{added_total} 词条"
        f"（种子库移植/语料分类/LLM 扩写三源，语料校验防幻觉）")
    errs = loader.validator(doc)
    if errs:
        raise SystemExit(f"validator 未通过：{errs}")
    if args.dry_run:
        print(f"[dry-run] 将新增 {added_total} 词条，version → {doc['version']}")
        return
    yaml.safe_dump(doc, open(path, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print(f"[written] {path} version={doc['version']} 新增 {added_total}")


if __name__ == "__main__":
    main()
