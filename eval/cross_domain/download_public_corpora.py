#!/usr/bin/env python3
"""跨领域公开语料下载（2026-10-11 论文实验扩展 E1）。

三域各取 10,000 篇英文摘要（与 KB1 英文刊库同语种，扫描器语义一致）：
  cs  = arXiv cs.CL（计算与语言）
  mat = arXiv cond-mat.mtrl-sci（材料科学）
  bio = PubMed 2025-05..2025-10 窗口（一般生物医学）
产物 eval/baselines/data/public_corpora/{cs,mat,bio}.jsonl：{"doc_id","text"}。
arXiv 礼貌间隔 3s/页（500/页×30 页）；PubMed usehistory+efetch 500/页。
定种子切分（下游统一）：construct 样本 3,000 / 测试集 100（隔离）/
重放样本 10×2,000（全不相交）——由 lexicon_cross_domain.py 消费时现算，
本脚本只落原始语料 + 行序随机化（seed 20261011-domain）。
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "public_corpora")
ARXIV = "http://export.arxiv.org/api/query"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch(url: str, retries: int = 3) -> bytes:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "knc-lexicon-study/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"  retry {i+1}: {e}", flush=True)
            time.sleep(10)


def arxiv_domain(cat: str, n: int, out: list[dict]) -> None:
    start = 0
    while len(out) < n:
        q = urllib.parse.quote(f"cat:{cat}")
        url = (f"{ARXIV}?search_query={q}&sortBy=submittedDate"
               f"&sortOrder=descending&start={start}&max_results=500")
        try:
            root = ET.fromstring(fetch(url))
        except Exception as e:  # noqa: BLE001  arXiv 深分页偶发 500：重试后仍败则收手
            print(f"  [{cat}] page fail at start={start}: {e}; stop with {len(out)}",
                  flush=True)
            break
        entries = root.findall("a:entry", NS)
        if not entries:
            break
        for e in entries:
            aid = e.findtext("a:id", "", NS).rsplit("/", 1)[-1]
            title = " ".join((e.findtext("a:title", "", NS) or "").split())
            abstract = " ".join((e.findtext("a:summary", "", NS) or "").split())
            if title and abstract:
                out.append({"doc_id": f"arxiv:{aid}", "text": title + ". " + abstract})
        start += 500
        print(f"  [{cat}] {len(out)}/{n}", flush=True)
        time.sleep(3)


def pubmed(n: int, out: list[dict]) -> None:
    term = urllib.parse.quote('(2025/05/01[PDAT] : 2025/10/01[PDAT]) AND english[lang]')
    r = ET.fromstring(fetch(f"{EUTILS}esearch.fcgi?db=pubmed&retmax={n}&term={term}&usehistory=y"))
    wm = r.findtext("WebEnv")
    qk = r.findtext("QueryKey")
    total = r.findtext("Count")
    print(f"  [pubmed] count={total}, fetching {n}", flush=True)
    start = 0
    while start < min(n, int(total or n)):
        url = (f"{EUTILS}efetch.fcgi?db=pubmed&rettype=abstract&retmode=xml"
               f"&retstart={start}&retmax=500&query_key={qk}&WebEnv={wm}")
        root = ET.fromstring(fetch(url))
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            ti = art.findtext(".//ArticleTitle") or ""
            abst = " ".join(t.text or "" for t in art.findall(".//Abstract/AbstractText"))
            ti = " ".join(ti.split())
            abst = " ".join(abst.split())
            if pmid and (ti or abst):
                out.append({"doc_id": f"pmid:{pmid}", "text": ti + ". " + abst})
        start += 500
        print(f"  [pubmed] {min(len(out), start)}/{n}", flush=True)
        time.sleep(1)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    jobs = {"cs": lambda o: arxiv_domain("cs.CL", 10000, o),
            "mat": lambda o: arxiv_domain("cond-mat.mtrl-sci", 10000, o),
            "bio": lambda o: pubmed(10000, o)}
    only = sys.argv[1:] or list(jobs)
    for dom in only:
        path = os.path.join(OUT, f"{dom}.jsonl")
        if os.path.exists(path):
            print(f"[{dom}] exists, skip", flush=True)
            continue
        out: list[dict] = []
        print(f"[{dom}] downloading ...", flush=True)
        jobs[dom](out)
        rng = random.Random(f"20261011-{dom}")
        rng.shuffle(out)
        with open(path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{dom}] saved {len(out)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
