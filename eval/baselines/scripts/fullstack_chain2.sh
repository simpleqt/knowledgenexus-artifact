#!/usr/bin/env bash
# 完全体断点续链（v2，LLM 并发 70）：synq@70 → topics → KNC 五臂 → HippoRAG-300@70
# → LightRAG-S 限时 → 全量评分。synq 幂等（内容哈希缓存），已完成文档秒过。
R=/home/inspur/myenv/knowledgenexus
S=$R/eval/baselines/scripts
LOG=/tmp/knc-bl-fullchain.log
KB=$($R/.venv/bin/python -c "import json;print(json.load(open('$R/eval/baselines/data/teststack_ne/kb.json'))['kb_id'])")
say() { echo "[chain2 $(date '+%H:%M:%S')] $*" | tee -a $LOG; }

say "3v2) synq doc2query（并发 70）"
$R/.venv/bin/python - "$KB" <<'PYEOF'
import json, sys, urllib.request, concurrent.futures as cf, time
ids = list(json.load(open("/home/inspur/myenv/knowledgenexus/eval/baselines/data/teststack_ne/idmap.json")).values())
def one(d):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:28111/api/ingest/synq/{d}", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return 1
    except Exception:
        return 0
t0, done, ok = time.time(), 0, 0
with cf.ThreadPoolExecutor(70) as ex:
    for res in ex.map(one, ids):
        done += 1; ok += res
        if done % 25 == 0:
            print(f"  synq {done}/{len(ids)} ok={ok} {done/max(1,time.time()-t0)*60:.0f} doc/min", flush=True)
print(f"synq done {ok}/{done}")
PYEOF

say "4) topics 主题树"
curl -s -m 900 -X POST http://127.0.0.1:28111/api/ingest/topics/$KB | head -c 200; echo

say "5) KNC 五臂评测（默认/lexical/dense/graph/synq）× nea+ner"
for arm in "" "lexical" "dense" "graph" "synq"; do
  $R/.venv/bin/python -u $S/teststack_ne.py eval $arm || true
done

say "6) HippoRAG 300 篇（openie 并发 70）"
$R/eval/baselines/venvs/hipporag/bin/python -u $S/run_hipporag.py --slice L --set nea || true
$R/eval/baselines/venvs/hipporag/bin/python -u $S/run_hipporag.py --slice L --set ner || true

say "7) LightRAG Slice-S 限时重试（2h）"
timeout 7200 $R/eval/baselines/venvs/lightrag/bin/python -u $S/run_lightrag.py --slice S --set nea_S || say "   LightRAG 超时/失败——按不可行归档"

say "8) 全量评分"
cd $R/eval/baselines && python3 scripts/score.py "runs/knc*_teststack.json" "runs/bm25_nea_L.json" "runs/bm25_ner_L.json" "runs/llamaindex_nea_L.json" "runs/llamaindex_ner_L.json" "runs/hipporag_nea_L.json" "runs/hipporag_ner_L.json" "runs/lightrag*_nea_S_S.json" --ref runs/knc_nea_teststack.json --out runs/summary_full.md > /tmp/knc-bl-final-score3.log 2>&1
say "DONE"
