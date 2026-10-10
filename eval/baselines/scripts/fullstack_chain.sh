#!/usr/bin/env bash
# ne 切片"完全体"总链：全量富化 → 实体向量 → synq → topics → KNC 五臂评测
# → HippoRAG-300 → LightRAG-S（限时）→ 全量评分。全程隔离测试栈（28xxx），LLM 串行分速。
R=/home/inspur/myenv/knowledgenexus
S=$R/eval/baselines/scripts
LOG=/tmp/knc-bl-fullchain.log
KB=$($R/.venv/bin/python -c "import json;print(json.load(open('$R/eval/baselines/data/teststack_ne/kb.json'))['kb_id'])")
say() { echo "[chain $(date '+%H:%M:%S')] $*" | tee -a $LOG; }

say "0) 重启测试 ingest-worker：全阶段（mentions,links,relations,graph,media）workers=8"
bash -c '
sval() { grep "^$1=" '"$R"'/deploy/.env.services 2>/dev/null | head -1 | cut -d= -f2 || true; }
export MYSQL_HOST=127.0.0.1 MYSQL_PORT=23306 MYSQL_DB=knc MYSQL_USER=knc MYSQL_PASSWORD=knc_test \
  OPENSEARCH_URL=https://127.0.0.1:29200 OPENSEARCH_USER=admin OPENSEARCH_PASSWORD="KncTest#2026" \
  REDIS_URL="redis://:knc_test@127.0.0.1:26379/0" \
  S3_ENDPOINT=http://127.0.0.1:29000 S3_ACCESS_KEY=knc_test S3_SECRET_KEY=knc_test S3_BUCKET=knc-raw-test \
  EMBEDDING_API_BASE="$(sval EMBEDDING_API_BASE)" EMBEDDING_API_KEY="$(sval EMBEDDING_API_KEY)" \
  EMBEDDING_MODEL="$(sval EMBEDDING_MODEL)" EMBEDDING_CONCURRENCY="$(sval EMBEDDING_CONCURRENCY)" \
  LLM_API_BASE="$(sval LLM_API_BASE)" LLM_API_KEY="$(sval LLM_API_KEY)" LLM_MODEL="$(sval LLM_MODEL)" \
  VL_EMBED_BASE_URL="$(sval VL_EMBED_BASE_URL)" VL_EMBED_MODEL="$(sval VL_EMBED_MODEL)" VL_EMBED_API_KEY="$(sval VL_EMBED_API_KEY)" \
  RETRIEVAL_PROFILE="$(sval RETRIEVAL_PROFILE)" \
  KNC_ENRICH_STAGES=mentions,links,relations,graph,media KNC_ENRICH_WORKERS=8
export PYTHONPATH="'"$R"'/libs/knc-core:'"$R"'/libs/engine:'"$R"'/libs/pipeline:'"$R"'/libs/schema"
fuser -k 28111/tcp 2>/dev/null; sleep 1
cd '"$R"'/services/ingest-worker && nohup '"$R"'/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 28111 > /tmp/knc-test-ingest.log 2>&1 &
echo ingest-worker restarted'
sleep 15

say "1) 富化任务排空（poller 自驱，~6k LLM in relations）"
while :; do
  N=$($R/.venv/bin/python -c "
import pymysql
c=pymysql.connect(host='127.0.0.1',port=23306,user='root',password='root_test',database='knc')
cur=c.cursor(); cur.execute(\"SELECT COUNT(*) FROM enrich_job WHERE kb_id=UNHEX('$KB') AND status IN ('PENDING','RUNNING')\")
print(cur.fetchone()[0])" 2>/dev/null || echo 999)
  [ "$N" = "0" ] && break
  sleep 60
done
say "   富化任务清零"

say "2) 实体向量（entity 通道）"
curl -s -m 600 -X POST http://127.0.0.1:28111/api/ingest/entity-vectors/$KB | head -c 200; echo

say "3) synq doc2query（~12k LLM 调用，并发 3 分速）"
$R/.venv/bin/python - "$KB" <<'PYEOF'
import json, sys, urllib.request, concurrent.futures as cf, time
kb = sys.argv[1]
ids = list(json.load(open("/home/inspur/myenv/knowledgenexus/eval/baselines/data/teststack_ne/idmap.json")).values())
def one(d):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:28111/api/ingest/synq/{d}", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1200) as r:
            return 1
    except Exception:
        return 0
t0, done, ok = time.time(), 0, 0
with cf.ThreadPoolExecutor(3) as ex:
    for res in ex.map(one, ids):
        done += 1; ok += res
        if done % 20 == 0:
            print(f"  synq {done}/{len(ids)} ok={ok} {done/max(1,time.time()-t0)*60:.0f} doc/h", flush=True)
print(f"synq done {ok}/{done}")
PYEOF

say "4) topics 主题树"
curl -s -m 900 -X POST http://127.0.0.1:28111/api/ingest/topics/$KB | head -c 200; echo

say "5) KNC 五臂评测（默认/lexical/dense/graph/synq）× nea+ner"
for arm in "" "lexical" "dense" "graph" "synq"; do
  $R/.venv/bin/python -u $S/teststack_ne.py eval $arm || true
done

say "6) HippoRAG 300 篇切片（索引 + nea/ner 查询）"
$R/eval/baselines/venvs/hipporag/bin/python -u $S/run_hipporag.py --slice L --set nea || true
$R/eval/baselines/venvs/hipporag/bin/python -u $S/run_hipporag.py --slice L --set ner || true

say "7) LightRAG Slice-S 限时重试（2h）"
timeout 7200 $R/eval/baselines/venvs/lightrag/bin/python -u $S/run_lightrag.py --slice S --set nea_S || say "   LightRAG 超时/失败——按不可行归档"

say "8) 全量评分"
cd $R/eval/baselines && python3 scripts/score.py "runs/knc*_teststack.json" "runs/bm25_nea_L.json" "runs/bm25_ner_L.json" "runs/llamaindex_nea_L.json" "runs/llamaindex_ner_L.json" "runs/hipporag_nea_L.json" "runs/hipporag_ner_L.json" "runs/lightrag*_nea_S_S.json" --ref runs/knc_nea_teststack.json --out runs/summary_full.md > /tmp/knc-bl-final-score3.log 2>&1
say "DONE"
