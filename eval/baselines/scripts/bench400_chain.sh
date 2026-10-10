#!/usr/bin/env bash
# 外部基准 400 篇小池 × 测试栈完整评测链（生产零接触）：
#   每 集：入库 → 仅保 mentions/links（其余阶段任务删除，防误触发 LLM 富化）
#         → 排空 → 实体向量 → synq@70 → KNC 三臂(默认/dense/lexical) → bm25 → llamaindex → hipporag@70
#   末尾：全量评分 summary_bench.md
R=/home/inspur/myenv/knowledgenexus
S=$R/eval/baselines/scripts
LOG=/tmp/knc-bl-bench400.log
say() { echo "[bench400 $(date '+%H:%M:%S')] $*" | tee -a $LOG; }

say "0) 测试 ingest-worker 调回 mentions,links"
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
  KNC_ENRICH_STAGES=mentions,links KNC_ENRICH_WORKERS=8
export PYTHONPATH="'"$R"'/libs/knc-core:'"$R"'/libs/engine:'"$R"'/libs/pipeline:'"$R"'/libs/schema"
fuser -k 28111/tcp 2>/dev/null; sleep 1
cd '"$R"'/services/ingest-worker && nohup '"$R"'/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 28111 > /tmp/knc-test-ingest.log 2>&1 &
echo worker-mentions-links'
sleep 15

for ds in hotpotqa 2wikimultihopqa musique; do
  say "[$ds] 1) 入库 400 篇池 → 测试栈"
  $R/.venv/bin/python -u $S/bench_ingest.py --dataset $ds \
      --admin http://127.0.0.1:28100 --ingest http://127.0.0.1:28111 --batch 100 || { say "[$ds] 入库失败"; continue; }
  KB=$($R/.venv/bin/python -c "import json;print(json.load(open('$R/eval/baselines/data/bench/$ds/kb.json'))['kb_id'].lower())")
  say "[$ds] 2) 裁任务：仅留 mentions/links（KB=$KB）"
  $R/.venv/bin/python -c "
import pymysql
c=pymysql.connect(host='127.0.0.1',port=23306,user='root',password='root_test',database='knc',autocommit=True)
cur=c.cursor()
cur.execute(\"DELETE FROM enrich_job WHERE kb_id=UNHEX('$KB') AND stage NOT IN ('mentions','links')\")
print('  裁掉', cur.rowcount, '条')"
  say "[$ds] 3) 等 mentions/links 排空"
  while :; do
    N=$($R/.venv/bin/python -c "
import pymysql
c=pymysql.connect(host='127.0.0.1',port=23306,user='root',password='root_test',database='knc')
cur=c.cursor(); cur.execute(\"SELECT COUNT(*) FROM enrich_job WHERE kb_id=UNHEX('$KB') AND status IN ('PENDING','RUNNING')\")
print(cur.fetchone()[0])" 2>/dev/null || echo 999)
    [ "$N" = "0" ] && break
    sleep 20
  done
  say "[$ds] 4) 实体向量 + synq@70"
  curl -s -m 600 -X POST http://127.0.0.1:28111/api/ingest/entity-vectors/$KB | head -c 120; echo
  $R/.venv/bin/python - "$ds" <<'PYEOF'
import json, sys, urllib.request, concurrent.futures as cf, time
ds = sys.argv[1]
ids = list(json.load(open(f"/home/inspur/myenv/knowledgenexus/eval/baselines/data/bench/{ds}/idmap.json")).values())
def one(d):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:28111/api/ingest/synq/{d}", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1200) as r:
            return 1
    except Exception:
        return 0
ok = done = 0
with cf.ThreadPoolExecutor(70) as ex:
    for res in ex.map(one, ids):
        done += 1; ok += res
print(f"  synq {ok}/{done}")
PYEOF
  say "[$ds] 5) KNC 三臂 + bm25 + llamaindex"
  for arm in "" "dense" "lexical"; do
    $R/.venv/bin/python -u $S/bench_run_knc.py --dataset $ds --api http://127.0.0.1:28121 ${arm:+--channels $arm} || true
  done
  $R/eval/baselines/venvs/lite/bin/python -u $S/run_bm25.py --bench $ds --set x || true
  $R/eval/baselines/venvs/lite/bin/python -u $S/run_llamaindex.py --bench $ds || true
  say "[$ds] 6) HippoRAG（70 并发）"
  $R/eval/baselines/venvs/hipporag/bin/python -u $S/run_hipporag.py --bench $ds --set x || true
done

say "7) 全量评分"
cd $R/eval/baselines && python3 scripts/score.py "runs/*_pool.json" --ref runs/knc_hotpotqa_pool.json --out runs/summary_bench.md > /tmp/knc-bl-bench-score.log 2>&1
say "DONE"
