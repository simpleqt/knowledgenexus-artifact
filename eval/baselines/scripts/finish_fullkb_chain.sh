#!/bin/bash
# 收尾全库实验链（服务保护版）：
#   金丝雀：每 60s 以防缓存随机查询探测 /api/retrieve，>5s 或失败 → 杀实验进程并退出
#   实验：llamaindex ner 全库（索引已持久化，仅查询）→ bm25 全库 nea/ner（纯本地 CPU）
R=/home/inspur/myenv/knowledgenexus
CANARY_LOG=/tmp/knc-bl-canary.log
MAIN_PID=$$

exp_kill() {
  pkill -P $MAIN_PID 2>/dev/null      # 杀实验子进程
  echo "[canary] $(date '+%F %T') 检索延迟越限/失败，已终止实验子进程" >> $CANARY_LOG
}
canary() {
  while true; do
    Q="健康探测$(date +%s%N | tail -c 8)"
    T=$(curl -s -m 10 -o /dev/null -w "%{time_total}" -X POST http://127.0.0.1:18121/api/retrieve \
        -H "Content-Type: application/json" \
        -d "{\"kb_id\":\"21bcbed8ea6b4a5bbf55f3e3e9988c1f\",\"query\":\"$Q\",\"top_k\":1}")
    OK=$?
    echo "$(date '+%F %T') t=$T rc=$OK" >> $CANARY_LOG
    if [ "$OK" != "0" ] || awk -v t="$T" 'BEGIN{exit !(t>5)}'; then
      exp_kill; exit 9
    fi
    sleep 60
  done
}
canary & CANARY_PID=$!
trap "kill $CANARY_PID 2>/dev/null" EXIT

echo "[chain] $(date '+%T') llamaindex ner 全库（重载索引+39 题）"
$R/eval/baselines/venvs/lite/bin/python -u $R/eval/baselines/scripts/run_llamaindex.py --slice full --set ner
RC1=$?
echo "[chain] llamaindex ner rc=$RC1; bm25 全库 nea"
$R/eval/baselines/venvs/lite/bin/python -u $R/eval/baselines/scripts/run_bm25.py --slice full --set nea
echo "[chain] bm25 全库 ner"
$R/eval/baselines/venvs/lite/bin/python -u $R/eval/baselines/scripts/run_bm25.py --slice full --set ner
RC=$?
echo "[chain] done rc=$RC; 评分"
cd $R/eval/baselines && python3 scripts/score.py "runs/*_full*.json" "runs/llamaindex_*_L.json" "runs/bm25_*_L.json" --out runs/summary_scale.md
echo FULLKB_CHAIN_DONE
