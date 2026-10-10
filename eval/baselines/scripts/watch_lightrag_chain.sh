#!/bin/bash
# 链式看门狗：等 lightrag-S 结束 → 依次跑 lightrag-L(nea/ner) → 全量评分落盘
R=/home/inspur/myenv/knowledgenexus
echo "[watcher] $(date '+%F %T') 等 lightrag-S 结束..."
while pgrep -f "run_lightrag.py --slice S" >/dev/null; do sleep 60; done
echo "[watcher] $(date '+%F %T') lightrag-S 已结束，检查产物："
ls -la $R/eval/baselines/runs/ | grep "lightrag.*_S"
echo "[watcher] $(date '+%F %T') 启动 lightrag-L (nea)"
$R/eval/baselines/venvs/lightrag/bin/python $R/eval/baselines/scripts/run_lightrag.py --slice L --set nea >> /tmp/knc-bl-lightrag-L.log 2>&1
echo "[watcher] $(date '+%F %T') 启动 lightrag-L (ner)"
$R/eval/baselines/venvs/lightrag/bin/python $R/eval/baselines/scripts/run_lightrag.py --slice L --set ner >> /tmp/knc-bl-lightrag-L.log 2>&1
echo "[watcher] $(date '+%F %T') 全量评分"
cd $R/eval/baselines && python3 scripts/score.py "runs/*.json" --ref runs/knc_nea_full-kb.json --out runs/summary.md > /tmp/knc-bl-final-score.log 2>&1
echo "[watcher] $(date '+%F %T') DONE"
