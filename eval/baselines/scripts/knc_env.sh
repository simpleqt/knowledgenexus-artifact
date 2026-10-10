#!/usr/bin/env bash
# KNC 进程内评测环境（镜像 deploy/run_services.sh，除 PROFILING/鉴权外不增不减）
R="${R:-/home/inspur/myenv/knowledgenexus}"
cd "$R/deploy" || exit 1   # val()/sval() 相对 deploy/.env（与 run_services.sh 同目录约定）
val()  { grep "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2 || true; }
sval() { grep "^$1=" .env.services 2>/dev/null | head -1 | cut -d= -f2 || true; }
export MYSQL_HOST=127.0.0.1 MYSQL_PORT=13306 MYSQL_DB=knc MYSQL_USER=knc \
  MYSQL_PASSWORD="$(val MYSQL_PASSWORD)" \
  OPENSEARCH_URL=https://127.0.0.1:19200 OPENSEARCH_USER=admin \
  OPENSEARCH_PASSWORD="$(val OPENSEARCH_ADMIN_PASSWORD)" \
  REDIS_URL="redis://:$(val REDIS_PASSWORD)@127.0.0.1:16379/0" \
  S3_ENDPOINT=http://127.0.0.1:19000 \
  S3_ACCESS_KEY="$(val MINIO_ROOT_USER)" S3_SECRET_KEY="$(val MINIO_ROOT_PASSWORD)" \
  S3_BUCKET=knc-raw NEBULA_GRAPHD=127.0.0.1:19669 \
  EMBEDDING_API_BASE="$(sval EMBEDDING_API_BASE)" \
  EMBEDDING_API_KEY="$(sval EMBEDDING_API_KEY)" \
  EMBEDDING_MODEL="$(sval EMBEDDING_MODEL)" \
  RERANK_API_BASE="$(sval RERANK_API_BASE)" \
  RERANK_API_KEY="$(sval RERANK_API_KEY)" \
  RERANK_MODEL="$(sval RERANK_MODEL)" \
  LLM_API_BASE="$(sval LLM_API_BASE)" \
  LLM_API_KEY="$(sval LLM_API_KEY)" \
  LLM_MODEL="$(sval LLM_MODEL)" \
  VL_EMBED_BASE_URL="$(sval VL_EMBED_BASE_URL)" \
  VL_EMBED_MODEL="$(sval VL_EMBED_MODEL)" \
  VL_EMBED_API_KEY="$(sval VL_EMBED_API_KEY)" \
  RETRIEVAL_PROFILE="$(sval RETRIEVAL_PROFILE)" \
  EMBEDDING_CONCURRENCY="$(sval EMBEDDING_CONCURRENCY)"
# 检索票权/门控调优 env（2026-09-30 补齐：此前进程内 harness 缺这些导致基线
# ≠完整生产配置——hit@20/MRR 差 4~6pt；臂间内部对照不受影响，但 O-4 的
# synq 提权等主杠杆必须在场）。空值不导出（保持"未设=旧行为"语义）。
for _k in RRF_SYNQ_BOOST_ASSOC KNC_DOCVEC_ON RRF_DOCVEC_ASSOC_W RRF_DOC_FUSE \
          RRF_DOC_FUSE_EMIT KNC_CHANNEL_DEPTH RRF_LEXICAL_EN_W KNC_LEX_ALIAS_EXPAND \
          RRF_DOCVEC_PRIOR RRF_DOC_CAP KNC_HYDE_DOCVEC \
          EMBEDDING_MRL_DIM EMBEDDING_DIM; do
  _v="$(sval "$_k")"
  [ -n "$_v" ] && export "$_k=$_v"
done
export PYTHONPATH="$R/libs/knc-core:$R/libs/engine:$R/libs/pipeline:$R/libs/schema:$R/services/retrieval-api"
