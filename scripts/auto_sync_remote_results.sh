#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${1:?"Usage: $0 <remote_host> [remote_base] [local_base] [run_id] [interval_sec]"}
REMOTE_BASE=${2:-/home/coder/backend-block}
LOCAL_BASE=${3:-/home/nosignalx2k/Загрузки/backend-block/results/remote_runs}
RUN_ID=${4:-$(date +%Y%m%d_%H%M%S)}
INTERVAL=${5:-60}
STOP_WHEN_DONE=${STOP_WHEN_DONE:-0}

LOCAL_RUN="${LOCAL_BASE}/${RUN_ID}"
mkdir -p "${LOCAL_RUN}"

targets=(
  "results/chunking_ablation:chunking"
  "results/table_ablation:tables"
  "results/noise:noise"
  "results/extended_sota_experiments:extended_sota"
)

echo "[sync] Remote: ${REMOTE_HOST}:${REMOTE_BASE}"
echo "[sync] Local : ${LOCAL_RUN}"
echo "[sync] Interval: ${INTERVAL}s"

while true; do
  for entry in "${targets[@]}"; do
    remote_rel="${entry%%:*}"
    local_rel="${entry##*:}"
    mkdir -p "${LOCAL_RUN}/${local_rel}"
    rsync -az -e "ssh -o StrictHostKeyChecking=no" \
      "${REMOTE_HOST}:${REMOTE_BASE}/${remote_rel}/" \
      "${LOCAL_RUN}/${local_rel}/" || true
  done

  if [[ "${STOP_WHEN_DONE}" == "1" ]]; then
    if ! ssh -o StrictHostKeyChecking=no "${REMOTE_HOST}" \
      "pgrep -af 'rag_micro.run_experiments|run_extended_sota_experiments|run_review_experiments' >/dev/null"; then
      echo "[sync] Remote jobs not found; stopping."
      break
    fi
  fi

  sleep "${INTERVAL}"
done
