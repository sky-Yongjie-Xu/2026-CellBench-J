#!/usr/bin/env bash
set -euo pipefail
set -m

datasets=(cortex covid lung liver pbmc12k ms Zheng68k hPancreas)
max_jobs=1

for data in "${datasets[@]}"; do
  # 控制并发数
    while (( $(pgrep -f "bash_cluster.sh" | wc -l) >= max_jobs )); do
      echo "⏳ 当前 zero 任务数已达上限 ($max_jobs)，等待中..."
      sleep 2
    done

  echo "[launch] $data"
  bash bash_cluster.sh "$data" &
done

wait
echo "✅ All few-shot cluster jobs finished"
