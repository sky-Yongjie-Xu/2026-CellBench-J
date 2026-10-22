#!/usr/bin/env bash
set -euo pipefail

# datasets=(pbmc12k ms Zheng68k hPancreas)
datasets=(cortex covid lung liver pbmc12k ms Zheng68k hPancreas)
seeds=(42 43 44 45 46)
topks=(1 3 5 9)
max_jobs=1

for data in "${datasets[@]}"; do
  for topk in "${topks[@]}"; do
    for seed in "${seeds[@]}"; do

      # 等待直到运行中任务数 < max_jobs
      while (( $(pgrep -f "bash_annotation.sh" | wc -l) >= max_jobs )); do
        echo "⏳ 当前 zero 任务数已达上限 ($max_jobs)，等待中..."
        sleep 2
      done

      echo "[launch] $data seed=$seed topk=$topk"
      bash bash_annotation.sh "$data" "$seed" "$topk" &
    done
  done
done

wait
echo "✅ All few-shot annotation jobs finished"
