#!/usr/bin/env bash
set -euo pipefail
set -m          # <── 关键：显式开启 Job Control
trap 'echo -e "\n[!] Ctrl-C, killing all…"; pkill -P $$' INT TERM

datasets=(norman adamson)
# datasets=(norman)
seeds=(42 43)
# seeds=42
# topks=(100 500 1000 5000 10000)
topks=(6 10)
# topks=(5000)
max_jobs=1

for data in "${datasets[@]}"; do
  for topk in "${topks[@]}"; do
    for seed in "${seeds[@]}"; do
      while (( $(pgrep -f "bash_pertub.sh" | wc -l) >= max_jobs )); do
        echo "⏳ 当前 zero 任务数已达上限 ($max_jobs)，等待中..."
        sleep 2
      done

      echo "[launch] $data seed$seed topk$topk" | tee -a "run.txt"  
      bash bash_pertub.sh "$data" "$seed" "$topk" &
    done
  done
done

wait
echo "✅ All reconstruct jobs finished"
