# 本地评测与结果

主程序直接运行，不需要账号、API Key 或跟踪服务：

```bash
python main.py --cfgs configs/genecompass.yaml --task embedding_extraction --dataset_name pbmc12k --gpu 0 --seed 41
```

注释、聚类、批次校正、重建和扰动的最终指标保存在
`output/metrics/<task>/<run_id>.json`。配置中的 `output_dir` 可以改变根目录。
每次启动分配独立运行 ID，记录模型、数据集、seed、topk、阶段、配置和指标。
同一运行的指标更新合并保存，重复实验保留各自文件。
非有限指标保存为 JSON `null`。Embedding 仍由模型保存至原有本地目录。
VCC 指标单独保存在 `output/metrics/perturbation_vcc/`。

原有 `xu_*.yaml` 保留参数组合，使用本地串行入口运行：

```bash
python scripts/run_batch.py xu_frontier_embedding_extraction.yaml --dry-run
python scripts/run_batch.py xu_frontier_embedding_extraction.yaml
python scripts/run_batch.py xu_perturbation_vcc.yaml --limit 1
```

`--dry-run` 只显示命令，`--limit N` 限制组合数。任意任务失败时，批量入口最终返回非零状态。
VCC 参数文件通过 `extra_args: [--vcc-eval]` 调用评估已有预测的流程。

指标统一导出，直接读取 `output/metrics/` 中对应任务的 v2 记录：

```bash
python scripts/export_metrics.py --task clustering
python scripts/export_metrics.py --task annotation --root output
```

输出 CSV 保存在 `output/exports/v2/`，包含分项指标、配对排名和覆盖情况；该命令不生成图表。使用自定义结果目录时，通过 `--root` 指定。历史本地结果和模型权重保留。

全部适用任务使用统一启动入口：

```bash
python main.py --dataset pbmc12k --dry-run
python main.py --dataset pbmc12k --cfgs configs/nicheformer.yaml
```
