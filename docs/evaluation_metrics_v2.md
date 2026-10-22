# 评估指标协议 v2

本次修改规范化指标及其直接依赖的预测路径。旧论文结果不能改名后作为 v2 结果；需要重新计算，扰动头与损失变化还需要重新训练。没有重新完成全部模型与数据集的评测。

## 五个任务

| 任务 | 主指标与辅助指标 | 定义 |
|---|---|---|
| 聚类 | ARI、NMI、cell-type ASW；辅助 ACC、AvgCC | 默认固定 Louvain resolution=1，kNN=15；ASW=(s+1)/2；AvgCC 仅作描述 |
| 注释 | Macro-F1、Accuracy；Precision、Recall、Balanced Accuracy、各类别结果；有概率时 AUROC/AP | 真值在前、预测在后；概率列对应明确类别；Macro 分数按测试集有支持的类别计算，记录缺失类别；AUPRC 使用 OVR average precision |
| 批次整合 | ARI、NMI、cell-type ASW、Batch ASW、scaled iLISI、kBET；辅助 scaled cLISI、图连通性 | 调用 scIB；Batch ASW 在 cell type 内计算 1-abs(s)；LISI 使用官方图算法并缩放到 [0,1] |
| 重建 | MSE、Pearson；辅助 MAE、Spearman、逐基因相关性、Top-K 表达重合 | 逐细胞与逐基因的相关性分开记录；常量向量不定义相关性，记录有效数量；Top-K 与 few-shot 样本数解耦 |
| 扰动 | 条件×细胞类型级 MAE/MSE、delta Pearson/Spearman、对照基线；VCC 单独报告 PDS/DES/MAE | 每组先平均，再对组等权；delta 是评价表达空间中的差，不再误称 LFC；方向只在真实非零效应基因上计算 |

不再计算 `(Pearson-MSE)/2`、`(PDS+DES-MAE)/3`，也不保留把错误定义改名继续输出的旧 LFC、全局 batch ASW 或负值 1-cLISI 字段。`AvgBio` 改为明确名称 `AvgCC`。重建的 Sign_agree 删除。

## 配置

`configs/default.yaml` 中 `evaluation_cfg` 控制：

- `clustering_mode: fixed`；只有显式设为 `oracle_nmi` 才允许参考标签优化分辨率。结果记录该模式，不能混合排名。
- `clustering_resolution: 1.0`、`n_neighbors: 15`。
- `correction_mode: harmony`：现有共同 PCA 后执行 Harmony；设为 `native` 时跳过 Harmony，但仍有共同 PCA 步骤。
- `compute_kbet: true`：尝试 scIB 的 R kBET；缺少依赖时记 null 并保存原因。`strict_optional_metrics: true` 可改为直接报错。
- `reconstruction_top_k: 50` 与 `perturbation_top_k: 50` 独立于 few-shot `topk`。
- `min_effect: 0.0`：扰动方向阈值，单位为评价表达空间；正式实验应预先指定。
- `vcc_threads: 1`：VCC 差异分析并行数。

VCC 的 DES 需要已验证的 log1p 归一化非负表达；在对应数据配置中明确设置：

```yaml
data_cfg:
  evaluation_expression_scale: log1p_normalized
```

不能仅凭数值范围推断后填写该字段；输入必须实际符合这一声明。未声明时 VCC 报错，其他误差仍以 `dataset_defined` 标记。PDS 是 `cell_eval.discrimination_score_l1`，DES 是 `cell_eval.overlap_at_N`，不是 Pearson delta。按细胞类型分别执行，每种类型至少需要两个扰动候选和对照；全部条件均纳入，不再按字母顺序截取前 300 个。

## 训练与结果对应关系

- 注释的训练/测试划分按类别分层；保存概率用于 OVR 指标。
- 聚类默认不使用测试标签选择分辨率。
- 重建在训练细胞上选择 400 个候选空间内 HVG。上游数据读取仍可能先在全数据选择候选 HVG；这不是完整的归纳式预处理改造。
- 扰动按条件分层划分训练/验证/测试；各集合共享同一个条件 ID 字典。仍是已见条件任务，不代表未见扰动泛化。
- 扰动残差输出允许正负值，最终表达截断到非负；去掉对可能已经取对数的表达再次计算“LFC”的损失。
- 重建/扰动最佳权重使用深拷贝，避免 `state_dict` 随后续训练改变。
- 扰动基础指标使用同细胞类型观测对照群体均值；VCC 文件包含相应完整对照群体，不只随机匹配的对照子集。模型训练仍以匹配对照细胞为输入。
- 若实验存在 donor/batch 混杂，仍应提供匹配批次的对照设计；此次没有自动把 donor/batch 留出加入训练划分。

## 缓存、缺失值与排名

JSON 记录含 `metric_version: 2`。默认读取/导出只选择 v2；旧结果保留，可显式 `load_runs(..., metric_version=1)` 查看。扰动预测和对照映射转入 `output/perturbation_results/v2/<signature>/`，签名包括输入、权重、代码、seed、shot 数与评估配置。

无法定义的相关性及不可用的可选指标是 null，并记录有效数量或原因；NaN/Inf 输入直接报错。不可用值不补零。

```bash
python scripts/export_metrics.py --task reconstruction
python scripts/export_metrics.py --task correction
python scripts/export_metrics.py --task perturbation_vcc
```

使用统一入口 `python scripts/export_metrics.py --task <task>` 导出。输出在 `output/exports/v2/`：分项分数、平均排名、完整比较块清单。不同 shot、phase、评估配置分开。只在方法与主指标齐全的 dataset×seed 块内排名，先平均同一数据集的 seed，再平均数据集与指标。缺失块明确列出；没有完整块就不生成数值排名。方法集合是该协议在结果中出现的方法，不自动声称涵盖所有已接入模型。

旧 `y_code/` 图表为历史 v1 结果脚本，不读取新的输出位置；本次未重绘论文图，也不把旧排名转成 v2 排名。

## 依赖与验证边界

本机 scIB 1.1.7 附带 LISI 程序要求较新 GLIBC；`bio/runtime/scib.py` 使用包内原始 C++ 源码在 `output/runtime_cache/scib/` 编译，临时重定向程序查找位置，未修改安装包或 `public_code`。标准 iLISI/cLISI 已实际运行通过。

本机缺少 R/rpy2/kBET：kBET 目前不可用，日志保存明确原因；包含 kBET 的完整批次综合排名不会静默降级为其他指标的平均。

单元测试覆盖公式、常量/缺失类别、概率列、条件等权、下调预测、官方 Batch ASW、缺依赖状态、版本隔离、尺度不变排名。另用合成数据实跑 scIB 与 cell_eval：完美预测得到 PDS=1、DES=1、MAE=0。测试不替代全数据 benchmark。
