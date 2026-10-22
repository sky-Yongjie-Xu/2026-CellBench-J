# CellBench 代码结构

`bio` 是项目 Python 包名，与现有运行环境和脚本保持一致。代码按职责分层，模型名、YAML 路径、数据/权重目录和命令行参数继续使用原有约定。

```text
bio/
├── cli.py                   # 参数解析和程序启动
├── config.py                # YAML 合并及路径解析
├── data/
│   ├── datasets.py          # 评测数据读取、标签及批次元数据
│   ├── foundation.py        # 预训练模型原始输入、基因匹配和资源检查
│   └── perturbation.py      # 对照匹配、扰动样本与 collate
├── models/
│   ├── registry.py          # 17 个模型的显式按需注册
│   ├── base.py              # 公共实验上下文
│   ├── adapter.py           # 三阶段接口、embedding 提取与缓存调用
│   ├── heads.py             # 分类、重建、扰动任务头
│   ├── loading.py           # 权重完整性检查
│   ├── adapters/            # 各模型实现，统一小写文件名
│   └── compat/              # Tabula、GeneCompass 及 tokenizer 兼容代码
├── tasks/
│   ├── runner.py            # 模型选择与任务分发
│   ├── embedding.py         # 提取、聚类、Harmony 整合流程
│   ├── annotation.py        # 注释训练与预测
│   ├── reconstruction.py   # 重建训练与预测
│   ├── perturbation.py      # 扰动训练、预测及结果保存
│   ├── common.py            # few-shot 抽样
│   └── evaluation.py        # 调用指标函数并记录结果
├── metrics/                 # 按任务划分的 v2 指标；不负责训练
├── io/
│   ├── paths.py             # 项目根目录与统一资源路径
│   ├── embeddings.py        # embedding 指纹、对齐验证、原子写入
│   ├── artifacts.py         # 预测结果的版本化路径
│   ├── results.py           # JSON 指标记录和读取
│   └── export.py            # 分项分数与完整比较块排名
├── assets/                  # 权重与官方源码下载、校验及来源记录
├── diagnostics/             # 资源、环境、适配器推理和协议审查
├── runtime/                 # 批量调度、隔离环境推理、随机种子及 scIB 兼容
├── vendor/                  # 本地第三方依赖副本及配套资源
├── utils/                   # 历史公共工具，配置/种子已转发到新位置
└── modeling/__init__.py     # 旧模型注册命名空间的轻量兼容入口
```

## 调用关系

```text
main.py / python -m bio
  → cli.main
  → config.config_loader
  → tasks.runner.run_model
  → models.registry.get_model
  → 被选择的模型适配器
  → tasks 中的训练/评测流程
  → metrics 中的指标函数
  → io.results / io.export
```

`models.BaseModel` 通过任务 mixin 保留原先的方法调用接口，训练实现实际存放于 `tasks/`。这是对已有适配器 API 的兼容安排，不把模型算法和训练循环重新复制到每个模型中。

模型仍实现三个接口：`load_pretrained`、`prepare_inputs`、`encode_batch`。根据模型词表和资源依赖，真实调用通常先加载模型/词表，再准备输入；六个隔离环境模型仍由子进程完成原生推理。

指标函数不导入模型，也不训练或写结果文件；结果记录与预测缓存不再混在指标模块。轻量 CLI、配置加载、模型列表查询均不导入 Torch 或所有模型依赖。选择 PCA 不会加载 Geneformer、scGPT 或 GEARS 头。

## 使用方式

现有命令继续有效：

```bash
python main.py --cfgs configs/scgpt.yaml --task clustering --dataset_name pbmc12k
```

也可以使用：

```bash
python -m bio --cfgs configs/scgpt.yaml --task clustering --dataset_name pbmc12k
python -m bio --help
```

省略 `--task` / `--dataset_name` 时使用 YAML 值，不再被旧硬编码默认值覆盖。显式 `--gpu` 在导入 Torch 前设置设备可见性；省略时沿用环境设置。`--log_to_file`、`--iter` 作为历史命令兼容参数保留，本次没有实现此前缺少的文件日志或恢复逻辑。

新代码导入：

```python
from bio.config import config_loader
from bio.models import get_model, list_models
from bio.metrics import annotation, reconstruction
from bio.io.results import log_metrics

model_class = get_model("Geneformer")
```

`from bio import modeling; modeling.Geneformer` 的旧注册入口仍可用。旧的内部文件路径（如 `bio.modeling.Geneformer`、`bio.evaluation`、`bio.local_results`）已迁移，项目内引用均已更新；外部自写脚本应使用新路径。两个历史实验脚本依赖的树聚类工具保留在 `bio.metrics.hierarchical`。

## 扩展规范

- 新增模型：在 `models/adapters/` 增加小写模块，在 `registry.py` 显式登记，配置与协议说明分别放在 `configs/` 和 `resources/model_protocols.json`。
- 新增任务：流程放在 `tasks/`，纯指标放在 `metrics/`，在 `tasks/runner.py` 注册入口。
- 修改路径/缓存：只在 `io/` 实现，避免模型内硬编码机器路径。
- 上游代码保持在 `public_code/`；实际需要的本地移植依赖在 `vendor/`，兼容逻辑在 `models/compat/` 或 `runtime/`。
- 下载、检查和批处理实现分别放在 `assets/`、`diagnostics/`、`runtime/`；`scripts/` 仅保留命令入口及环境安装 shell 脚本。

旧 `BaseModel` 中重复的 embedding 缓存、未使用的占位训练接口和“模型失败后退回原始表达”的兜底已移除，所有注册模型继续使用统一 `FoundationAdapter`。

目录变化会改变源码指纹，旧 embedding / 预测文件保留但不会自动命中新缓存。此次重构不改变 v2 指标公式，也不代表完成全数据训练或复现论文数值。

## 外部脚本归入包

| 原入口（继续可用） | 实现位置 |
|---|---|
| `scripts/download_weights.py` | `bio/assets/weights.py` |
| `scripts/download_frontier_weights.py` | `bio/assets/frontier.py` |
| `scripts/download_public_code.py` | `bio/assets/sources.py` |
| `scripts/check_*.py` | `bio/diagnostics/` 下对应检查模块 |
| `scripts/audit_model_adapters.py` | `bio/diagnostics/protocols.py` |
| `scripts/run_batch.py` | `bio/runtime/batch.py` |
| `scripts/extract_frontier_embeddings.py` | `bio/runtime/extract.py` |

模块也支持 `python -m bio.runtime.batch …` 等调用。包内代码通过 `bio.io.paths.PROJECT_ROOT` 定位资源，不依赖脚本所在层级。模型后端保持按需加载，继续使用各自的 Python 环境。

`datasets/`、`weights/`、`output/` 是数据和运行产物；`configs/`、`resources/` 是配置和清单；`public_code/` 是官方源码；它们不归入 Python 实现包。`cell_eval/` 保持独立导入命名空间，历史实验与绘图脚本也保留在外部，避免把旧评测流程作为正式接口。

## 一个入口运行数据集的全部适用任务

```bash
# 先查看计划，不加载模型、不训练
python main.py --dataset pbmc12k --dry-run
# 默认全部 17 个模型，每个模型顺序执行 embedding、聚类、注释、重建
python main.py --dataset pbmc12k
# 选择模型及 GPU
python main.py --dataset uc_epi --cfgs configs/scgpt.yaml --gpu 0
# 老参数也可以显式选择全部任务
python main.py --dataset_name norman --task all --cfgs configs/geneformer.yaml
# 只运行一个任务，原用法继续有效
python main.py --dataset pbmc12k --task clustering --cfgs configs/pca.yaml
```

`--dataset` 默认启动全部适用任务；`--dataset_name` 保留原来的单任务行为，可配合 `--task all`。全部任务模式默认遍历注册表中的 17 个模型（14 个基础模型、PCA/UMAP/scVI 三个基线），每个模型使用自己的 YAML；指定 `--cfgs` 则只运行这个配置。一次使用一个 seed 和一个 topk，默认分别为 41 和 1，可用 `--seed`、`--topk` 修改。单任务模式省略配置时仍保留原默认配置。

任务计划在 `bio/runtime/suite.py` 集中定义：普通数据运行 embedding、聚类、注释、重建；已有批次映射的数据增加 correction；扰动数据运行 embedding、perturbation。批次映射可由 `data_cfg.dataset_batch_keys` 补充。这里只安排数据集协议允许的任务，实际数据文件、标签、批次数量和权重仍由各任务运行时检查。

VCC 仅在扰动数据配置显式声明 `evaluation_expression_scale: log1p_normalized` 时加入，且依赖本次 perturbation 成功。必须先核实表达尺度，不能仅为启用任务而填写这一声明。不适用任务及原因会打印出来。

任务使用顺序子进程，完成后释放该进程的模型与显存。每个任务的详细输出放在 `output/suites/<run_id>/<model>/<task>.log`，状态与命令记录在 `output/suites/<run_id>/summary.json`。默认独立任务失败后继续执行剩余任务；某模型 embedding 失败后跳过它的下游任务，继续下一个模型。模型缺少权重或运行环境也会明确记录失败，最终以非零退出码报告失败；添加 `--fail-fast` 可在首次失败后停止。旧 `xu_*.yaml` 仍可用于多模型、多 seed 或多 shot 参数网格，但运行单个数据集已不需要逐个调用这些文件。

已删除无调用方的旧后端/兼容转发文件及重复导出入口。六模型后端直接位于 `bio/runtime/frontier_backends.py`，GeneCompass 兼容实现位于 `bio/models/compat/genecompass.py`；指标导出统一使用 `scripts/export_metrics.py --task <task>`。

对于 `pbmc12k`，默认计划共 17 × 4 = 68 个模型任务步骤。模型按注册顺序串行执行，不同时加载全部模型，也不自动下载缺失权重或安装环境。`--fail-fast` 会在任意模型首次失败后停止整个计划。embedding 缓存仍按数据集和模型分目录保存。
