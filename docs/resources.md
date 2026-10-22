# 数据与模型资源路径

所有相对路径均以 CellBench 项目根目录为基准，不依赖启动命令所在目录。

## 数据集

配置中的 `data_cfg.root_path` 已统一为 `datasets`。现有文件可直接保持平铺结构：

| dataset_name | 文件 |
| --- | --- |
| pbmc12k | datasets/pbmc_12k.h5ad |
| hPancreas | datasets/human_pancreas_norm_complexBatch.h5ad |
| cortex | datasets/Perirhinal_Cortex.h5ad |
| ms | datasets/filtered_ms_adata.h5ad |
| Zheng68k | datasets/Zheng68K.h5ad |
| covid | datasets/covid.h5ad |
| immune | datasets/immune.h5ad |
| adamson | datasets/adamson.h5ad |
| norman | datasets/norman.h5ad |
| uc_epi | datasets/ucepi.h5ad + uc_epi_*.tsv |
| uc_imm | datasets/ucimm.h5ad + uc_imm_*.tsv |

UC 数据需要 cell、celltype、gene、batch_health、batch_location、batch_patient 六个 TSV 文件。
现有 UC h5ad 使用从 0 开始的数字索引，因此按 TSV 顺序补齐真实细胞 ID 和基因名；
代码校验所有文件长度，并比对 h5ad 已有的标签和批次字段。已有真实细胞 ID 时则按 ID 对齐。
若元数据与表达矩阵不匹配，程序报错，不会静默覆盖。

UC 默认以患者字段 `batch_p` 作为 `batch`；可在配置中切换为 `batch_h`（健康状态）或
`batch_l`（位置）：

```yaml
data_cfg:
  root_path: datasets
  dataset_batch_keys:
    uc_epi: batch_p
    uc_imm: batch_p
```

`ms` 在现有目录下只读取 `filtered_ms_adata.h5ad`，不再要求另一个 `c_data.h5ad`。
`hPancreas` 读取现有完整胰腺文件，不再要求 `demo_train.h5ad` 和 `demo_test.h5ad`。
缺少平铺文件时，相关读取函数仍可回退至旧版子目录布局。
当前尚未提供 liver、lung、rgc、celegan 和 Replogle 文件。

## 模型权重

| 模型 | 本地目录 | 下载来源/版本 |
| --- | --- | --- |
| CellPLM | weights/CellPLM | [官方仓库](https://github.com/OmicsML/CellPLM)，20230926_85M |
| Geneformer | weights/Geneformer | [官方 Hugging Face](https://huggingface.co/ctheodoris/Geneformer)，固定旧版 revision 31bf641d0f173a18a5e06d9a6574bb0d3fe5edc1 的 gf-12L-30M-i2048 |
| LangCell | weights/LangCell | [官方仓库](https://github.com/PharMolix/LangCell)链接的 cell_bert 和 cell_proj；配套 30M 词表/中位数 |
| scFoundation | weights/scFoundation | [官方仓库](https://github.com/biomap-research/scFoundation)链接的 SharePoint models.ckpt，配套 19264 基因索引 |
| scGPT | weights/scGPT/human | [官方仓库](https://github.com/bowang-lab/scGPT)链接的 whole-human 权重、args.json、vocab.json |
| scMulan | weights/scMulan | [官方仓库](https://github.com/SuperBianC/scMulan)链接的清华云 ckpt_scMulan.pt、meta_info.pt 和基因映射资源 |
| Nicheformer | weights/Nicheformer | [官方 Hugging Face](https://huggingface.co/theislab/Nicheformer)，固定 revision 0ba4ba162f9ad3d6c8b713761954c31be2fa69ed；附技术均值文件 |
| Tabula | weights/Tabula | [官方仓库](https://github.com/aristoteleo/tabula)链接的 pancreas.pth、词表、配置和 source/ 源码 |

Geneformer 与 LangCell 使用各自目录下的 30M tokenizer 资源，避免误用 95M/104M 词表。
PCA、UMAP、scVI 基线不需要下载上述基础模型的预训练权重。

## 下载与检查

下载器只需要 `requests`、`gdown` 和 Git，不需要 PyTorch。建议在自己的环境中安装：

```bash
pip install requests gdown
python scripts/download_weights.py
python scripts/download_weights.py --models scMulan
python scripts/check_resources.py
```

已有文件会复用，未完成的 HTTP 下载保留 `.part` 以便重试。失败文件不会伪装成成功文件。
下载来源、文件大小、SHA-256 和失败原因记录在 `weights/download_manifest.json`。
Tabula 源码版本记录在 `weights/source_revisions.json`。
CellPLM 的 Dropbox 压缩包会保留，所需 checkpoint 已单独解压。

2026-10-21 下载结果：七个模型的必需资源已下载；scMulan 的主权重链接出现 TLS 连接错误，
尚缺 `weights/scMulan/ckpt_scMulan.pt`。其余 scMulan 元数据和基因映射文件已下载。
可用上述 `--models scMulan` 命令重试，或从作者链接下载主权重后放到该位置。

安装 `h5py` 后，还可检查现有数据及 UC TSV 的行列数：

```bash
python scripts/check_resources.py --datasets --output weights/resource_report.json
```

上述检查确认资源完整性及元数据结构，不代表模型已经成功加载或训练；运行模型仍需安装
各模型对应的依赖。`cellbench` 环境配置见 [environment.md](environment.md)；
`main.py` 已改为默认执行模型任务，VCC 评估通过 `--vcc-eval` 显式选择。

## 标准化公开数据补充

公开源文件的固定下载地址、版本、官方大小和可用 MD5 记录于 `resources/dataset_assets.json`；下载回执同时记录 SHA-256。原文件保存在 `datasets/source_files/`，评测入口读取 `datasets/<数据集名>.h5ad`。

```bash
python -m bio.assets.datasets
python -m bio.assets.prepare_datasets --ids bmmc hlca heart kidney seaad_mtg replogleK562 replogleRPE1 --limit 30000 --seed 41
python main.py --dataset bmmc --dry-run
python main.py --dataset hlca --cfgs configs/pca.yaml
```

新图谱默认生成至多 30,000 个细胞的固定计算基准子集。以供体×细胞类型分层，分配等额配额并重新分配未用配额；扰动数据按细胞类型×条件分层。保留来源提供的全部 RNA 基因，绝不将标准化值四舍五入作为计数。BMMC 的蛋白特征明确排除。每份文件旁的 `.provenance.json` 记录来源、计数层、抽样方法、原始与保留细胞数，以及细胞 ID 指纹；原始文件保留，后续可以重建更大的子集。

这些子集是固定分层抽样的计算基准，不代表原始自然细胞组成，不能据此估计自然群体细胞比例。公开的完整源文件也不意味着已确认与模型预训练语料独立。按供体留出、按研究留出和未见扰动划分仍需在后续评测协议中明确；当前训练任务不会因为下载了多供体数据就自动切换划分策略。

新数据角色：BMMC（骨髓、多站点）、HLCA core（肺、多研究）、Heart（心脏、多供体）、Kidney（肾脏单细胞、多来源）、SEA-AD MTG（单脑区、多供体）、Replogle K562/RPE1（遗传扰动）。Adamson/Norman 的可靠来源版本另以 `adamson_full` / `norman_full` 下载 ID 管理，准备后替换 `adamson.h5ad` / `norman.h5ad`；`full` 指源下载标识，评测文件仍是显式抽样子集。

Replogle 的两个子集各保留 3,000 个对照细胞，其余 27,000 个按扰动条件分层。Adamson 使用来源明确标注的三个阴性对照构建 `ctrl`，随后去除条件中的 guide 后缀；原始 `perturbation` 字段保留。BMMC、Replogle 等来源本身可能已筛选基因，保留全部来源基因不等于获得完整转录组。

清理与保留决策见 `resources/dataset_selection.json`。旧 `covid.h5ad` 因供体缺失和类型/批次混杂，在 HLCA 验证后删除，记录于 `datasets/covid.retired.json`。Adamson/Norman 的旧版在新版验证后原位替换，替换记录为对应的 `.replacement.json`。PBMC12k 用于调试，cortex 和 UC 用于补充分析。hPancreas、immune、MS、Zheng68K 暂保留，但原始计数来源问题尚未解决，不能据此声称这些旧数据已经适配所有要求原始计数的模型。
