# 新增六个通用单细胞模型

已增加六个冻结嵌入适配器，模型 ID 分别为 `UCE`、`scPRINT2`、
`TranscriptFormer`、`CellFM`、`GeneCompass`、`scLong`。
它们通过独立 Python 进程调用作者模型，嵌入返回 `cellbench` 后，复用现有聚类、
Harmony 批次校正、注释分类头和表达重建评测。全量微调尚未实现，调用时会明确报错。

| 配置 | 官方代码 | 本次接入的检查点 | 默认后端 |
| --- | --- | --- | --- |
| `configs/uce.yaml` | https://github.com/snap-stanford/UCE | 4 层 UCE | 主 cellbench，自动选择 GPU |
| `configs/scprint2.yaml` | https://github.com/cantinilab/scPRINT-2 | `small-v2.ckpt` | 独立 scprint2 环境 |
| `configs/transcriptformer.yaml` | https://github.com/czi-ai/transcriptformer | TF-Sapiens | 独立 Python 3.11，CPU |
| `configs/cellfm.yaml` | https://github.com/biomed-AI/CellFM | `base_weight.ckpt` | 独立 MindSpore，CPU |
| `configs/genecompass.yaml` | https://github.com/xCompass-AI/GeneCompass | GeneCompass-Small，6 层 | 独立 Transformers 4.30 |
| `configs/sclong.yaml` | https://github.com/BaiDing1234/scLong | 官方 1B 检查点 | 主 cellbench，CPU |

具体源码 commit 固定在 `scripts/download_public_code.py`，本机来源记录在
`public_code/sources.json`。模型名称与检查点版本不能任意替换；适配器会拒绝不支持的版本。

## 资源下载

```bash
conda activate cellbench
python scripts/download_public_code.py --models UCE scPRINT2 TranscriptFormer CellFM GeneCompass scLong
python scripts/download_frontier_weights.py
python scripts/check_resources.py --output output/resources_check.json
```

权重、词表和生物先验都保存到 `weights/<模型ID>/`。
下载器检查空响应、HTML、Git LFS 指针、HTTP 长度，并记录 SHA-256；支持 `.part` 断点续传。
压缩包解压拒绝路径穿越和符号链接。不会下载作者的预训练细胞数据集。
`weights/frontier*_manifest.json` 保存各批次结果，失败记录不代表检查点可用。

GeneCompass-Small 检查点已由用户补齐，MD5 为
`9763fa1e30ee37fe5ee637fe056f452b`，与官方下载页面一致。
该检查点已包含四类先验矩阵与同源索引，冻结推理直接使用其中的 buffer，
无需重新下载缺失的外部先验 pickle（重建先验或训练仍可能需要）。
当前官方 main 未连接配置中的表达值编码器，项目通过
`bio/models/compat/genecompass.py` 恢复该变体：复用官方 ContinuousValueEncoder
的 Linear→ReLU→Linear→LayerNorm 运算，并修正其强制 CUDA 调用；
按 `use_value_emb=true` 将拼接层恢复为 `[256, 1536]`。
按 `concat_values_first=false` 将值编码放到四类先验之后；顺序为
基因、启动子、共表达、基因家族、PECA、表达值。此顺序是依据保存配置恢复的，
未取得作者原训练实现或参考输出，尚未验证逐值等价，正式研究应披露该兼容实现。
检查点全部 144 个状态项名称与形状匹配，以 strict=True 加载；
两个真实 PBMC 细胞的 CPU 推理成功，输出 `2 × 256`，细胞对齐与有限数检查通过。
不需要为此改换 Base。严格加载不等同于论文结果复现。
详细实测见 `output/genecompass_checkpoint_check.json`。
来源入口为 https://www.scidb.cn/en/anonymous/SUZOdk1y 和
https://github.com/xCompass-AI/GeneCompass/blob/main/genecompass/knowledge_embeddings.py 。

## 独立环境

```bash
bash scripts/setup_frontier_environments.sh genecompass
bash scripts/setup_frontier_environments.sh transcriptformer
bash scripts/setup_frontier_environments.sh scprint2
bash scripts/setup_frontier_environments.sh cellfm
```

新增环境通过项目 `runtime_envs/` 访问，不升级原来的 `cellbench`。
安装脚本默认将实体放到 Linux 用户目录 `~/.cache/cellbench/runtime_envs/`，
项目内使用软链接，避免 WSL Windows 挂载盘的小文件安装开销。
可通过 `CELLBENCH_ENV_HOME` 更改环境实体位置；已有项目内环境会保留。
UCE/scLong 复用现有环境。其他环境通过配置中的 `backend_python` 指定；也可设置
`CELLBENCH_UCE_PYTHON`、`CELLBENCH_SCPRINT2_PYTHON`、
`CELLBENCH_TRANSCRIPTFORMER_PYTHON`、`CELLBENCH_CELLFM_PYTHON`、
`CELLBENCH_GENECOMPASS_PYTHON`、`CELLBENCH_SCLONG_PYTHON` 覆盖解释器路径。

TranscriptFormer 使用 PyTorch 2.5.1；scPRINT-2 使用 PyTorch 2.8.0。
安装脚本默认使用官方 CPU wheel，便于在当前机器验证。
有合适显卡时可设置 `CELLBENCH_TORCH_INDEX` 到对应的官方 CUDA wheel 索引，
并修改 `backend_device`。不要把两个新版 PyTorch 安装到原来的主环境。

TF-Sapiens 默认 `max_length: 1023`，通过官方配置覆盖接口限制输入上下文，
以降低当前机器的 CPU 推理内存。网络和权重保持原样；此设置可能截断较长细胞序列，
必须随结果记录。在更大内存机器上可设为预训练上限 `2047`。

CellFM 官方参考依赖是 MindSpore 2.2.10，其历史安装包节点本次 TLS 失败。
脚本提供可获取的 MindSpore 2.6.0 CPU 环境用于兼容性检查；这不代表与论文原环境已验证等价。
可用 `CELLBENCH_MINDSPORE_SPEC` 指定已有的官方 2.2.10 wheel 路径或 URL。
当前机器只有约 11 GiB 主机内存；9.66 GB CellFM 检查点的解析存在内存不足风险，
后端会在加载前进行保守内存检查。请在更大内存机器执行完整推理。

本适配器按检查点内的基因表对齐 scPRINT-2 输入，使用官方 Embedder 和 Collator，
冻结嵌入推理不需要下载完整 ontology 数据库。Lamin 设置与缓存隔离在
`weights/scPRINT2/ontology_settings` 和 `ontology_cache`。
如果要使用官方其他任务、分类指标或训练流程，需要按官方说明初始化相应 ontology 数据：
https://github.com/cantinilab/scPRINT-2#to-know-about-laminai 。

## 运行与预处理

```bash
python main.py --cfgs configs/uce.yaml \
  --task embedding_extraction --dataset_name pbmc12k --gpu 0 --seed 41
python main.py --cfgs configs/sclong.yaml \
  --task clustering --dataset_name pbmc12k --gpu 0 --seed 41
python scripts/check_frontier_models.py
```

`xu_frontier_embedding_extraction.yaml` 是六个新配置的本地批量参数文件，从本地已有的 PBMC 数据开始。
资源未齐的模型会明确报错，不会回退到随机参数或 PCA。

新增模型的输入从原始 h5ad 重新读取完整基因，避免现有数据读取器提前做 HVG/归一化。
默认按 `layers['counts']`、`.raw.X`、`.X` 的顺序查找原始表达，随后检查非负、有限和整数。
可设置 `expression_layer: counts`、`raw`、`X` 或其他 layer 名，
也可通过 `data_cfg.foundation_input_path` 指向真正的原始 counts h5ad。
归一化数据不会通过取整伪装成 counts。

scLong 按官方要求使用 normalize-total 1e4 + log1p；如果输入本身已经是该形式，
需显式设置 `input_is_log1p: true`。Ensembl/符号转换优先使用文件内的标识，
并使用固定版本的 CellFM/HGNC 表补充人类映射。相同基因 ID 合并计数。
当前六份配置只声明支持人类数据，跨物种扩展需要另外验证。

预处理后的 h5ad 不向模型提供真实细胞类型或批次标签。返回的细胞 ID 必须与输入完全对应，
允许重排，但拒绝增加、丢失、重复细胞及非有限嵌入。缓存包含输入、权重、源码版本、
运行代码、配置和种子的签名。推理参数或资源变更会使缓存失效。

UCE 使用官方分词及网络，采用 mmap/参数赋值加载以减少重复 ESM 表的主机内存占用。
GeneCompass 使用最后一层非 padding 基因的平均表示；CellFM 使用 CLS。
scLong 原代码输出每基因表示，本适配器对输入中匹配的基因 decoder 表示取均值，排除 PAD。
scPRINT-2 默认 `use_knn: false`，以独立细胞输入参加统一评测；检查点的网络结构保持不变。
按检查点完整基因表对齐后，以 `inference_chunk_size: 256` 分块调用官方 Embedder，
限制其 SimpleAnnDataset 的稠密矩阵内存占用。

新增六个模型使用原始全基因输入，旧适配器中部分使用 HVG 输入。
正式汇总模型性能时需统一输入政策，或明确报告各模型的原生预处理协议。
本次小样本推理检查用于验证接入，不构成性能排名或完整 benchmark。

## 验证记录

`output/frontier_status.json` 分别报告源码、资源、依赖导入和实际推理状态。
`output/frontier_smoke/` 保存两个真实 PBMC 细胞的输入及实际输出。
目前 UCE 得到 `2 × 1280`、scPRINT-2 得到 `2 × 424`、
TF-Sapiens（1023 上下文）得到 `2 × 2048`、scLong 得到 `2 × 200`，
均通过细胞 ID 和有限数检查。CellFM 未进行完整推理；GeneCompass 使用上述兼容实现得到 `2 × 256`，严格加载与输入输出检查通过。
依赖版本快照位于 `resources/frontier_environments/`，TF/scPRINT-2 提供 CPU 依赖锁定文件。

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_frontier_contract.py' -v
```

六项测试覆盖返回细胞重排、丢失/重复/非有限值拒绝、完整 counts 读取、重复基因合并、
源数据与评测顺序对齐，以及拒绝把归一化表达当作 counts。
