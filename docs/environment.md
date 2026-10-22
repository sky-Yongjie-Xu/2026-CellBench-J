# cellbench 运行环境

环境位置：`/root/anaconda3/envs/cellbench`。

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate cellbench
cd /mnt/e/CellBench
```

本机使用 Python 3.10、PyTorch 2.2.0 + CUDA 11.8，并匹配 torchvision 0.17.0、
torchaudio 2.2.0 和 torchtext 0.17.0。GPU 是 RTX 2070（8GB）。
核心依赖版本记录在 `requirements-cellbench.txt`，基础 Conda 配置为 `environment-cellbench.yml`。
完整 pip 版本锁定清单为 `requirements-cellbench.lock.txt`，安装脚本优先使用该清单；
Conda 原生包的 Linux 平台锁定清单为 `environment-cellbench.explicit.txt`。

## 重建与检查

```bash
bash scripts/setup_cellbench.sh
conda run --no-capture-output -n cellbench python scripts/check_cellbench_environment.py
conda run --no-capture-output -n cellbench python -m pip check
```

安装脚本安装项目依赖及 LangCell 仓库配套的 geneformer 0.0.1，
并将已下载的 30M 词表和基因中位数资源复制到该环境的 geneformer 包中。
`public_code/` 中的原始文件保持不变。

环境变量 `PYTHONNOUSERSITE=1` 由 Conda 在激活环境时自动设置，
避免 `~/.local/lib/python3.10/site-packages` 中的其他包混入。
如直接使用解释器绝对路径而不激活环境，需显式设置该变量。

环境检查报告写入 `output/environment_check.json`，包括实际模块导入和 GPU 张量运算结果。

2026-10-21 已验证：依赖一致性检查、全部 11 个模型适配器导入、CUDA 矩阵计算，
以及通过实际主入口提取 PBMC12k 的 PCA 嵌入。结果形状为 11,990 × 256，
文件位于 `output/embeddings/pbmc12k_PCA_model/all.pkl`。
scGPT 的本地权重加载亦已验证：普通注意力层的 Q/K/V 参数与原始 checkpoint 完全一致。
上述验证未执行完整的基础模型训练或基准评测。

## 运行入口

例如，提取 PBMC12k 的 PCA 嵌入：

```bash
python main.py \
  --cfgs configs/pca.yaml \
  --task embedding_extraction \
  --dataset_name pbmc12k \
  --gpu 0
```

默认直接执行模型任务。对已经保存的扰动预测进行 VCC 评测时，显式添加 `--vcc-eval`：

```bash
python main.py \
  --cfgs configs/nicheformer.yaml \
  --task perturbation \
  --dataset_name adamson \
  --vcc-eval
```

预测与缓存统一写入项目 `output/`，可用 YAML 顶层 `output_dir` 修改。
嵌入缓存位于 `output/embeddings/`，扰动预测位于 `output/perturbation_results/`。

## 本机适配

RTX 2070 属于 Turing 架构，不满足 [FlashAttention-2 的 Ampere 或更新架构要求](https://github.com/Dao-AILab/flash-attention#installation-and-features)。
scGPT 使用普通 PyTorch 注意力，并用上游加载函数转换预训练注意力参数名称。
Tabula 使用现有适配器的 FastFormer 回退分支；该分支与 FlashAttention 权重的匹配程度
需在具体实验前核查，环境导入通过不能替代模型权重和结果验证。

scMulan 的主权重下载状态可通过 `python scripts/check_resources.py` 检查；
缺少该文件不会妨碍环境配置和其他模型的依赖检查。
