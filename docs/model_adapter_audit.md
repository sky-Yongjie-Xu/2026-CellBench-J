# 模型适配器源码审查

范围：14 个基础模型的数据、加载、embedding 三阶段；PCA/UMAP/scVI 三个基线也统一接口与缓存。

本表区分官方 API、协议移植和兼容重建，不把“可以运行”当作“数值等价”。官方源码保持原样；基准缓存属于项目自有代码。

| 模型 | 数据与基因匹配 | 模型加载 | embedding | 本次完整适配器小样本 |
|---|---|---|---|---|
| CellPLM | 全基因计数、官方 common_preprocess，禁止重改潜在层配置 | 官方 OmicsFormer + EmbedderHead；CPU 加载端口 | 官方 CellEmbeddingPipeline.predict，保留顺序 | 通过 [2, 512] |
| Geneformer | V1 计数/总量/中位数排名编码，保留源行 ID | BertForMaskedLM，检查未初始化参数 | 倒数第二层、非 padding 均值 | 通过 [2, 512] |
| LangCell | V1 排名编码 + CLS 与 padding，无截断式对齐 | cell_bert + cell_proj.bin，检查骨干缺失 | CLS 投影与 L2 归一化 | 通过 [2, 256] |
| scFoundation | 19264 基因补齐、normalize/log1p、t4 计数提示 | 严格加载原模型；CPU 设备端口 | 逐细胞四路池化，3072 维 | 通过 [2, 3072] |
| scGPT | 本地基因映射、normalize/log1p、51 分箱与 1200 长度 | 官方兼容权重映射，验证全部编码器参数 | 关闭 batch/分类模块，CLS 后 L2 归一化 | 通过 [2, 512] |
| scMulan | 基因名统一与表达分箱；拒绝丢失细胞 | 严格加载；当前主权重缺失 | 官方 PCT 生成隐藏状态取 -2 token；未实测 | 未重跑；见已有后端记录 |
| Nicheformer | 全词表、1e4 标准化、technology mean、上下文前缀 | 作者 HF 模型代码；接口异于 Lightning 源码 | 去前三个上下文位置后均值，与 HF/源码定义一致 | 通过 [2, 512] |
| Tabula | 官方 check_vocab/Preprocessor，保留评测表达空间 | 官方模型 + 等价 SDPA，自注意力权重完整性检查 | 官方 cell_embed，拒绝 NaN/Inf 清零 | 通过 [2, 192] |
| UCE | 官方 AnndataProcessor、染色体/基因采样 | 官方 TransformerModel，mmap/meta 严格赋值 | 官方 forward 细胞表示 | 未重跑；见已有后端记录 |
| scPRINT2 | checkpoint 词表重排；官方 Embedder/Collator | 官方构造器与 checkpoint hook，严格加载 | 官方 scprint_emb，关闭 knn/分类，分块 | 未重跑；见已有后端记录 |
| TranscriptFormer | 官方 CLI/tokenizer；缺测序 assay 时 unknown | 官方推理 CLI，TF-Sapiens | 官方 embedding；上下文限制 1023 | 未重跑；见已有后端记录 |
| CellFM | 公开 data_process 协议的 CPU/MindSpore 端口 | 官方 CellFM 与严格参数加载；内存不足未实测 | 官方 CLS 路径；未实测 | 未重跑；见已有后端记录 |
| GeneCompass | 计数/中位数、log2 与排序；使用 checkpoint 先验 | 恢复公开源码缺失的值编码器；144 状态项严格加载 | 最后一层非 padding 基因均值；原训练实现等价未验证 | 通过 [2, 256] |
| scLong | 固定词表补齐、normalize/log1p | 官方 DualEncoderSCFM，严格加载 | 匹配基因 decoder 均值是 benchmark 自定义池化 | 未重跑；见已有后端记录 |

## 统一结构与缓存

`FoundationAdapter` 统一 `prepare_inputs`、`load_pretrained`、`encode_batch`；兼容原来的训练/评测入口。
14 个基础模型与 3 个基线共享源行 ID 校验、有限数校验、缓存读取和原子写入。六个独立后端在子进程中实际加载权重。
模型输入从全基因计数重新读取；下游表达目标继续使用评测基因空间，二者不混用。标签不会传入预训练编码器。
缓存位置为 `output/embeddings/<dataset>_<model>/v2/<signature>/all.pkl`；签名包括输入/侧文件、权重文件状态、配置、seed、本地与上游 Python 源码。
旧缓存不删除，但不自动复用；非有限数、重复/丢失细胞及标签错位均明确失败。

PCA/UMAP/scVI 使用 `StatisticalAdapter`，在当前数据上拟合，不加载基础模型检查点。scVI 必须使用真实计数。PCA 已通过两细胞流程；UMAP/scVI 本次未执行完整拟合。

复现：`python scripts/check_adapter_inference.py --model Geneformer --input datasets/pbmc_12k.h5ad --cells 2`。
两细胞测试验证加载、输出结构、细胞顺序和缓存复读，不代表全数据评测或与官方实现的逐数值对照。

## 逐模型差异与依据

### CellPLM

**审查类型：** `native_api`。

官方完整潜在层与零样本 Pipeline；仅增加 CPU 加载及骨干完整性检查。

- 去掉原先把 gmvae 改为 ae、修改 decoder/dropout 的覆盖。
- 固定本地 HGNC 匹配；官方默认在线匹配可有差异。

源码：[__init__.py](../public_code/CellPLM/CellPLM/pipeline/__init__.py)、[cell_embedding.py](../public_code/CellPLM/CellPLM/pipeline/cell_embedding.py)

固定提交：`b59ee7688b1bd2a745856a610248b46f15019a22`；源码干净：`True`。

### Geneformer

**审查类型：** `protocol_port`。

V1 30M 排名编码、BertForMaskedLM、倒数第二层非 padding 均值。

- 使用本地 in-memory tokenizer 端口；不声称与所有新版本 tokenizer 字节级相同。
- 保留 cell_id；移除随机分类头作为预训练骨干的加载方式。

源码：[tokenizer.py](../public_code/Geneformer/geneformer/tokenizer.py)、[emb_extractor.py](../public_code/Geneformer/geneformer/emb_extractor.py)

固定提交：`31bf641d0f173a18a5e06d9a6574bb0d3fe5edc1`；源码干净：`True`。

### LangCell

**审查类型：** `protocol_port`。

Geneformer V1 排名、CLS 前缀、训练好的投影与 L2 归一化。

- CLS/padding collator 移植，规避新版 Transformers tokenizer 接口不兼容。
- 拒绝旧代码中的截断对齐和越界 token 截断。

源码：[preprocess.py](../public_code/LangCell/data_preprocess/preprocess.py)、[utils.py](../public_code/LangCell/LangCell-annotation-zeroshot/utils.py)

固定提交：`69e41ef2fae485b67703294b50f3150bb1d5bb9b`；源码干净：`True`。

### scFoundation

**审查类型：** `protocol_port`。

19264 基因补零；原始计数 normalize/log1p；t4 提示；四路池化。

- 单细胞批次大小固定为 1，保持官方逐细胞尾 token/均值池化语义。
- 加载 helper 保留严格权重检查，允许 CPU；原先无条件 CUDA。

源码：[get_embedding.py](../public_code/scFoundation/model/get_embedding.py)、[load.py](../public_code/scFoundation/model/load.py)

固定提交：`397631c495eddf9ad6644fc00c6ea8139e651245`；源码干净：`True`。

### scGPT

**审查类型：** `protocol_port`。

关闭额外 batch 模块，分箱与 CLS，L2 归一化，原生注意力兼容。

- 运行时使用 cellbench 中的 scgpt 包；公开源码的默认协议已逐项核对，包版本并非源码逐文件拷贝。
- tokenize_and_pad_batch 与官方 DataCollator 的随机采样实现不同；固定 seed 与长度，不宣称逐值相同。

源码：[cell_emb.py](../public_code/scGPT/scgpt/tasks/cell_emb.py)、[preprocess.py](../public_code/scGPT/scgpt/preprocess.py)、[util.py](../public_code/scGPT/scgpt/utils/util.py)

固定提交：`cebd6fae655b9c585a4807daa3ac31bb764f06b4`；源码干净：`True`。

### scMulan

**审查类型：** `protocol_port_unverified`。

基因统一、表达分箱、PCT 提示及生成隐藏状态的倒数第二 token。

- 主检查点缺失，不能实测；仅静态核对。
- 严格拒绝预处理后细胞缺失/重排；DataFrame 列按实际基因顺序。

源码：[scMulan.py](../public_code/scMulan/scMulan/scMulan.py)、[model.py](../public_code/scMulan/scMulan/model/model.py)

固定提交：`d64da5ed91fd3c730c4c975e898e5e5cc360d872`；源码干净：`True`。

### Nicheformer

**审查类型：** `hf_port`。

完整词表顺序、每细胞 1e4 标准化、technology mean、三类上下文 token、去上下文均值。

- 加载作者 HF 版本的模型代码，接口与 public_code 中 Lightning 版本不同。
- technology mean 缺失即失败；零均值按官方替换为 1，保留官方包含 padding 位置的池化定义。

源码：[dataset.py](../public_code/Nicheformer/src/nicheformer/data/dataset.py)、[_nicheformer.py](../public_code/Nicheformer/src/nicheformer/models/_nicheformer.py)

固定提交：`485cadbc5caa15119adfd54228f8a8af835fcabc`；源码干净：`True`。

### Tabula

**审查类型：** `native_api_with_attention_compat`。

官方预处理与模型；SDPA 保留 FlashAttention 参数布局和标准自注意力运算。

- 不再把 FlashAttention 检查点切到不同的 FastFormer 架构。
- FP32 SDPA 已与 PyTorch 标准多头注意力做数值比较；不声称和 FlashAttention FP16 逐位一致。
- pancreas.pth 的具体训练范围需要单独确认，正式比较需明确检查点。

源码：[preprocessor.py](../public_code/Tabula/tabula/finetune/preprocessor.py)、[transformer.py](../public_code/Tabula/tabula/model/transfomer/transformer.py)、[attention.py](../public_code/Tabula/tabula/model/transfomer/attention.py)

固定提交：`65b5f7ebf36a534da42de94570c108539af05541`；源码干净：`True`。

### UCE

**审查类型：** `native_api`。

官方 AnndataProcessor/采样器/模型与细胞表示。

- mmap/meta assign 降低加载内存；仅加载所选物种蛋白资源；不更改网络。

源码：[evaluate.py](../public_code/UCE/evaluate.py)、[eval_data.py](../public_code/UCE/eval_data.py)、[model.py](../public_code/UCE/model.py)

固定提交：`9c416007be15ad6753dc84af4468c1dc10421ab9`；源码干净：`True`。

### scPRINT2

**审查类型：** `native_api`。

官方模型、on_load_checkpoint、Embedder；按 checkpoint 物种词表补齐。

- 关闭分类与 knn；分块限制内存；这些是显式推理设置。

源码：[cell_emb.py](../public_code/scPRINT2/scprint2/tasks/cell_emb.py)、[model.py](../public_code/scPRINT2/scprint2/model/model.py)

固定提交：`db0ef5f1fd120084a4323572bd54f8f2a2af2566`；源码干净：`True`。

### TranscriptFormer

**审查类型：** `native_api`。

官方 CLI 完成词表处理、模型加载与 embedding 输出。

- 目前为 1023 上下文，发布最大值 2047；CPU 内存限制下的显式配置。
- 没有 assay 元数据时用 unknown，不伪造实际测序技术。

源码：[inference.py](../public_code/TranscriptFormer/src/transcriptformer/cli/inference.py)、[inference.py](../public_code/TranscriptFormer/src/transcriptformer/model/inference.py)、[tokenizer.py](../public_code/TranscriptFormer/src/transcriptformer/tokenizer/tokenizer.py)

固定提交：`c943a89a4de9511a8ab1010715bf1cfa8827cc99`；源码干净：`True`。

### CellFM

**审查类型：** `protocol_port_unverified`。

官方 MindSpore 模型，计数匹配、log-normalization 与 CLS 输出。

- 依赖可导入，但当前可用内存不足，完整模型推理未验证。
- 使用 MindSpore 2.6 CPU 兼容环境；与作者版本的数值一致性未验证。

源码：[model.py](../public_code/CellFM/model.py)、[data_process.py](../public_code/CellFM/data_process.py)、[config.py](../public_code/CellFM/config.py)

固定提交：`72c9f4a9580a3716058c184900ed14a65151ed8f`；源码干净：`True`。

### GeneCompass

**审查类型：** `reconstructed_unverified_equivalence`。

检查点先验 buffers；中位数缩放/log2 排序；恢复保存配置中的值编码器。

- 公开 main 与 Small 发布权重不匹配，兼容实现恢复 value encoder 和 1536 宽拼接。
- concat_values_first=false 按保存配置解释为值编码置于先验之后；未验证作者训练实现的逐值等价。

源码：[modeling_bert.py](../public_code/GeneCompass/genecompass/modeling_bert.py)、[knowledge_embeddings.py](../public_code/GeneCompass/genecompass/knowledge_embeddings.py)、[preprocess.py](../public_code/GeneCompass/preprocess/preprocess.py)

固定提交：`59e5e485cb940038a7db040c2f31c8a2ce76ba53`；源码干净：`True`。

### scLong

**审查类型：** `custom_cell_pooling`。

官方 DualEncoderSCFM 严格加载；对匹配基因的 merged_decodings 均值池化。

- 源码零样本脚本调用 get_cell_emb，但当前类未定义该方法。
- 当前细胞级均值池化是 benchmark 明确定义，不能标为官方原样细胞 embedding。

源码：[ding_models.py](../public_code/scLong/performer_pytorch_cont/ding_models.py)、[scLong_zero_shot.py](../public_code/scLong/zero-shot-batch/scLong_zero_shot.py)

固定提交：`41b72021540918e4386c4f2264351d7a6cbeed99`；源码干净：`True`。

