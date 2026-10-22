# 模型适配器规范

每个基础模型在自己的模块中导出同名类，并继承 `FoundationAdapter`。
模型特有的数学协议保留；公共接口、身份校验和缓存只有一套实现。
PCA、UMAP、scVI 继承同一接口，但属于数据集上拟合的统计基线，没有预训练检查点。

```python
from bio.models.adapter import FoundationAdapter

__all__ = ['ExampleModel']

class ExampleModel(FoundationAdapter):
    def load_pretrained(self, model_cfg):
        # 使用固定 public_code 版本或明确记录的兼容端口。
        # 验证编码器参数完整，未使用的任务头可以显式排除。
        self.model = ...
        self.setup_task_heads(embedding_dim)

    def prepare_inputs(self, data_cfg, train=True):
        self.adata_all = self.read_adata(data_cfg['dataset_name'])
        native = self.read_model_input()  # 全基因计数，与评测表达目标独立
        self.all_loader = ...  # 每批携带整数 cell_id，保持源行身份

    def encode_batch(self, batch):
        # 仅把模型需要的基因/表达/掩码等输入传给编码器，不传细胞标签。
        # 使用模型官方定义的 tokenization 和 pooling。
        return evaluation_expression, cell_embeddings, cell_labels
```

数据准备依赖词表，因此框架实际先构建模型/词表，再准备输入。
独立环境模型在 `encode_batch` 调用子进程，由后端实际加载权重并执行原生预处理。

要求：

- 基因标识符、合并重复基因、计数来源由公共输入层处理；各模型自行执行原生缩放、分箱、排序。
- 没有原始计数时明确失败，不使用四舍五入、猜测 log 状态或填零来掩盖输入问题。
- 下游目标使用 `evaluation_expression()`；不得把全基因模型输入替换到 HVG 重建目标里。
- 每个输出细胞都必须有源行 ID。丢失、重复、未知身份、标签错位及非有限值均失败。
- 不覆盖 `extract_cell_embeddings`、`save_embeddings`、`load_embeddings` 和缓存路径。
- 缓存按输入、权重、参数、seed 与源码指纹隔离，原子写入；旧无签名缓存不复用。
- 在 `resources/model_protocols.json` 记录三个阶段的参考源码、协议和已知差异。
- 运行 `scripts/audit_model_adapters.py`、契约测试及真实小样本检查。

```bash
python scripts/audit_model_adapters.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/check_adapter_inference.py --model Geneformer --input datasets/pbmc_12k.h5ad
```

本规范针对冻结 embedding 的评测管线；完整骨干微调须另行验证，不能由小样本提取成功推断其可用。
