import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig

from geneformer import DataCollatorForCellClassification
from geneformer import TranscriptomeTokenizer
from geneformer.tokenizer import tokenize_cell
from geneformer.collator_for_classification import PrecollatorForGeneAndCellClassification
from geneformer.pretrainer import token_dictionary

# 将<cls>标记添加到token字典中
token_dictionary = dict(token_dictionary)
token_dictionary.setdefault('<cls>', len(token_dictionary))

logger = logging.get_logger(__name__)


class GeneformerTranscriptomeTokenizer(TranscriptomeTokenizer):
    """
    语言细胞转录组分词器，继承自TranscriptomeTokenizer
    用于处理单细胞RNA测序数据并将其转换为token序列
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize_anndata(self, data):
        """
        将AnnData对象(单细胞RNA测序数据)转换为token序列
        
        参数:
            data: AnnData对象，包含单细胞RNA测序数据
            
        返回:
            tokenized_cells: 每个细胞的token序列列表
            file_cell_metadata: 细胞元数据字典(如果有)
        """
        # 如果有自定义属性名称字典，初始化元数据存储
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        # 获取编码miRNA的位置
        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in data.var["ensembl_id"]]
        )[0]
        
        # 计算标准化因子向量
        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in data.var["ensembl_id"][coding_miRNA_loc]
            ]
        )
        
        # 获取编码miRNA的ID和对应的token
        coding_miRNA_ids = data.var["ensembl_id"][coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        # 检查是否有filter_pass列，用于筛选细胞
        var_exists = False
        if var_exists is True:
            filter_pass_loc = np.where(
                [True if i == 1 else False for i in data.obs["filter_pass"]]
            )[0]
        elif var_exists is False:
            print(
                f"data has no column attribute 'filter_pass'; tokenizing all cells."
            )
            filter_pass_loc = np.array([i for i in range(data.shape[0])])

        tokenized_cells = []

        # 获取子视图(通过筛选的细胞和编码miRNA)
        subview = data[filter_pass_loc, coding_miRNA_loc]

        # 计算标准化表达矩阵
        subview_norm_array = (
            subview.X.toarray().T
            / subview.obs.n_counts.to_numpy()
            * 10_000
            / norm_factor_vector[:, None]
        )

        # 对每个细胞进行分词
        tokenized_cells += [
            tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
            for i in range(subview_norm_array.shape[1])
        ]

        # 收集元数据(如果有)
        if self.custom_attr_name_dict is not None:
            for k in file_cell_metadata.keys():
                file_cell_metadata[k] += subview.obs[k].tolist()
        else:
            file_cell_metadata = None

        return tokenized_cells, file_cell_metadata