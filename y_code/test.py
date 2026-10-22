import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.io import mmread
import os

# ==========================================
# 👇 用户配置区
# ==========================================

# Embedding 文件路径
EMBEDDING_PATH = '/usr/storage/yongjie/embeddings_update/uc_epi_Nicheformer/all.pkl'

# 原始数据路径（batch_data 目录）
DATA_DIR = '/usr/data/yongjie/data_zoo/Used_data/batch_data'
DATASET_NAME = 'uc_epi'

# ==========================================
# 🚀 主程序
# ==========================================

def load_batch_data(data_dir, dataset_name):
    """从 batch_data 目录加载原始数据"""
    mtx_path = os.path.join(data_dir, f"{dataset_name}.mtx")
    celltype_path = os.path.join(data_dir, f"{dataset_name}_celltype.tsv")
    batch_path = os.path.join(data_dir, f"{dataset_name}_batch.tsv")
    gene_path = os.path.join(data_dir, f"{dataset_name}_gene.tsv")
    
    print(f"  Loading: {mtx_path}")
    mtx_data = mmread(mtx_path)
    if sp.issparse(mtx_data):
        x = sp.csr_matrix(mtx_data.T)
    else:
        x = sp.csr_matrix(np.array(mtx_data).T)
    
    celltype_df = pd.read_csv(celltype_path, sep='\t', header=None)
    celltype_labels = celltype_df.iloc[:, 0].values
    
    batch_df = pd.read_csv(batch_path, sep='\t', header=None)
    batch_labels = batch_df.iloc[:, 0].values
    
    gene_df = pd.read_csv(gene_path, sep='\t', header=None)
    genes = gene_df.iloc[:, 0].values
    
    return x, celltype_labels, batch_labels, genes

def main():
    print("=" * 60)
    print("🔍 批次标签检查脚本")
    print("=" * 60)
    
    # --- Step 1: 加载 Embedding ---
    print(f"\n[1/4] Loading embeddings from: {EMBEDDING_PATH}")
    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(f"❌ 找不到文件: {EMBEDDING_PATH}")

    with open(EMBEDDING_PATH, 'rb') as f:
        data = pickle.load(f)

    print(f"  Keys: {data.keys()}")
    
    cell_embs = data['cell_embs']
    labels = data['labels']
    raw_exprs = data['raw_exprs']
    cell_ids = data['cell_ids']
    
    print(f"  cell_embs: {cell_embs.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  raw_exprs: {raw_exprs.shape}")
    print(f"  cell_ids: {cell_ids.shape}")
    print(f"  cell_ids 范围: [{cell_ids.min()}, {cell_ids.max()}]")
    print(f"  cell_ids 前10个: {cell_ids[:10]}")
    print(f"  cell_ids 后10个: {cell_ids[-10:]}")
    
    # --- Step 2: 加载原始批次数据 ---
    print(f"\n[2/4] Loading original batch data from: {DATA_DIR}/{DATASET_NAME}")
    x_orig, celltype_orig, batch_orig, genes_orig = load_batch_data(DATA_DIR, DATASET_NAME)
    
    print(f"  原始数据形状: {x_orig.shape}")
    print(f"  celltype 数量: {len(celltype_orig)}")
    print(f"  batch 数量: {len(batch_orig)}")
    print(f"  genes 数量: {len(genes_orig)}")
    
    # --- Step 3: 检查数据对齐 ---
    print(f"\n[3/4] 检查数据对齐...")
    
    # 检查 cell_ids 是否是有效索引
    if cell_ids.max() >= len(batch_orig):
        print(f"  ⚠️ 警告: cell_ids.max()={cell_ids.max()} >= 原始数据细胞数={len(batch_orig)}")
        print(f"  这说明 embedding 使用的是处理后的数据索引，而非原始数据索引")
        
        # 检查是否是连续索引（0, 1, 2, ...）
        if np.array_equal(cell_ids, np.arange(len(cell_ids))):
            print(f"  ✅ cell_ids 是连续索引 [0, {len(cell_ids)-1}]，说明需要使用 adata_all 的索引")
    else:
        print(f"  ✅ cell_ids 在有效范围内 [0, {len(batch_orig)-1}]")
    
    # --- Step 4: 检查批次标签 ---
    print(f"\n[4/4] 检查批次标签...")
    
    # 获取原始批次标签的分布
    batch_unique, batch_counts = np.unique(batch_orig, return_counts=True)
    print(f"  原始数据批次分布:")
    for b, c in zip(batch_unique, batch_counts):
        print(f"    {b}: {c} cells")
    
    # 如果 cell_ids 是连续索引，说明 embedding 对应的是过滤/处理后的数据
    # 需要模拟 dataset.py 中的处理流程
    print(f"\n  模拟 dataset.py 处理流程...")
    
    # 模拟 read_uc_epi 的处理
    from scipy.io import mmread
    mtx_path = os.path.join(DATA_DIR, f"{DATASET_NAME}.mtx")
    mtx_data = mmread(mtx_path)
    if sp.issparse(mtx_data):
        x = sp.csr_matrix(mtx_data.T)
    else:
        x = sp.csr_matrix(np.array(mtx_data).T)
    
    print(f"    原始矩阵: {x.shape}")
    
    # 模拟 normalize + log1p
    if x.max() > 10:
        print(f"    x.max() = {x.max():.2f} > 10，会做 normalize_total + log1p")
    
    # 模拟 highly_variable_genes (这会过滤基因，但不会过滤细胞)
    print(f"    HVG 过滤后基因数: 3000 (不影响细胞数)")
    
    # 检查最终的批次标签
    print(f"\n  最终检查:")
    print(f"    Embedding 细胞数: {len(cell_embs)}")
    print(f"    原始 batch 标签数: {len(batch_orig)}")
    
    if len(cell_embs) == len(batch_orig):
        print(f"    ✅ 细胞数完全匹配！可以直接使用 batch_orig 作为批次标签")
        
        # 检查用于 Harmony 的批次标签
        batch_labels = pd.Categorical(batch_orig)
        print(f"\n  用于 Harmony 的批次分布:")
        print(batch_labels.value_counts())
        
        unique_batches = batch_labels.unique()
        if len(unique_batches) < 2:
            print(f"    ❌ 只有 {len(unique_batches)} 个批次，Harmony 无法运行！")
        else:
            print(f"    ✅ 有 {len(unique_batches)} 个批次，可以运行 Harmony")
    else:
        print(f"    ❌ 细胞数不匹配！需要检查数据处理流程")
        print(f"    差异: {len(cell_embs) - len(batch_orig)} 个细胞")
        
        # 尝试找出原因
        if len(cell_embs) < len(batch_orig):
            print(f"    可能原因: 某些细胞在模型处理时被过滤掉了")
        else:
            print(f"    可能原因: 数据合并或其他处理导致细胞数增加")

    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
