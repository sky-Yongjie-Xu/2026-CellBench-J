import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bio.data.dataset import read_ms, read_Zheng68k
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
from bio.metrics.hierarchical import cluster_acc, build_tree, dendrogram_purity, leaf_purity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_reduction_results(X_256, true_labels, preds, method_name, save_dir='./results'):
    """
    先用256维聚类，再降到2维画真实标签和聚类标签（标题用英文）
    """
    os.makedirs(save_dir, exist_ok=True)
    if method_name == 'PCA':
        reducer = PCA(n_components=2)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X_256)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=true_labels, cmap='tab20', s=5, alpha=0.7)
    plt.title(f'{method_name} 2D (True Labels)')
    plt.subplot(1, 2, 2)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=preds, cmap='tab20', s=5, alpha=0.7)
    plt.title(f'{method_name} 2D (Cluster Labels)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{method_name}_2d_visualization.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(true_labels, preds, method_name, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title(f'{method_name} Clustering Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{method_name}_confusion_matrix.png'), dpi=300)
    plt.close()

def evaluate_dimension_reduction(adata_test, n_components=256, prefix=''):
    """
    评估PCA和UMAP降维到256维后的聚类效果
    """
    # 获取真实标签并转换为数字
    if 'celltype' in adata_test.obs:
        label_encoder = LabelEncoder()
        true_labels = label_encoder.fit_transform(adata_test.obs['celltype'].values)
    else:
        raise ValueError("数据集中没有找到'celltype'标签")
    
    # 准备数据
    if scipy.sparse.issparse(adata_test.X):
        X = adata_test.X.toarray()
    else:
        X = adata_test.X
    
    # PCA降到256维
    pca_256 = PCA(n_components=n_components)
    X_pca_256 = pca_256.fit_transform(X)
    # UMAP降到256维
    umap_256 = umap.UMAP(n_components=n_components, random_state=42)
    X_umap_256 = umap_256.fit_transform(X)
    
    n_clusters = len(np.unique(true_labels))
    # 聚类
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42).fit(X_pca_256)
    preds_pca = kmeans_pca.labels_
    centers_pca = kmeans_pca.cluster_centers_
    kmeans_umap = KMeans(n_clusters=n_clusters, random_state=42).fit(X_umap_256)
    preds_umap = kmeans_umap.labels_
    centers_umap = kmeans_umap.cluster_centers_
    
    # 评估
    def calculate_metrics(true_labels, preds, centers):
        acc_value = cluster_acc(true_labels, preds)
        acc_nmi = normalized_mutual_info_score(true_labels, preds)
        acc_ari = adjusted_rand_score(true_labels, preds)
        node_root = build_tree(centers, preds)
        zzl_leaves = node_root.list_all_leaf()
        ind_samples_of_leaves = [
            [zzl_leaves[i], np.where(preds == i)[0]] for i in range(len(zzl_leaves))
        ]
        dp = dendrogram_purity(node_root, true_labels, ind_samples_of_leaves)
        lp = leaf_purity(node_root, true_labels, ind_samples_of_leaves)
        return {
            "acc": acc_value,
            "nmi": acc_nmi,
            "ari": acc_ari,
            "dp": dp,
            "lp": lp
        }
    metrics_pca = calculate_metrics(true_labels, preds_pca, centers_pca)
    metrics_umap = calculate_metrics(true_labels, preds_umap, centers_umap)
    # 可视化
    plot_reduction_results(X_pca_256, true_labels, preds_pca, f'{prefix}PCA')
    plot_reduction_results(X_umap_256, true_labels, preds_umap, f'{prefix}UMAP')
    plot_confusion_matrix(true_labels, preds_pca, f'{prefix}PCA')
    plot_confusion_matrix(true_labels, preds_umap, f'{prefix}UMAP')
    return {
        f'{prefix}PCA': metrics_pca,
        f'{prefix}UMAP': metrics_umap
    }

def main():
    data_cfg = {
        "root_path": "/root/commusim/data_zoo/Used_data"
    }
    print("正在加载MS数据集...")
    adata_train_ms, adata_valid_ms, adata_test_ms = read_ms(data_cfg)
    print(f"MS测试数据形状: {adata_test_ms.shape}")
    print("\n正在评估MS数据集降维效果...")
    metrics_ms = evaluate_dimension_reduction(adata_train_ms, prefix='ms_')
    print("\nMS数据集PCA降维到256维的聚类结果：")
    for metric_name, value in metrics_ms["ms_PCA"].items():
        print(f"{metric_name}: {value:.4f}")
    print("\nMS数据集UMAP降维到256维的聚类结果：")
    for metric_name, value in metrics_ms["ms_UMAP"].items():
        print(f"{metric_name}: {value:.4f}")
    results_df_ms = pd.DataFrame({
        'Metric': list(metrics_ms['ms_PCA'].keys()),
        'PCA': list(metrics_ms['ms_PCA'].values()),
        'UMAP': list(metrics_ms['ms_UMAP'].values())
    })
    results_df_ms.to_csv('ms_dimension_reduction_results.csv', index=False)
    print("\nMS结果已保存到 ms_dimension_reduction_results.csv 和 results 目录")

    print("\n正在加载Zheng68k数据集...")
    adata_train_zheng, adata_valid_zheng, adata_test_zheng = read_Zheng68k(data_cfg)
    print(f"Zheng68k测试数据形状: {adata_test_zheng.shape}")
    print("\n正在评估Zheng68k数据集降维效果...")
    metrics_zheng = evaluate_dimension_reduction(adata_train_zheng, prefix='zheng_')
    print("\nZheng68k数据集PCA降维到256维的聚类结果：")
    for metric_name, value in metrics_zheng["zheng_PCA"].items():
        print(f"{metric_name}: {value:.4f}")
    print("\nZheng68k数据集UMAP降维到256维的聚类结果：")
    for metric_name, value in metrics_zheng["zheng_UMAP"].items():
        print(f"{metric_name}: {value:.4f}")
    results_df_zheng = pd.DataFrame({
        'Metric': list(metrics_zheng['zheng_PCA'].keys()),
        'PCA': list(metrics_zheng['zheng_PCA'].values()),
        'UMAP': list(metrics_zheng['zheng_UMAP'].values())
    })
    results_df_zheng.to_csv('zheng_dimension_reduction_results.csv', index=False)
    print("\nZheng68k结果已保存到 zheng_dimension_reduction_results.csv 和 results 目录")

if __name__ == "__main__":
    main() 