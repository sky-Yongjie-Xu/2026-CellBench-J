import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 使用相对路径读取CSV文件
csv_path = os.path.join(script_dir, '../output/perturbation_vcc/perturbation_vcc_stats.csv')
df = pd.read_csv(csv_path)

# 解析mean±std列，提取均值和标准差
def parse_mean_std(value):
    parts = str(value).split('±')
    mean = float(parts[0])
    std = float(parts[1]) if len(parts) > 1 else 0.0
    return mean, std

# 解析三个指标
df[['des_mean', 'des_std']] = df['DES_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))
df[['mae_mean', 'mae_std']] = df['MAE_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))
df[['pds_mean', 'pds_std']] = df['PDS_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))

# 将top值转换为数值
top_mapping = {'top1': 1, 'top3': 3, 'top5': 5, 'top7': 7, 'top9': 9}
df['top_num'] = df['top'].map(top_mapping)

# 定义数据集顺序
datasets_order = ['adamson', 'norman', 'replogleK562', 'replogleRPE1']
datasets_labels = ['Adamson', 'Norman', 'Replogle K562', 'Replogle RPE1']

# 定义模型顺序和样式（与参考脚本保持一致）
models_order = ['PCA', 'UMAP', 'scVI', 'scGPT', 'CellPLM', 'Geneformer', 'LangCell', 'scMulan', 'scFoundation']
colors = {
    'PCA': '#2ca02c',           # 绿色
    'UMAP': '#5dade2',          # 浅蓝色
    'scVI': '#0d47a1',          # 深蓝色
    'scGPT': '#9b59b6',         # 紫色
    'CellPLM': '#ff9800',       # 橙色
    'Geneformer': '#e74c3c',     # 红色
    'LangCell': '#ec407a',       # 粉色
    'scMulan': '#8b4513',        # 棕色
    'scFoundation': '#f48fb1'    # 浅粉色
}
markers = {
    'PCA': 'o',          # 圆形
    'UMAP': 's',         # 方形
    'scVI': '^',         # 三角形
    'scGPT': 'D',        # 菱形
    'CellPLM': 'v',      # 倒三角
    'Geneformer': 's',   # 方形带十字
    'LangCell': '*',     # 星形
    'scMulan': 'x',      # 十字
    'scFoundation': '*'  # 填充星形
}
marker_sizes = {
    'PCA': 6,
    'UMAP': 6,
    'scVI': 7,
    'scGPT': 6,
    'CellPLM': 6,
    'Geneformer': 7,
    'LangCell': 7,
    'scMulan': 7,
    'scFoundation': 8
}

# 定义自适应Y轴范围的函数
def get_ylim(df_subset, metric_mean_col, metric_std_col):
    metric_max = (df_subset[metric_mean_col] + df_subset[metric_std_col]).max()
    metric_min = max(0, (df_subset[metric_mean_col] - df_subset[metric_std_col]).min())
    # 添加一些边距
    metric_max = metric_max * 1.1
    metric_min = max(0, metric_min * 0.9)
    return (metric_min, metric_max)

# 绘制所有3个指标的3x4布局图
def plot_all_metrics():
    # 创建图形（3x4布局，3个指标 x 4个数据集）- 调整尺寸使子图更扁
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    
    # 定义三个指标
    metrics_info = [
        ('DES', 'des_mean', 'des_std', 'DES'),
        ('MAE', 'mae_mean', 'mae_std', 'MAE'),
        ('PDS', 'pds_mean', 'pds_std', 'PDS')
    ]
    
    # 为每个指标和数据集绘制子图
    for metric_idx, (metric_name, metric_mean_col, metric_std_col, ylabel) in enumerate(metrics_info):
        for dataset_idx, (dataset, dataset_label) in enumerate(zip(datasets_order, datasets_labels)):
            ax = axes[metric_idx, dataset_idx]
        
            # 获取该数据集的数据
            df_dataset = df[df['dataset'] == dataset].copy()
            
            # 获取自适应Y轴范围
            ylim = get_ylim(df_dataset, metric_mean_col, metric_std_col)
            
            # 为每个模型绘制折线
            for model in models_order:
                df_model = df_dataset[df_dataset['method'] == model]
                if len(df_model) == 0:
                    continue
                
                # 按top数排序
                df_model = df_model.sort_values('top_num')
                
                x = df_model['top_num'].values
                y = df_model[metric_mean_col].values
                yerr = df_model[metric_std_col].values
                
                # 绘制置信区间（半透明阴影）
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=colors[model])
                
                # 绘制折线和标记点
                marker_style = markers[model]
                marker_size = marker_sizes[model]
                if model == 'Geneformer':
                    # Geneformer使用方形带十字的特殊标记
                    ax.plot(x, y, color=colors[model], marker='s', markersize=marker_size+1, 
                           linestyle='-', linewidth=2, label=model, markeredgewidth=1.5, 
                           markeredgecolor=colors[model], markerfacecolor='white', zorder=3)
                    # 添加十字标记
                    ax.plot(x, y, color=colors[model], marker='+', markersize=marker_size+3, 
                           linestyle='None', markeredgewidth=2, zorder=4)
                elif model == 'scFoundation':
                    # scFoundation使用填充星形
                    ax.plot(x, y, color=colors[model], marker='*', markersize=marker_size, 
                           linestyle='-', linewidth=2, label=model, markerfacecolor=colors[model],
                           markeredgecolor='white', markeredgewidth=0.5, zorder=3)
                elif model == 'LangCell':
                    # LangCell使用星形
                    ax.plot(x, y, color=colors[model], marker='*', markersize=marker_size,
                           linestyle='-', linewidth=2, label=model, markeredgewidth=1.5,
                           markerfacecolor='white', markeredgecolor=colors[model], zorder=3)
                else:
                    ax.plot(x, y, color=colors[model], marker=marker_style, markersize=marker_size,
                           linestyle='-', linewidth=2, label=model, markeredgewidth=1.5,
                           markerfacecolor='white', markeredgecolor=colors[model], zorder=3)
            
            # 设置坐标轴
            ax.set_xlabel('top', fontsize=14)
            # 只在每行的第一列显示y轴标签
            if dataset_idx == 0:
                ax.set_ylabel(ylabel, fontsize=14)
            # 只在第一行显示数据集标题
            if metric_idx == 0:
                ax.set_title(dataset_label, fontsize=15, fontweight='bold')
            ax.set_xlim(0.5, 9.5)  # 减少两边空隙
            ax.set_xticks([1, 3, 5, 7, 9])
            ax.set_ylim(ylim[0], ylim[1])
            # 设置刻度标签字体大小
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.3, color='gray')
            ax.set_axisbelow(True)
    
    # 创建共享图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # 根据模型顺序重新排列图例
    handles_dict = dict(zip(labels, handles))
    ordered_handles = [handles_dict[model] for model in models_order if model in handles_dict]
    ordered_labels = [model for model in models_order if model in handles_dict]
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为图例留出更多空间
    
    # 创建共享图例（位于整个图的顶部）
    fig.legend(ordered_handles, ordered_labels, loc='upper center', ncol=9, 
              fontsize=16, frameon=False, 
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=2.0)
    
    # 保存图片（使用相对路径）
    output_dir = os.path.join(script_dir, '../output/perturbation_vcc')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'perturbation_vcc_all_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"所有3个指标的图片已保存至: {output_path}")
    
    plt.close()

# 生成3x4布局的大图
plot_all_metrics()

print("图片已生成完成！")

