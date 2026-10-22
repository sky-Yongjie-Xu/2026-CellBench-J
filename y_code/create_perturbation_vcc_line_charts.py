import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置专业的matplotlib样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 读取数据
df = pd.read_csv('/usr/data/yongjie/output/perturbation_vcc/perturbation_vcc_stats.csv')

# 解析 mean±std 格式
def parse_mean_std(value_str):
    if pd.isna(value_str):
        return np.nan, np.nan
    parts = str(value_str).split('±')
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    return float(parts[0]), 0.0

# 为三个指标解析数据
for metric_name in ['DES', 'MAE', 'PDS']:
    df[[f'{metric_name}_mean', f'{metric_name}_std']] = df[f'{metric_name}_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))

# 将top值转换为数值
top_mapping = {'top1': 1, 'top3': 3, 'top5': 5, 'top7': 7, 'top9': 9}
df['top_num'] = df['top'].map(top_mapping)

# 定义数据集顺序
datasets_order = ['adamson', 'norman']
datasets_labels = ['Adamson', 'Norman']

# 定义模型顺序和配色方案（保持原始颜色）
models_order = ['PCA', 'UMAP', 'scVI', 'scGPT', 'CellPLM', 'Geneformer', 'LangCell', 'scMulan', 'scFoundation']

# 使用与 plot_perturbation_stats_comparison.py 相同的配色方案
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
    'Geneformer': 's',   # 方形
    'LangCell': '*',     # 星形
    'scMulan': 'x',      # 十字
    'scFoundation': '*'  # 星形
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

# 获取自适应Y轴范围
def get_ylim(df_subset, metric_mean_col, metric_std_col):
    metric_max = (df_subset[metric_mean_col] + df_subset[metric_std_col]).max()
    metric_min = max(0, (df_subset[metric_mean_col] - df_subset[metric_std_col]).min())
    metric_max = metric_max * 1.08
    metric_min = max(0, metric_min * 0.95)
    return (metric_min, metric_max)

# 绘制单个指标的图
def plot_metric(metric_name, metric_mean_col, metric_std_col, ylabel, output_filename):
    # 创建图形（1x2布局，两个数据集）- 适合论文单栏宽度
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # 为每个数据集绘制子图
    for idx, (dataset, dataset_label) in enumerate(zip(datasets_order, datasets_labels)):
        ax = axes[idx]
        
        # 获取该数据集的数据
        df_dataset = df[df['dataset'] == dataset].copy()
        
        # 获取自适应Y轴范围
        ylim = get_ylim(df_dataset, metric_mean_col, metric_std_col)
        
        # 为每个模型绘制折线
        for model in models_order:
            df_model = df_dataset[df_dataset['method'] == model].sort_values('top_num')
            
            if df_model.empty:
                continue
            
            x = df_model['top_num'].values
            y = df_model[metric_mean_col].values
            yerr = df_model[metric_std_col].values
            
            # 绘制置信区间
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=colors[model])
            
            # 绘制折线和标记点
            marker_style = markers[model]
            marker_size = marker_sizes[model]
            
            ax.plot(x, y, color=colors[model], marker=marker_style, markersize=marker_size-1,
                   linestyle='-', linewidth=2.0, label=model, markeredgewidth=1.5,
                   markerfacecolor='white', markeredgecolor=colors[model], zorder=3,
                   alpha=0.9)
        
        # 设置坐标轴
        ax.set_xlabel('Top-k', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(dataset_label, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlim(0.5, 9.5)
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.set_ylim(ylim[0], ylim[1])
        ax.tick_params(axis='both', which='major', labelsize=9, width=1.2, length=4)
        
        # 优化网格样式
        ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    # 创建共享图例
    handles, labels = axes[0].get_legend_handles_labels()
    
    # 根据模型顺序重新排列图例
    handles_dict = dict(zip(labels, handles))
    ordered_handles = [handles_dict[model] for model in models_order if model in handles_dict]
    ordered_labels = [model for model in models_order if model in handles_dict]
    
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    
    # 创建共享图例（位于整个图的顶部，2行布局）
    legend = fig.legend(ordered_handles, ordered_labels, loc='upper center', ncol=5, 
                       fontsize=9, frameon=False, 
                       bbox_to_anchor=(0.5, 0.99), columnspacing=1.2, handlelength=2.0,
                       handletextpad=0.4, borderpad=0.4)
    
    # 保存图片（只保存PNG格式）
    output_path_png = f'/usr/data/yongjie/output/{output_filename}'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"{metric_name} 图片已保存至: {output_path_png}")
    
    plt.close()

# 定义指标配置
metrics_config = [
    ('DES', 'DES_mean±std', 'DES_mean', 'DES'),
    ('MAE', 'MAE_mean±std', 'MAE_mean', 'MAE'),
    ('PDS', 'PDS_mean±std', 'PDS_mean', 'PDS')
]

# 为所有指标分别生成图
for metric_name, metric_col, metric_mean_col, ylabel in metrics_config:
    output_filename = f'perturbation_vcc_{metric_name.lower()}_line_comparison.png'
    plot_metric(metric_name, f'{metric_name}_mean', f'{metric_name}_std', ylabel, output_filename)

print("所有VCC图片已生成完成！")
