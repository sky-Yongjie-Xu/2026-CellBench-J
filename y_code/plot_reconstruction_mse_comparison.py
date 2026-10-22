import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv('/usr/data/yongjie/output/xyj_bench_reconstruction_stats.csv')

# 解析MSE_expr_mean±std列，提取均值和标准差
def parse_mean_std(value):
    parts = str(value).split('±')
    mean = float(parts[0])
    std = float(parts[1]) if len(parts) > 1 else 0.0
    return mean, std

df[['mse_mean', 'mse_std']] = df['MSE_expr_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))

# 将top值转换为数值
top_mapping = {'top100': 100, 'top300': 300, 'top500': 500, 'top700': 700, 'top900': 900}
df['top_num'] = df['top'].map(top_mapping)

# 定义数据集顺序（按照用户指定的顺序）
datasets_order = ['pbmc12k', 'hPancreas', 'cortex', 'ms', 'Zheng68k', 'lung', 'covid', 'immune']
datasets_labels = ['PBMC12k', 'hPancreas', 'Cortex', 'MS', 'Zheng68k', 'Lung', 'Covid', 'Immune']

# 定义模型顺序和样式
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
def get_ylim(dataset_name, df_subset):
    mse_max = (df_subset['mse_mean'] + df_subset['mse_std']).max()
    mse_min = max(0, (df_subset['mse_mean'] - df_subset['mse_std']).min())
    # 添加一些边距
    mse_max = mse_max * 1.1
    mse_min = max(0, mse_min * 0.9)
    return (mse_min, mse_max)

# 创建图形（2x4布局，8个数据集）
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
# 不添加总标题，符合顶级会议论文风格

axes_flat = axes.flatten()

# 为每个数据集绘制子图
for idx, (dataset, dataset_label) in enumerate(zip(datasets_order, datasets_labels)):
    ax = axes_flat[idx]
    
    # 获取该数据集的数据
    df_dataset = df[df['dataset'] == dataset].copy()
    
    # 获取自适应Y轴范围
    ylim = get_ylim(dataset, df_dataset)
    
    # 为每个模型绘制折线
    for model in models_order:
        df_model = df_dataset[df_dataset['method'] == model]
        if len(df_model) == 0:
            continue
        
        # 按top数排序
        df_model = df_model.sort_values('top_num')
        
        x = df_model['top_num'].values
        y = df_model['mse_mean'].values
        yerr = df_model['mse_std'].values
        
        # 绘制置信区间（半透明阴影）
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=colors[model])
        
        # 绘制折线和标记点
        marker_style = markers[model]
        marker_size = marker_sizes[model]
        # 只在第一个子图添加label，用于创建共享图例
        label = model if idx == 0 else None
        
        if model == 'Geneformer':
            # Geneformer使用方形带十字的特殊标记
            ax.plot(x, y, color=colors[model], marker='s', markersize=marker_size+1, 
                   linestyle='-', linewidth=2, label=label, markeredgewidth=1.5, 
                   markeredgecolor=colors[model], markerfacecolor='white', zorder=3)
            # 添加十字标记
            ax.plot(x, y, color=colors[model], marker='+', markersize=marker_size+3, 
                   linestyle='None', markeredgewidth=2, zorder=4)
        elif model == 'scFoundation':
            # scFoundation使用填充星形
            ax.plot(x, y, color=colors[model], marker='*', markersize=marker_size, 
                   linestyle='-', linewidth=2, label=label, markerfacecolor=colors[model],
                   markeredgecolor='white', markeredgewidth=0.5, zorder=3)
        elif model == 'LangCell':
            # LangCell使用星形
            ax.plot(x, y, color=colors[model], marker='*', markersize=marker_size,
                   linestyle='-', linewidth=2, label=label, markeredgewidth=1.5,
                   markerfacecolor='white', markeredgecolor=colors[model], zorder=3)
        else:
            ax.plot(x, y, color=colors[model], marker=marker_style, markersize=marker_size,
                   linestyle='-', linewidth=2, label=label, markeredgewidth=1.5,
                   markerfacecolor='white', markeredgecolor=colors[model], zorder=3)
    
    # 设置坐标轴
    ax.set_xlabel('Top-k', fontsize=22, fontweight='bold')
    # 只在每行的第一个子图显示Y轴标签
    if idx % 4 == 0:
        ax.set_ylabel('MSE', fontsize=22, fontweight='bold')
    ax.set_title(dataset_label, fontsize=24, fontweight='bold')
    ax.set_xlim(50, 950)
    ax.set_xticks([100, 300, 500, 700, 900])
    ax.set_ylim(ylim[0], ylim[1])
    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

# 创建共享图例
handles, labels = axes_flat[0].get_legend_handles_labels()
# 根据模型顺序重新排列图例
handles_dict = dict(zip(labels, handles))
ordered_handles = [handles_dict[model] for model in models_order if model in handles_dict]
ordered_labels = [model for model in models_order if model in handles_dict]

# 创建共享图例（位于整个图的顶部）
fig.legend(ordered_handles, ordered_labels, loc='upper center', ncol=9, 
          fontsize=20, frameon=False, 
          bbox_to_anchor=(0.5, 1.02), columnspacing=1.5, handlelength=2.5, 
          handletextpad=0.5, borderpad=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为图例留出更多空间

# 保存图片
output_path = '/usr/data/yongjie/output/reconstruction_mse_comparison_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {output_path}")

plt.show()
