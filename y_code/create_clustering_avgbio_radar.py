import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# 读取数据
df = pd.read_csv('/usr/data/yongjie/output/xyj_bench_clustering_results.csv')

# 所有9个数据集
datasets = ['pbmc12k', 'hPancreas', 'cortex', 'ms', 'liver', 'Zheng68k', 'lung', 'covid', 'immune']
df_filtered = df[df['dataset'].isin(datasets)].copy()

# 定义模型顺序
models_order = ['PCA', 'UMAP', 'scVI', 'CellPLM', 'Geneformer', 'LangCell', 'scGPT', 'scMulan', 'scFoundation']

# 数据集标签映射
dataset_labels = {
    'pbmc12k': 'PBMC12k',
    'hPancreas': 'hPancreas',
    'cortex': 'Cortex',
    'ms': 'MS',
    'liver': 'Liver',
    'Zheng68k': 'Zheng68k',
    'lung': 'Lung',
    'covid': 'Covid',
    'immune': 'Immune'
}

# 使用与其他任务一致的颜色（与 plot_perturbation_stats_comparison.py 相同）
model_colors = {
    'PCA': '#2ca02c',           # 绿色
    'UMAP': '#5dade2',          # 浅蓝色
    'scVI': '#0d47a1',          # 深蓝色
    'scGPT': '#9b59b6',         # 紫色
    'CellPLM': '#ff9800',       # 橙色
    'Geneformer': '#e74c3c',    # 红色
    'LangCell': '#ec407a',      # 粉色
    'scMulan': '#8b4513',       # 棕色
    'scFoundation': '#f48fb1'   # 浅粉色
}

# 准备数据
radar_data = {}
for model in models_order:
    values = []
    for dataset in datasets:
        model_data = df_filtered[(df_filtered['dataset'] == dataset) & 
                                 (df_filtered['method'] == model)]
        if len(model_data) > 0:
            values.append(model_data.iloc[0]['AvgBio'])
        else:
            values.append(0)
    radar_data[model] = values

# 创建雷达图 - 优化版本
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='polar')

# 设置角度
num_vars = len(datasets)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 分组绘制：基线方法、预训练模型、我们的模型
baseline_models = ['PCA', 'UMAP', 'scVI']
pretrained_models = ['CellPLM', 'Geneformer', 'LangCell', 'scGPT']
our_models = ['scMulan', 'scFoundation']

# 为每个模型绘制雷达图
for model in models_order:
    values = radar_data[model]
    values += values[:1]  # 闭合
    
    # 根据模型类型设置线宽和透明度
    if model in our_models:
        linewidth = 2.5
        alpha_fill = 0.08
        alpha_line = 1.0
        zorder = 3
    elif model in baseline_models:
        linewidth = 1.5
        alpha_fill = 0.05
        alpha_line = 0.7
        zorder = 1
    else:
        linewidth = 1.8
        alpha_fill = 0.06
        alpha_line = 0.8
        zorder = 2
    
    ax.plot(angles, values, '-', linewidth=linewidth, label=model, 
            color=model_colors[model], alpha=alpha_line, zorder=zorder)
    ax.fill(angles, values, alpha=alpha_fill, color=model_colors[model])

# 设置标签 - 更大更清晰
ax.set_xticks(angles[:-1])
ax.set_xticklabels([dataset_labels[d] for d in datasets], 
                    fontsize=12, fontweight='bold')

# 设置y轴范围和刻度
ax.set_ylim(0, 0.9)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10, color='gray')

# 优化网格样式
ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8, color='gray')
# 去除最外圈的圆线
ax.spines['polar'].set_visible(False)

# 设置背景色
ax.set_facecolor('#fafafa')

# 添加标题 - 更简洁
plt.title('Clustering Performance (AvgBio)', 
          fontsize=15, fontweight='bold', pad=25, y=1.08)

# 优化图例 - 分组显示
legend = plt.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05), 
                    fontsize=10, frameon=False, ncol=1)
# legend.get_frame().set_facecolor('white')
# legend.get_frame().set_alpha(0.95)

# 保存图片
output_path = '/usr/data/yongjie/output/clustering_avgbio_radar.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"雷达图已保存至: {output_path}")

plt.close()
