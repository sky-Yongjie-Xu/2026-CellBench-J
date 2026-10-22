import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('/usr/data/yongjie/output/xyj_bench_clustering_results.csv')

# 所有9个数据集
datasets = ['pbmc12k', 'cortex','hPancreas', 'immune', 'Zheng68k', 'ms', 'covid', 'lung','liver']
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

# 使用与其他任务一致的颜色
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

# 4个指标
metrics = ['nmi', 'ari', 'asw', 'AvgBio']
metric_titles = {
    'nmi': 'NMI',
    'ari': 'ARI',
    'asw': 'ASW',
    'AvgBio': 'AvgBio'
}

# 准备数据
radar_data = {}
for metric in metrics:
    radar_data[metric] = {}
    for model in models_order:
        values = []
        for dataset in datasets:
            model_data = df_filtered[(df_filtered['dataset'] == dataset) & 
                                     (df_filtered['method'] == model)]
            if len(model_data) > 0:
                values.append(model_data.iloc[0][metric])
            else:
                values.append(0)
        radar_data[metric][model] = values

# 创建4个子图的雷达图
fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(projection='polar'))

# 设置角度（逆时针旋转20度）
num_vars = len(datasets)
rotation_angle = np.radians(0)  # 20度转换为弧度
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles = [angle + rotation_angle for angle in angles]  # 逆时针旋转20度
angles += angles[:1]  # 闭合

# 分组绘制
baseline_models = ['PCA', 'UMAP', 'scVI']
pretrained_models = ['CellPLM', 'Geneformer', 'LangCell', 'scGPT']
our_models = ['scMulan', 'scFoundation']

# 为每个指标创建雷达图
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # 为每个模型绘制雷达图
    for model in models_order:
        values = radar_data[metric][model]
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
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    if idx == 0:
        # 只在第一个子图显示完整标签
        ax.set_xticklabels([dataset_labels[d] for d in datasets], 
                          fontsize=12, fontweight='bold')
    else:
        # 其他子图显示简短标签或不显示
        ax.set_xticklabels([dataset_labels[d] for d in datasets], 
                          fontsize=12, fontweight='bold')
    
    # 设置y轴范围和刻度
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, color='gray')
    
    # 优化网格样式
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8, color='gray')
    # 去除最外圈的圆线
    ax.spines['polar'].set_visible(False)
    
    # 设置背景色
    ax.set_facecolor('#fafafa')
    
    # 添加子图标题
    ax.set_title(metric_titles[metric], fontsize=15, fontweight='bold', pad=15)

# 添加图例（横向排列放到子图上方）- 去掉黑框
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
           fontsize=14, frameon=False, ncol=9,
           columnspacing=1.0, handlelength=2.0)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存图片
output_path = '/usr/data/yongjie/output/clustering_radar_4metrics.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"4指标雷达图已保存至: {output_path}")

plt.close()
