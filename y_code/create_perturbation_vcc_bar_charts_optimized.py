import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 读取数据
df = pd.read_csv('/usr/data/yongjie/output/perturbation_vcc/perturbation_vcc_stats.csv')

# 定义指标
metrics = {
    'DES': 'DES_mean±std',
    'MAE': 'MAE_mean±std', 
    'PDS': 'PDS_mean±std'
}

# 解析 mean±std 格式
def parse_mean_std(value_str):
    if pd.isna(value_str):
        return np.nan, np.nan
    parts = str(value_str).split('±')
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    return float(parts[0]), 0.0

# 为每个指标解析数据
for metric_name, metric_col in metrics.items():
    df[[f'{metric_name}_mean', f'{metric_name}_std']] = df[metric_col].apply(lambda x: pd.Series(parse_mean_std(x)))

# 筛选 adamson 数据集
df_adamson = df[df['dataset'] == 'adamson'].copy()

# 定义模型顺序和top值
models_order = ['PCA', 'UMAP', 'scVI', 'scGPT', 'CellPLM', 'Geneformer', 'LangCell', 'scMulan', 'scFoundation']
tops = ['top1', 'top3', 'top5', 'top7', 'top9']

# 将十六进制颜色转换为RGB
def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

# 生成颜色渐变
def get_color_gradient(base_color, n_colors=5):
    """Generate gradient colors from darker to lighter"""
    colors = []
    for i in range(n_colors):
        # 从较深到较浅 (0.6 到 1.0)
        factor = 0.6 + (i * 0.4 / (n_colors - 1))
        # 混合白色来增加亮度
        colors.append(tuple(base_color[j] + (1.0 - base_color[j]) * (1 - factor) for j in range(3)))
    return colors

# 使用与折线图相同的基础颜色（与 plot_perturbation_stats_comparison.py 相同）
base_colors_hex = {
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

# 转换为RGB格式
base_colors = {model: hex_to_rgb(color) for model, color in base_colors_hex.items()}

# 创建图表
metric_titles = {
    'DES': 'DES (↑)',
    'MAE': 'MAE (↓)',
    'PDS': 'PDS (↑)'
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (metric_name, metric_key) in enumerate(metrics.items()):
    ax = axes[idx]
    
    # 为每个模型绘制柱状图
    x_positions = np.arange(len(models_order))
    bar_width = 0.15
    
    for i, top in enumerate(tops):
        values = []
        colors_for_this_top = []
        
        for model in models_order:
            model_data = df_adamson[(df_adamson['method'] == model) & (df_adamson['top'] == top)]
            if not model_data.empty:
                values.append(model_data[f'{metric_name}_mean'].values[0])
            else:
                values.append(0)
            
            # 为每个模型生成渐变色
            gradient = get_color_gradient(base_colors[model], len(tops))
            colors_for_this_top.append(gradient[i])
        
        # 绘制柱状图
        positions = x_positions + (i - 2) * bar_width
        ax.bar(positions, values, bar_width, color=colors_for_this_top, 
               edgecolor='white', linewidth=0.5)
    
    # 设置标题和标签
    ax.set_title(metric_titles[metric_name], fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models_order, rotation=45, ha='right', fontsize=9)
    if idx == 0:
        ax.set_ylabel('Average Value', fontsize=10)
    
    # 自适应y轴
    ax.set_ylim(0, df_adamson[f'{metric_name}_mean'].max() * 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

# 创建图例（使用蓝色渐变表示 top 值）
legend_elements = []
legend_base_color = (0.3, 0.5, 0.9)  # 蓝色用于图例
legend_gradient = get_color_gradient(legend_base_color, len(tops))
for i, top in enumerate(tops):
    legend_elements.append(Patch(facecolor=legend_gradient[i], label=top, edgecolor='none'))

# 调整布局，为图例留出空间
plt.tight_layout(rect=[0, 0, 0.85, 1])

# 将图例放在右侧，更靠近子图
fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.86, 0.5), 
           title='Top Values', fontsize=9, title_fontsize=10, frameon=False)

# 保存图片
output_path = '/usr/data/yongjie/output/perturbation_vcc_bar_charts.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"柱状图已保存至: {output_path}")

plt.close()
