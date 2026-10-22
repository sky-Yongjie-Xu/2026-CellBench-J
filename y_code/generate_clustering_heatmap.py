import csv
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data_dict = {}
with open('/usr/data/yongjie/output/xyj_bench_clustering_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['seed'] != '41':
            continue
        dataset = row['dataset']
        method = row['method']
        avgbio = float(row['AvgBio'])
        
        if dataset not in data_dict:
            data_dict[dataset] = {}
        data_dict[dataset][method] = avgbio

# 定义数据集和方法顺序
dataset_order = ['pbmc12k', 'hPancreas', 'cortex', 'ms', 'liver', 'Zheng68k', 'lung', 'covid', 'immune']
method_order = ['PCA', 'UMAP', 'scVI', 'Geneformer', 'CellPLM', 'scMulan', 'scGPT', 'LangCell', 'scFoundation']

# 创建数据矩阵
data_matrix = []
for dataset in dataset_order:
    row = []
    for method in method_order:
        if dataset in data_dict and method in data_dict[dataset]:
            val = data_dict[dataset][method]
            row.append(val)
        else:
            row.append(np.nan)
    data_matrix.append(row)
data_matrix = np.array(data_matrix)

# 归一化颜色映射
min_val = np.nanmin(data_matrix)
max_val = np.nanmax(data_matrix)

# 定义形状映射和颜色
def get_shape_and_color(value):
    if np.isnan(value):
        return None, None
    
    # 归一化到0-1
    normalized = (value - min_val) / (max_val - min_val)
    
    # 根据数值范围选择形状
    if value < 0.5:
        shape = 'o'  # 圆形
    elif value < 0.6:
        shape = 's'  # 方形
    elif value < 0.7:
        shape = '^'  # 三角形
    elif value < 0.8:
        shape = 'D'  # 菱形
    else:
        shape = 'p'  # 五边形
    
    # 颜色从浅蓝到深蓝
    color = plt.cm.Blues(normalized)
    
    return shape, color, normalized

# 绘制热图
fig, ax = plt.subplots(figsize=(14, 10))

# 设置背景色
ax.set_facecolor('white')

# 绘制形状 - 更大尺寸
for i in range(len(dataset_order)):
    for j in range(len(method_order)):
        val = data_matrix[i, j]
        shape, color, norm_val = get_shape_and_color(val)
        
        if shape:
            x_pos = j
            y_pos = i
            ax.scatter(x_pos, y_pos, marker=shape, s=2000, 
                      color=color, edgecolors='black', linewidths=1.5, alpha=0.9)

# 移除数值文本显示

# 设置刻度
ax.set_xticks(np.arange(len(method_order)))
ax.set_yticks(np.arange(len(dataset_order)))
ax.set_xticklabels(method_order, fontsize=18, fontweight='bold', rotation=45, ha='right')
ax.set_yticklabels(dataset_order, fontsize=18, fontweight='bold')

# 添加网格
ax.set_xticks(np.arange(len(method_order)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(dataset_order)) - 0.5, minor=True)
ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
ax.tick_params(which="minor", size=0)

# 反转y轴
ax.invert_yaxis()

ax.set_title('Clustering Performance Heatmap (AvgBio)', fontsize=20, fontweight='bold', pad=15)
ax.set_xlabel('Method', fontsize=18, fontweight='bold')
ax.set_ylabel('Dataset', fontsize=18, fontweight='bold')

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min_val, vmax=max_val))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='AvgBio', shrink=0.8)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('AvgBio', fontsize=18, fontweight='bold')

plt.tight_layout()
plt.savefig('clustering_heatmap_shapes.png', dpi=300, bbox_inches='tight')
print("热图已保存到: clustering_heatmap_shapes.png")

# 创建图例
fig_legend, ax_legend = plt.subplots(figsize=(8, 5))
ax_legend.axis('off')

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEF0', 
               markersize=20, markeredgecolor='black', markeredgewidth=1.5,
               label='AvgBio < 0.5 (Circle)', linewidth=0),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#5B9BD5', 
               markersize=20, markeredgecolor='black', markeredgewidth=1.5,
               label='0.5 ≤ AvgBio < 0.6 (Square)', linewidth=0),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#3665B3', 
               markersize=20, markeredgecolor='black', markeredgewidth=1.5,
               label='0.6 ≤ AvgBio < 0.7 (Triangle)', linewidth=0),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#1F3A8F', 
               markersize=20, markeredgecolor='black', markeredgewidth=1.5,
               label='0.7 ≤ AvgBio < 0.8 (Diamond)', linewidth=0),
    plt.Line2D([0], [0], marker='p', color='w', markerfacecolor='#0A0F5C', 
               markersize=25, markeredgecolor='black', markeredgewidth=1.5,
               label='AvgBio ≥ 0.8 (Pentagon)', linewidth=0)
]

ax_legend.legend(handles=legend_elements, loc='center', fontsize=18, 
                frameon=False, ncol=1)
ax_legend.set_title('AvgBio Range Symbols', fontsize=20, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('clustering_heatmap_legend.png', dpi=300, bbox_inches='tight')
print("图例已保存到: clustering_heatmap_legend.png")

