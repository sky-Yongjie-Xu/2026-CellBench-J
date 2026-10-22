import pandas as pd
import numpy as np

# 读取CSV文件
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

# 准备数据
table_data = {}
for dataset in datasets:
    table_data[dataset] = {}
    for model in models_order:
        model_data = df_filtered[(df_filtered['dataset'] == dataset) & 
                                 (df_filtered['method'] == model)]
        if len(model_data) > 0:
            row = model_data.iloc[0]
            table_data[dataset][model] = {
                'nmi': row['nmi'],
                'ari': row['ari'],
                'asw': row['asw'],
                'avgbio': row['AvgBio']
            }

# 找出最佳值（用于高亮）
def find_best_values(dataset_name):
    best_nmi = None
    best_ari = None
    best_asw = None
    best_avgbio = None
    best_nmi_val = -float('inf')
    best_ari_val = -float('inf')
    best_asw_val = -float('inf')
    best_avgbio_val = -float('inf')
    
    for model in models_order:
        if model in table_data[dataset_name]:
            nmi_val = table_data[dataset_name][model]['nmi']
            ari_val = table_data[dataset_name][model]['ari']
            asw_val = table_data[dataset_name][model]['asw']
            avgbio_val = table_data[dataset_name][model]['avgbio']
            
            if nmi_val > best_nmi_val:
                best_nmi_val = nmi_val
                best_nmi = model
            
            if ari_val > best_ari_val:
                best_ari_val = ari_val
                best_ari = model
            
            if asw_val > best_asw_val:
                best_asw_val = asw_val
                best_asw = model
            
            if avgbio_val > best_avgbio_val:
                best_avgbio_val = avgbio_val
                best_avgbio = model
    
    return best_nmi, best_ari, best_asw, best_avgbio

# 计算每个数据集的模型数量
def count_models(dataset_name):
    return len([m for m in models_order if m in table_data[dataset_name]])

# 生成LaTeX表格 - 2行布局：第一行5个数据集，第二行4个数据集
latex_lines = []
latex_lines.append("\\begin{table*}[htbp]")
latex_lines.append("")
latex_lines.append("  \\centering")
latex_lines.append("")
latex_lines.append("  \\scriptsize")
latex_lines.append("")
latex_lines.append("  \\caption{Performance of various models on clustering task (NMI, ARI, ASW, AvgBio).}")
latex_lines.append("")
latex_lines.append("  \\label{tab:clustering_performance_all}")
latex_lines.append("")
latex_lines.append("  \\setlength{\\tabcolsep}{2pt}")
latex_lines.append("")

# 生成表头 - 每行5个数据集，共2行
# 将9个数据集分成2组，第一组5个，第二组4个
datasets_groups = [
    datasets[0:5],  # pbmc12k, hPancreas, cortex, ms, liver
    datasets[5:9]   # Zheng68k, lung, covid, immune
]

# 第一组（5个数据集）
latex_lines.append("  \\begin{tabular}{l|cccc|cccc|cccc|cccc|cccc}")
latex_lines.append("")
latex_lines.append("    \\toprule")
latex_lines.append("    \\multirow{2}{*}{Model}")

# 第一行表头：数据集名称
header_parts1 = []
for dataset in datasets_groups[0]:
    dataset_label = dataset_labels[dataset]
    header_parts1.append(f"\\multicolumn{{4}}{{c}}{{{dataset_label}}}")

latex_lines.append("      & " + " & ".join(header_parts1) + " \\\\")
latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\\cmidrule(lr){14-17}\\cmidrule(lr){18-21}")

# 第二行表头：4个指标
header_line2 = "      & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio"
latex_lines.append(header_line2 + " \\\\")
latex_lines.append("    \\midrule")
latex_lines.append("")

# 为每个模型生成行（第一组2个数据集）
for model in models_order:
    model_exists = any(model in table_data[dataset] for dataset in datasets_groups[0])
    if not model_exists:
        continue
    
    model_label = model
    if model == 'UMAP':
        model_label = 'UMAP'
    elif model == 'scVI':
        model_label = 'scVI'
    
    cells = []
    for dataset in datasets_groups[0]:
        best_nmi, best_ari, best_asw, best_avgbio = find_best_values(dataset)
        
        if model in table_data[dataset]:
            data = table_data[dataset][model]
            nmi_val = data['nmi'] * 100
            ari_val = data['ari'] * 100
            asw_val = data['asw'] * 100
            avgbio_val = data['avgbio'] * 100
            
            nmi_text = f"{nmi_val:.2f}"
            if model == best_nmi:
                nmi_text = f"\\cellcolor{{gray!25}}{{{nmi_text}}}"
            
            ari_text = f"{ari_val:.2f}"
            if model == best_ari:
                ari_text = f"\\cellcolor{{gray!25}}{{{ari_text}}}"
            
            asw_text = f"{asw_val:.2f}"
            if model == best_asw:
                asw_text = f"\\cellcolor{{gray!25}}{{{asw_text}}}"
            
            avgbio_text = f"{avgbio_val:.2f}"
            if model == best_avgbio:
                avgbio_text = f"\\cellcolor{{gray!25}}{{{avgbio_text}}}"
            
            cells.append(nmi_text)
            cells.append(ari_text)
            cells.append(asw_text)
            cells.append(avgbio_text)
        else:
            cells.append("—")
            cells.append("—")
            cells.append("—")
            cells.append("—")
    
    model_cell = f"    {model_label}"
    row = model_cell + "      & " + " & ".join(cells) + " \\\\"
    latex_lines.append(row)

latex_lines.append("    \\bottomrule")
latex_lines.append("  \\end{tabular}")
latex_lines.append("")

# 生成其余4组表格
for group_idx in range(1, len(datasets_groups)):
    latex_lines.append("  \\vspace{0.3cm}")
    latex_lines.append("")
    
    current_group = datasets_groups[group_idx]
    num_datasets_in_group = len(current_group)
    
    # 根据数据集数量调整表格列数
    if num_datasets_in_group == 5:
        latex_lines.append("  \\begin{tabular}{l|cccc|cccc|cccc|cccc|cccc}")
    elif num_datasets_in_group == 4:
        latex_lines.append("  \\begin{tabular}{l|cccc|cccc|cccc|cccc}")
    else:  # 只有1个数据集
        latex_lines.append("  \\begin{tabular}{l|cccc}")
    
    latex_lines.append("")
    latex_lines.append("    \\toprule")
    latex_lines.append("    \\multirow{2}{*}{Model}")
    
    # 表头：数据集名称
    header_parts = []
    for dataset in current_group:
        dataset_label = dataset_labels[dataset]
        header_parts.append(f"\\multicolumn{{4}}{{c}}{{{dataset_label}}}")
    
    latex_lines.append("      & " + " & ".join(header_parts) + " \\\\")
    
    # cmidrule
    if num_datasets_in_group == 5:
        latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\\cmidrule(lr){14-17}\\cmidrule(lr){18-21}")
        header_line = "      & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio"
    elif num_datasets_in_group == 4:
        latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\\cmidrule(lr){14-17}")
        header_line = "      & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio & NMI & ARI & ASW & AvgBio"
    else:
        latex_lines.append("    \\cmidrule(lr){2-5}")
        header_line = "      & NMI & ARI & ASW & AvgBio"
    
    latex_lines.append(header_line + " \\\\")
    latex_lines.append("    \\midrule")
    latex_lines.append("")
    
    # 为每个模型生成行
    for model in models_order:
        model_exists = any(model in table_data[dataset] for dataset in current_group)
        if not model_exists:
            continue
        
        model_label = model
        if model == 'UMAP':
            model_label = 'UMAP'
        elif model == 'scVI':
            model_label = 'scVI'
        
        cells = []
        for dataset in current_group:
            best_nmi, best_ari, best_asw, best_avgbio = find_best_values(dataset)
            
            if model in table_data[dataset]:
                data = table_data[dataset][model]
                nmi_val = data['nmi'] * 100
                ari_val = data['ari'] * 100
                asw_val = data['asw'] * 100
                avgbio_val = data['avgbio'] * 100
                
                nmi_text = f"{nmi_val:.2f}"
                if model == best_nmi:
                    nmi_text = f"\\cellcolor{{gray!25}}{{{nmi_text}}}"
                
                ari_text = f"{ari_val:.2f}"
                if model == best_ari:
                    ari_text = f"\\cellcolor{{gray!25}}{{{ari_text}}}"
                
                asw_text = f"{asw_val:.2f}"
                if model == best_asw:
                    asw_text = f"\\cellcolor{{gray!25}}{{{asw_text}}}"
                
                avgbio_text = f"{avgbio_val:.2f}"
                if model == best_avgbio:
                    avgbio_text = f"\\cellcolor{{gray!25}}{{{avgbio_text}}}"
                
                cells.append(nmi_text)
                cells.append(ari_text)
                cells.append(asw_text)
                cells.append(avgbio_text)
            else:
                cells.append("—")
                cells.append("—")
                cells.append("—")
                cells.append("—")
        
        model_cell = f"    {model_label}"
        row = model_cell + "      & " + " & ".join(cells) + " \\\\"
        latex_lines.append(row)
    
    latex_lines.append("    \\bottomrule")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append("")
latex_lines.append("\\end{table*}")

# 输出LaTeX代码
latex_code = "\n".join(latex_lines)
print(latex_code)

# 保存到文件
with open('/usr/data/yongjie/output/clustering_table_all_datasets.tex', 'w') as f:
    f.write(latex_code)

print("\n\nLaTeX表格已保存至: /usr/data/yongjie/output/clustering_table_all_datasets.tex")

