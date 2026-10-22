import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('output/xyj_bench_batch_correction_results.csv')

# 所有数据集（根据实际数据调整）
datasets = ['uc_imm', 'uc_epi']  # 当前只有2个数据集，如果有更多数据集可以添加
df_filtered = df[df['dataset'].isin(datasets)].copy()

# 定义模型顺序（10个模型）
models_order = ['PCA', 'UMAP', 'scVI', 'CellPLM', 'Geneformer', 'LangCell', 'scGPT', 'scMulan', 'scFoundation', 'Nicheformer']

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
    'immune': 'Immune',
    'uc_imm': 'UC\\_Imm',
    'uc_epi': 'UC\\_Epi'
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
                'ASW_celltype': row['ASW_celltype'],
                '1-ASW_batch': row['1-ASW_batch'],
                'ilisi_batch': row['ilisi_batch'],
                '1-cLISI_mean': row['1-cLISI_mean']
            }

# 找出最佳值（用于高亮）
def find_best_values(dataset_name):
    best_asw_celltype = None
    best_asw_batch = None
    best_ilisi = None
    best_clisi = None
    best_asw_celltype_val = -float('inf')
    best_asw_batch_val = -float('inf')
    best_ilisi_val = -float('inf')
    best_clisi_val = -float('inf')
    
    for model in models_order:
        if model in table_data[dataset_name]:
            asw_celltype_val = table_data[dataset_name][model]['ASW_celltype']
            asw_batch_val = table_data[dataset_name][model]['1-ASW_batch']
            ilisi_val = table_data[dataset_name][model]['ilisi_batch']
            clisi_val = table_data[dataset_name][model]['1-cLISI_mean']
            
            if asw_celltype_val > best_asw_celltype_val:
                best_asw_celltype_val = asw_celltype_val
                best_asw_celltype = model
            
            if asw_batch_val > best_asw_batch_val:
                best_asw_batch_val = asw_batch_val
                best_asw_batch = model
            
            if ilisi_val > best_ilisi_val:
                best_ilisi_val = ilisi_val
                best_ilisi = model
            
            if clisi_val > best_clisi_val:
                best_clisi_val = clisi_val
                best_clisi = model
    
    return best_asw_celltype, best_asw_batch, best_ilisi, best_clisi

# 计算每个数据集的模型数量
def count_models(dataset_name):
    return len([m for m in models_order if m in table_data[dataset_name]])

# 生成LaTeX表格 - 根据数据集数量动态调整布局
latex_lines = []
latex_lines.append("\\begin{table*}[htbp]")
latex_lines.append("")
latex_lines.append("  \\centering")
latex_lines.append("")
latex_lines.append("  \\scriptsize")
latex_lines.append("")
latex_lines.append("  \\caption{Performance of various models on batch correction task (ASW\\_celltype, 1-ASW\\_batch, iLISI\\_batch, 1-cLISI).}")
latex_lines.append("")
latex_lines.append("  \\label{tab:correction_performance_all}")
latex_lines.append("")
latex_lines.append("  \\setlength{\\tabcolsep}{2pt}")
latex_lines.append("")

# 生成表头 - 根据数据集数量分组
# 如果数据集数量<=5，只生成一个表格；如果>5，分成两组
if len(datasets) <= 5:
    datasets_groups = [datasets]
else:
    datasets_groups = [
        datasets[0:5],  # 前5个数据集
        datasets[5:]    # 剩余数据集
    ]

# 第一组数据集
current_group = datasets_groups[0]
num_datasets_in_group = len(current_group)

# 根据数据集数量调整表格列数
if num_datasets_in_group == 5:
    latex_lines.append("  \\begin{tabular}{l|cccc|cccc|cccc|cccc|cccc}")
elif num_datasets_in_group == 4:
    latex_lines.append("  \\begin{tabular}{l|cccc|cccc|cccc|cccc}")
elif num_datasets_in_group == 3:
    latex_lines.append("  \\begin{tabular}{l|cccc|cccc|cccc}")
elif num_datasets_in_group == 2:
    latex_lines.append("  \\begin{tabular}{l|cccc|cccc}")
else:  # 只有1个数据集
    latex_lines.append("  \\begin{tabular}{l|cccc}")

latex_lines.append("")
latex_lines.append("    \\toprule")
latex_lines.append("    \\multirow{2}{*}{Model}")

# 第一行表头：数据集名称
header_parts1 = []
for dataset in current_group:
    dataset_label = dataset_labels.get(dataset, dataset)
    header_parts1.append(f"\\multicolumn{{4}}{{c}}{{{dataset_label}}}")

latex_lines.append("      & " + " & ".join(header_parts1) + " \\\\")

# cmidrule - 根据数据集数量动态生成
if num_datasets_in_group == 5:
    latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\\cmidrule(lr){14-17}\\cmidrule(lr){18-21}")
elif num_datasets_in_group == 4:
    latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\\cmidrule(lr){14-17}")
elif num_datasets_in_group == 3:
    latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}")
elif num_datasets_in_group == 2:
    latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}")
else:
    latex_lines.append("    \\cmidrule(lr){2-5}")

# 第二行表头：4个指标（重复每个数据集）
header_parts2 = []
for _ in current_group:
    header_parts2.append("ASW\\_ct & 1-ASW\\_b & iLISI & 1-cLISI")
header_line2 = "      & " + " & ".join(header_parts2)
latex_lines.append(header_line2 + " \\\\")
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
        best_asw_celltype, best_asw_batch, best_ilisi, best_clisi = find_best_values(dataset)
        
        if model in table_data[dataset]:
            data = table_data[dataset][model]
            # ASW_celltype 和 1-ASW_batch 转换为百分比
            asw_celltype_val = data['ASW_celltype'] * 100
            asw_batch_val = data['1-ASW_batch'] * 100
            # ilisi_batch 和 1-cLISI_mean 保持原始值
            ilisi_val = data['ilisi_batch']
            clisi_val = data['1-cLISI_mean']
            
            asw_celltype_text = f"{asw_celltype_val:.2f}"
            if model == best_asw_celltype:
                asw_celltype_text = f"\\cellcolor{{gray!25}}{{{asw_celltype_text}}}"
            
            asw_batch_text = f"{asw_batch_val:.2f}"
            if model == best_asw_batch:
                asw_batch_text = f"\\cellcolor{{gray!25}}{{{asw_batch_text}}}"
            
            ilisi_text = f"{ilisi_val:.2f}"
            if model == best_ilisi:
                ilisi_text = f"\\cellcolor{{gray!25}}{{{ilisi_text}}}"
            
            clisi_text = f"{clisi_val:.2f}"
            if model == best_clisi:
                clisi_text = f"\\cellcolor{{gray!25}}{{{clisi_text}}}"
            
            cells.append(asw_celltype_text)
            cells.append(asw_batch_text)
            cells.append(ilisi_text)
            cells.append(clisi_text)
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

# 生成其余组表格（如果有）
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
    elif num_datasets_in_group == 3:
        latex_lines.append("  \\begin{tabular}{l|cccc|cccc|cccc}")
    elif num_datasets_in_group == 2:
        latex_lines.append("  \\begin{tabular}{l|cccc|cccc}")
    else:  # 只有1个数据集
        latex_lines.append("  \\begin{tabular}{l|cccc}")
    
    latex_lines.append("")
    latex_lines.append("    \\toprule")
    latex_lines.append("    \\multirow{2}{*}{Model}")
    
    # 表头：数据集名称
    header_parts = []
    for dataset in current_group:
        dataset_label = dataset_labels.get(dataset, dataset)
        header_parts.append(f"\\multicolumn{{4}}{{c}}{{{dataset_label}}}")
    
    latex_lines.append("      & " + " & ".join(header_parts) + " \\\\")
    
    # cmidrule
    if num_datasets_in_group == 5:
        latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\\cmidrule(lr){14-17}\\cmidrule(lr){18-21}")
    elif num_datasets_in_group == 4:
        latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\\cmidrule(lr){14-17}")
    elif num_datasets_in_group == 3:
        latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}")
    elif num_datasets_in_group == 2:
        latex_lines.append("    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}")
    else:
        latex_lines.append("    \\cmidrule(lr){2-5}")
    
    # 第二行表头
    header_parts2 = []
    for _ in current_group:
        header_parts2.append("ASW\\_ct & 1-ASW\\_b & iLISI & 1-cLISI")
    header_line = "      & " + " & ".join(header_parts2)
    
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
            best_asw_celltype, best_asw_batch, best_ilisi, best_clisi = find_best_values(dataset)
            
            if model in table_data[dataset]:
                data = table_data[dataset][model]
                # ASW_celltype 和 1-ASW_batch 转换为百分比
                asw_celltype_val = data['ASW_celltype'] * 100
                asw_batch_val = data['1-ASW_batch'] * 100
                # ilisi_batch 和 1-cLISI_mean 保持原始值
                ilisi_val = data['ilisi_batch']
                clisi_val = data['1-cLISI_mean']
                
                asw_celltype_text = f"{asw_celltype_val:.2f}"
                if model == best_asw_celltype:
                    asw_celltype_text = f"\\cellcolor{{gray!25}}{{{asw_celltype_text}}}"
                
                asw_batch_text = f"{asw_batch_val:.2f}"
                if model == best_asw_batch:
                    asw_batch_text = f"\\cellcolor{{gray!25}}{{{asw_batch_text}}}"
                
                ilisi_text = f"{ilisi_val:.2f}"
                if model == best_ilisi:
                    ilisi_text = f"\\cellcolor{{gray!25}}{{{ilisi_text}}}"
                
                clisi_text = f"{clisi_val:.2f}"
                if model == best_clisi:
                    clisi_text = f"\\cellcolor{{gray!25}}{{{clisi_text}}}"
                
                cells.append(asw_celltype_text)
                cells.append(asw_batch_text)
                cells.append(ilisi_text)
                cells.append(clisi_text)
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
with open('output/correction_table_all_datasets.tex', 'w') as f:
    f.write(latex_code)

print("\n\nLaTeX表格已保存至: output/correction_table_all_datasets.tex")
