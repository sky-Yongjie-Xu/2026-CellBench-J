import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('/usr/data/yongjie/output/xyj_bench_reconstruction_stats.csv')

# 解析mean±std列
def parse_mean_std(value):
    parts = str(value).split('±')
    mean = float(parts[0])
    std = float(parts[1]) if len(parts) > 1 else 0.0
    return mean, std

df[['mse_mean', 'mse_std']] = df['MSE_expr_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))
df[['pearson_mean', 'pearson_std']] = df['Pearson_expr_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))

# 所有9个数据集
datasets = ['pbmc12k', 'hPancreas', 'cortex', 'ms', 'liver', 'Zheng68k', 'lung', 'covid', 'immune']
df_filtered = df[df['dataset'].isin(datasets)].copy()

# 定义模型顺序（按照用户示例）
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

# 使用真实的top值（只使用Top100, Top500, Top700, Top900）
top_mapping = {
    'Top100': 'top100',
    'Top500': 'top500',
    'Top700': 'top700',
    'Top900': 'top900'
}

# 准备数据
table_data = {}
for dataset in datasets:
    table_data[dataset] = {}
    for model in models_order:
        model_data = df_filtered[(df_filtered['dataset'] == dataset) & 
                                 (df_filtered['method'] == model)]
        if len(model_data) == 0:
            continue
        
        row_data = {'mse': {}, 'pearson': {}}
        
        # 提取各个top值的数据
        for top_label, top_value in top_mapping.items():
            top_row = model_data[model_data['top'] == top_value]
            if len(top_row) > 0:
                row_data['mse'][top_label] = (top_row.iloc[0]['mse_mean'], top_row.iloc[0]['mse_std'])
                row_data['pearson'][top_label] = (top_row.iloc[0]['pearson_mean'], top_row.iloc[0]['pearson_std'])
        
        # 计算Mean（使用所有top值的平均值）
        if row_data['mse']:
            mse_values = [v[0] for v in row_data['mse'].values()]
            pearson_values = [v[0] for v in row_data['pearson'].values()]
            row_data['mse']['Mean'] = np.mean(mse_values)
            row_data['pearson']['Mean'] = np.mean(pearson_values)
        
        table_data[dataset][model] = row_data

# 找出最佳值（用于高亮）
def find_best_values(dataset_name):
    best_mse = {}
    best_pearson = {}
    
    for top_label in ['Top100', 'Top500', 'Top700', 'Top900', 'Mean']:
        best_mse_val = float('inf')
        best_pearson_val = -float('inf')
        
        for model in models_order:
            if model in table_data[dataset_name]:
                if top_label in table_data[dataset_name][model]['mse']:
                    mse_val = table_data[dataset_name][model]['mse'][top_label]
                    if isinstance(mse_val, tuple):
                        mse_val = mse_val[0]
                    if mse_val < best_mse_val:
                        best_mse_val = mse_val
                        best_mse[top_label] = model
                
                if top_label in table_data[dataset_name][model]['pearson']:
                    pearson_val = table_data[dataset_name][model]['pearson'][top_label]
                    if isinstance(pearson_val, tuple):
                        pearson_val = pearson_val[0]
                    if pearson_val > best_pearson_val:
                        best_pearson_val = pearson_val
                        best_pearson[top_label] = model
    
    return best_mse, best_pearson

# 计算每个数据集的模型数量
def count_models(dataset_name):
    return len([m for m in models_order if m in table_data[dataset_name]])

# 生成LaTeX表格
latex_lines = []
latex_lines.append("\\begin{table*}[htbp]")
latex_lines.append("")
latex_lines.append("  \\centering")
latex_lines.append("")
latex_lines.append("  \\renewcommand\\arraystretch{1.1}")
latex_lines.append("")
latex_lines.append("  \\scriptsize")
latex_lines.append("")
latex_lines.append("  \\caption{Performance of various models on gene reconstruction task. Metrics grouped by MSE (↓) and Pearson (↑), with results reported across Top100/500/700/900, and the mean.}")
latex_lines.append("")
latex_lines.append("  \\label{tab:topk_grouped_metric_all}")
latex_lines.append("")
latex_lines.append("  \\setlength{\\tabcolsep}{4pt}")
latex_lines.append("")
latex_lines.append("  \\begin{tabular}{cl|ccccc|ccccc}")
latex_lines.append("")
latex_lines.append("    \\toprule")
latex_lines.append("    \\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Model}")
latex_lines.append("      & \\multicolumn{5}{c|}{MSE ↓}")
latex_lines.append("      & \\multicolumn{5}{c}{Pearson ↑} \\\\")
latex_lines.append("    \\cmidrule(lr){3-7} \\cmidrule(lr){8-12}")
latex_lines.append("     & & Top100 & Top500 & Top700 & Top900 & Mean")
latex_lines.append("       & Top100 & Top500 & Top700 & Top900 & Mean \\\\")
latex_lines.append("    \\midrule")
latex_lines.append("")

# 为每个数据集生成行
for idx, dataset in enumerate(datasets):
    dataset_label = dataset_labels[dataset]
    best_mse, best_pearson = find_best_values(dataset)
    num_models = count_models(dataset)
    
    # 在第5个数据集（liver）之后插入分页
    # 数据集顺序：pbmc12k, hPancreas, cortex, ms, liver, Zheng68k, lung, covid, immune
    # 索引4是liver，在它之后分页
    if idx == 5:  # Zheng68k之前插入分页（前5个数据集在第一页，后4个在第二页）
        latex_lines.append("    \\end{tabular}")
        latex_lines.append("  \\end{table*}")
        latex_lines.append("")
        latex_lines.append("\\begin{table*}[htbp]")
        latex_lines.append("")
        latex_lines.append("  \\centering")
        latex_lines.append("")
        latex_lines.append("  \\renewcommand\\arraystretch{1.1}")
        latex_lines.append("")
        latex_lines.append("  \\scriptsize")
        latex_lines.append("")
        latex_lines.append("  \\caption{Performance of various models on gene reconstruction task (continued). Metrics grouped by MSE (↓) and Pearson (↑), with results reported across Top100/500/700/900, and the mean.}")
        latex_lines.append("")
        latex_lines.append("  \\label{tab:topk_grouped_metric_all_cont}")
        latex_lines.append("")
        latex_lines.append("  \\setlength{\\tabcolsep}{4pt}")
        latex_lines.append("")
        latex_lines.append("  \\begin{tabular}{cl|ccccc|ccccc}")
        latex_lines.append("")
        latex_lines.append("    \\toprule")
        latex_lines.append("    \\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Model}")
        latex_lines.append("      & \\multicolumn{5}{c|}{MSE ↓}")
        latex_lines.append("      & \\multicolumn{5}{c}{Pearson ↑} \\\\")
        latex_lines.append("    \\cmidrule(lr){3-7} \\cmidrule(lr){8-12}")
        latex_lines.append("     & & Top100 & Top500 & Top700 & Top900 & Mean")
        latex_lines.append("       & Top100 & Top500 & Top700 & Top900 & Mean \\\\")
        latex_lines.append("    \\midrule")
        latex_lines.append("")
    
    latex_lines.append(f"    \\multirow{{{num_models}}}{{*}}{{{dataset_label}}}")
    latex_lines.append("")
    
    for model in models_order:
        if model not in table_data[dataset]:
            continue
        
        model_label = model
        if model == 'UMAP':
            model_label = 'umap'
        elif model == 'scVI':
            model_label = 'SCVI'
        
        data = table_data[dataset][model]
        
        # 检查是否需要高亮整行（scMulan或scFoundation）
        highlight_row = (model == 'scMulan' or model == 'scFoundation')
        
        # 生成MSE列
        mse_cells = []
        for top_label in ['Top100', 'Top500', 'Top700', 'Top900', 'Mean']:
            if top_label in data['mse']:
                mse_val = data['mse'][top_label]
                if isinstance(mse_val, tuple):
                    mean_val, std_val = mse_val
                    cell_text = f"{mean_val:.3f}$\\pm${std_val:.3f}"
                else:
                    cell_text = f"{mse_val:.3f}"
                
                # 检查是否需要高亮这个单元格
                if model == best_mse.get(top_label):
                    cell_text = f"\\cellcolor{{gray!25}}{cell_text}"
                
                mse_cells.append(cell_text)
            else:
                mse_cells.append("—")
        
        # 生成Pearson列
        pearson_cells = []
        for top_label in ['Top100', 'Top500', 'Top700', 'Top900', 'Mean']:
            if top_label in data['pearson']:
                pearson_val = data['pearson'][top_label]
                if isinstance(pearson_val, tuple):
                    mean_val, std_val = pearson_val
                    cell_text = f"{mean_val:.3f}$\\pm${std_val:.3f}"
                else:
                    cell_text = f"{pearson_val:.3f}"
                
                # 检查是否需要高亮这个单元格
                if model == best_pearson.get(top_label):
                    cell_text = f"\\cellcolor{{gray!25}}{cell_text}"
                
                pearson_cells.append(cell_text)
            else:
                pearson_cells.append("—")
        
        # 组合行
        if highlight_row:
            model_cell = f"    & \\cellcolor{{gray!25}}\\textbf{{{model_label}}}"
        else:
            model_cell = f"    & {model_label}"
        
        row = model_cell + "      & " + " & ".join(mse_cells) + "      & " + " & ".join(pearson_cells) + " \\\\"
        latex_lines.append(row)
    
    latex_lines.append("    \\midrule")
    latex_lines.append("")

latex_lines.append("    \\bottomrule")
latex_lines.append("  \\end{tabular}")
latex_lines.append("\\end{table*}")

# 输出LaTeX代码
latex_code = "\n".join(latex_lines)
print(latex_code)

# 保存到文件
with open('/usr/data/yongjie/output/reconstruction_table_all_datasets.tex', 'w') as f:
    f.write(latex_code)

print("\n\nLaTeX表格已保存至: /usr/data/yongjie/output/reconstruction_table_all_datasets.tex")

