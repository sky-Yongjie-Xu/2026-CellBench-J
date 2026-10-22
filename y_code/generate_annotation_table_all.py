import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('/usr/data/yongjie/output/xyj_bench_annotation_adapter_finetuning_stats.csv')

# 解析mean±std列
def parse_mean_std(value):
    parts = str(value).split('±')
    mean = float(parts[0])
    std = float(parts[1]) if len(parts) > 1 else 0.0
    return mean, std

df[['acc_mean', 'acc_std']] = df['acc_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))
df[['f1_mean', 'f1_std']] = df['f1_m_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))
df[['precision_mean', 'precision_std']] = df['precision_marco_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))
df[['recall_mean', 'recall_std']] = df['recall_marco_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))

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

# 只使用top1, top5, top9
shot_mapping = {
    'Top1': 'top1',
    'Top5': 'top5',
    'Top9': 'top9'
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
        
        row_data = {'acc': {}, 'f1': {}, 'precision': {}, 'recall': {}}
        
        # 提取各个shot值的数据
        for shot_label, shot_value in shot_mapping.items():
            shot_row = model_data[model_data['shot'] == shot_value]
            if len(shot_row) > 0:
                row_data['acc'][shot_label] = (shot_row.iloc[0]['acc_mean'], shot_row.iloc[0]['acc_std'])
                row_data['f1'][shot_label] = (shot_row.iloc[0]['f1_mean'], shot_row.iloc[0]['f1_std'])
                row_data['precision'][shot_label] = (shot_row.iloc[0]['precision_mean'], shot_row.iloc[0]['precision_std'])
                row_data['recall'][shot_label] = (shot_row.iloc[0]['recall_mean'], shot_row.iloc[0]['recall_std'])
        
        table_data[dataset][model] = row_data

# 找出最佳值（用于高亮）- 对于每个指标和每个shot值
def find_best_values(dataset_name):
    best_values = {}
    for metric in ['acc', 'f1', 'precision', 'recall']:
        best_values[metric] = {}
        for shot_label in ['Top1', 'Top5', 'Top9']:
            best_val = -float('inf')
            best_model = None
            
            for model in models_order:
                if model in table_data[dataset_name]:
                    if shot_label in table_data[dataset_name][model][metric]:
                        val = table_data[dataset_name][model][metric][shot_label]
                        if isinstance(val, tuple):
                            val = val[0]
                        if val > best_val:
                            best_val = val
                            best_model = model
            
            best_values[metric][shot_label] = best_model
    
    return best_values

# 计算每个数据集的模型数量
def count_models(dataset_name):
    return len([m for m in models_order if m in table_data[dataset_name]])

# 生成LaTeX表格
latex_lines = []
latex_lines.append("\\begin{table*}[htbp]")
latex_lines.append("")
latex_lines.append("  \\centering")
latex_lines.append("")
latex_lines.append("  \\scriptsize")
latex_lines.append("")
latex_lines.append("  \\caption{Performance of various models on four single-cell benchmarks (Acc\\,\\%, F$_1$\\,\\%, Precision\\,\\%, Recall\\,\\%).}")
latex_lines.append("")
latex_lines.append("  \\label{tab:performance_acc_f1_precision_recall_all}")
latex_lines.append("")
latex_lines.append("  \\setlength{\\tabcolsep}{4pt}")
latex_lines.append("")
latex_lines.append("  \\begin{tabular}{llcccccccccccc}")
latex_lines.append("")
latex_lines.append("    \\toprule")
latex_lines.append("    Dataset   & Model")
latex_lines.append("      & \\multicolumn{4}{c}{Top1}    & \\multicolumn{4}{c}{Top5}")
latex_lines.append("      & \\multicolumn{4}{c}{Top9}    \\\\")
latex_lines.append("    \\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}")
latex_lines.append("              & ")
latex_lines.append("      & Acc    & F$_1$    & Prec    & Recall")
latex_lines.append("      & Acc    & F$_1$    & Prec    & Recall")
latex_lines.append("      & Acc    & F$_1$    & Prec    & Recall   \\\\")
latex_lines.append("    \\midrule")
latex_lines.append("")

# 为每个数据集生成行
for idx, dataset in enumerate(datasets):
    dataset_label = dataset_labels[dataset]
    best_values = find_best_values(dataset)
    num_models = count_models(dataset)
    
    # 在第5个数据集（liver）之后插入分页
    if idx == 5:  # Zheng68k之前插入分页（前5个数据集在第一页，后4个在第二页）
        latex_lines.append("    \\end{tabular}")
        latex_lines.append("  \\end{table*}")
        latex_lines.append("")
        latex_lines.append("\\begin{table*}[htbp]")
        latex_lines.append("")
        latex_lines.append("  \\centering")
        latex_lines.append("")
        latex_lines.append("  \\scriptsize")
        latex_lines.append("")
        latex_lines.append("  \\caption{Performance of various models on four single-cell benchmarks (continued) (Acc\\,\\%, F$_1$\\,\\%, Precision\\,\\%, Recall\\,\\%).}")
        latex_lines.append("")
        latex_lines.append("  \\label{tab:performance_acc_f1_precision_recall_all_cont}")
        latex_lines.append("")
        latex_lines.append("  \\setlength{\\tabcolsep}{4pt}")
        latex_lines.append("")
        latex_lines.append("  \\begin{tabular}{llcccccccccccc}")
        latex_lines.append("")
        latex_lines.append("    \\toprule")
        latex_lines.append("    Dataset   & Model")
        latex_lines.append("      & \\multicolumn{4}{c}{Top1}    & \\multicolumn{4}{c}{Top5}")
        latex_lines.append("      & \\multicolumn{4}{c}{Top9}    \\\\")
        latex_lines.append("    \\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}")
        latex_lines.append("              & ")
        latex_lines.append("      & Acc    & F$_1$    & Prec    & Recall")
        latex_lines.append("      & Acc    & F$_1$    & Prec    & Recall")
        latex_lines.append("      & Acc    & F$_1$    & Prec    & Recall   \\\\")
        latex_lines.append("    \\midrule")
        latex_lines.append("")
    
    latex_lines.append(f"    \\multirow{{{num_models}}}{{*}}{{{dataset_label}}}")
    latex_lines.append("")
    
    for model in models_order:
        if model not in table_data[dataset]:
            continue
        
        model_label = model
        if model == 'UMAP':
            model_label = 'UMAP'
        elif model == 'scVI':
            model_label = 'scVI'
        
        data = table_data[dataset][model]
        
        # 检查是否需要高亮整行（scMulan或scFoundation）
        highlight_row = (model == 'scMulan' or model == 'scFoundation')
        
        # 生成每个shot值的单元格
        cells = []
        for shot_label in ['Top1', 'Top5', 'Top9']:
            for metric in ['acc', 'f1', 'precision', 'recall']:
                if shot_label in data[metric]:
                    val = data[metric][shot_label]
                    if isinstance(val, tuple):
                        mean_val, std_val = val
                        # 转换为百分比并保留2位小数
                        mean_val_pct = mean_val * 100
                        std_val_pct = std_val * 100
                        cell_text = f"{mean_val_pct:.2f}"
                    else:
                        cell_text = f"{val * 100:.2f}"
                    
                    # 检查是否需要高亮这个单元格
                    if model == best_values[metric].get(shot_label):
                        cell_text = f"\\cellcolor{{gray!25}}{{{cell_text}}}"
                    else:
                        cell_text = f"  {cell_text}"
                    
                    cells.append(cell_text)
                else:
                    cells.append("  —")
        
        # 组合行
        if highlight_row:
            model_cell = f"      & \\cellcolor{{gray!25}}\\textbf{{{model_label}}}"
        else:
            model_cell = f"      & {model_label}"
        
        row = model_cell + "      & " + " & ".join(cells) + "  \\\\"
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
with open('/usr/data/yongjie/output/annotation_table_all_datasets.tex', 'w') as f:
    f.write(latex_code)

print("\n\nLaTeX表格已保存至: /usr/data/yongjie/output/annotation_table_all_datasets.tex")



