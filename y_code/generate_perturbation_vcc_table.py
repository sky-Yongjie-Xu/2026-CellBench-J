import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('/usr/data/yongjie/output/perturbation_vcc/perturbation_vcc_stats.csv')

# 解析mean±std列
def parse_mean_std(value):
    parts = str(value).split('±')
    mean = float(parts[0])
    std = float(parts[1]) if len(parts) > 1 else 0.0
    return mean, std

df[['des_mean', 'des_std']] = df['DES_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))
df[['mae_mean', 'mae_std']] = df['MAE_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))
df[['pds_mean', 'pds_std']] = df['PDS_mean±std'].apply(lambda x: pd.Series(parse_mean_std(x)))

# 两个数据集
datasets = ['adamson', 'norman']
df_filtered = df[df['dataset'].isin(datasets)].copy()

# 定义模型顺序
models_order = ['PCA', 'UMAP', 'scVI', 'scGPT', 'CellPLM', 'Geneformer', 'LangCell', 'scMulan', 'scFoundation']

# 数据集标签映射
dataset_labels = {
    'adamson': 'Adamson',
    'norman': 'Norman'
}

# 使用真实的top值
top_mapping = {
    'Top1': 'top1',
    'Top3': 'top3',
    'Top5': 'top5',
    'Top7': 'top7',
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
        
        row_data = {'des': {}, 'mae': {}, 'pds': {}}
        
        # 提取各个top值的数据
        for top_label, top_value in top_mapping.items():
            top_row = model_data[model_data['top'] == top_value]
            if len(top_row) > 0:
                row_data['des'][top_label] = (top_row.iloc[0]['des_mean'], top_row.iloc[0]['des_std'])
                row_data['mae'][top_label] = (top_row.iloc[0]['mae_mean'], top_row.iloc[0]['mae_std'])
                row_data['pds'][top_label] = (top_row.iloc[0]['pds_mean'], top_row.iloc[0]['pds_std'])
        
        # 计算Mean（使用所有top值的平均值）
        if row_data['des']:
            des_values = [v[0] for v in row_data['des'].values()]
            mae_values = [v[0] for v in row_data['mae'].values()]
            pds_values = [v[0] for v in row_data['pds'].values()]
            row_data['des']['Mean'] = np.mean(des_values)
            row_data['mae']['Mean'] = np.mean(mae_values)
            row_data['pds']['Mean'] = np.mean(pds_values)
        
        table_data[dataset][model] = row_data

# 找出最佳值（用于高亮）
def find_best_values(dataset_name):
    best_des = {}
    best_mae = {}
    best_pds = {}
    
    for top_label in ['Top1', 'Top5', 'Top9', 'Mean']:
        best_des_val = -float('inf')  # DES越高越好
        best_mae_val = float('inf')   # MAE越低越好
        best_pds_val = -float('inf')  # PDS越高越好
        
        for model in models_order:
            if model in table_data[dataset_name]:
                if top_label in table_data[dataset_name][model]['des']:
                    des_val = table_data[dataset_name][model]['des'][top_label]
                    if isinstance(des_val, tuple):
                        des_val = des_val[0]
                    if des_val > best_des_val:
                        best_des_val = des_val
                        best_des[top_label] = model
                
                if top_label in table_data[dataset_name][model]['mae']:
                    mae_val = table_data[dataset_name][model]['mae'][top_label]
                    if isinstance(mae_val, tuple):
                        mae_val = mae_val[0]
                    if mae_val < best_mae_val:
                        best_mae_val = mae_val
                        best_mae[top_label] = model
                
                if top_label in table_data[dataset_name][model]['pds']:
                    pds_val = table_data[dataset_name][model]['pds'][top_label]
                    if isinstance(pds_val, tuple):
                        pds_val = pds_val[0]
                    if pds_val > best_pds_val:
                        best_pds_val = pds_val
                        best_pds[top_label] = model
    
    return best_des, best_mae, best_pds

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
latex_lines.append("  \\caption{Performance of various models on gene perturbation prediction task. Metrics include DES (↑), MAE (↓), and PDS (↑), with results reported across Top1/5/9, and the mean.}")
latex_lines.append("")
latex_lines.append("  \\label{tab:perturbation_vcc_results}")
latex_lines.append("")
latex_lines.append("  \\setlength{\\tabcolsep}{3pt}")
latex_lines.append("")
latex_lines.append("  \\begin{tabular}{cl|cccc|cccc|cccc}")
latex_lines.append("")
latex_lines.append("    \\toprule")
latex_lines.append("    \\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Model}")
latex_lines.append("      & \\multicolumn{4}{c|}{DES ↑}")
latex_lines.append("      & \\multicolumn{4}{c|}{MAE ↓}")
latex_lines.append("      & \\multicolumn{4}{c}{PDS ↑} \\\\")
latex_lines.append("    \\cmidrule(lr){3-6} \\cmidrule(lr){7-10} \\cmidrule(lr){11-14}")
latex_lines.append("     & & Top1 & Top5 & Top9 & Mean")
latex_lines.append("       & Top1 & Top5 & Top9 & Mean")
latex_lines.append("       & Top1 & Top5 & Top9 & Mean \\\\")
latex_lines.append("    \\midrule")
latex_lines.append("")

# 为每个数据集生成行
for idx, dataset in enumerate(datasets):
    dataset_label = dataset_labels[dataset]
    best_des, best_mae, best_pds = find_best_values(dataset)
    num_models = count_models(dataset)
    
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
        
        # 生成DES列
        des_cells = []
        for top_label in ['Top1', 'Top5', 'Top9', 'Mean']:
            if top_label in data['des']:
                des_val = data['des'][top_label]
                if isinstance(des_val, tuple):
                    mean_val, std_val = des_val
                    cell_text = f"{mean_val:.3f}$\\pm${std_val:.3f}"
                else:
                    cell_text = f"{des_val:.3f}"
                
                # 检查是否需要高亮这个单元格
                if model == best_des.get(top_label):
                    cell_text = f"\\cellcolor{{gray!25}}{cell_text}"
                
                des_cells.append(cell_text)
            else:
                des_cells.append("—")
        
        # 生成MAE列
        mae_cells = []
        for top_label in ['Top1', 'Top5', 'Top9', 'Mean']:
            if top_label in data['mae']:
                mae_val = data['mae'][top_label]
                if isinstance(mae_val, tuple):
                    mean_val, std_val = mae_val
                    cell_text = f"{mean_val:.3f}$\\pm${std_val:.3f}"
                else:
                    cell_text = f"{mae_val:.3f}"
                
                # 检查是否需要高亮这个单元格
                if model == best_mae.get(top_label):
                    cell_text = f"\\cellcolor{{gray!25}}{cell_text}"
                
                mae_cells.append(cell_text)
            else:
                mae_cells.append("—")
        
        # 生成PDS列
        pds_cells = []
        for top_label in ['Top1', 'Top5', 'Top9', 'Mean']:
            if top_label in data['pds']:
                pds_val = data['pds'][top_label]
                if isinstance(pds_val, tuple):
                    mean_val, std_val = pds_val
                    cell_text = f"{mean_val:.3f}$\\pm${std_val:.3f}"
                else:
                    cell_text = f"{pds_val:.3f}"
                
                # 检查是否需要高亮这个单元格
                if model == best_pds.get(top_label):
                    cell_text = f"\\cellcolor{{gray!25}}{cell_text}"
                
                pds_cells.append(cell_text)
            else:
                pds_cells.append("—")
        
        # 组合行
        if highlight_row:
            model_cell = f"    & \\cellcolor{{gray!25}}\\textbf{{{model_label}}}"
        else:
            model_cell = f"    & {model_label}"
        
        row = model_cell + "      & " + " & ".join(des_cells) + "      & " + " & ".join(mae_cells) + "      & " + " & ".join(pds_cells) + " \\\\"
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
with open('/usr/data/yongjie/output/perturbation_vcc_table.tex', 'w') as f:
    f.write(latex_code)

print("\n\nLaTeX表格已保存至: /usr/data/yongjie/output/perturbation_vcc_table.tex")
