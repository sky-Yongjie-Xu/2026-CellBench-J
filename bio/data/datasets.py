import os
import scanpy as sc
import numpy as np
import hdf5plugin
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import pickle
from anndata import AnnData
import scipy.sparse as sp
import anndata
import pandas as pd
from tqdm import tqdm
from scipy.io import mmread
from bio.io.paths import project_path


def _attach_uc_metadata(adata, root, prefix):
    """Attach the supplied one-column TSV files, checking cell/gene alignment."""
    def column(suffix, expected):
        path = root / f"{prefix}_{suffix}.tsv"
        values = pd.read_csv(path, sep='\t', header=None, dtype=str, keep_default_na=False)
        if values.shape != (expected, 1):
            raise ValueError(f"{path}: expected {expected} rows and one column, got {values.shape}")
        return values.iloc[:, 0].to_numpy()

    if not (root / f"{prefix}_cell.tsv").is_file():
        return
    cell_ids = column("cell", adata.n_obs)
    if len(set(cell_ids)) != adata.n_obs:
        raise ValueError(f"{prefix}_cell.tsv contains duplicate cell IDs")
    positional = np.array_equal(adata.obs_names.astype(str), np.arange(adata.n_obs).astype(str))
    if positional:
        order = np.arange(adata.n_obs)
    else:
        order = pd.Index(cell_ids).get_indexer(adata.obs_names.astype(str))
        if np.any(order < 0):
            raise ValueError(f"{prefix}_cell.tsv does not match the h5ad cell IDs")
    for suffix, key in (("celltype", "celltype"), ("batch_health", "batch_h"),
                        ("batch_location", "batch_l"), ("batch_patient", "batch_p")):
        values = column(suffix, adata.n_obs)[order]
        if key in adata.obs and not np.array_equal(adata.obs[key].astype(str).to_numpy(), values):
            raise ValueError(f"{prefix}_{suffix}.tsv disagrees with h5ad column {key}; check row order")
        adata.obs[key] = values
    adata.obs["cell_id"] = cell_ids[order]
    if positional:
        adata.obs_names = cell_ids
    genes = column("gene", adata.n_vars)
    if np.array_equal(adata.var_names.astype(str), np.arange(adata.n_vars).astype(str)):
        adata.var_names = genes
    elif not np.array_equal(adata.var_names.astype(str), genes):
        raise ValueError(f"{prefix}_gene.tsv disagrees with the h5ad gene order")
    adata.var["gene_name"] = genes


def _read_flat_dataset(data_cfg, filename, label_fields=("celltype",),
                       gene_fields=("gene_name",), batch_fields=(),
                       n_top_genes=3000, subset_hvg=True):
    """Read the flat datasets layout, preserving existing preprocessing choices.

    Return None when the flat file is absent so legacy layouts still work.
    Missing biological labels are errors; batch columns must not stand in for labels.
    """
    path = project_path(data_cfg.get("root_path", "datasets")) / filename
    if not path.is_file():
        return None
    adata = sc.read_h5ad(str(path))
    name = data_cfg.get("dataset_name", "")
    if filename in ("ucepi.h5ad", "ucimm.h5ad"):
        prefix = "uc_epi" if filename == "ucepi.h5ad" else "uc_imm"
        _attach_uc_metadata(adata, path.parent, prefix)
    label_key = data_cfg.get("dataset_label_keys", {}).get(name)
    candidates = (label_key,) if label_key else label_fields
    for key in candidates:
        if key in adata.obs:
            adata.obs["celltype"] = adata.obs[key]
            break
    else:
        raise ValueError(
            f"{path} has no cell-type labels. Expected one of {candidates}; "
            f"available obs columns: {list(adata.obs.columns)}. "
            "Add the actual labels or configure data_cfg.dataset_label_keys; "
            "batch identifiers are not cell-type labels."
        )
    for key in gene_fields:
        if key in adata.var:
            adata.var["gene_name"] = adata.var[key].astype(str).values
            break
    else:
        adata.var["gene_name"] = adata.var_names.astype(str)
    batch_key = data_cfg.get("dataset_batch_keys", {}).get(name)
    if batch_key:
        if batch_key not in adata.obs:
            raise ValueError(f"Configured batch column {batch_key!r} is absent from {path}")
        adata.obs["batch"] = adata.obs[batch_key]
    elif "batch" not in adata.obs:
        for key in batch_fields:
            if key in adata.obs:
                adata.obs["batch"] = adata.obs[key]
                break
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    if subset_hvg:
        adata = adata[:, adata.var.highly_variable].copy()
    return adata


######## 单细胞转录组数据
def read_pbmc12k(data_cfg):
    flat = _read_flat_dataset(data_cfg, "pbmc_12k.h5ad", ("str_labels", "celltype"), ("gene_symbols", "gene_name"))
    if flat is not None:
        return flat
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_path = root_path + "/pbmc12k/pbmc_12k.h5ad"
    adata = sc.read_h5ad(adata_path)
    
    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    
    adata.obs["celltype"] = adata.obs["str_labels"]
    adata.var["gene_name"] = adata.var["gene_symbols"]
    return adata

def read_hPancreas(data_cfg):
    flat = _read_flat_dataset(data_cfg, "human_pancreas_norm_complexBatch.h5ad",
                              ("celltype", "Celltype"), batch_fields=("tech",),
                              n_top_genes=400, subset_hvg=False)
    if flat is not None:
        return flat
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_train = sc.read_h5ad(root_path + "/demo_train.h5ad")
    adata_test = sc.read_h5ad(root_path + "/demo_test.h5ad")
    adata = adata_train.concatenate(adata_test, batch_key="batch", batch_categories=["train", "test"])
    adata.X = sp.csr_matrix(adata.X)

    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=400)

    adata.obs["celltype"] = adata.obs["Celltype"]
    adata.var["gene_name"] = adata.var["Gene Symbol"]
    return adata
    
def read_cortex(data_cfg):
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_path = root_path + "/Perirhinal_Cortex.h5ad"
    adata = sc.read_h5ad(adata_path)

    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]

    adata.obs["celltype"] = adata.obs["cell_type"]
    adata.var["gene_name"] = adata.var["Gene"]
    return adata

def read_ms(data_cfg):
    flat = _read_flat_dataset(data_cfg, "filtered_ms_adata.h5ad", batch_fields=("str_batch",),
                              n_top_genes=400, subset_hvg=False)
    if flat is not None:
        return flat
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata = sc.read_h5ad(root_path + "/c_data.h5ad")
    adata_test = sc.read_h5ad(root_path + "/filtered_ms_adata.h5ad")

    adata_new = AnnData(X=adata.X, obs=adata.obs[['celltype']].copy(), var=adata.var[['gene_name']].copy())
    adata_test_new = AnnData(X=adata_test.X, obs=adata_test.obs[['celltype']].copy(), var=adata_test.var[['gene_name']].copy())
    adata_new.var_names = adata_new.var['gene_name'].values
    adata_test_new.var_names = adata_test_new.var['gene_name'].values
    adata = adata_new.concatenate(adata_test_new)

    sc.pp.highly_variable_genes(adata, n_top_genes=400)
    return adata

def read_liver(data_cfg):
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_path = root_path + "/GSE151530_Liver_ensg.h5ad"
    adata = sc.read_h5ad(adata_path)

    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4) 
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=400)

    adata.obs["celltype"] = adata.obs["Type"]
    return adata

def read_Zheng68k(data_cfg):
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_path = root_path + "/Zheng68K.h5ad"
    adata = sc.read_h5ad(adata_path)

    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=3000)   
    adata = adata[:, adata.var.highly_variable]
    return adata

def read_lung(data_cfg):
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_path = root_path + "/GSE131907_Lung_ensg.h5ad"
    adata = sc.read_h5ad(adata_path)

    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4) 
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=400)

    adata.obs["celltype"] = adata.obs["Cell_type"]
    return adata

def read_immune(data_cfg):
    flat = _read_flat_dataset(data_cfg, "immune.h5ad", ("Manually_curated_celltype", "celltype"),
                              batch_fields=("Donor",))
    if flat is not None:
        return flat
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_path = root_path + "/immune/global.h5ad"
    adata = sc.read_h5ad(adata_path)

    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=3000)   
    adata = adata[:, adata.var.highly_variable]

    adata.obs["celltype"] = adata.obs["Manually_curated_celltype"]
    return adata
    

############# 单细胞扰动数据
def read_adamson(data_cfg):
    flat = _read_flat_dataset(data_cfg, "adamson.h5ad", ("cell_type", "celltype"))
    if flat is not None:
        return flat
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_path = root_path + "/adamson/perturb_processed.h5ad"
    adata = sc.read_h5ad(adata_path)

    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=3000)   
    adata = adata[:, adata.var.highly_variable]

    adata.obs["celltype"] = adata.obs["cell_type"]
    return adata

def read_norman(data_cfg):
    flat = _read_flat_dataset(data_cfg, "norman.h5ad", ("cell_type", "celltype"))
    if flat is not None:
        return flat
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_path = root_path + "/norman/perturb_processed.h5ad"
    adata = sc.read_h5ad(adata_path)

    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=3000)   
    adata = adata[:, adata.var.highly_variable]

    adata.obs["celltype"] = adata.obs["cell_type"]
    return adata

def read_replogleK562(data_cfg):
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_subset_path = root_path + "/replogle/K562_essential_raw_singlecell_01_subset.h5ad"

    # adata_path = root_path + "/replogle/K562_essential_raw_singlecell_01.h5ad"
    # adata = anndata.read_h5ad(adata_path)
    # coordinates_path = root_path + "/replogle/annotated_embedding_coordinates.csv"
    # df_sub = pd.read_csv(coordinates_path)
    
    # adata_ = adata[adata.obs['gene'].isin(list(df_sub.gene.values)+['non-targeting']), :]

    # target_cells = 100
    # cluster_key = 'gene'
    # adatas = [adata_[adata_.obs[cluster_key]==clust] for clust in adata_.obs[cluster_key].cat.categories]
    # for dat in tqdm(adatas):
    #     if dat.n_obs > target_cells:
    #         if dat.obs[cluster_key].cat.categories.values[0] != 'non-targeting':
    #             sc.pp.subsample(dat, n_obs=target_cells)
    #         else:
    #             sc.pp.subsample(dat, n_obs=2500)
    # adata_downsampled_ = adatas[0].concatenate(*adatas[1:])

    # sc.pp.normalize_total(adata_downsampled_)
    # sc.pp.log1p(adata_downsampled_)
    # sc.pp.highly_variable_genes(adata_downsampled_, n_top_genes=3000, subset=False)
    # hvg_flag_ = adata_downsampled_.var['highly_variable'].values
    # gene_flag_ = adata_downsampled_.var['gene_name'].isin(adata_downsampled_.obs['gene'].values).values
    # select_flag_ = np.logical_or(hvg_flag_, gene_flag_)
    # condition_flag_ = adata_downsampled_.obs['gene'].isin(adata_downsampled_.var['gene_name'].values.tolist()+['non-targeting']).values
    # adata_subset = adata_downsampled_[condition_flag_, select_flag_]

    # adata_subset.obs['condition'] = [i+'+ctrl' for i in adata_subset.obs['gene'].values]
    # adata_subset.obs['condition'] = adata_subset.obs['condition'].replace({'non-targeting+ctrl': 'ctrl'})
    # adata_subset.obs['celltype'] = 'K562'
    # adata_subset.write_h5ad(adata_subset_path)

    adata = anndata.read_h5ad(adata_subset_path)
    adata.X = sp.csr_matrix(adata.X)
    adata.var = adata.var.drop('gene_name', axis=1)
    adata.obs['control'] = (adata.obs['condition'] == 'ctrl').astype(int)
    return adata

def read_replogleRPE1(data_cfg):
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_subset_path = root_path + "/replogle/rpe1_raw_singlecell_01_subset.h5ad"

    # adata_path = root_path + "/replogle/rpe1_raw_singlecell_01.h5ad"
    # adata = sc.read_h5ad(adata_path)
    # coordinates_path = root_path + "/replogle/annotated_embedding_coordinates.csv"
    # df_sub = pd.read_csv(coordinates_path)

    # adata_ = adata[adata.obs['gene'].isin(list(df_sub.gene.values)+['non-targeting']), :]

    # target_cells = 100
    # cluster_key = 'gene'
    # adatas = [adata_[adata_.obs[cluster_key]==clust] for clust in adata_.obs[cluster_key].cat.categories]
    # for dat in tqdm(adatas):
    #     if dat.n_obs > target_cells:
    #         if dat.obs[cluster_key].cat.categories.values[0] != 'non-targeting':
    #             sc.pp.subsample(dat, n_obs=target_cells)
    #         else:
    #             sc.pp.subsample(dat, n_obs=2500)
    # adata_downsampled_ = adatas[0].concatenate(*adatas[1:])

    # sc.pp.normalize_total(adata_downsampled_)
    # sc.pp.log1p(adata_downsampled_)
    # sc.pp.highly_variable_genes(adata_downsampled_, n_top_genes=3000, subset=False)
    # hvg_flag_ = adata_downsampled_.var['highly_variable'].values
    # gene_flag_ = adata_downsampled_.var['gene_name'].isin(adata_downsampled_.obs['gene'].values).values
    # select_flag_ = np.logical_or(hvg_flag_, gene_flag_)
    # condition_flag_ = adata_downsampled_.obs['gene'].isin(adata_downsampled_.var['gene_name'].values.tolist()+['non-targeting']).values
    # adata_subset = adata_downsampled_[condition_flag_, select_flag_]

    # adata_subset.obs['condition'] = [i+'+ctrl' for i in adata_subset.obs['gene'].values]
    # adata_subset.obs['condition'] = adata_subset.obs['condition'].replace({'non-targeting+ctrl': 'ctrl'})
    # adata_subset.obs['celltype'] = 'RPE1'
    # adata_subset.write_h5ad(adata_subset_path)

    adata = anndata.read_h5ad(adata_subset_path)
    adata.X = sp.csr_matrix(adata.X)
    adata.var = adata.var.drop('gene_name', axis=1)
    adata.obs['control'] = (adata.obs['condition'] == 'ctrl').astype(int)
    return adata

############# 单细胞多批次数据
def read_rgc(data_cfg):
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    mtx_path = root_path + "/batch_data/rgc.mtx"
    celltype_path = root_path + "/batch_data/rgc_celltype.tsv"
    batch_path = root_path + "/batch_data/rgc_batch.tsv"
    gene_path = root_path + "/batch_data/rgc_gene.tsv"
    mtx_data = mmread(mtx_path)
    if sp.issparse(mtx_data):
        x = sp.csr_matrix(mtx_data.T)
    else:
        x = sp.csr_matrix(np.array(mtx_data).T)
    celltype_df = pd.read_csv(celltype_path, sep='\t', header=None)
    celltype_labels = celltype_df.iloc[:, 0].values
    batch_df = pd.read_csv(batch_path, sep='\t', header=None)
    batch_labels = batch_df.iloc[:, 0].values
    gene_df = pd.read_csv(gene_path, sep='\t', header=None)
    genes = gene_df.iloc[:, 0].values

    adata = sc.AnnData(x)
    adata.var["gene_name"] = genes
    # 设置 var_names 为实际基因名称（某些模型优先使用 var_names 而不是 gene_name）
    adata.var_names = genes
    
    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    
    # 确保返回稀疏矩阵格式
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    
    adata.obs["celltype"] = celltype_labels
    adata.obs['batch'] = batch_labels
    return adata

def read_celegan(data_cfg):
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    mtx_path = root_path + "/batch_data/celegan.mtx"
    celltype_path = root_path + "/batch_data/celegan_celltype.tsv"
    batch_path = root_path + "/batch_data/celegan_batch.tsv"
    gene_path = root_path + "/batch_data/celegan_gene.tsv"
    mtx_data = mmread(mtx_path)
    if sp.issparse(mtx_data):
        x = sp.csr_matrix(mtx_data.T)
    else:
        x = sp.csr_matrix(np.array(mtx_data).T)
    celltype_df = pd.read_csv(celltype_path, sep='\t', header=None)
    celltype_labels = celltype_df.iloc[:, 0].values
    batch_df = pd.read_csv(batch_path, sep='\t', header=None)
    batch_labels = batch_df.iloc[:, 0].values
    gene_df = pd.read_csv(gene_path, sep='\t', header=None)
    genes = gene_df.iloc[:, 0].values
    
    # 过滤掉 celltype 为 'NA' 或 NaN 的细胞
    valid_mask = (celltype_labels != 'NA') & (~pd.isna(celltype_labels))
    if sp.issparse(x):
        x = x[valid_mask]
    else:
        x = x[valid_mask]
    celltype_labels = celltype_labels[valid_mask]
    batch_labels = batch_labels[valid_mask]

    adata = sc.AnnData(x)
    adata.var["gene_name"] = genes
    # 设置 var_names 为实际基因名称（某些模型优先使用 var_names 而不是 gene_name）
    adata.var_names = genes
    
    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    
    # 确保返回稀疏矩阵格式
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    
    adata.obs["celltype"] = celltype_labels
    adata.obs['batch'] = batch_labels
    return adata

def read_uc_epi(data_cfg):
    flat = _read_flat_dataset(data_cfg, "ucepi.h5ad", ("celltype", "cell_type", "label"), batch_fields=("batch_p",))
    if flat is not None:
        return flat
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    mtx_path = root_path + "/batch_data/uc_epi.mtx"
    celltype_path = root_path + "/batch_data/uc_epi_celltype.tsv"
    batch_path = root_path + "/batch_data/uc_epi_batch.tsv"
    gene_path = root_path + "/batch_data/uc_epi_gene.tsv"
    mtx_data = mmread(mtx_path)
    if sp.issparse(mtx_data):
        x = sp.csr_matrix(mtx_data.T)
    else:
        x = sp.csr_matrix(np.array(mtx_data).T)
    celltype_df = pd.read_csv(celltype_path, sep='\t', header=None)
    celltype_labels = celltype_df.iloc[:, 0].values
    batch_df = pd.read_csv(batch_path, sep='\t', header=None)
    batch_labels = batch_df.iloc[:, 0].values
    gene_df = pd.read_csv(gene_path, sep='\t', header=None)
    genes = gene_df.iloc[:, 0].values

    adata = sc.AnnData(x)
    adata.var["gene_name"] = genes
    # 设置 var_names 为实际基因名称（某些模型优先使用 var_names 而不是 gene_name）
    adata.var_names = genes
    
    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    
    # 确保返回稀疏矩阵格式
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    
    adata.obs["celltype"] = celltype_labels
    adata.obs['batch'] = batch_labels
    return adata

def read_uc_imm1(data_cfg):
    flat = _read_flat_dataset(data_cfg, "ucimm.h5ad", ("celltype", "cell_type", "label"), batch_fields=("batch_p",))
    if flat is not None:
        return flat
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    mtx_path = root_path + "/batch_data/uc_imm.mtx"
    celltype_path = root_path + "/batch_data/uc_imm_celltype.tsv"
    batch_path = root_path + "/batch_data/uc_imm_batch.tsv"
    gene_path = root_path + "/batch_data/uc_imm_gene.tsv"
    mtx_data = mmread(mtx_path)
    if sp.issparse(mtx_data):
        x = sp.csr_matrix(mtx_data.T)
    else:
        x = sp.csr_matrix(np.array(mtx_data).T)
    celltype_df = pd.read_csv(celltype_path, sep='\t', header=None)
    celltype_labels = celltype_df.iloc[:, 0].values
    batch_df = pd.read_csv(batch_path, sep='\t', header=None)
    batch_labels = batch_df.iloc[:, 0].values
    gene_df = pd.read_csv(gene_path, sep='\t', header=None)
    genes = gene_df.iloc[:, 0].values

    adata = sc.AnnData(x)
    adata.var["gene_name"] = genes
    # 设置 var_names 为实际基因名称（某些模型优先使用 var_names 而不是 gene_name）
    adata.var_names = genes
    
    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    
    # 确保返回稀疏矩阵格式
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    
    adata.obs["celltype"] = celltype_labels
    adata.obs['batch'] = batch_labels
    return adata

def read_uc_imm(data_cfg):
    flat = _read_flat_dataset(data_cfg, "ucimm.h5ad", ("celltype", "cell_type", "label"), batch_fields=("batch_p",))
    if flat is not None:
        return flat
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    mtx_path = root_path + "/batch_data/uc_imm.mtx"
    celltype_path = root_path + "/batch_data/uc_imm_celltype.tsv"
    batch_path = root_path + "/batch_data/uc_imm_batch.tsv"
    gene_path = root_path + "/batch_data/uc_imm_gene.tsv"
    mtx_data = mmread(mtx_path)
    if sp.issparse(mtx_data):
        x = sp.csr_matrix(mtx_data.T)
    else:
        x = sp.csr_matrix(np.array(mtx_data).T)
    celltype_df = pd.read_csv(celltype_path, sep='\t', header=None)
    celltype_labels = celltype_df.iloc[:, 0].values
    batch_df = pd.read_csv(batch_path, sep='\t', header=None)
    batch_labels = batch_df.iloc[:, 0].values
    gene_df = pd.read_csv(gene_path, sep='\t', header=None)
    genes = gene_df.iloc[:, 0].values

    adata = sc.AnnData(x)
    adata.var["gene_name"] = genes
    # 设置 var_names 为实际基因名称（某些模型优先使用 var_names 而不是 gene_name）
    adata.var_names = genes
    
    if adata.X.max() > 10:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    
    # 确保返回稀疏矩阵格式
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    
    adata.obs["celltype"] = celltype_labels
    adata.obs['batch'] = batch_labels
    return adata



############# 其他数据

def read_templte(data_cfg):
    np.random.seed(42)  # 你可以选择任意整数作为 seed 值
    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_path = str(project_path(root_path) / "adata.h5ad")  # 注意大小写
    adata = sc.read_h5ad(adata_path)

    train_num = adata.shape[0]
    tr = np.random.permutation(train_num)
    train_indices = tr[:int(train_num * 0.8)]
    valid_indices = tr[int(train_num * 0.8):int(train_num*0.9)]
    test_indices = tr[int(train_num * 0.9):]
    
    adata_train = adata[train_indices]
    adata_valid = adata[valid_indices]
    adata_test = adata[test_indices]

    return adata_train, adata_valid, adata_test, adata_all

def read_humanpancreas(data_cfg):
    np.random.seed(42)

    root_path = str(project_path(data_cfg.get("root_path", "datasets")))
    adata_path = str(project_path(data_cfg.get("root_path", "datasets")) / "human_pancreas_norm_complexBatch.h5ad")
    adata = sc.read_h5ad(adata_path)

    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    train_num = adata.shape[0]

    tr = np.random.permutation(train_num)
    train_indices = tr[:int(train_num * 0.8)]
    valid_indices = tr[int(train_num * 0.8):int(train_num*0.9)]
    test_indices = tr[int(train_num * 0.9):]
    
    adata_train = adata[train_indices]
    adata_valid = adata[valid_indices]
    adata_test = adata[test_indices]
    adata_all = adata.copy()

    return adata_train, adata_valid, adata_test, adata_all

# 示例用法
if __name__ == "__main__":
    data_cfg = {
        "root_path": str(project_path("datasets"))
    }
    
    adata_train, adata_valid, adata_test  = read_norman(data_cfg)

    
