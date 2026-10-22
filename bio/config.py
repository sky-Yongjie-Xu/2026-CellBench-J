"""YAML configuration merge and project-relative paths."""
import yaml

def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if isinstance(src[k], dict) and isinstance(dst[k], dict):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v

def config_loader(path):
    from bio.io.paths import project_path
    path = project_path(path)
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    with open(project_path("configs/default.yaml"), 'r') as stream:
        dst_cfgs = yaml.safe_load(stream)
    MergeCfgsDict(src_cfgs, dst_cfgs)
    for key in ('root_path', 'data_path'):
        if dst_cfgs['data_cfg'].get(key):
            dst_cfgs['data_cfg'][key] = str(project_path(dst_cfgs['data_cfg'][key]))
    for key in ('ckpt', 'ckpt_path', 'pretrain_directory', 'hf_dir',
                'technology_mean_path', 'vocab_file', 'config_file',
                'meta_info_path', 'gene_list_path', 'token_dictionary_file', 'gene_median_file'):
        if dst_cfgs['model_cfg'].get(key):
            dst_cfgs['model_cfg'][key] = str(project_path(dst_cfgs['model_cfg'][key]))
    return dst_cfgs
