export PYTHONPATH=$PYTHONPATH:.
data=$1
seed=$2
gpu_id=0

# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/scvi.yaml --phase zero_shot --task cluster --dataset_name $data
# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/pca.yaml --phase zero_shot --task cluster --dataset_name $data
# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/umap.yaml --phase zero_shot --task cluster --dataset_name $data
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --cfgs ./configs/cellplm.yaml --phase zero_shot --task cluster --dataset_name $data
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --cfgs ./configs/scgpt.yaml --phase zero_shot --task cluster --dataset_name $data
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --cfgs ./configs/scfoundation.yaml --phase zero_shot --task cluster --dataset_name $data
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --cfgs ./configs/langcell.yaml --phase zero_shot --task cluster --dataset_name $data
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --cfgs ./configs/geneformer.yaml --phase zero_shot --task cluster --dataset_name $data
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --cfgs ./configs/scmulan.yaml --phase zero_shot --task cluster --dataset_name $data
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --cfgs ./configs/cellfm.yaml --phase zero_shot --task cluster --dataset_name $data


datasets=(cortex covid lung liver)
CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/geneformer.yaml --phase zero_shot --task cluster --dataset_name liver
CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/geneformer.yaml --phase zero_shot --task cluster --dataset_name liver
CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/geneformer.yaml --phase zero_shot --task cluster --dataset_name liver
CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/cellplm.yaml --phase zero_shot --task cluster --dataset_name covid
