export PYTHONPATH=$PYTHONPATH:.
export TRANSFORMERS_DEFAULT_CACHE="./hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"
data=$1
seed=$2
topk=$3
gpu_id=$(( seed - 42 ))

# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/scvi.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed $seed
# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed $seed
CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/umap.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed $seed

# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/cellplm.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed $seed
# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/langcell.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed $seed
# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/geneformer.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed $seed
# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/scgpt.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed $seed
# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/scfoundation.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed $seed
# CUDA_VISIBLE_DEVICES=$gpu_id python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed $seed


# CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scfoundation.yaml --phase few_shot --task restruct --dataset_name cortex --topk 500 --seed 42
