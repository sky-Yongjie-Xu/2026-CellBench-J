# bash  bash_ms.sh
# bash  bash_hpancreas.sh
# bash  bash_pbmc12k.sh
# bash  bash_zheng68k.sh

# tmux new -s commusim
# cd benchmark
# conda activate benchmark

# CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/geneformer.yaml --phase few_shot --task annotation --dataset_name ms --topk 9 --seed 46

export PYTHONPATH=$PYTHONPATH:/root/commusim/benchmark



data=$1

# seed=42
topk=1
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 42
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 43
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 44
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 45
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 46

topk=3
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 42
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 43
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 44
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 45
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 46

topk=5
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 42
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 43
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 44
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 45
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 46

topk=9
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 42
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 43
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 44
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 45
CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/pca.yaml --phase few_shot --task annotation --dataset_name $data --topk $topk --seed 46


