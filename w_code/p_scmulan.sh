export PYTHONPATH=$PYTHONPATH:/root/commusim/benchmark


# CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase zero_shot --task cluster --dataset_name pbmc12k &
# CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase zero_shot --task cluster --dataset_name norman &

topk=100
data=norman


CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 42 &
CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 43 &
CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 44 &
CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 45 &
CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 46 

data=adamson


CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 42 &
CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 43 &
CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 44 &
CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 45 &
CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 46 

topk=500
data=norman


CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 42 &
CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 43 &
CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 44 &
CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 45 &
CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 46 

data=adamson


CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 42 &
CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 43 &
CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 44 &
CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 45 &
CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 46 

topk=1000
data=norman


CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 42 &
CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 43 &
CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 44 &
CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 45 &
CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 46 

data=adamson


CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 42 &
CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 43 &
CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 44 &
CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 45 &
CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 46 

topk=5000
data=norman
CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 42 &
CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 43 &
CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 44 &
CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 45 &
CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 46 

data=adamson
CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 42 &
CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 43 &
CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 44 &
CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 45 &
CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 46 

topk=10000
data=norman
CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 42 &
CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 43 &
CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 44 &
CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 45 &
CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 46 

data=adamson

CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 42 &
CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 43 &
CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 44 &
CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 45 &
CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase few_shot --task restruct --dataset_name $data --topk $topk --seed 46 