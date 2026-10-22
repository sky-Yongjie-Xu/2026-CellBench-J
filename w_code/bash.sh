export PYTHONPATH=$PYTHONPATH:.
export TRANSFORMERS_DEFAULT_CACHE="./hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"

echo "✅ Init"
seed=42

topk=10

# data=pbmc12k
datasets=(lung)
# datasets=(pbmc12k pbmc12k)
# datasets=(ms Zheng68k)

for data in "${datasets[@]}"; do
    echo "✅ $data"
    # CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/pca.yaml --phase zero_shot --task annotation --dataset_name $data --topk $topk --seed $seed &
    # CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/umap.yaml --phase zero_shot --task annotation --dataset_name $data --topk $topk --seed $seed &
    CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/scvi.yaml --phase zero_shot --task annotation --dataset_name $data --topk $topk --seed $seed &

    # CUDA_VISIBLE_DEVICES=0 python bio/main.py --cfgs ./configs/langcell.yaml --phase zero_shot --task annotation --dataset_name $data --topk $topk --seed $seed &
    # CUDA_VISIBLE_DEVICES=1 python bio/main.py --cfgs ./configs/geneformer.yaml --phase zero_shot --task annotation --dataset_name $data --topk $topk --seed $seed &
    # CUDA_VISIBLE_DEVICES=2 python bio/main.py --cfgs ./configs/scgpt.yaml --phase zero_shot --task annotation --dataset_name $data --topk $topk --seed $seed &
    # CUDA_VISIBLE_DEVICES=3 python bio/main.py --cfgs ./configs/scfoundation.yaml --phase zero_shot --task annotation --dataset_name $data --topk $topk --seed $seed &
    # CUDA_VISIBLE_DEVICES=5 python bio/main.py --cfgs ./configs/cellplm.yaml --phase zero_shot --task annotation --dataset_name $data --topk $topk --seed $seed &
    # CUDA_VISIBLE_DEVICES=4 python bio/main.py --cfgs ./configs/scmulan.yaml --phase zero_shot --task annotation --dataset_name $data --topk $topk --seed $seed 
done


echo "✅ All annotation jobs finished"


