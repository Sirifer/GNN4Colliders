#!/bin/bash

# Pretraining
datasets=("ttH" "tHjb" "ggF" "VBF" "WH" "ZH" "ttyy" "tttt" "SingleT_schan" "ttbar" "ttW" "ttt")
chunks=3

for data in "${datasets[@]}"; do
    python scripts/prep_data.py --config configs/stats_100K/pretraining_multiclass.yaml --dataset "$data" --shuffle_mode --chunk 0
    for ((i=0; i<chunks; i++)); do
        python scripts/prep_data.py --config configs/stats_100K/pretraining_multiclass.yaml --dataset "$data" --shuffle_mode --chunk "$i"
    done
done

python scripts/training_script.py --config configs/stats_100K/pretraining_multiclass.yaml --preshuffle --nocompile --lazy

# Finetuning

datasets=("ttH_CP_even" "ttH_CP_odd")
chunks=3

for data in "${datasets[@]}"; do
    python scripts/prep_data.py --config configs/stats_100K/finetuning_ttH_CP_even_vs_odd.yaml --dataset "$data" --shuffle_mode --chunk 0
    for ((i=0; i<chunks; i++)); do
        python scripts/prep_data.py --config configs/stats_100K/finetuning_ttH_CP_even_vs_odd.yaml --dataset "$data" --shuffle_mode --chunk "$i"
    done
done

python scripts/training_script.py --config configs/stats_100K/finetuning_ttH_CP_even_vs_odd.yaml --preshuffle --nocompile --lazy


# Inference

# python scripts/inference.py \
#     --target "/global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/data/ntuples/Hyy_pretraining/multilabel_10K/ttH_NLO.root" \
#     --destination "/global/cfs/projectdirs/atlas/joshua/GNN4Colliders/root_gnn_dgl/scores/ttH_NLO.root" \
#     --config "configs/demo/finetuning_ttH_CP_Even_vs_Odd.yaml" \
#     --chunks 1 \
#     --chunkno 0 \
#     --write \
#     --branch 'GNN_Score'

# python scripts/inference.py \
#     --target "/global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/data/ntuples/Hyy_pretraining/multilabel_10K/ttH_CPodd.root" \
#     --destination "/global/cfs/projectdirs/atlas/joshua/GNN4Colliders/root_gnn_dgl/scores/ttH_CPodd.root" \
#     --config "configs/demo/finetuning_ttH_CP_Even_vs_Odd.yaml" \
#     --chunks 1 \
#     --chunkno 0 \
#     --write \
#     --branch 'GNN_Score'