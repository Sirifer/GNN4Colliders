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

# From Scratch Training

datasets=("ttH_CP_even" "ttH_CP_odd")
chunks=3

for data in "${datasets[@]}"; do
    python scripts/prep_data.py --config configs/stats_100K/ttH_CP_even_vs_odd.yaml --dataset "$data" --shuffle_mode --chunk 0
    for ((i=0; i<chunks; i++)); do
        python scripts/prep_data.py --config configs/stats_100K/ttH_CP_even_vs_odd.yaml --dataset "$data" --shuffle_mode --chunk "$i"
    done
done

python scripts/training_script.py --config configs/stats_100K/ttH_CP_even_vs_odd.yaml --preshuffle --nocompile --lazy

# Finetuning Training

python scripts/training_script.py --config configs/stats_100K/finetuning_ttH_CP_even_vs_odd.yaml --preshuffle --nocompile --lazy

# Inference: Writing GNN Scores for from-scratch training and finetuned training to root files
files=(
    "ttH_NLO.root"
    "ttH_CPodd.root"
)

config=(
    "configs/stats_100K/ttH_CP_even_vs_odd.yaml"
    "configs/stats_100K/finetuning_ttH_CP_even_vs_odd.yaml"
)

branch_name=( 
    "cp_score"
    "finetuning_cp_score"
)

for ((j=0; j<${#files[@]}; j++))
do
    python scripts/inference.py \
        --target "/global/cfs/projectdirs/atlas/joshua/gnn_data/stats_100K/${files[j]}" \
        --destination "/global/cfs/projectdirs/atlas/joshua/gnn_data/scores/stats_100K/${files[j]}" \
        --config "${config[@]}" \
        --branch_name "${branch_name[@]}" \
        --chunks 1 \
        --chunkno 0 \
        --write
    echo ''
done