#!/bin/bash

# Pretraining

# Data Preparation
datasets=("ttH" "tHjb" "ggF" "VBF" "WH" "ZH" "ttyy" "tttt" "SingleT_schan" "ttbar" "ttW" "ttt")
chunks=3

for data in "${datasets[@]}"; do
    python scripts/prep_data.py --config configs/demo/pretraining_multiclass.yaml --dataset "$data" --shuffle_mode --chunk 0
    for ((i=0; i<chunks; i++)); do
        python scripts/prep_data.py --config configs/demo/pretraining_multiclass.yaml --dataset "$data" --shuffle_mode --chunk "$i"
    done
done

# Training

python scripts/training_script.py --config configs/demo/pretraining_multiclass.yaml --preshuffle --nocompile --lazy

# Inference

