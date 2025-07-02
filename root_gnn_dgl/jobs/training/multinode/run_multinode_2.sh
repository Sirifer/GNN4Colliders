#!/bin/bash

TORCHRUN_ARGUMENTS="$1 $2 $3 $4 $5"
shift 5
TRAIN_ARGUMENTS="$@"

# Print the entire argument string
echo "TORCHRUN_ARGUMENTS: $TORCHRUN_ARGUMENTS"
echo "TRAIN_ARGUMENTS: $TRAIN_ARGUMENTS"

cd /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/

DIRECTORY="/global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/configs/model_configs/"
COMMAND="/global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/scripts/training_script.py --preshuffle --nocompile --lazy --config $DIRECTORY$TRAIN_ARGUMENTS"

eval "$(conda shell.bash hook)"
conda init bash
conda activate /opt/conda/envs/dgl

echo $COMMAND

torchrun \
    $TORCHRUN_ARGUMENTS \
    $COMMAND