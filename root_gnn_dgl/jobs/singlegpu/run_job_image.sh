#!/bin/bash

CONFIG=$1
shift
ARGUEMENTS="$*"

DIRECTORY="/global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/configs/model_configs/"
BASE_COMMAND="/global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/scripts/training_script.py $ARGUEMENTS --preshuffle --nocompile --lazy --config $DIRECTORY"

echo "launched image"
cd /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/

COMMAND="$BASE_COMMAND$CONFIG"

eval "$(conda shell.bash hook)"
conda init bash
conda activate /opt/conda/envs/dgl

echo "Running my script now"
echo $COMMAND
python -u $COMMAND
echo "Done"