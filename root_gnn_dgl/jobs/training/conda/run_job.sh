#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 15:00:00
#SBATCH -A atlas
#SBATCH -o /global/cfs/projectdirs/atlas/joshua/gnn/root_gnn_dgl/jobs/slurm/%j.out # STDOUT

CONFIG=$1
shift
ARGUEMENTS="$*"

DIRECTORY="/global/cfs/projectdirs/atlas/joshua/gnn/root_gnn_dgl/"
BASE_COMMAND="$DIRECTORY/training_script.py $ARGUEMENTS --preshuffle --nocompile --lazy --config $DIRECTORY"

echo "launched image"
cd $DIRECTORY

eval "$(conda shell.bash hook)"
conda init bash
conda activate /opt/conda/envs/dgl

COMMAND="$BASE_COMMAND$CONFIG"

echo "Running my script now"
echo $COMMAND
python -u $COMMAND
echo "Done"