#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 15:00:00
#SBATCH -A atlas
#SBATCH -o /global/cfs/projectdirs/atlas/joshua/gnn/root_gnn_dgl/jobs/slurm/%j.out # STDOUT

ARGUEMENTS="$*"

echo "Arguements: $ARGUEMENTS"




source /global/homes/j/joshuaho/launch_image.sh "--entrypoint /global/cfs/projectdirs/atlas/joshua/GNN4Colliders/root_gnn_dgl/jobs/training/singlegpu/run_job_image.sh" $ARGUEMENTS

#!/bin/bash

CONFIG=$1
shift
ARGUEMENTS="$*"

DIRECTORY="/global/cfs/projectdirs/atlas/joshua/GNN4Colliders/root_gnn_dgl/"
COMMAND="$DIRECTORY"scripts/training_script.py $ARGUEMENTS --preshuffle --nocompile --lazy --config $DIRECTORY$CONFIG

echo "Running my script now"
echo $COMMAND
python -u $COMMAND
echo "Done"