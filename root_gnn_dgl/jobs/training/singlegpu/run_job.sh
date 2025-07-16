#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --mail-user=ho22joshua@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 15:00:00
#SBATCH -A trn007
#SBATCH -o /global/cfs/projectdirs/atlas/joshua/GNN4Colliders/root_gnn_dgl/jobs/slurm/%j.out # STDOUT

ARGUEMENTS="$*"

echo "Arguements: $ARGUEMENTS"
echo "launching image"
source /global/homes/j/joshuaho/launch_image.sh "--entrypoint /global/cfs/projectdirs/atlas/joshua/GNN4Colliders/root_gnn_dgl/jobs/training/singlegpu/run_job_image.sh" $ARGUEMENTS