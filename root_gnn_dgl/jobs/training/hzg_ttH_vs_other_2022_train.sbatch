#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg_ttH_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=v100:1
#SBATCH --time=8:00:00
#SBATCH -o /scratchfs/cms/huangsuyun/GNN4Colliders/logs/hzg-train-%j.out

set -e

source /cms/user/huangsuyun/app/miniconda3/etc/profile.d/conda.sh
conda activate /publicfs/cms/user/huangsuyun/conda/envs/pytorch

export MPLCONFIGDIR=/scratchfs/cms/huangsuyun/GNN4Colliders/cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "$MPLCONFIGDIR"

cd /cms/user/huangsuyun/GNN4Colliders/root_gnn_dgl

srun python scripts/training_script.py \
  --config configs/hzg/ttH_vs_other_small.yaml \
  --preshuffle \
  --nocompile \
  --lazy \
  --restart
