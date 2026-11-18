#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-merge-trees
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/merge_trees-%j.out

set -euo pipefail

mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs

export PYTHONUNBUFFERED=1

PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python
SCRIPT=/scratchfs2/cms/huangsuyun/GNN4Colliders/tools/code/merge_trees_all.py

echo "=== Start merge trees ==="
echo "Python: ${PYTHON}"
echo "Script: ${SCRIPT}"

${PYTHON} ${SCRIPT}

echo "=== All done ==="