python scripts/prep_data.py \
    --config configs/hzg/你的配置.yaml \
    --dataset ttH_SR_ttHhad \
    --shuffle_mode \
    --chunk 0

python scripts/training_script.py \
  --config configs/hzg/ttH_vs_other_small.yaml \
  --preshuffle \
  --nocompile \
  --lazy \
  --restart


  python scripts/prep_data.py \
  --config configs/hzg/ttH_vs_other_small.yaml \
  --dataset ttH_SR_VH3l \
  --chunk 0

python scripts/prep_data.py \
  --config configs/hzg/ttH_vs_other_small.yaml \
  --dataset ttH_SR_VH3l \
  --chunk 1

python scripts/prep_data.py \
  --config configs/hzg/ttH_vs_other_small.yaml \
  --dataset ttH_SR_VH3l \
  --shuffle_mode \
  --drop_last \
  --chunk 0


  cd /cms/user/huangsuyun/GNN4Colliders/root_gnn_dgl

MAX_JOBS=1 sbatch \
jobs/prep_data/hzg_ttH_vs_other_small_prep_simple.sbatch