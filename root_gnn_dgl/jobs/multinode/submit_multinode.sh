date

DIRECTORY="/global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/configs/model_configs/"

configs=(
  'pretraining_multilabel/multilabel_5_particle_counting.yaml --restart --multinode'
  'pretraining_multilabel/multilabel_17_higgs_kinematics.yaml --restart --multinode'
  'pretraining_multilabel/multilabel_29_top_kinematics.yaml --restart --multinode'
  'pretraining_multilabel/multilabel_41_higgs_tops_all_kinematics.yaml --restart --multinode'
)

counter=0

for job in "${configs[@]}"
do
  sbatch --job-name="$job" \
        /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/jobs/run_multinode_1.sh "$job"
  ((counter++))
done

echo "Total jobs submitted: $counter"