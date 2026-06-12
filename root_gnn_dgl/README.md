
# root_gnn_dgl

Pretrained DGL-based ROOT graph neural network. 

Pretrained model location: `/global/cfs/projectdirs/atlas/joshua/Pretrained_GNN/multiclass_pretrained_model_12/`
To use the pretrained model, take a look at a finetuning config in `configs`.
Replace `pretraining_path:` with `/global/cfs/projectdirs/atlas/joshua/Pretrained_GNN/multiclass_pretrained_model_12/model_epoch_71.pt`.

## Overview
- Stable release with pretrained model weights.

## Conda setup

The conda environment is required for the inference step: applying the GNN onto root files and saving GNN scores as an additional branch. This is because the infereces script uses pyROOT.

```bash
cd setup
conda env create -f environment.yml
conda activate pytorch
cd ..
python setup/test_setup.py 
```

##  Container Setup (Podman-HPC)

- NERSC Perlmutter environment with `podman-hpc` available.
- Access to `joshuaho/pytorch:1.0` on Docker Hub [https://hub.docker.com/r/joshuaho/pytorch](https://hub.docker.com/r/joshuaho/pytorch)

The inference step requires the conda environment, since the container does not contain ROOT.

### Pull the Prebuilt Image

```bash
podman-hpc pull docker.io/joshuaho/pytorch:1.0
```

Or, you can build your own container here:

```bash
cd setup
source build_image.sh
```

Run the image and mount the paths you need, replaceing `<source>` with source directory path and `<target>` with the path for when you are inside the container.
```bash

podman-hpc run \
  -it \
  --mount type=bind,source=<source>,target=<target> \
  --rm \
  --network host \
  --gpu \
  --userns keep-id \
  --shm-size=32g \
  joshuaho/pytorch:1.0
  ```

### Test the Environment
Run the `setup/test_setup.py` script to confirm that all packages needed for training are properly set up.
```bash
python setup/test_setup.py
```


## Running the Demo
The demo training is an example of our ML workflow, consisting of training a pretrained model, then finetuning it for an analysis task, while also training a model for the analysis task from scratch. The config files for the demo are located in the directory `configs/stats_100K/`. The demo can be run on a login node on Perlmutter (if enough GPU memory is availble).

To check login node GPU memory availability, use the command `nvidia-smi`. If there is not enough memory available, you can switch to another login node with the command `ssh login**`, where `**` is a number between 0 and 39.

For better performance, it is recommended to run the training and inference of the demo on a shared interactive node, where you have access to one exclusive GPU. An interactive node can be requested using the shell script in `jobs/interactive.sh`.

The pretraining for the demo is a multiclass classification training on 12 datasets corresponding to 12 distinct physics processes, containing 100,000 simulated collision events each. The pretraining is then fintuned on a binary classification task between two datasets containing 100,000 simulated collision events each for two different processes, called ttH CP Even and ttH CP Odd.

The entire demo can be ran with the command
```bash
source run_demo.sh
``` 

This shell script can also be used as an example to run the entire workflow.

## Data Preparation
The first step in the process is to convert the events stored in ROOT files into DGL graph objects. This conversion is handled automatically by the Dataset objects during their creation, provided the graph data has not already been saved to disk. To accomplish this, a simple script is used to initialize the relevant Dataset object and then exit. This script needs to be executed for each data chunk in each dataset being used for training.

Below is an example of how to use the `scripts/prep_data.py` script:

```bash
datasets=("ttH" "tHjb" "ggF" "VBF" "WH" "ZH" "ttyy" "tttt" "SingleT_schan" "ttbar" "ttW" "ttt")
chunks=3

for data in "${datasets[@]}"; do
    python scripts/prep_data.py --config configs/demo/pretraining_multiclass.yaml --dataset "$data" --shuffle_mode --chunk 0
    for ((i=0; i<chunks; i++)); do
        python scripts/prep_data.py --config configs/demo/pretraining_multiclass.yaml --dataset "$data" --shuffle_mode --chunk "$i"
    done
done
```

The `--shuffle_mode` flag performs shuffling and pre-batches the graphs in each chunk, since holding the entire dataset in memory and shuffling it together can be prohibitive for large datasets.

You will probably see a `list index out of range error`. This is expected, and the program will still save your graphs to disk before throwing this error. Fixing this error is a WIP, but for now can be ignored
```python
dgl.save_graphs(str(graph_path).replace('.bin', f'_{self.process_chunks[i]}.bin'), self.graph_chunks[i], {'labels': self.label_chunks[i], 'tracking': self.tracking_chunks[i], 'global': self.global_chunks[i]})
IndexError: list index out of range
``` 

To make sure you have all the necessary graphs to train, you can use the `scripts/check_dataset_files.py` script to ensure all graphs are properly processed. Using the `--rerun` runtime arguement will tell the script to automically re-processes any missing files.

### Selections
Dataset-level event selections are defined in the config file at the same level as `args:` and `folding:`. They are applied during data preparation before graphs are built, and they are also reported as a cutflow after processing.

Example:
```yaml
Datasets:
  ttH_CP_even:
    module: root_gnn_base.dataset
    class: LazyDataset
    args:
      name: ttH_CP_even
      # ...
    selections:
      - "N_jet >= 5"
      - "(123 < higgs_m) & (higgs_m < 127)"
    folding:
      n_folds: 4
      test: [0]
      train: [1, 2, 3]
```

Supported selection syntax:
- Simple comparisons like `"N_jet >= 5"`, `"higgs_m < 200"`, or `"MET_met == 0"`
- Multiple conditions combined with `&` and parentheses, for example `"(123 < higgs_m) & (higgs_m < 127)"`

To quickly test that the selections in a config are valid and to print a fast cutflow for every dataset, run:
```bash
python scripts/selections.py --config configs/stats_100K/ttH_CP_even_vs_odd.yaml
```

This script reads only the branches needed for the selections and prints:
- the status of each selection expression
- the yield after each cut
- the individual efficiency of each cut
- the total efficiency relative to the uncut sample

## Training
Training is run by `scripts/training_script`. `--preshuffle` tells it to use the preshuffled and batched graphs rather than shuffling and batching on the fly, and `--restart` can be used to force the training to start from the beginning rather than from the last available checkpoint.

Using the `--nocompile` arguement is also recommended, as using `torch.compile()` requires padding the graphs beforehand during data processing.

```bash
python scripts/training_script.py --config configs/stats_100K/pretraining_multiclass.yaml --preshuffle --nocompile --lazy
```

This step should produce the training directory `trainings/stats_100K/pretraining_multiclass/` containing a copy of the config file, checkpoints (`model_epoch_*.pt`) with the model weights after each epoch of training, npz files with the GNN outputs for each event after each epoch of training, and two files `training.log` and `training.png` which summarize the model performance and convergence.

## Inference
Inference is done by `scripts/inference.py`. This script applies the model defined by `--config` onto the samples located at `--target`. A new set of samples with the GNN scores saved as the `--branch` in the ntuples will be created at `--destination`. The `--chunks` arguement will handel the inference in specified chunks.

```bash
python scripts/inference.py \
    --target "/global/cfs/projectdirs/atlas/joshua/gnn_data/stats_100K/ttH_NLO.root" \
    --destination "/global/cfs/projectdirs/atlas/joshua/gnn_data/scores/stats_100K/ttH_NLO.root" \
    --config "configs/stats_100K/finetuning_ttH_CP_even_vs_odd.yaml" \
    --branch_name "GNN_Score" \
    --chunks 1 \
    --chunkno 0 \
    --write
```

You can also input a list as the `--config` and the `--branch_name` to simultaneously apply multiple models onto the same set of samples. An example on how to do this in shell script is in the `run_demo.sh` file.

## ONNX Export
The `scripts/onnx_export.py` script exports a trained model to ONNX and validates the exported model against PyTorch on a real batch from the processed dataset.

Example:
```bash
python scripts/onnx_export.py --config configs/stats_100K/ttH_CP_even_vs_odd.yaml --name ttH.onnx
```

To export a specific checkpoint instead of the best `Test_AUC` epoch, pass `--epoch`:
```bash
python scripts/onnx_export.py --config configs/stats_100K/ttH_CP_even_vs_odd.yaml --name ttH_epoch68.onnx --epoch 68
```

What it does:
- Reads the model definition from the provided config
- Automatically chooses the matching model class from the config, including the finetuning or from-scratch path
- Loads the first prebatched chunk from the dataset save directory
- Uses the graph node count per event as `global_features`
- Exports an ONNX model with inputs in this order:
  1. `node_features`
  2. `edge_features`
  3. `global_features`
  4. `edge_index`
  5. `node_batch`
- Checks the ONNX graph with `onnx.checker`
- Runs ONNX Runtime and compares the output to PyTorch

If `--epoch` is omitted, the script loads the checkpoint from the best `Test_AUC` entry in `training.log`, which matches the previous behavior.

The validation is a three-way comparison:
1. The original DGL model from training
2. The tensor-based PyTorch translation of the same GNN
3. The exported ONNX model

The script compares both logits and final sigmoid scores across all three models. It also produces a diagnostic plot showing:
- score distributions for the DGL, tensor, and ONNX models
- residual distributions for tensor - DGL and ONNX - DGL

The plot is saved automatically next to the ONNX file using the same stem, with `_onnx.png` appended.

Input shapes follow the notebook convention:
- `node_features`: `[num_nodes, node_feature_dim]`
- `edge_features`: `[num_edges, edge_feature_dim]`
- `global_features`: `[batch_size, 1]`
- `edge_index`: `[2, num_edges]`
- `node_batch`: `[num_nodes]`

The validation prints the max and mean absolute differences for both logits and sigmoid probabilities. For a healthy export, those differences should be very small, typically around `1e-5` or better for logits.

## Running Jobs + Parallelization

Perlmutter job scripts are located in `jobs/`. Job scripts are separated into 3 categories: `prep_data`, `training`, and `inference`. 

The different shell scripts show how to request GPU or CPU nodes from Perlmutter, which are reqired for running jobs. 

### Data Prep Parallelization

The preparation of data can be parallelized across several threads on a CPU. The parallelization is handled by python's `concurrent.futures.ThreadPoolExecutor`.

### Training Parallelization

Parallelization of GNN training is implemented with `torch.DistributedDataParallel`. The job submission script is in `jobs/training/multinode/submit.sh`. 

When running a multinode training, remember to use the `--multinode` run-time arguement for the training script.

### Inference Parallelization

Model inference parallelization is done with `mpi4py` (currently not listed in the conda environment requirements). You can run the parallel inference script with `mpirun -np <num_nodes> python jobs/inference/run_inference.py`.
