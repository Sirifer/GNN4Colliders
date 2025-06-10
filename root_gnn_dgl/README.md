# root_gnn_dgl

## Environment Setup

The environment dependencies for this project are listed in `setup/environment.yml`. Follow the steps below to set up the environment:

### Step 1: Install Conda
If you don’t already have Conda installed, install either Miniconda (lightweight) or Anaconda (full version):

- **Miniconda**: Download and install from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
- **Anaconda**: Download and install from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).

### Step 2: Clone the Repository
Clone this repository to your local machine:
```bash
git init
git lfs install
git clone https://huggingface.co/HWresearch/GNN4Colliders
```
If you want to clone without large files - just their pointers
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/HWresearch/GNN4Colliders
```

### Step 3: Create the Conda Environment
Use the `environment.yml` file to create the Conda environment:
```bash
conda env create -f setup/environment.yml -n <environment_name>
```

### Step 4: Activate the Environment
Activate the newly created environment:
```bash
conda activate <environment_name>
```
Replace <environment_name> with the name of the environment specified in Step 4.

### Step 5: Test the Environment
Run the `setup/test_setup.py` script to confirm that all packages needed for training are properly set up.
```bash
python setup/test_setup.py
```
## Running the Demo
The demo training is an example of our ML workflow, consisting of training a pretrained model, then finetuning it for an analysis task. The config files for the demo are located in the directory `configs/demo/`. The demo can be run on a login node.

The pretraining for the demo is a multiclass classification training on 12 datasets corresponding to 12 distinct physics processes, containing 10,000 simulated collision events each. The pretraining is then fintuned on a binary classification task between two datasets containing 10,000 simulated collision events each for two different processes, called ttH CP Even and ttH CP Odd.

The entire demo can be ran with the command
```bash
source run_demo.sh
``` 


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

## Training
