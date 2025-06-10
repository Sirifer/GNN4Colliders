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

## Data Preparation
The first step in the process is to convert the events stored in ROOT files into DGL graph objects. This conversion is handled automatically by the Dataset objects during their creation, provided the graph data has not already been saved to disk. To accomplish this, a simple script is used to initialize the relevant Dataset object and then exit. This script needs to be executed for each data chunk in each dataset being used for training.

Below is an example of how to use the `scripts/prep_data.py` script:

```bash
<insert exmaple here>
```

The `--shuffle_mode` flag performs shuffling and pre-batches the graphs in each chunk, since holding the entire dataset in memory and shuffling it together can be prohibitive for large datasets.

## Training
