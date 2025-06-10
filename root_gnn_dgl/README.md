# root_gnn_dgl

## Environment Setup

The environment dependencies for this project are listed in `environment.yml`. Follow the steps below to set up the environment:

### Step 1: Install Conda
If you don’t already have Conda installed, install either Miniconda (lightweight) or Anaconda (full version):

- **Miniconda**: Download and install from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
- **Anaconda**: Download and install from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).

### Step 2: Clone the Repository
Clone this repository to your local machine:
```bash
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
