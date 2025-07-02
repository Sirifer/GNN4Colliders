import os
import subprocess
from mpi4py import MPI

# Define the number of GPUs per node
GPUS_PER_NODE = 4

# Define your files, configs, and branch namesfiles = [
files = [
    "merged_aMCPy8_tHjb125_CP_0_AF3_had_scaled.root",
    "merged_aMCPy8_tHjb125_CP_0_AF3_lep_scaled.root",
    "merged_aMCPy8_tHjb125_CP_0_FullSim_had_scaled.root",
    "merged_aMCPy8_tHjb125_CP_0_FullSim_lep_scaled.root",
    "merged_aMCPy8_tHjb125_CP_15_AF3_had_scaled.root",
    "merged_aMCPy8_tHjb125_CP_15_AF3_lep_scaled.root",
    "merged_aMCPy8_tHjb125_CP_30_AF3_had_scaled.root",
    "merged_aMCPy8_tHjb125_CP_30_AF3_lep_scaled.root",
    "merged_aMCPy8_tHjb125_CP_45_AF3_had_scaled.root",
    "merged_aMCPy8_tHjb125_CP_45_AF3_lep_scaled.root",
    "merged_aMCPy8_tHjb125_CP_45_FullSim_had_scaled.root",
    "merged_aMCPy8_tHjb125_CP_45_FullSim_lep_scaled.root",
    "merged_aMCPy8_tHjb125_CP_60_AF3_had_scaled.root",
    "merged_aMCPy8_tHjb125_CP_60_AF3_lep_scaled.root",
    "merged_aMCPy8_tHjb125_CP_75_AF3_had_scaled.root",
    "merged_aMCPy8_tHjb125_CP_75_AF3_lep_scaled.root",
    "merged_aMCPy8_tHjb125_CP_90_AF3_had_scaled.root",
    "merged_aMCPy8_tHjb125_CP_90_AF3_lep_scaled.root",
    "merged_aMCPy8_ttH125_CP_0_AF3_had_scaled.root",
    "merged_aMCPy8_ttH125_CP_0_AF3_lep_scaled.root",
    "merged_aMCPy8_ttH125_CP_0_FullSim_had_scaled.root",
    "merged_aMCPy8_ttH125_CP_0_FullSim_lep_scaled.root",
    "merged_aMCPy8_ttH125_CP_15_AF3_had_scaled.root",
    "merged_aMCPy8_ttH125_CP_15_AF3_lep_scaled.root",
    "merged_aMCPy8_ttH125_CP_30_AF3_had_scaled.root",
    "merged_aMCPy8_ttH125_CP_30_AF3_lep_scaled.root",
    "merged_aMCPy8_ttH125_CP_45_AF3_had_scaled.root",
    "merged_aMCPy8_ttH125_CP_45_AF3_lep_scaled.root",
    "merged_aMCPy8_ttH125_CP_45_FullSim_had_scaled.root",
    "merged_aMCPy8_ttH125_CP_45_FullSim_lep_scaled.root",
    "merged_aMCPy8_ttH125_CP_60_AF3_had_scaled.root",
    "merged_aMCPy8_ttH125_CP_60_AF3_lep_scaled.root",
    "merged_aMCPy8_ttH125_CP_75_AF3_had_scaled.root",
    "merged_aMCPy8_ttH125_CP_75_AF3_lep_scaled.root",
    "merged_aMCPy8_ttH125_CP_90_AF3_had_scaled.root",
    "merged_aMCPy8_ttH125_CP_90_AF3_lep_scaled.root",
    "merged_aMCPy8_tWH125_CP_0_AF3_had_scaled.root",
    "merged_aMCPy8_tWH125_CP_0_AF3_lep_scaled.root",
    "merged_aMCPy8_tWH125_CP_0_FullSim_had_scaled.root",
    "merged_aMCPy8_tWH125_CP_0_FullSim_lep_scaled.root",
    "merged_aMCPy8_tWH125_CP_15_AF3_had_scaled.root",
    "merged_aMCPy8_tWH125_CP_15_AF3_lep_scaled.root",
    "merged_aMCPy8_tWH125_CP_30_AF3_had_scaled.root",
    "merged_aMCPy8_tWH125_CP_30_AF3_lep_scaled.root",
    "merged_aMCPy8_tWH125_CP_45_AF3_had_scaled.root",
    "merged_aMCPy8_tWH125_CP_45_AF3_lep_scaled.root",
    "merged_aMCPy8_tWH125_CP_45_FullSim_had_scaled.root",
    "merged_aMCPy8_tWH125_CP_45_FullSim_lep_scaled.root",
    "merged_aMCPy8_tWH125_CP_60_AF3_had_scaled.root",
    "merged_aMCPy8_tWH125_CP_60_AF3_lep_scaled.root",
    "merged_aMCPy8_tWH125_CP_75_AF3_had_scaled.root",
    "merged_aMCPy8_tWH125_CP_75_AF3_lep_scaled.root",
    "merged_aMCPy8_tWH125_CP_90_AF3_had_scaled.root",
    "merged_aMCPy8_tWH125_CP_90_AF3_lep_scaled.root",
    "merged_hadnti_data.root",
    "merged_hadti_data.root",
    "merged_lepnti_data.root",
    "merged_lepti_data.root",
    "merged_MGPy8_ttyy_combined_FullSim_had_scaled.root",
    "merged_MGPy8_ttyy_combined_FullSim_lep_scaled.root",
    "merged_PhPy8_4FS_A14NNPDF_bbH125_FullSim_had_scaled.root",
    "merged_PhPy8_4FS_A14NNPDF_bbH125_FullSim_lep_scaled.root",
    "merged_PhPy8_ggH125_NNLOPS_FullSim_had_scaled.root",
    "merged_PhPy8_ggH125_NNLOPS_FullSim_lep_scaled.root",
    "merged_PhPy8_ggZH125_FullSim_had_scaled.root",
    "merged_PhPy8_ggZH125_FullSim_lep_scaled.root",
    "merged_PhPy8_VBFH125_FullSim_had_scaled.root",
    "merged_PhPy8_VBFH125_FullSim_lep_scaled.root",
    "merged_PhPy8_WmH125J_MINLO_FullSim_had_scaled.root",
    "merged_PhPy8_WmH125J_MINLO_FullSim_lep_scaled.root",
    "merged_PhPy8_WpH125J_MINLO_FullSim_had_scaled.root",
    "merged_PhPy8_WpH125J_MINLO_FullSim_lep_scaled.root",
    "merged_PhPy8_ZH125J_MINLO_FullSim_had_scaled.root",
    "merged_PhPy8_ZH125J_MINLO_FullSim_lep_scaled.root",
]
# Configurations for each type of file
config = [
    "configs/model_configs/run_3_ttH/v06/sb_yukawa_no_weights.yaml",
    "configs/model_configs/run_3_ttH/v06/sb_yukawa_weights.yaml",
    "configs/model_configs/run_3_ttH/v06/tHjb_CP_no_weights.yaml",
    "configs/model_configs/run_3_ttH/v06/tHjb_CP_weights.yaml",
    "configs/model_configs/run_3_ttH/v06/ttH_CP_no_weights.yaml",
    "configs/model_configs/run_3_ttH/v06/ttH_CP_weights.yaml",
    "configs/model_configs/run_3_ttH/v06/ttH_tHjb_tWH_CP_no_weights.yaml",
    "configs/model_configs/run_3_ttH/v06/ttH_tHjb_tWH_CP_weights.yaml",
    "configs/model_configs/run_3_ttH/v06/tWH_CP_no_weights.yaml",
    "configs/model_configs/run_3_ttH/v06/tWH_CP_weights.yaml",

]

# Branch names for each type of file
branch_name = [
    "sb_yukawa_no_weights_score",
    "sb_yukawa_weights_score",
    "tHjb_CP_no_weights_score",
    "tHjb_CP_weights_score",
    "ttH_CP_no_weights_score",
    "ttH_CP_weights_score",
    "ttH_tHjb_tWH_CP_no_weights_score",
    "ttH_tHjb_tWH_CP_weights_score",
    "tWH_CP_no_weights_score",
    "tWH_CP_weights_score",
]

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank (unique ID for each process)
size = comm.Get_size()  # Total number of processes across all nodes

# Master process manages the task queue
def master():
    num_workers = size - 1  # Exclude the master process
    task_queue = files[:]  # Copy the list of files
    workers_busy = [False] * num_workers  # Track if workers are busy

    print(f"Master process starting with {num_workers} workers.")

    # Distribute initial tasks to workers
    for worker_rank in range(1, size):
        if task_queue:
            # Send a file to the worker
            file = task_queue.pop(0)
            comm.send(file, dest=worker_rank, tag=0)
            workers_busy[worker_rank - 1] = True
        else:
            # No tasks left, send termination signal
            comm.send(None, dest=worker_rank, tag=1)

    # Process results and assign new tasks dynamically
    while any(workers_busy):
        # Receive results from workers
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker_rank = status.Get_source()
        workers_busy[worker_rank - 1] = False  # Worker is now free

        # Assign a new task if available
        if task_queue:
            file = task_queue.pop(0)
            comm.send(file, dest=worker_rank, tag=0)
            workers_busy[worker_rank - 1] = True
        else:
            # No tasks left, send termination signal
            comm.send(None, dest=worker_rank, tag=1)

    print("Master process: All tasks completed.")

# Worker processes handle tasks dynamically
def worker():
    gpu_id = (rank - 1) % GPUS_PER_NODE  # Assign GPU ID based on rank
    print(f"Worker {rank} starting, GPU ID: {gpu_id}", flush=True)

    while True:
        # Receive a task from the master
        file = comm.recv(source=0, tag=MPI.ANY_TAG)
        if file is None:
            # Termination signal received
            print(f"Worker {rank}: Termination signal received.")
            break

        # Initialize config and branch_name for each file
        local_config = config[:]  # Create a copy of the global config
        local_branch_name = branch_name[:]  # Create a copy of the global branch_name

        # Determine file type and append appropriate configurations
        if "had" in file:
            local_config += had_config  # Append had-specific configurations
            local_branch_name += had_branch_name  # Append had-specific branch names
        elif "lep" in file:
            local_config += lep_config  # Append lep-specific configurations
            local_branch_name += lep_branch_name  # Append lep-specific branch names

        # Process the file
        print(f"Worker {rank}: Processing file {file} on GPU {gpu_id}", flush=True)
        print(f"branches = {local_branch_name}")
        target = f"/global/cfs/projectdirs/atlas/joshua/ttHCP/ntuples/v02/preselection/merged/{file}"
        destination = f"/global/cfs/projectdirs/atlas/joshua/ttHCP/ntuples/scores/gnn_v06/{file}"
        
        # Set the GPU for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Build the command
        command = [
            "python", "scripts/inference.py",
            "--target", target,
            "--destination", destination,
            "--config", *local_config,
            "--branch_name", *local_branch_name,
            "--chunks", "1",
            "--chunkno", "0",
            "--write"
        ]
        
        # Execute the command
        subprocess.run(command)
        
        # Notify the master that the task is complete
        comm.send(f"Worker {rank}: Finished {file}", dest=0, tag=0)

# Main function
if __name__ == "__main__":
    if rank == 0:
        master()
    else:
        worker()