from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import os
import subprocess

def extract_dataset_keys_and_chunks(yaml_file):
    """
    Extract keys and shuffle_chunks under the 'Datasets' section from a YAML file.
    """
    try:
        # Open and parse the YAML file
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        # Check if 'Datasets' exists in the YAML file
        if 'Datasets' in config:
            datasets = config['Datasets']
            results = []
            for key, value in datasets.items():
                # Extract shuffle_chunks if available
                shuffle_chunks = value.get('shuffle_chunks', None)
                results.append((key, shuffle_chunks))
            return results
        else:
            print(f"No 'Datasets' section found in {yaml_file}.")
            return []
    except Exception as e:
        print(f"Error reading {yaml_file}: {e}")
        return []

def call_bash_script(config, dataset_key, shuffle_chunks, bash_script):
    """
    Call a bash script with dataset_key and shuffle_chunks as arguments.
    """
    try:
        # Construct the command to call the bash script
        command = f"{bash_script} {config} {dataset_key} {shuffle_chunks}"

        print(f"Executing command: {command}")  # Debugging: Print the command being executed

        # Run the command
        result = subprocess.run(command, shell=True, check=True)
        return f"Success: {dataset_key}"
    except subprocess.CalledProcessError as e:
        return f"Error executing bash script for dataset {dataset_key}: {e.stderr.decode()}"

def process_yaml_file(config, base_directory, bash_script):
    """
    Process a single YAML file by extracting datasets and calling the bash script in parallel.
    """
    yaml_file = os.path.join(base_directory, config)  # Construct full path
    if os.path.exists(yaml_file):
        print(f"Processing file: {config}")
        datasets = extract_dataset_keys_and_chunks(yaml_file)

        # Use ThreadPoolExecutor to parallelize bash script calls for datasets
        max_workers = min(len(datasets), os.cpu_count())  # Limit workers to number of datasets or CPU cores
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(call_bash_script, config, key, shuffle_chunks, bash_script)
                for key, shuffle_chunks in datasets
            ]

            # Collect and print results as they complete
            results = []
            for future in as_completed(futures):
                results.append(future.result())
            return results
    else:
        return [f"File not found: {yaml_file}"]

def main():
    # Base directory containing the YAML files
    base_directory = os.getcwd() + "/"

    # List of YAML configuration files (relative to the base directory)
    configs = [
        # "configs/stats_100K/pretraining_multiclass.yaml",
        # "configs/stats_100K/ttH_CP_even_vs_odd.yaml",
        "configs/stats_all/pretraining_multiclass.yaml",
        "configs/stats_all/ttH_CP_even_vs_odd.yaml",
        # "configs/attention/ttH_CP_even_vs_odd.yaml",
    ]

    # Path to the bash script to be called
    bash_script = base_directory + "jobs/prep_data/prep_data.sh"

    # Use ThreadPoolExecutor to process YAML files concurrently
    max_workers = os.cpu_count()  # Use all available CPU cores
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_yaml_file, config, base_directory, bash_script)
            for config in configs
        ]

        # Collect and print results as they complete
        for future in as_completed(futures):
            results = future.result()
            for result in results:
                print(result)

if __name__ == "__main__":
    main()