from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import yaml
import os
import subprocess
from pathlib import Path

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
        command = [bash_script, config, dataset_key, str(shuffle_chunks)]

        print(f"Executing command: {' '.join(command)}")

        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        return f"Success: {dataset_key}"
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else "")
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode() if e.stdout else "")
        message = f"Error executing bash script for dataset {dataset_key}: {stderr or stdout or str(e)}"
        return message

def process_yaml_file(config, base_directory, bash_script):
    """
    Process a single YAML file by extracting datasets and calling the bash script in parallel.
    """
    yaml_file = os.path.join(base_directory, config)
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
    parser = argparse.ArgumentParser(description="Run prep_data.sh for every dataset in one or more configs.")
    parser.add_argument("configs", nargs="+", help="YAML config files, relative to the repo root or absolute.")
    parser.add_argument(
        "--bash-script",
        default=None,
        help="Path to prep_data.sh. Defaults to jobs/prep_data/prep_data.sh under the repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    base_directory = str(repo_root) + "/"
    configs = [str(Path(config).resolve()) if Path(config).is_absolute() else config for config in args.configs]
    bash_script = args.bash_script or str(repo_root / "jobs/prep_data/prep_data.sh")

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
