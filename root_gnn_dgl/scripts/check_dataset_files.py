import yaml
import os
import subprocess
import argparse

def check_dataset_files(yaml_file, rerun = False):
    """
    Check if all required .bin files exist for each dataset in the YAML file.
    """
    try:
        # Open and parse the YAML file
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        # Check if 'Datasets' exists in the YAML file
        if 'Datasets' not in config:
            print(f"No 'Datasets' section found in {yaml_file}.")
            return

        datasets = config['Datasets']
        all_files_exist = True

        for dataset_name, dataset_config in datasets.items():
            # Extract required information
            save_dir = dataset_config['args']['save_dir']
            chunks = dataset_config['args']['chunks']
            folding = dataset_config.get('folding', {})
            n_folds = folding.get('n_folds', 0)
            test_folds = folding.get('test', [])
            train_folds = folding.get('train', [])

            # Check for chunk files
            missing_files = []
            for chunk in range(chunks):
                chunk_file = os.path.join(save_dir, f"{dataset_name}_{chunk}.bin")
                if not os.path.exists(chunk_file):
                    missing_files.append(chunk_file)

            # Check for prebatched files for each fold
            for fold in range(n_folds):
                # Test files
                if fold in test_folds:
                    test_file = os.path.join(save_dir, f"{dataset_name}_prebatched_padded_{fold}_n_{n_folds}_f_{'_'.join(map(str, test_folds))}.bin")
                    if not os.path.exists(test_file):
                        missing_files.append(test_file)

                # Train files
                if fold in train_folds:
                    train_file = os.path.join(save_dir, f"{dataset_name}_prebatched_padded_{fold}_n_{n_folds}_f_{'_'.join(map(str, train_folds))}.bin")
                    if not os.path.exists(train_file):
                        missing_files.append(train_file)

            # Print results for the current dataset
            if missing_files:
                all_files_exist = False
                print(f"Missing files for dataset '{dataset_name}':")
                for missing_file in missing_files:
                    print(f"  - {missing_file}")
                
                # Reprocess the dataset with drop_last=False
                if (rerun):
                    print(f"Reprocessing dataset '{dataset_name}")
                    prep_command = f"bash/prep_data.sh {yaml_file} {dataset_name} {chunks}"
                    try:
                        subprocess.run(prep_command, shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Could Not Reprocess '{dataset_name}': {e}")
            else:
                print(f"All files exist for dataset '{dataset_name}'.")

        # Final summary
        if all_files_exist:
            print("All required files exist for all datasets.")
        else:
            print("Some files are missing.")

    except Exception as e:
        print(f"Error processing {yaml_file}: {e}")

def main(pargs):
    # Base directory containing the YAML files
    base_directory = os.getcwd() + "/configs/"

    if (pargs.configs):
        # If pargs is a comma-separated list, split it
        configs = [p.strip() for p in pargs.configs.split(',')]

    else:
        # List of YAML configuration files (relative to the base directory)
        configs = [
            "attention/ttH_CP_even_vs_odd.yaml",

            "stats_100K/finetuning_ttH_CP_even_vs_odd.yaml",
            "stats_100K/pretraining_multiclass.yaml",
            "stats_100K/ttH_CP_even_vs_odd.yaml",

            "stats_all/finetuning_ttH_CP_even_vs_odd.yaml",
            "stats_all/pretraining_multiclass.yaml",
            "stats_all/ttH_CP_even_vs_odd.yaml",
        ]

    # Loop through each config file
    for config in configs:
        yaml_file = os.path.join(base_directory, config)  # Construct full path
        if os.path.exists(yaml_file):
            print(f"Processing file: {config}")
            check_dataset_files(yaml_file, pargs.rerun)
        else:
            print(f"File not found: {yaml_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check YAML config files")
    parser.add_argument(
        "--configs", "-c",
        type=str,
        required=False,
        help="Comma-separated list of YAML config paths relative to base directory"
    )
    parser.add_argument(
        "--rerun", "-r",
        type=bool,
        default=False,
        required=False,
        help="Automatically re-runs data processing to fix missing files"
    )
    args = parser.parse_args()
    main(args)