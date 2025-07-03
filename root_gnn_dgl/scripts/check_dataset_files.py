import yaml
import os
import subprocess
import argparse

def check_dataset_files(yaml_file, rerun=False):
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

            print(f"\n== Checking dataset: {dataset_name} ==")
            print(f"  save_dir: {save_dir}")
            print(f"  chunks: {chunks}")
            print(f"  n_folds: {n_folds}")
            print(f"  test_folds: {test_folds}")
            print(f"  train_folds: {train_folds}")

            missing_files = []

            # 1. Check for chunk files
            for chunk in range(chunks):
                chunk_file = os.path.join(save_dir, f"{dataset_name}_{chunk}.bin")
                if not os.path.exists(chunk_file):
                    missing_files.append(chunk_file)

            # 2. Check for prebatched fold files (test and train)
            #    Naming: dataset_name_prebatched_padded_{fold}_n_{n_folds}_f_{foldlist}.bin
            fold_types = [('test', test_folds), ('train', train_folds)]
            for fold_type, folds in fold_types:
                if not folds:
                    continue
                foldlist_str = '_'.join(map(str, folds))
                for i in range(chunks):
                    prebatched_file = os.path.join(
                        save_dir,
                        f"{dataset_name}_prebatched_padded_{i}_n_{n_folds}_f_{foldlist_str}.bin"
                    )
                    if not os.path.exists(prebatched_file):
                        missing_files.append(prebatched_file)

            # Print results for the current dataset
            if missing_files:
                all_files_exist = False
                print(f"  Missing files for dataset '{dataset_name}':")
                for missing_file in missing_files:
                    print(f"    - {missing_file}")

                # Optionally rerun data prep
                if rerun:
                    print(f"  Reprocessing dataset '{dataset_name}' ...")
                    prep_command = f"bash/prep_data.sh {yaml_file} {dataset_name} {chunks}"
                    try:
                        subprocess.run(prep_command, shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"  Could NOT reprocess '{dataset_name}': {e}")
            else:
                print(f"  All files exist for dataset '{dataset_name}'.")

        # Final summary
        if all_files_exist:
            print("\nAll required files exist for all datasets.")
        else:
            print("\nSome files are missing.")

    except Exception as e:
        print(f"Error processing {yaml_file}: {e}")

def main(pargs):
    # Base directory containing the YAML files
    base_directory = os.getcwd() + "/configs/"

    if pargs.configs:
        configs = [p.strip() for p in pargs.configs.split(',')]
    else:
        configs = [
            "attention/ttH_CP_even_vs_odd.yaml",

            "stats_100K/finetuning_ttH_CP_even_vs_odd.yaml",
            "stats_100K/pretraining_multiclass.yaml",
            "stats_100K/ttH_CP_even_vs_odd.yaml",

            "stats_all/finetuning_ttH_CP_even_vs_odd.yaml",
            "stats_all/pretraining_multiclass.yaml",
            "stats_all/ttH_CP_even_vs_odd.yaml",
        ]

    for config in configs:
        yaml_file = os.path.join(base_directory, config)
        if os.path.exists(yaml_file):
            print(f"\nProcessing file: {config}")
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
        action='store_true',   # Correct way for a boolean flag
        help="Automatically re-run data processing to fix missing files"
    )
    args = parser.parse_args()
    main(args)