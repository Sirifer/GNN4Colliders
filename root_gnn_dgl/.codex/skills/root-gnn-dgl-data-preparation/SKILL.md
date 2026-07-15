---
name: root-gnn-dgl-data-preparation
description: Use when the user asks to build graphs or prebatched .bin files, rerun failed data prep, verify missing graph chunks, or use scripts/prep_data.py, scripts/check_dataset_files.py, or jobs/prep_data/prep_data.sh before training in root_gnn_dgl.
---

# root-gnn-dgl-data-preparation

Use this skill for graph creation and graph-readiness checks before training.

## Primary entry point

Run from the repo root:

```bash
python scripts/prep_data.py --config <config.yaml> --dataset <dataset_name> --chunk <chunk_index>
```

Use `--shuffle_mode` when you want preshuffled, prebatched graph files for training:

```bash
python scripts/prep_data.py --config <config.yaml> --dataset <dataset_name> --shuffle_mode --chunk <chunk_index>
```

## Recommended run pattern

- Read dataset names from `config["Datasets"]`.
- Read the chunk count from each dataset's `args.chunks`.
- When matching the repo's own wrappers, run chunk `0` once before the full loop. Both `run_demo.sh` and `jobs/prep_data/prep_data.sh` do this.

Example pattern:

```bash
python scripts/prep_data.py --config configs/stats_100K/pretraining_multiclass.yaml --dataset ttH --shuffle_mode --chunk 0
for i in 0 1 2; do
  python scripts/prep_data.py --config configs/stats_100K/pretraining_multiclass.yaml --dataset ttH --shuffle_mode --chunk "$i"
done
```

Use the repo wrapper when you want the standard loop:

```bash
bash jobs/prep_data/prep_data.sh <config> <dataset> <chunks> [extra_args]
```

## Important flags and caveats

- `--shuffle_mode` creates the prebatched artifacts consumed by `scripts/training_script.py --preshuffle`.
- `--drop_last` is inverted by the CLI definition: passing the flag sets `drop_last=False`.
- Dataset configs can override training batch size during prebatching with a dataset-level `batch_size`.
- The README says a `list index out of range` after graph saving is currently expected in some prep runs. Treat it as non-fatal if the output `.bin` files were written successfully.

## Dataset selections

Datasets may define event selections at the dataset level:

```yaml
Datasets:
  signal:
    args:
      ...
    selections:
      - [n_jets, 4, ">="]
      - "met_met_NOSYS > 30000"
```

Selection behavior:

- Selections are applied during data prep before graph chunking.
- `scripts/prep_data.py` prints a cutflow for each dataset before processing.
- Tuple/list selections use `[branch, cut, op]`, where `op` can be `>`, `>=`, `<`, `<=`, `==`, or `!=`.
- String selections are evaluated against loaded ROOT branches, so referenced branch names must exist.
- Selection branches are added automatically to the branch list through `selection_branches()`.
- Empty or omitted `selections` means all events pass.
- If a selection references vector branches, verify the result is one boolean per event; jagged per-object masks will not index event arrays correctly.

## Audit the outputs

Run from the repo root:

```bash
python scripts/check_dataset_files.py --configs stats_100K/pretraining_multiclass.yaml
```

The `--configs` argument must be a comma-separated list of paths relative to `configs/`.

This checker validates:

- chunk files named `${dataset}_${chunk}.bin`
- prebatched fold files named `${dataset}_prebatched_padded_${i}_n_${n_folds}_f_${foldlist}.bin`

Use rerun mode to repair missing artifacts:

```bash
python scripts/check_dataset_files.py --configs stats_100K/pretraining_multiclass.yaml --rerun
```

For bulk prep over every dataset in one or more configs, use:

```bash
python jobs/prep_data/run_processing.py configs/run_3_ttH/scratch.yaml configs/run_3_ttH/finetuning.yaml
```

This calls `jobs/prep_data/prep_data.sh` for each dataset using the dataset-level `shuffle_chunks` value.

Treat data prep as ready only if:

- every required chunk file exists
- every required prebatched fold file exists when training will use `--preshuffle`
- save paths match the config
- any post-save `IndexError` did not prevent the files from being written
