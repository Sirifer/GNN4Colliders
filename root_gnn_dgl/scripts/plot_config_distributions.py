#!/usr/bin/env python3

import argparse
import glob
import math
import os
import re
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import uproot
import yaml


OPERATORS = {
    ">": lambda values, cut: values > cut,
    ">=": lambda values, cut: values >= cut,
    "<": lambda values, cut: values < cut,
    "<=": lambda values, cut: values <= cut,
    "==": lambda values, cut: values == cut,
    "!=": lambda values, cut: values != cut,
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def selection_branches(selection):
    if isinstance(selection, str):
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", selection)
        keywords = {"and", "or", "not", "True", "False"}
        return [token for token in tokens if token not in keywords]
    if isinstance(selection, (list, tuple)) and len(selection) > 0:
        return [selection[0]]
    return []


def feature_branches(node_branch_names):
    branches = []
    for feature in node_branch_names:
        if not isinstance(feature, list):
            continue
        for branch in feature:
            if isinstance(branch, str) and branch != "CALC_E":
                branches.append(branch)
    return branches


def branches_for_dataset(dataset_config):
    args = dataset_config["args"]
    branches = feature_branches(args.get("node_branch_names", []))
    for selection in dataset_config.get("selections", []):
        branches.extend(selection_branches(selection))
    return sorted(set(branches))


def resolve_files(args):
    raw_dir = args["raw_dir"]
    file_names = args["file_names"]
    files = []
    if isinstance(file_names, str):
        files.extend(glob.glob(os.path.join(raw_dir, file_names)))
    else:
        for file_name in file_names:
            files.extend(glob.glob(os.path.join(raw_dir, file_name)))
    return sorted(files)


def load_arrays(dataset_config, max_events=None):
    args = dataset_config["args"]
    tree_name = args.get("tree_name", "nominal_Loose")
    branches = branches_for_dataset(dataset_config)
    arrays = []
    events_read = 0

    for file_name in resolve_files(args):
        with uproot.open(file_name) as root_file:
            tree = root_file[tree_name]
            entry_stop = None
            if max_events is not None:
                remaining = max_events - events_read
                if remaining <= 0:
                    break
                entry_stop = remaining
            array = tree.arrays(branches, library="ak", entry_stop=entry_stop)
            arrays.append(array)
            if branches:
                events_read += len(array[branches[0]])
            if max_events is not None and events_read >= max_events:
                break

    if not arrays:
        pattern = os.path.join(args["raw_dir"], str(args["file_names"]))
        raise FileNotFoundError(f"No files found for pattern {pattern}")
    return ak.concatenate(arrays, axis=0)


def selection_mask(data, selections):
    first_field = data.fields[0] if len(data.fields) > 0 else None
    if first_field is None:
        return None

    mask = np.ones(len(data[first_field]), dtype=bool)
    for selection in selections:
        if isinstance(selection, str):
            current_mask = eval(selection, {"__builtins__": {}}, data)
        else:
            branch, cut, op = selection
            if op not in OPERATORS:
                raise ValueError(f"Unknown selection operator: {op}")
            current_mask = OPERATORS[op](data[branch], cut)
        mask = mask & ak.to_numpy(current_mask)
    return mask


def ensure_node_array(value, reference):
    if isinstance(value, (int, float, complex)):
        return ak.full_like(reference, value)
    return value


def branch_array(data, branch, node_type, reference):
    value = ensure_node_array(branch, reference)
    if not isinstance(branch, str):
        return value
    value = data[branch]
    if node_type == "single":
        return ak.singletons(value)
    return value


def clean_label(value):
    return str(value).replace("/", "_")


def per_type_feature_label(feature_spec, type_index):
    return clean_label(feature_spec[type_index])


def flatten_values(values):
    flat_values = np.asarray(ak.to_numpy(ak.ravel(values)), dtype=float)
    return flat_values[np.isfinite(flat_values)]


def build_feature_values(data, dataset_config):
    args = dataset_config["args"]
    node_branch_names = args["node_branch_names"]
    node_branch_types = args["node_branch_types"]
    node_feature_scales = [float(scale) for scale in args["node_feature_scales"]]
    n_types = len(node_branch_names[0])

    references = []
    for type_index in range(n_types):
        branch = node_branch_names[0][type_index]
        node_type = node_branch_types[type_index]
        if isinstance(branch, str):
            reference = data[branch]
            if node_type == "single":
                reference = ak.singletons(reference)
        else:
            raise ValueError("The first node feature must use real branches to define node counts.")
        references.append(reference)

    features = {}
    pt_parts = []
    eta_parts = []

    for type_index in range(n_types):
        pt_parts.append(branch_array(data, node_branch_names[0][type_index], node_branch_types[type_index], references[type_index]))
        eta_parts.append(branch_array(data, node_branch_names[1][type_index], node_branch_types[type_index], references[type_index]))

    for feature_index, feature_spec in enumerate(node_branch_names):
        if not isinstance(feature_spec, list):
            continue

        per_type_parts = []
        for type_index in range(n_types):
            branch = feature_spec[type_index]
            if not isinstance(branch, str) or branch == "CALC_E":
                per_type_parts.append(None)
                continue

            reference = references[type_index]
            node_type = node_branch_types[type_index]
            per_type_parts.append(branch_array(data, branch, node_type, reference))

        for type_index, part in enumerate(per_type_parts):
            if part is None:
                continue
            scaled_part = part * node_feature_scales[feature_index]
            scaled_pt = pt_parts[type_index] * node_feature_scales[0]
            scaled_part = scaled_part[scaled_pt != 0]
            features[per_type_feature_label(feature_spec, type_index)] = flatten_values(scaled_part)

    return features


def common_bins(datasets, feature_name, bins):
    values = np.concatenate([features[feature_name] for features in datasets.values() if len(features[feature_name]) > 0])
    if len(values) == 0:
        return np.linspace(0, 1, bins + 1)

    unique_values = np.unique(values)
    if len(unique_values) <= 20 and np.allclose(unique_values, np.round(unique_values)):
        low = math.floor(values.min())
        high = math.ceil(values.max())
        return np.arange(low - 0.5, high + 1.5, 1)

    low, high = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = values.min(), values.max()
    if low == high:
        low -= 0.5
        high += 0.5
    return np.linspace(low, high, bins + 1)


def plot_feature(feature_name, datasets, bins):
    fig, ax = plt.subplots(figsize=(8, 6))
    hist_bins = common_bins(datasets, feature_name, bins)

    for dataset_name, features in datasets.items():
        values = features[feature_name]
        if len(values) == 0:
            continue
        ax.hist(
            values,
            bins=hist_bins,
            histtype="step",
            density=True,
            linewidth=1.8,
            label=f"{dataset_name} (n={len(values)})",
        )

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Normalized entries")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Plot model-input node feature distributions for every dataset in a config."
    )
    parser.add_argument("--config", required=True, help="YAML config containing Datasets.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for optional PNG outputs. Defaults to plots/<config-stem>_distributions.",
    )
    parser.add_argument(
        "--output-pdf",
        default=None,
        help="Path for the multi-page PDF. Defaults to plots/<config-stem>_distributions.pdf.",
    )
    parser.add_argument("--write-pngs", action="store_true", help="Also write one PNG per plot.")
    parser.add_argument("--bins", type=int, default=80, help="Number of bins for continuous features.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional maximum events per dataset.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path("plots") / f"{Path(args.config).stem}_distributions"
    output_pdf = Path(args.output_pdf) if args.output_pdf else Path("plots") / f"{Path(args.config).stem}_distributions.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if args.write_pngs:
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset_features = {}
    for dataset_name, dataset_config in config["Datasets"].items():
        print(f"Loading {dataset_name}", flush=True)
        data = load_arrays(dataset_config, max_events=args.max_events)
        mask = selection_mask(data, dataset_config.get("selections", []))
        if mask is not None:
            data = data[mask]
        dataset_features[dataset_name] = build_feature_values(data, dataset_config)

    feature_names = list(next(iter(dataset_features.values())).keys())
    with PdfPages(output_pdf) as pdf:
        for feature_name in feature_names:
            fig = plot_feature(feature_name, dataset_features, args.bins)
            pdf.savefig(fig)
            if args.write_pngs:
                output_path = output_dir / f"{safe_filename(feature_name)}.png"
                fig.savefig(output_path, dpi=160)
                print(f"Wrote {output_path}", flush=True)
            plt.close(fig)
    print(f"Wrote {output_pdf}", flush=True)


if __name__ == "__main__":
    main()
