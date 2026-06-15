#!/usr/bin/env python3
"""Export a trained DGL GNN checkpoint to ONNX and validate one graph/event at a time.

Usage:
    python scripts/export_onnx.py --config configs/stats_100K/ttH_CP_even_vs_odd.yaml --name ttH.onnx

Defaults:
    - infer best epoch from training log via root_gnn_base.utils.get_best_epoch
    - export ONNX using one real graph/event
    - validate with real data, one graph/event at a time
    - compare DGL -> tensor and tensor -> ONNX
    - save diagnostic plot next to ONNX file
"""

from __future__ import annotations

import argparse
import inspect
import importlib
import os
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, Iterator, Optional, Tuple

file_path = os.getcwd()
sys.path.append(file_path)

import dgl
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from dgl.dataloading import GraphDataLoader
from torch_scatter import scatter_mean, scatter_sum

try:
    from root_gnn_base import utils
except Exception as exc:
    utils = None
    _UTILS_IMPORT_ERROR = exc
else:
    _UTILS_IMPORT_ERROR = None


# -------------------------
# Config / checkpoint utils
# -------------------------


def load_config(config_file: str | os.PathLike[str]) -> Dict[str, Any]:
    config_path = Path(config_file)
    with config_path.open() as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    if conf is None:
        raise ValueError(f"Empty config: {config_file}")

    include_config(conf, config_path.parent)
    return conf


def include_config(conf: Dict[str, Any], base_dir: Path) -> None:
    includes = conf.pop("include", None)
    if not includes:
        return

    if isinstance(includes, (str, os.PathLike)):
        includes = [includes]

    for inc in includes:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = base_dir / inc_path

        with inc_path.open() as f:
            included = yaml.load(f, Loader=yaml.FullLoader) or {}

        include_config(included, inc_path.parent)
        conf.update(included)


def find_model_class(model_cfg: Dict[str, Any]) -> str:
    return str(model_cfg.get("class", "")).split(".")[-1]


def infer_global_size(model_args: Dict[str, Any]) -> int:
    for key in ("global_size", "global_in_size", "global_dim", "n_global", "sample_global"):
        if key in model_args:
            return int(model_args[key])
    return 1


def load_best_checkpoint(conf: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    if utils is None:
        raise RuntimeError(
            "Could not import root_gnn_base.utils, which is needed for utils.get_best_epoch. "
            f"Original import error: {_UTILS_IMPORT_ERROR}"
        )

    try:
        return utils.get_best_epoch(conf, mode="max")
    except TypeError:
        return utils.get_best_epoch(conf)


def load_checkpoint(conf: Dict[str, Any], epoch: Optional[int]) -> Tuple[int, Dict[str, Any]]:
    if epoch is None:
        return load_best_checkpoint(conf)

    training_dir = Path(conf["Training_Directory"])
    checkpoint_path = training_dir / f"model_epoch_{epoch}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return epoch, checkpoint


# -------------------------
# MLP helpers
# -------------------------


def make_slp(in_size: int, out_size: int, activation=nn.ReLU, dropout: float = 0) -> list[nn.Module]:
    return [nn.Linear(in_size, out_size), activation(), nn.Dropout(dropout)]


def make_mlp(
    in_size: int,
    hid_size: int,
    out_size: int,
    n_layers: int,
    activation=nn.ReLU,
    dropout: float = 0,
) -> nn.Sequential:
    layers: list[nn.Module] = []

    if n_layers > 1:
        layers += make_slp(in_size, hid_size, activation, dropout)
        for _ in range(n_layers - 2):
            layers += make_slp(hid_size, hid_size, activation, dropout)
        layers += make_slp(hid_size, out_size, activation, dropout)
    else:
        layers += make_slp(in_size, out_size, activation, dropout)

    layers.append(nn.LayerNorm(out_size))
    return nn.Sequential(*layers)


def broadcast_global_to_nodes(h_global: torch.Tensor, node_batch: torch.Tensor) -> torch.Tensor:
    if h_global.dim() == 1:
        h_global = h_global.unsqueeze(0)
    return h_global[node_batch.to(torch.long)]


def broadcast_global_to_edges(h_global: torch.Tensor, edge_batch: torch.Tensor) -> torch.Tensor:
    if h_global.dim() == 1:
        h_global = h_global.unsqueeze(0)
    return h_global[edge_batch.to(torch.long)]


def copy_v_udf(edges):
    return {"m_v": edges.dst["h"]}


def make_node_batch_ids(batch_num_nodes: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(len(batch_num_nodes), device=batch_num_nodes.device, dtype=torch.long),
        batch_num_nodes.to(torch.long),
    )


def make_edge_batch_ids(batch_num_edges: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(len(batch_num_edges), device=batch_num_edges.device, dtype=torch.long),
        batch_num_edges.to(torch.long),
    )


# -------------------------
# Tensor / ONNX model copies
# -------------------------


class EdgeNetworkONNX(nn.Module):
    """ONNX-friendly tensor implementation of the DGL Edge_Network."""

    def __init__(
        self,
        sample_graph: Any,
        sample_global: int,
        hid_size: int,
        out_size: int,
        n_layers: int,
        n_proc_steps: int,
        dropout: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if kwargs:
            print(f"Unused args while creating EdgeNetworkONNX: {kwargs}")

        self.n_proc_steps = n_proc_steps

        node_in = int(sample_graph.ndata["features"].shape[1])
        edge_in = int(sample_graph.edata["features"].shape[1])
        gl_size = int(sample_global)

        self.layers = nn.ModuleList()
        self.node_encoder = make_mlp(node_in, hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_encoder = make_mlp(edge_in, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_encoder = make_mlp(gl_size, hid_size, hid_size, n_layers, dropout=dropout)

        self.node_update = make_mlp(3 * hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_update = make_mlp(4 * hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = make_mlp(3 * hid_size, hid_size, hid_size, n_layers, dropout=dropout)

        self.global_decoder = make_mlp(hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = nn.Linear(hid_size, out_size)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        global_feats: torch.Tensor,
        edge_index: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)
        node_batch = node_batch.to(torch.long)

        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features)
        h_global = self.global_encoder(global_feats)

        num_graphs = global_feats.size(0)

        for _ in range(self.n_proc_steps):
            edge_batch = node_batch[dst]

            e = self.edge_update(
                torch.cat(
                    [
                        e,
                        h[src],
                        h[dst],
                        broadcast_global_to_edges(h_global, edge_batch),
                    ],
                    dim=1,
                )
            )

            h_e = scatter_sum(e, dst, dim=0, dim_size=h.size(0))

            h = self.node_update(
                torch.cat(
                    [
                        h,
                        h_e,
                        broadcast_global_to_nodes(h_global, node_batch),
                    ],
                    dim=1,
                )
            )

            mean_n = scatter_mean(h, node_batch, dim=0, dim_size=num_graphs)
            mean_e = scatter_mean(e, edge_batch, dim=0, dim_size=num_graphs)

            h_global = self.global_update(torch.cat([h_global, mean_n, mean_e], dim=1))

        return self.classify(self.global_decoder(h_global))


class TransferredLearningFinetuningONNX(nn.Module):
    """ONNX-friendly tensor implementation of Transferred_Learning_Finetuning."""

    def __init__(
        self,
        pretraining_path: str,
        pretraining_model_args: Dict[str, Any],
        sample_graph: Any,
        sample_global: int,
        hid_size: int,
        out_size: int,
        n_layers: int,
        n_proc_steps: int,
        dropout: float = 0,
        frozen_pretraining: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if kwargs:
            print(f"Unused args while creating TransferredLearningFinetuningONNX: {kwargs}")

        self.n_proc_steps = n_proc_steps

        pre_args = dict(pretraining_model_args)
        pre_args.setdefault("dropout", dropout)

        self.pretrained_model = EdgeNetworkONNX(
            sample_graph=sample_graph,
            sample_global=sample_global,
            **pre_args,
        )

        checkpoint = torch.load(pretraining_path, map_location="cpu")
        self.pretrained_model.load_state_dict(checkpoint["model_state_dict"])

        self.pretrained_model = nn.Sequential(*list(self.pretrained_model.children())[:-1])

        print(f"Freeze Pretraining = {frozen_pretraining}")

        if frozen_pretraining:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model[7].parameters():
                param.requires_grad = True

        torch.manual_seed(2)
        self.classify = nn.Linear(hid_size, out_size)

    def _backbone_forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        global_feats: torch.Tensor,
        edge_index: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)
        node_batch = node_batch.to(torch.long)

        node_enc = self.pretrained_model[1]
        edge_enc = self.pretrained_model[2]
        glob_enc = self.pretrained_model[3]
        node_upd = self.pretrained_model[4]
        edge_upd = self.pretrained_model[5]
        glob_upd = self.pretrained_model[6]
        glob_dec = self.pretrained_model[7]

        h = node_enc(node_features)
        e = edge_enc(edge_features)
        h_global = glob_enc(global_feats)

        num_graphs = global_feats.size(0)

        for _ in range(self.n_proc_steps):
            edge_batch = node_batch[dst]

            e = edge_upd(
                torch.cat(
                    [
                        e,
                        h[src],
                        h[dst],
                        broadcast_global_to_edges(h_global, edge_batch),
                    ],
                    dim=1,
                )
            )

            h_e = scatter_sum(e, dst, dim=0, dim_size=h.size(0))

            h = node_upd(
                torch.cat(
                    [
                        h,
                        h_e,
                        broadcast_global_to_nodes(h_global, node_batch),
                    ],
                    dim=1,
                )
            )

            mean_n = scatter_mean(h, node_batch, dim=0, dim_size=num_graphs)
            mean_e = scatter_mean(e, edge_batch, dim=0, dim_size=num_graphs)

            h_global = glob_upd(torch.cat([h_global, mean_n, mean_e], dim=1))

        return glob_dec(h_global)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        global_feats: torch.Tensor,
        edge_index: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.classify(
            self._backbone_forward(
                node_features,
                edge_features,
                global_feats,
                edge_index,
                node_batch,
            )
        )


# -------------------------
# Model construction
# -------------------------


def make_sample_graph(node_features: int, edge_features: int) -> Any:
    return SimpleNamespace(
        ndata={"features": torch.zeros(2, node_features, dtype=torch.float32)},
        edata={"features": torch.zeros(2, edge_features, dtype=torch.float32)},
    )


def build_tensor_model(conf: Dict[str, Any]) -> nn.Module:
    model_cfg = conf["Model"]
    model_args = dict(model_cfg.get("args", {}))
    class_name = find_model_class(model_cfg)

    node_in = int(model_args.get("in_size", 7))
    edge_in = int(model_args.get("edge_in_size", 3))
    global_in = infer_global_size(model_args)

    sample_graph = make_sample_graph(node_in, edge_in)

    common = {
        "sample_graph": sample_graph,
        "sample_global": global_in,
        "hid_size": int(model_args["hid_size"]),
        "out_size": int(model_args["out_size"]),
        "n_layers": int(model_args["n_layers"]),
        "n_proc_steps": int(model_args["n_proc_steps"]),
        "dropout": float(model_args.get("dropout", 0)),
    }

    if class_name == "Edge_Network":
        return EdgeNetworkONNX(**common)

    if class_name == "Transferred_Learning_Finetuning":
        pretraining_model = model_args.get("pretraining_model", {})
        pre_args = dict(pretraining_model.get("args", {}))

        pre_args.pop("in_size", None)
        pre_args.pop("edge_in_size", None)

        return TransferredLearningFinetuningONNX(
            pretraining_path=model_args["pretraining_path"],
            pretraining_model_args=pre_args,
            frozen_pretraining=bool(model_args.get("frozen_pretraining", False)),
            **common,
        )

    raise ValueError(
        f"Unsupported Model.class={class_name!r}. "
        "Expected Edge_Network or Transferred_Learning_Finetuning."
    )


def build_dgl_model(conf: Dict[str, Any], sample_graph: dgl.DGLGraph, sample_global: torch.Tensor) -> nn.Module:
    if utils is None:
        raise RuntimeError(
            "Could not import root_gnn_base.utils, which is needed to build the DGL model. "
            f"Original import error: {_UTILS_IMPORT_ERROR}"
        )

    return utils.buildFromConfig(
        conf["Model"],
        {
            "sample_graph": sample_graph,
            "sample_global": sample_global,
        },
    )


def patch_finetuning_pretrained_output(model: nn.Module) -> nn.Module:
    """Patch older finetuning models so Pretrained_Output can accept explicit globals.

    The repo has moved through a few signatures for the finetuning DGL model.
    Some checkpoints still load a class whose forward() calls Pretrained_Output(g.clone())
    while the body expects a global_feats tensor. This adapter preserves the original
    module weights but makes the instance callable from the exporter in either style.
    """

    if not hasattr(model, "TL_node_encoder") or not hasattr(model, "TL_global_encoder"):
        return model

    original = getattr(model, "Pretrained_Output", None)
    if original is None:
        return model

    try:
        signature = inspect.signature(original)
        # Bound methods exclude "self".
        if len(signature.parameters) > 1:
            return model
    except (TypeError, ValueError):
        pass

    def _patched_pretrained_output(self, g, global_feats=None):
        h = self.TL_node_encoder(g.ndata["features"])
        e = self.TL_edge_encoder(g.edata["features"])
        g.ndata["h"] = h
        g.edata["e"] = e

        if global_feats is None:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)

        h_global = self.TL_global_encoder(global_feats)
        node_batch = make_node_batch_ids(g.batch_num_nodes())
        edge_batch = make_edge_batch_ids(g.batch_num_edges())

        for _ in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u("h", "m_u"))
            g.apply_edges(copy_v_udf)
            g.edata["e"] = self.TL_edge_update(
                torch.cat(
                    (
                        g.edata["e"],
                        g.edata["m_u"],
                        g.edata["m_v"],
                        broadcast_global_to_edges(h_global, edge_batch),
                    ),
                    dim=1,
                )
            )
            g.update_all(dgl.function.copy_e("e", "m"), dgl.function.sum("m", "h_e"))
            g.ndata["h"] = self.TL_node_update(
                torch.cat((g.ndata["h"], g.ndata["h_e"], broadcast_global_to_nodes(h_global, node_batch)), dim=1)
            )
            h_global = self.TL_global_update(
                torch.cat((h_global, dgl.mean_nodes(g, "h"), dgl.mean_edges(g, "e")), dim=1)
            )

        return self.TL_global_decoder(h_global)

    model.Pretrained_Output = MethodType(_patched_pretrained_output, model)
    return model


# -------------------------
# Dataset / graph utilities
# -------------------------


def build_dataset_from_config(conf: Dict[str, Any]):
    if utils is None:
        raise RuntimeError(
            "Could not import root_gnn_base.utils, which is needed to build the dataset. "
            f"Original import error: {_UTILS_IMPORT_ERROR}"
        )

    dset_name = list(conf["Datasets"].keys())[0]
    dset_conf = dict(conf["Datasets"][dset_name])
    dataset = utils.buildFromConfig(dset_conf)
    return dset_name, dataset


def single_graph_loader(conf: Dict[str, Any]) -> Tuple[str, GraphDataLoader]:
    dset_name, dataset = build_dataset_from_config(conf)

    loader = GraphDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return dset_name, loader


def get_global_features(batch: dgl.DGLGraph) -> torch.Tensor:
    candidates = []

    for attr in ("global_features", "global_feats", "globals"):
        if hasattr(batch, attr):
            candidates.append(getattr(batch, attr))

    for key in ("global_features", "global_feats", "globals", "features"):
        try:
            if key in batch.ndata and False:
                pass
        except Exception:
            pass

    for candidate in candidates:
        if isinstance(candidate, torch.Tensor) and candidate.numel() > 0:
            if candidate.dim() == 1:
                candidate = candidate.unsqueeze(0)
            return candidate.to(torch.float32)

    return batch.batch_num_nodes().to(torch.float32).unsqueeze(1)


def tensorize_single_graph(batch: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch.batch_num_nodes()) != 1:
        raise ValueError(
            f"Expected a single graph/event, but got batched graph with {len(batch.batch_num_nodes())} graphs"
        )

    node_features = batch.ndata["features"].detach().cpu().to(torch.float32)
    edge_features = batch.edata["features"].detach().cpu().to(torch.float32)

    src, dst = batch.edges()
    edge_index = torch.stack([src.detach().cpu(), dst.detach().cpu()], dim=0).to(torch.long)

    node_batch = torch.zeros(node_features.shape[0], dtype=torch.long)
    global_feats = get_global_features(batch).detach().cpu().to(torch.float32)

    if global_feats.dim() == 1:
        global_feats = global_feats.unsqueeze(0)

    if global_feats.shape[0] != 1:
        global_feats = global_feats.reshape(1, -1)

    return node_features, edge_features, global_feats, edge_index, node_batch


def first_real_event_inputs(conf: Dict[str, Any]) -> Tuple[str, dgl.DGLGraph, Tuple[torch.Tensor, ...]]:
    dset_name, loader = single_graph_loader(conf)

    batch, labels, tracking, extra = next(iter(loader))
    _ = labels, tracking, extra

    inputs = tensorize_single_graph(batch)
    return dset_name, batch, inputs


# -------------------------
# ONNX export / runtime
# -------------------------


def export_onnx(model: nn.Module, inputs: Tuple[torch.Tensor, ...], out_path: str) -> None:
    output = Path(out_path)

    if output.parent and str(output.parent) != ".":
        output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        inputs,
        str(output),
        input_names=[
            "node_features",
            "edge_features",
            "global_features",
            "edge_index",
            "node_batch",
        ],
        output_names=["logits"],
        dynamic_axes={
            "node_features": {0: "num_nodes"},
            "edge_features": {0: "num_edges"},
            "edge_index": {1: "num_edges"},
            "node_batch": {0: "num_nodes"},
        },
        opset_version=16,
    )


def make_onnx_session(onnx_path: str) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()

    # Avoid Perlmutter / CPU affinity warnings from ONNX Runtime.
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    return ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def run_onnx(sess: ort.InferenceSession, inputs: Tuple[torch.Tensor, ...]) -> np.ndarray:
    node_features, edge_features, global_feats, edge_index, node_batch = inputs

    ort_inputs = {
        "node_features": node_features.numpy().astype(np.float32),
        "edge_features": edge_features.numpy().astype(np.float32),
        "global_features": global_feats.numpy().astype(np.float32),
        "edge_index": edge_index.numpy().astype(np.int64),
        "node_batch": node_batch.numpy().astype(np.int64),
    }

    ort_input_names = {inp.name for inp in sess.get_inputs()}
    ort_inputs = {k: v for k, v in ort_inputs.items() if k in ort_input_names}

    return sess.run(None, ort_inputs)[0]


# -------------------------
# Real-data validation loop
# -------------------------


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_real_data_test(
    conf: Dict[str, Any],
    tensor_model: nn.Module,
    onnx_path: str,
    epoch: int,
    checkpoint: Dict[str, Any],
    max_events: int,
    tol_dgl_tensor: float,
    tol_tensor_onnx: float,
) -> None:
    dset_name, loader = single_graph_loader(conf)

    first_batch, labels, tracking, extra = next(iter(loader))
    _ = labels, tracking, extra

    first_inputs = tensorize_single_graph(first_batch)
    first_global = first_inputs[2]

    dgl_model = build_dgl_model(conf, first_batch, first_global)
    dgl_model.load_state_dict(checkpoint["model_state_dict"])
    dgl_model = patch_finetuning_pretrained_output(dgl_model)
    dgl_model.eval().cpu()

    tensor_model.eval().cpu()

    sess = make_onnx_session(onnx_path)

    all_dgl_logits = []
    all_tensor_logits = []
    all_onnx_logits = []

    all_dgl_prob = []
    all_tensor_prob = []
    all_onnx_prob = []

    dgl_tensor_max_diffs = []
    tensor_onnx_max_diffs = []

    n_tested = 0

    # Recreate loader so event 0 is included.
    _, loader = single_graph_loader(conf)

    for item in loader:
        batch, labels, tracking, extra = item
        _ = labels, tracking, extra

        inputs = tensorize_single_graph(batch)
        node_features, edge_features, global_feats, edge_index, node_batch = inputs

        with torch.no_grad():
            dgl_logits = dgl_model(batch, global_feats).detach().cpu().numpy()
            tensor_logits = tensor_model(*inputs).detach().cpu().numpy()

        onnx_logits = run_onnx(sess, inputs)

        dgl_prob = sigmoid_np(dgl_logits)
        tensor_prob = sigmoid_np(tensor_logits)
        onnx_prob = sigmoid_np(onnx_logits)

        all_dgl_logits.append(dgl_logits.reshape(-1))
        all_tensor_logits.append(tensor_logits.reshape(-1))
        all_onnx_logits.append(onnx_logits.reshape(-1))

        all_dgl_prob.append(dgl_prob.reshape(-1))
        all_tensor_prob.append(tensor_prob.reshape(-1))
        all_onnx_prob.append(onnx_prob.reshape(-1))

        dgl_tensor_max_diffs.append(float(np.max(np.abs(dgl_logits - tensor_logits))))
        tensor_onnx_max_diffs.append(float(np.max(np.abs(tensor_logits - onnx_logits))))

        n_tested += 1

        if n_tested % 100 == 0:
            print(f"Validated {n_tested} single-event graphs...")

        if max_events > 0 and n_tested >= max_events:
            break

    if n_tested == 0:
        raise RuntimeError("No events were available for validation.")

    dgl_logits_all = np.concatenate(all_dgl_logits)
    tensor_logits_all = np.concatenate(all_tensor_logits)
    onnx_logits_all = np.concatenate(all_onnx_logits)

    dgl_prob_all = np.concatenate(all_dgl_prob)
    tensor_prob_all = np.concatenate(all_tensor_prob)
    onnx_prob_all = np.concatenate(all_onnx_prob)

    dgl_vs_tensor = np.abs(dgl_logits_all - tensor_logits_all)
    tensor_vs_onnx = np.abs(tensor_logits_all - onnx_logits_all)

    dgl_vs_tensor_prob = np.abs(dgl_prob_all - tensor_prob_all)
    tensor_vs_onnx_prob = np.abs(tensor_prob_all - onnx_prob_all)

    print(f"\n== Real Data Test: {dset_name} ==")
    print(f"Epoch                          : {epoch}")
    print(f"Single-event graphs tested     : {n_tested}")
    print(f"DGL output shape               : {dgl_logits_all.shape}")
    print(f"Tensor output shape            : {tensor_logits_all.shape}")
    print(f"ONNX output shape              : {onnx_logits_all.shape}")

    print("\nLogit comparisons")
    print(f"max abs diff DGL->Tensor       : {dgl_vs_tensor.max():.8g}")
    print(f"mean abs diff DGL->Tensor      : {dgl_vs_tensor.mean():.8g}")
    print(f"max abs diff Tensor->ONNX      : {tensor_vs_onnx.max():.8g}")
    print(f"mean abs diff Tensor->ONNX     : {tensor_vs_onnx.mean():.8g}")

    print("\nScore comparisons")
    print(f"max abs diff DGL->Tensor       : {dgl_vs_tensor_prob.max():.8g}")
    print(f"mean abs diff DGL->Tensor      : {dgl_vs_tensor_prob.mean():.8g}")
    print(f"max abs diff Tensor->ONNX      : {tensor_vs_onnx_prob.max():.8g}")
    print(f"mean abs diff Tensor->ONNX     : {tensor_vs_onnx_prob.mean():.8g}")

    print("\nPer-event max logit-diff summaries")
    print(f"DGL->Tensor max over events    : {np.max(dgl_tensor_max_diffs):.8g}")
    print(f"DGL->Tensor mean over events   : {np.mean(dgl_tensor_max_diffs):.8g}")
    print(f"Tensor->ONNX max over events   : {np.max(tensor_onnx_max_diffs):.8g}")
    print(f"Tensor->ONNX mean over events  : {np.mean(tensor_onnx_max_diffs):.8g}")

    save_comparison_plot(
        onnx_path=onnx_path,
        sample_name=dset_name,
        dgl_prob=dgl_prob_all,
        tensor_prob=tensor_prob_all,
        onnx_prob=onnx_prob_all,
    )

    failed = False

    if dgl_vs_tensor.max() > tol_dgl_tensor:
        failed = True
        print(
            f"\nFAIL: DGL->Tensor max diff {dgl_vs_tensor.max():.8g} "
            f"> tolerance {tol_dgl_tensor:.8g}"
        )

    if tensor_vs_onnx.max() > tol_tensor_onnx:
        failed = True
        print(
            f"\nFAIL: Tensor->ONNX max diff {tensor_vs_onnx.max():.8g} "
            f"> tolerance {tol_tensor_onnx:.8g}"
        )

    if failed:
        raise RuntimeError("Real-data validation failed.")

    print("\nReal-data validation passed")


def save_comparison_plot(
    onnx_path: str,
    sample_name: str,
    dgl_prob: np.ndarray,
    tensor_prob: np.ndarray,
    onnx_prob: np.ndarray,
) -> None:
    score_bins = np.linspace(0.0, 1.0, 41)

    residuals_onnx = onnx_prob.reshape(-1) - dgl_prob.reshape(-1)
    residuals_tensor = tensor_prob.reshape(-1) - dgl_prob.reshape(-1)

    combined_residuals = np.concatenate([residuals_onnx, residuals_tensor])

    if np.all(combined_residuals == combined_residuals[0]):
        diff_bins = np.linspace(combined_residuals[0] - 1e-8, combined_residuals[0] + 1e-8, 80)
    else:
        diff_bins = np.histogram_bin_edges(combined_residuals, bins=80)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4))

    ax_left.hist(
        dgl_prob.reshape(-1),
        bins=score_bins,
        histtype="step",
        linewidth=2.0,
        label="DGL",
    )
    ax_left.hist(
        tensor_prob.reshape(-1),
        bins=score_bins,
        histtype="step",
        linewidth=2.0,
        label="Tensor",
    )
    ax_left.hist(
        onnx_prob.reshape(-1),
        bins=score_bins,
        histtype="step",
        linewidth=2.0,
        label="ONNX",
    )
    ax_left.set_title(f"Score Distributions: {sample_name}")
    ax_left.set_xlabel("Score")
    ax_left.set_ylabel("Events / bin")
    ax_left.legend()

    ax_right.hist(
        residuals_onnx,
        bins=diff_bins,
        histtype="step",
        linewidth=1.8,
        label="ONNX - DGL",
    )
    ax_right.hist(
        residuals_tensor,
        bins=diff_bins,
        histtype="step",
        linewidth=1.8,
        label="Tensor - DGL",
    )
    ax_right.set_title(f"Differences vs DGL: {sample_name}")
    ax_right.set_xlabel("Score difference")
    ax_right.set_ylabel("Events / bin")
    ax_right.set_yscale("log")
    ax_right.legend()

    plt.tight_layout()

    plot_path = os.path.splitext(onnx_path)[0] + "_onnx.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison plot to {plot_path}")


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export root_gnn_base GCN models to ONNX.")

    parser.add_argument("--config", required=True, help="YAML training config.")
    parser.add_argument("--name", required=True, help='Output ONNX filename, e.g. "ttH.onnx".')

    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Checkpoint epoch to export. Default: best Test_AUC epoch.",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip real-data validation and plotting.",
    )
    parser.add_argument(
        "--max-test-events",
        type=int,
        default=1000,
        help="Number of single-event graphs to validate. Use 0 for all events. Default: 1000.",
    )
    parser.add_argument(
        "--tol-dgl-tensor",
        type=float,
        default=1e-8,
        help="Max allowed logit difference for DGL vs tensor model.",
    )
    parser.add_argument(
        "--tol-tensor-onnx",
        type=float,
        default=5e-5,
        help="Max allowed logit difference for tensor model vs ONNX.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conf = load_config(args.config)

    tensor_model = build_tensor_model(conf)
    epoch, checkpoint = load_checkpoint(conf, args.epoch)

    tensor_model.load_state_dict(checkpoint["model_state_dict"])
    tensor_model.eval().cpu()

    if args.no_test:
        model_args = conf["Model"].get("args", {})

        node_in = int(model_args.get("in_size", 7))
        edge_in = int(model_args.get("edge_in_size", 3))
        global_in = infer_global_size(model_args)

        node_features = torch.randn(4, node_in, dtype=torch.float32)
        src = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        dst = torch.tensor([1, 2, 0, 3, 0, 3, 1, 2], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        edge_features = torch.randn(edge_index.shape[1], edge_in, dtype=torch.float32)
        global_features = torch.ones(1, global_in, dtype=torch.float32)
        node_batch = torch.zeros(node_features.shape[0], dtype=torch.long)

        export_inputs = (
            node_features,
            edge_features,
            global_features,
            edge_index,
            node_batch,
        )
    else:
        dset_name, first_batch, export_inputs = first_real_event_inputs(conf)
        print(f"Using one real event from {dset_name} as the ONNX export example input.")

    with torch.no_grad():
        _ = tensor_model(*export_inputs)

    export_onnx(tensor_model, export_inputs, args.name)

    print(f"Exported epoch {epoch} to {args.name}")

    if not args.no_test:
        run_real_data_test(
            conf=conf,
            tensor_model=tensor_model,
            onnx_path=args.name,
            epoch=epoch,
            checkpoint=checkpoint,
            max_events=args.max_test_events,
            tol_dgl_tensor=args.tol_dgl_tensor,
            tol_tensor_onnx=args.tol_tensor_onnx,
        )


if __name__ == "__main__":
    main()
