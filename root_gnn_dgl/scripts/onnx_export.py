import argparse
import os
import sys

file_path = os.getcwd()
sys.path.append(file_path)

import numpy as np
import torch
import dgl
import yaml
import matplotlib.pyplot as plt

import onnx
import onnxruntime as ort
from torch_scatter import scatter_sum, scatter_mean
from dgl.dataloading import GraphDataLoader

from root_gnn_base import utils


def include_config(conf):
    if 'include' in conf:
        for i in conf['include']:
            with open(i) as f:
                conf.update(yaml.load(f, Loader=yaml.FullLoader))
        del conf['include']


def load_config(config_file):
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    include_config(conf)
    return conf


def load_sample_graph(config):
    dset_name = list(config["Datasets"].keys())[0]
    dset_conf = config["Datasets"][dset_name]
    save_dir = dset_conf["args"]["save_dir"]
    chunkno = 0
    fold_conf = dset_conf["folding"]
    suffix = utils.fold_selection_name(fold_conf, "train")
    prebatch_path = os.path.join(
        save_dir,
        f'{dset_conf["args"]["name"]}_prebatched_padded_{chunkno}_{suffix}.bin',
    )
    graphs, labels = dgl.load_graphs(prebatch_path)
    sample_graph = graphs[0]
    sample_batch_num_nodes = labels["batch_num_nodes"][0]
    sample_global = sample_batch_num_nodes.to(torch.float32).unsqueeze(1)
    return sample_graph, sample_global, sample_batch_num_nodes


def get_sample_name(config):
    dset_name = list(config["Datasets"].keys())[0]
    return config["Datasets"][dset_name]["args"]["name"]


def load_dgl_reference_batch(config, batch_size=1024):
    dset_name = list(config["Datasets"].keys())[0]
    dset_conf = config["Datasets"][dset_name]
    dset = utils.buildFromConfig(dset_conf)
    loader = GraphDataLoader(dset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    batch, _, _, global_feats = next(iter(loader))
    if global_feats is None or global_feats.numel() == 0:
        global_feats = batch.batch_num_nodes().to(torch.float32).unsqueeze(1)
    return batch, global_feats


def summarize_graph_batch(graph):
    batch_num_nodes = graph.batch_num_nodes()
    batch_num_edges = graph.batch_num_edges()
    print("Sample graph batch_num_nodes shape:", tuple(batch_num_nodes.shape), "sum:", int(batch_num_nodes.sum().item()) if batch_num_nodes.numel() else 0)
    print("Sample graph batch_num_edges shape:", tuple(batch_num_edges.shape), "sum:", int(batch_num_edges.sum().item()) if batch_num_edges.numel() else 0)


def broadcast_global_to_nodes(h_global: torch.Tensor, node_batch: torch.Tensor) -> torch.Tensor:
    return h_global[node_batch]


def broadcast_global_to_edges(h_global: torch.Tensor, edge_batch: torch.Tensor) -> torch.Tensor:
    return h_global[edge_batch]


def Make_SLP(in_size, out_size, activation=torch.nn.ReLU, dropout=0):
    return [torch.nn.Linear(in_size, out_size), activation(), torch.nn.Dropout(dropout)]


def Make_MLP(in_size, hid_size, out_size, n_layers, activation=torch.nn.ReLU, dropout=0):
    layers = []
    if n_layers > 1:
        layers += Make_SLP(in_size, hid_size, activation, dropout)
        for _ in range(n_layers - 2):
            layers += Make_SLP(hid_size, hid_size, activation, dropout)
        layers += Make_SLP(hid_size, out_size, activation, dropout)
    else:
        layers += Make_SLP(in_size, out_size, activation, dropout)
    layers.append(torch.nn.LayerNorm(out_size))
    return torch.nn.Sequential(*layers)


class Edge_Network(torch.nn.Module):
    def __init__(self, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        if len(sample_global) == 0:
            self.has_global = False
        else:
            self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        self.node_encoder = Make_MLP(sample_graph.ndata['features'].shape[1], hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_encoder = Make_MLP(sample_graph.edata['features'].shape[1], hid_size, hid_size, n_layers, dropout=dropout)
        self.global_encoder = Make_MLP(gl_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.node_update = Make_MLP(3 * hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_update = Make_MLP(4 * hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = Make_MLP(3 * hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_decoder = Make_MLP(hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = torch.nn.Linear(hid_size, out_size)

    def forward(self, node_features, edge_features, global_feats, edge_index, node_batch):
        src, dst = edge_index
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features)

        if (global_feats is None) or (global_feats.shape[1] == 0):
            ones = torch.ones_like(node_batch, dtype=torch.float)
            num_v = scatter_sum(ones.unsqueeze(-1), node_batch, 0)
            global_feats = num_v

        h_global = self.global_encoder(global_feats)

        for _ in range(self.n_proc_steps):
            m_u = h[src]
            m_v = h[dst]
            e_input = torch.cat([e, m_u, m_v, broadcast_global_to_edges(h_global, node_batch[dst])], dim=1)
            e = self.edge_update(e_input)

            h_e = scatter_sum(e, dst, dim=0, dim_size=h.size(0))
            n_input = torch.cat([h, h_e, broadcast_global_to_nodes(h_global, node_batch)], dim=1)
            h = self.node_update(n_input)

            mean_n = scatter_mean(h, node_batch, dim=0)
            mean_e = scatter_mean(e, node_batch[dst], dim=0)
            g_input = torch.cat([h_global, mean_n, mean_e], dim=1)
            h_global = self.global_update(g_input)

        h_global = self.global_decoder(h_global)
        return self.classify(h_global)


class Model_With_Globals(torch.nn.Module):
    def __init__(self, pretraining_path, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, frozen_pretraining=False, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps

        if len(sample_global) == 0:
            self.has_global = False
            gl_size = 1
        else:
            self.has_global = sample_global.shape[1] != 0
            gl_size = sample_global.shape[1]

        self.pretrained_model = Edge_Network(sample_graph, [], 64, 12, 4, 4)
        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint["model_state_dict"])
        pretrained_layers = list(self.pretrained_model.children())[:-1]
        self.pretrained_model = torch.nn.Sequential(*pretrained_layers)

        print(f"Freeze Pretraining = {frozen_pretraining}")
        if frozen_pretraining:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model[7]:
                param.requires_grad = True

        torch.manual_seed(2)
        self.pretrained_model[3] = Make_MLP(gl_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_encoder = Make_MLP(gl_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = Make_MLP(3 * hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_decoder = Make_MLP(hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = torch.nn.Linear(64, out_size)

    def _backbone_forward(self, node_features, edge_features, global_feats, edge_index, node_batch):
        src = edge_index[0]
        dst = edge_index[1]
        node_enc = self.pretrained_model[1]
        edge_enc = self.pretrained_model[2]
        glob_enc = self.global_encoder
        node_upd = self.pretrained_model[4]
        edge_upd = self.pretrained_model[5]
        glob_upd = self.global_update
        glob_dec = self.global_decoder
        n_steps = 4

        h = node_enc(node_features)
        e = edge_enc(edge_features)

        h_global = glob_enc(global_feats)
        for _ in range(n_steps):
            m_u = h[src]
            m_v = h[dst]
            e_in = torch.cat([e, m_u, m_v, broadcast_global_to_edges(h_global, node_batch[dst])], dim=1)
            e = edge_upd(e_in)

            h_e = scatter_sum(e, dst, dim=0, dim_size=h.size(0))
            n_in = torch.cat([h, h_e, broadcast_global_to_nodes(h_global, node_batch)], dim=1)
            h = node_upd(n_in)

            mean_n = scatter_mean(h, node_batch, dim=0)
            mean_e = scatter_mean(e, node_batch[dst], dim=0)
            g_in = torch.cat([h_global, mean_n, mean_e], dim=1)
            h_global = glob_upd(g_in)

        h_global = glob_dec(h_global)
        return h_global

    def forward(self, node_features, edge_features, global_feats, edge_index, node_batch):
        h_global = self._backbone_forward(node_features, edge_features, global_feats, edge_index, node_batch)
        return self.classify(h_global)


class Transferred_Learning_Finetuning_With_Globals(torch.nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, frozen_pretraining=False, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps

        if len(sample_global) == 0:
            self.has_global = False
            gl_size = 1
        else:
            self.has_global = sample_global.shape[1] != 0
            gl_size = sample_global.shape[1]

        self.pretrained_model = utils.buildFromConfig(pretraining_model, {'sample_graph': sample_graph, 'sample_global': sample_global})
        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_layers = list(self.pretrained_model.children())[:-1]
        self.pretrained_model = torch.nn.Sequential(*pretrained_layers)

        print(f"Freeze Pretraining = {frozen_pretraining}")
        if frozen_pretraining:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model[7]:
                param.requires_grad = True

        torch.manual_seed(2)
        self.node_encoder = Make_MLP(sample_graph.ndata['features'].shape[1], hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_encoder = Make_MLP(sample_graph.edata['features'].shape[1], hid_size, hid_size, n_layers, dropout=dropout)
        self.global_encoder = Make_MLP(gl_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.node_update = Make_MLP(3 * hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_update = Make_MLP(4 * hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = Make_MLP(3 * hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_decoder = Make_MLP(hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = torch.nn.Linear(pretraining_model['args']['hid_size'], out_size)

    def Pretrained_Output(self, node_features, edge_features, global_feats, edge_index, node_batch):
        src, dst = edge_index
        h = self.pretrained_model[1](node_features)
        e = self.pretrained_model[2](edge_features)
        h_global = self.pretrained_model[3](global_feats)
        for _ in range(4):
            m_u = h[src]
            m_v = h[dst]
            e = self.pretrained_model[5](torch.cat([e, m_u, m_v, broadcast_global_to_edges(h_global, node_batch[dst])], dim=1))
            h_e = scatter_sum(e, dst, dim=0, dim_size=h.size(0))
            h = self.pretrained_model[4](torch.cat([h, h_e, broadcast_global_to_nodes(h_global, node_batch)], dim=1))
            mean_n = scatter_mean(h, node_batch, dim=0)
            mean_e = scatter_mean(e, node_batch[dst], dim=0)
            h_global = self.pretrained_model[6](torch.cat([h_global, mean_n, mean_e], dim=1))
        return self.pretrained_model[7](h_global)

    def forward(self, node_features, edge_features, global_feats, edge_index, node_batch):
        pretrained_global = self.Pretrained_Output(node_features, edge_features, global_feats, edge_index, node_batch)
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features)
        h_global = self.global_encoder(global_feats)
        src, dst = edge_index
        for _ in range(self.n_proc_steps):
            e = self.edge_update(torch.cat([e, h[src], h[dst], broadcast_global_to_edges(h_global, node_batch[dst])], dim=1))
            h_e = scatter_sum(e, dst, dim=0, dim_size=h.size(0))
            h = self.node_update(torch.cat([h, h_e, broadcast_global_to_nodes(h_global, node_batch)], dim=1))
            mean_n = scatter_mean(h, node_batch, dim=0)
            mean_e = scatter_mean(e, node_batch[dst], dim=0)
            h_global = self.global_update(torch.cat([h_global, mean_n, mean_e], dim=1))
        h_global = self.global_decoder(h_global)
        return self.classify(torch.cat([pretrained_global, h_global], dim=1))


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, node_features, edge_features, global_features, edge_index, node_batch):
        return self.model(
            node_features,
            edge_features,
            global_features,
            edge_index,
            node_batch,
        )


def build_model(config, sample_graph, sample_global, epoch=None):
    model_conf = config["Model"]
    model_class = model_conf["class"]
    model_args = model_conf["args"].copy()
    model_args.update({"sample_graph": sample_graph, "sample_global": sample_global})
    if model_class == "Edge_Network":
        model = Edge_Network(
            sample_graph=sample_graph,
            sample_global=sample_global,
            hid_size=model_args["hid_size"],
            out_size=model_args["out_size"],
            n_layers=model_args["n_layers"],
            n_proc_steps=model_args["n_proc_steps"],
            dropout=model_args.get("dropout", 0),
        )
    elif model_class == "Transferred_Learning_Finetuning":
        model = Transferred_Learning_Finetuning_With_Globals(
            pretraining_path=model_args["pretraining_path"],
            pretraining_model=model_args["pretraining_model"],
            sample_graph=sample_graph,
            sample_global=sample_global,
            hid_size=model_args["hid_size"],
            out_size=model_args["out_size"],
            n_layers=model_args["n_layers"],
            n_proc_steps=model_args["n_proc_steps"],
            dropout=model_args.get("dropout", 0),
            frozen_pretraining=model_args.get("frozen_pretraining", False),
        )
    else:
        raise ValueError(f"Unsupported model class for ONNX export: {model_class}")
    if epoch is None:
        ep, checkpoint = utils.get_best_epoch(config, var="Test_AUC", mode="max")
    else:
        ep, checkpoint = utils.get_specific_epoch(config, epoch)
    if checkpoint is None:
        raise RuntimeError("No checkpoint found for the provided config.")
    state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        state_dict[new_key] = value
    model.load_state_dict(state_dict)
    model.eval()
    return model, ep


def build_dgl_model(config, sample_graph, sample_global, epoch=None):
    model_conf = config["Model"]
    model = utils.buildFromConfig(
        model_conf,
        {"sample_graph": sample_graph, "sample_global": sample_global},
    )
    if epoch is None:
        ep, checkpoint = utils.get_best_epoch(config, var="Test_AUC", mode="max")
    else:
        ep, checkpoint = utils.get_specific_epoch(config, epoch)
    if checkpoint is None:
        raise RuntimeError("No checkpoint found for the provided config.")
    state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        state_dict[new_key] = value
    model.load_state_dict(state_dict)
    model.eval()
    return model, ep


def export_and_test(config_file, onnx_name, epoch=None):
    config = load_config(config_file)
    sample_name = get_sample_name(config)
    sample_graph, sample_global, sample_batch_num_nodes = load_sample_graph(config)
    summarize_graph_batch(sample_graph)
    dgl_batch, dgl_global_feats = load_dgl_reference_batch(config, batch_size=1024)
    dgl_model, resolved_epoch = build_dgl_model(config, sample_graph, sample_global, epoch=epoch)
    tensor_model, _ = build_model(config, sample_graph, sample_global, epoch=resolved_epoch)

    wrapper = ONNXWrapper(tensor_model).cpu().eval()

    node_features = dgl_batch.ndata["features"].cpu()
    edge_features = dgl_batch.edata["features"].cpu()
    global_features = dgl_global_feats.cpu()
    src, dst = dgl_batch.edges()
    edge_index = torch.stack([src.cpu(), dst.cpu()], dim=0)
    node_batch = torch.repeat_interleave(
        torch.arange(len(dgl_batch.batch_num_nodes()), dtype=torch.int64),
        dgl_batch.batch_num_nodes().cpu().to(torch.int64),
    ).cpu()

    onnx_path = onnx_name if onnx_name.endswith(".onnx") else f"{onnx_name}.onnx"
    # ONNX input order matches the notebook convention:
    # node_features, edge_features, global_features, edge_index, node_batch.
    torch.onnx.export(
        wrapper,
        (node_features, edge_features, global_features, edge_index, node_batch),
        onnx_path,
        input_names=["node_features", "edge_features", "global_features", "edge_index", "node_batch"],
        output_names=["logits"],
        dynamic_axes={
            "node_features": {0: "num_nodes"},
            "edge_features": {0: "num_edges"},
            "global_features": {0: "batch_size"},
            "edge_index": {1: "num_edges"},
            "node_batch": {0: "num_nodes"},
        },
        opset_version=16,
    )
    print(f"Exported ONNX model to {onnx_path} from epoch {resolved_epoch}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed")

    ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "node_features": node_features.numpy().astype(np.float32),
        "edge_features": edge_features.numpy().astype(np.float32),
        "global_features": global_features.numpy().astype(np.float32),
        "edge_index": edge_index.numpy().astype(np.int64),
        "node_batch": node_batch.numpy().astype(np.int64),
    }
    ort_input_names = {inp.name for inp in ort_sess.get_inputs()}
    ort_inputs = {name: value for name, value in ort_inputs.items() if name in ort_input_names}
    print("ONNX runtime inputs:", sorted(ort_input_names))
    ort_out = ort_sess.run(None, ort_inputs)[0]

    with torch.no_grad():
        try:
            dgl_out = dgl_model(dgl_batch.to("cpu"), dgl_global_feats.to("cpu")).cpu().numpy()
        except Exception as exc:
            raise RuntimeError(
                "DGL reference forward failed. The loaded graph's batching metadata "
                "may not match the global feature shape expected by the original model."
            ) from exc
        tensor_out = tensor_model(
            node_features,
            edge_features,
            global_features,
            edge_index,
            node_batch,
        ).cpu().numpy()

    dgl_vs_tensor = np.abs(dgl_out - tensor_out)
    tensor_vs_onnx = np.abs(tensor_out - ort_out)
    dgl_prob = 1.0 / (1.0 + np.exp(-dgl_out))
    tensor_prob = 1.0 / (1.0 + np.exp(-tensor_out))
    ort_prob = 1.0 / (1.0 + np.exp(-ort_out))
    dgl_vs_tensor_prob = np.abs(dgl_prob - tensor_prob)
    tensor_vs_onnx_prob = np.abs(tensor_prob - ort_prob)

    print("DGL output shape    :", dgl_out.shape)
    print("Tensor output shape :", tensor_out.shape)
    print("ONNX output shape   :", ort_out.shape)
    print("max abs diff DGL->Tensor (logit):", dgl_vs_tensor.max())
    print("mean abs diff DGL->Tensor(logit):", dgl_vs_tensor.mean())
    print("max abs diff Tensor->ONNX (logit):", tensor_vs_onnx.max())
    print("mean abs diff Tensor->ONNX(logit):", tensor_vs_onnx.mean())
    print("max abs diff DGL->Tensor (prob)  :", dgl_vs_tensor_prob.max())
    print("mean abs diff DGL->Tensor(prob)  :", dgl_vs_tensor_prob.mean())
    print("max abs diff Tensor->ONNX (prob)  :", tensor_vs_onnx_prob.max())
    print("mean abs diff Tensor->ONNX(prob)  :", tensor_vs_onnx_prob.mean())

    scores_dgl = dgl_prob.reshape(-1)
    scores_tensor = tensor_prob.reshape(-1)
    scores_onnx = ort_prob.reshape(-1)
    bins = np.linspace(0.0, 1.0, 41)
    diff_tensor_dgl = scores_tensor - scores_dgl
    diff_onnx_dgl = scores_onnx - scores_dgl

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4))
    ax_left.hist(scores_dgl, bins=bins, histtype="step", linewidth=2.0, label="DGL", color="black", linestyle="-", alpha=0.95)
    ax_left.hist(scores_tensor, bins=bins, histtype="step", linewidth=2.0, label="Tensor", color="tab:blue", linestyle="--", alpha=0.8)
    ax_left.hist(scores_onnx, bins=bins, histtype="step", linewidth=2.0, label="ONNX", color="tab:orange", linestyle=(0, (3, 1, 1, 1)), alpha=0.8)
    ax_left.set_xlabel("Score")
    ax_left.set_ylabel("Events / bin")
    ax_left.set_title(f"Score Distributions: {sample_name}")
    ax_left.legend()

    diff_bins = np.histogram_bin_edges(diff_onnx_dgl, bins=80)
    ax_right.hist(diff_onnx_dgl, bins=diff_bins, histtype="step", linewidth=1.8, label="ONNX - DGL", color="tab:orange")
    ax_right.hist(diff_tensor_dgl, bins=diff_bins, histtype="step", linewidth=1.8, label="Tensor - DGL", color="tab:blue")
    ax_right.set_xlabel("Score difference")
    ax_right.set_ylabel("Events / bin")
    ax_right.set_title(f"Differences vs DGL: {sample_name}")
    ax_right.set_yscale("log")
    ax_right.legend()

    plt.tight_layout()
    plot_path = os.path.splitext(onnx_path)[0] + "_onnx.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved comparison plot to {plot_path}")
    plt.show()
    print("Export test passed")


def main():
    parser = argparse.ArgumentParser(description="Export a model to ONNX and validate it with ONNX Runtime")
    parser.add_argument("--config", required=True, help="Path to the training config")
    parser.add_argument("--name", required=True, help="Name of the output .onnx file")
    parser.add_argument("--epoch", type=int, default=None, help="Specific checkpoint epoch to export; defaults to best Test_AUC if omitted")
    args = parser.parse_args()

    export_and_test(args.config, args.name, epoch=args.epoch)


if __name__ == "__main__":
    main()
