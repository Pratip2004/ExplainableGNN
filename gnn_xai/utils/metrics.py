"""
utils/metrics.py
-----------------
Quantitative evaluation of GNN explanations.

Implements:
  - Fidelity+   : accuracy drop when top-k explanation edges are KEPT (higher = better)
  - Fidelity-   : accuracy drop when top-k explanation edges are REMOVED (lower = better)
  - Sparsity    : fraction of edges not in explanation (higher = more compact)
  - AUC-ROC     : if ground-truth edge masks exist (e.g., BA-2Motifs)
  - Precision@k : fraction of top-k edges that are in ground truth motif
  - Summary table of all metrics across methods
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.utils import subgraph as pyg_subgraph


# ─────────────────────────────────────────────────────────────
# Helper: model prediction on subgraph / masked graph
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def _predict(model, data: Data, edge_mask_bool: torch.Tensor,
             task: str, device: str) -> int:
    """
    Run model on graph with only selected edges (edge_mask_bool=True).
    """
    data = data.to(device)
    keep_idx = edge_mask_bool.nonzero(as_tuple=True)[0]
    if keep_idx.numel() == 0:
        return -1

    edge_index_sub = data.edge_index[:, keep_idx]

    if task == "graph":
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        out = model(data.x, edge_index_sub, batch)
    else:
        out = model(data.x, edge_index_sub)

    return out.argmax(dim=1).squeeze().item()


@torch.no_grad()
def _original_pred(model, data: Data, task: str, device: str) -> int:
    data = data.to(device)
    if task == "graph":
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        out = model(data.x, data.edge_index, batch)
    else:
        out = model(data.x, data.edge_index)
    return out.argmax(dim=1).squeeze().item()


# ─────────────────────────────────────────────────────────────
# Fidelity metrics
# ─────────────────────────────────────────────────────────────
def fidelity_plus(model, data: Data, edge_mask: torch.Tensor,
                  top_k: int = 10, task: str = "graph",
                  device: str = "cpu") -> float:
    """
    Fidelity+ : keep only explanation edges → does prediction stay the same?
    Higher = explanation is more sufficient.
    """
    orig = _original_pred(model, data, task, device)

    # Select top-k edges
    num_edges = edge_mask.size(0)
    top_k = min(top_k, num_edges)
    _, top_idx = edge_mask.topk(top_k)
    mask = torch.zeros(num_edges, dtype=torch.bool)
    mask[top_idx] = True

    kept_pred = _predict(model, data, mask, task, device)
    return float(kept_pred == orig)


def fidelity_minus(model, data: Data, edge_mask: torch.Tensor,
                   top_k: int = 10, task: str = "graph",
                   device: str = "cpu") -> float:
    """
    Fidelity- : remove explanation edges → does prediction change?
    Lower = explanation captures something important.
    (We report 1 - fidelity_minus so higher is always better.)
    """
    orig = _original_pred(model, data, task, device)

    num_edges = edge_mask.size(0)
    top_k = min(top_k, num_edges)
    _, top_idx = edge_mask.topk(top_k)
    mask = torch.ones(num_edges, dtype=torch.bool)
    mask[top_idx] = False   # remove explanation edges

    removed_pred = _predict(model, data, mask, task, device)
    # If removing changes prediction → explanation was important
    changed = float(removed_pred != orig)
    return changed   # higher means explanation is more necessary


def sparsity(edge_mask: torch.Tensor, top_k: int = 10) -> float:
    """
    Sparsity: fraction of edges NOT in the explanation (0=explain all, 1=explain none).
    """
    n = edge_mask.size(0)
    if n == 0:
        return 0.0
    return 1.0 - min(top_k, n) / n


# ─────────────────────────────────────────────────────────────
# Ground-truth metrics (when gt edge masks are available)
# ─────────────────────────────────────────────────────────────
def auc_roc(edge_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    AUC-ROC between predicted edge mask scores and binary ground-truth masks.
    """
    scores = edge_mask.cpu().numpy()
    labels = gt_mask.cpu().numpy()

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return float("nan")

    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return float("nan")


def precision_at_k(edge_mask: torch.Tensor, gt_mask: torch.Tensor,
                   k: int = 10) -> float:
    """
    Precision@k : of the top-k predicted edges, what fraction are in ground truth?
    """
    num_edges = edge_mask.size(0)
    k = min(k, num_edges)
    _, top_idx = edge_mask.topk(k)
    gt_binary = (gt_mask > 0.5).float().cpu()
    precision = gt_binary[top_idx].mean().item()
    return precision


# ─────────────────────────────────────────────────────────────
# Batch evaluation across a dataset
# ─────────────────────────────────────────────────────────────
def evaluate_explainer(model,
                       data_list: list,
                       explanations: list,
                       task: str = "graph",
                       top_k: int = 10,
                       device: str = "cpu",
                       has_gt: bool = False) -> dict:
    """
    Aggregate explanation quality metrics across a dataset.

    Parameters
    ----------
    model        : trained GNN
    data_list    : list of Data objects
    explanations : list of explanation dicts (from any explainer)
    task         : 'graph' | 'node'
    top_k        : how many top edges to consider
    has_gt       : whether Data objects have 'edge_mask_gt' attribute

    Returns
    -------
    dict of aggregated metrics (mean ± std)
    """
    fid_plus_list = []
    fid_minus_list = []
    sparsity_list = []
    auc_list = []
    precision_list = []

    for data, exp in zip(data_list, explanations):
        em = exp.get("edge_mask", None)
        if em is None:
            continue

        fp = fidelity_plus(model, data, em, top_k=top_k, task=task, device=device)
        fm = fidelity_minus(model, data, em, top_k=top_k, task=task, device=device)
        sp = sparsity(em, top_k=top_k)

        fid_plus_list.append(fp)
        fid_minus_list.append(fm)
        sparsity_list.append(sp)

        if has_gt and hasattr(data, "edge_mask_gt") and data.edge_mask_gt is not None:
            gt = data.edge_mask_gt
            if gt.size(0) == em.size(0):
                auc_list.append(auc_roc(em, gt))
                precision_list.append(precision_at_k(em, gt, k=top_k))

    def _stats(lst):
        lst = [x for x in lst if not np.isnan(x)]
        if not lst:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(lst)), "std": float(np.std(lst))}

    results = {
        "fidelity_plus":  _stats(fid_plus_list),
        "fidelity_minus": _stats(fid_minus_list),
        "sparsity":       _stats(sparsity_list),
        "auc_roc":        _stats(auc_list) if auc_list else {"mean": float("nan"), "std": float("nan")},
        "precision_at_k": _stats(precision_list) if precision_list else {"mean": float("nan"), "std": float("nan")},
    }
    return results


# ─────────────────────────────────────────────────────────────
# Pretty-print comparison table
# ─────────────────────────────────────────────────────────────
def print_comparison_table(results_dict: dict):
    """
    results_dict: {method_name: metrics_dict}
    """
    methods = list(results_dict.keys())
    metrics = ["fidelity_plus", "fidelity_minus", "sparsity", "auc_roc", "precision_at_k"]
    metric_labels = {
        "fidelity_plus": "Fidelity+↑",
        "fidelity_minus": "Fidelity-↑",
        "sparsity": "Sparsity↑",
        "auc_roc": "AUC-ROC↑",
        "precision_at_k": "Prec@K↑",
    }

    col_w = 18
    header = f"{'Method':<20}" + "".join(f"{metric_labels[m]:>{col_w}}" for m in metrics)
    print("\n" + "=" * (20 + col_w * len(metrics)))
    print("  Explainability Metrics Comparison")
    print("=" * (20 + col_w * len(metrics)))
    print(header)
    print("-" * (20 + col_w * len(metrics)))

    for method in methods:
        row = f"{method:<20}"
        for m in metrics:
            val = results_dict[method].get(m, {})
            mean = val.get("mean", float("nan"))
            std = val.get("std", float("nan"))
            if np.isnan(mean):
                cell = "N/A"
            else:
                cell = f"{mean:.3f}±{std:.3f}"
            row += f"{cell:>{col_w}}"
        print(row)

    print("=" * (20 + col_w * len(metrics)))
    print("↑ = higher is better")
    print()


# ─────────────────────────────────────────────────────────────
# Bar plot comparison
# ─────────────────────────────────────────────────────────────
def plot_metrics_comparison(results_dict: dict, save_path: str = None):
    """
    Grouped bar chart comparing methods across metrics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(results_dict.keys())
    metrics = ["fidelity_plus", "fidelity_minus", "sparsity"]
    metric_labels = ["Fidelity+", "Fidelity-", "Sparsity"]

    # Add AUC and Precision if available
    has_auc = any(not np.isnan(results_dict[m].get("auc_roc", {}).get("mean", float("nan")))
                  for m in methods)
    if has_auc:
        metrics.append("auc_roc")
        metric_labels.append("AUC-ROC")

    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / max(len(methods), 1)
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), 5))

    for i, (method, color) in enumerate(zip(methods, colors)):
        means = []
        stds = []
        for m in metrics:
            val = results_dict[method].get(m, {})
            means.append(val.get("mean", 0) or 0)
            stds.append(val.get("std", 0) or 0)

        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=method,
                      color=color, edgecolor="k", alpha=0.85)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    ecolor="black", capsize=3, linewidth=1)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("GNN Explainability Method Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved comparison plot → {save_path}")
    plt.close()
