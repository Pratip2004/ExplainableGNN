"""
utils/report.py  —  Auto-generate Markdown report with X-Node as primary method.
"""

import os
import json
import datetime
import numpy as np


def save_results_json(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)):    return int(obj)
        if isinstance(obj, np.ndarray):           return obj.tolist()
        if isinstance(obj, dict):  return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    print(f"  Saved results → {path}")


def generate_markdown_report(
    dataset_name, model_arch, gnn_metrics, explainer_metrics,
    config, plot_paths, save_path="results/report.md"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _fmt(metrics_dict, key):
        v = metrics_dict.get(key, {})
        m, s = v.get("mean", float("nan")), v.get("std", float("nan"))
        return "N/A" if np.isnan(m) else f"{m:.3f} ± {s:.3f}"

    lines = [
        "# GNN Explainability Experiment Report",
        f"> Generated: {now}",
        "",
        "---",
        "",
        "## 1. Experiment Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Dataset | `{dataset_name}` |",
        f"| Primary Model | **{model_arch.upper()} (X-Node Self-Explaining)** |",
        f"| Backbone | `{config.get('backbone','gin')}` |",
        f"| Hidden Channels | {config.get('hidden_channels','—')} |",
        f"| GNN Layers | {config.get('num_layers','—')} |",
        f"| Explanation Dim | {config.get('expl_dim', config.get('explanation_dim','—'))} |",
        f"| Faithfulness Weight λ | {config.get('faith_w', config.get('faith_weight','—'))} |",
        f"| Dropout | {config.get('dropout','—')} |",
        f"| Training Epochs | {config.get('epochs','—')} |",
        f"| Learning Rate | {config.get('lr','—')} |",
        f"| Task | {config.get('task','—')} |",
        "",
        "---",
        "",
        "## 2. GNN Predictive Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in gnn_metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | **{v:.4f}** |")

    if plot_paths.get("training_curve"):
        lines += ["", f"![Training Curves]({plot_paths['training_curve']})", ""]
    if plot_paths.get("confusion_matrix"):
        lines += ["", f"![Confusion Matrix]({plot_paths['confusion_matrix']})", ""]

    lines += [
        "---",
        "",
        "## 3. X-Node — Self-Explaining GNN (Primary Method)",
        "",
        "> **X-Node** (Sengupta & Rekik, GRAIL@MICCAI 2025 — arXiv:2508.10461)",
        ">",
        "> Unlike post-hoc methods, X-Node generates explanations *as part of* the",
        "> forward pass via three components:",
        "> 1. **Context Vector** c_v — 7 interpretable topological cues per node",
        "> 2. **Reasoner** r_θ — lightweight MLP mapping context → explanation vector e_v",
        "> 3. **Text Injection** — e_v fed back into message-passing pipeline",
        ">",
        "> Training loss: `L = L_CrossEntropy + λ × L_Faithfulness`",
        "> where L_Faithfulness = MSE(decode(e_v), h_v)",
        "",
        "### X-Node Context Features",
        "",
        "| # | Feature | Description |",
        "|---|---------|-------------|",
        "| 0 | Degree (norm) | Node degree / max degree in graph |",
        "| 1 | Clustering Coefficient | Local clustering around the node |",
        "| 2 | Eigenvector Centrality | Global importance via power iteration |",
        "| 3 | Degree Centrality | degree / (N-1) |",
        "| 4 | 2-hop Label Agreement | Fraction of 2-hop neighbours with same label |",
        "| 5 | Avg Edge Weight | Mean cosine similarity to 1-hop neighbours |",
        "| 6 | Feature Saliency | L2 norm of node feature vector (normalised) |",
        "",
    ]

    if plot_paths.get("xnode_viz"):
        lines += [
            "### X-Node Explanation Visualization",
            f"![X-Node]({plot_paths['xnode_viz']})",
            "",
            "*(Left) Graph coloured by explanation importance. "
            "(Centre) Context heatmap for top-k nodes. "
            "(Right) Explanation strength bar chart.*",
            "",
        ]
    if plot_paths.get("xnode_radar"):
        lines += [
            "### X-Node Context Radar Chart (top node)",
            f"![X-Node Radar]({plot_paths['xnode_radar']})",
            "",
        ]

    lines += [
        "---",
        "",
        "## 4. Explainability Method Comparison",
        "",
        "| Method | Type | Fidelity+ ↑ | Fidelity- ↑ | Sparsity ↑ | AUC-ROC ↑ | Prec@K ↑ |",
        "|--------|------|------------|------------|-----------|----------|---------|",
    ]

    method_types = {
        "X-Node": "Self-Explaining (OURS)",
        "GNNExplainer": "Post-hoc / Mask Opt.",
        "PGExplainer": "Post-hoc / Parameterized",
        "PGM-Explainer": "Post-hoc / Probabilistic",
        "SubgraphX": "Post-hoc / MCTS+Shapley",
    }
    for method, mets in explainer_metrics.items():
        mtype = method_types.get(method, "Post-hoc")
        row = (f"| **{method}** | {mtype} "
               f"| {_fmt(mets,'fidelity_plus')} "
               f"| {_fmt(mets,'fidelity_minus')} "
               f"| {_fmt(mets,'sparsity')} "
               f"| {_fmt(mets,'auc_roc')} "
               f"| {_fmt(mets,'precision_at_k')} |")
        lines.append(row)

    if plot_paths.get("metrics_comparison"):
        lines += ["", f"![Metrics Comparison]({plot_paths['metrics_comparison']})", ""]

    lines += [
        "---",
        "",
        "## 5. Post-hoc Method Visualizations",
        "",
    ]
    for method in ["gnnexplainer","pgexplainer","pgm_explainer","subgraphx"]:
        key = f"{method}_viz"
        if plot_paths.get(key):
            lines += [
                f"### {method.replace('_',' ').title()}",
                f"![{method}]({plot_paths[key]})",
                "",
            ]

    lines += [
        "---",
        "",
        "## 6. Method Summary & Trade-offs",
        "",
        "| Method | When To Use | Strength | Limitation |",
        "|--------|-------------|----------|------------|",
        "| **X-Node** | When interpretability must be part of the model | Zero extra inference cost; per-node; faithful by construction | Slightly more complex training loss |",
        "| GNNExplainer | Post-hoc, any frozen GNN | Simple, general | Per-instance optimisation; can be brittle |",
        "| PGExplainer | Post-hoc, need speed at inference | Generalizes across instances | Requires training the explainer network |",
        "| PGM-Explainer | When dependency structure matters | Richer causal-style relationships | Expensive for large graphs |",
        "| SubgraphX | Molecular / structural data with functional groups | Shapley-valued, interaction-aware | Computationally heavy (MCTS) |",
        "",
        "---",
        "",
        "## 7. Conclusion",
        "",
        f"This experiment implemented and evaluated XAI methods for GNNs on the **{dataset_name}** dataset.",
        "X-Node was used as the primary self-explaining framework, integrating interpretable",
        "context vectors and a Reasoner module into the GNN's message-passing pipeline.",
        "Post-hoc methods (GNNExplainer, PGExplainer, PGM-Explainer, SubgraphX) were",
        "evaluated for comparison, revealing the trade-offs between faithfulness, efficiency,",
        "and interpretability depth.",
        "",
        "---",
        "*Report generated by the GNN-XAI Framework*",
        "*Diptarshi Das & Pratip Modak | Supervisor: Dr. Ananda Shankar Chowdhury*",
    ]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved report → {save_path}")
    return save_path
