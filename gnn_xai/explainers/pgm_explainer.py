"""
explainers/pgm_explainer.py
----------------------------
PGM-Explainer (Vu & Thai, NeurIPS 2020) — Probabilistic Graphical Model approach.

Builds a Bayesian Network approximating the GNN's conditional behavior for a
given prediction. Outputs explicit probabilistic dependency structures among
important nodes/features rather than just importance scores.

Method:
  1. Perturb node features randomly → collect GNN predictions
  2. Identify nodes whose perturbations most change the prediction (chi-squared)
  3. Fit a simple Bayesian Network (Naive Bayes approximation) on selected nodes
  4. Visualize the dependency structure
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from scipy.stats import chi2_contingency
from itertools import combinations


class PGMExplainer:
    """
    Probabilistic Graphical Model explainer for GNNs.

    Perturbs node features and uses chi-squared tests to find which
    nodes significantly influence the prediction, then builds
    a simple dependency graph.
    """

    def __init__(self,
                 model,
                 n_samples: int = 100,
                 perturbation_std: float = 0.1,
                 num_top_nodes: int = 8,
                 task: str = "graph",
                 p_threshold: float = 0.05,
                 device: str = None):
        """
        Parameters
        ----------
        n_samples        : number of perturbation samples
        perturbation_std : std of Gaussian perturbation added to node features
        num_top_nodes    : how many most-influential nodes to include in PGM
        p_threshold      : chi-squared p-value threshold for edge inclusion
        """
        self.model = model
        self.n_samples = n_samples
        self.perturbation_std = perturbation_std
        self.num_top_nodes = num_top_nodes
        self.task = task
        self.p_threshold = p_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    # ──────────────────────────────────────────────────────────
    # Perturbation sampling
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def _sample_perturbations(self, data: Data) -> tuple:
        """
        Generate perturbed versions of the input graph and collect GNN predictions.

        Returns
        -------
        perturb_matrix : [n_samples, N] binary — was node i perturbed in sample s?
        pred_changes   : [n_samples]    binary — did prediction change vs original?
        orig_label     : int            original predicted class
        """
        data = data.to(self.device)
        x, edge_index = data.x, data.edge_index
        N, F = x.shape

        # Original prediction
        if self.task == "graph":
            batch = torch.zeros(N, dtype=torch.long, device=self.device)
            orig_out = self.model(x, edge_index, batch)
        else:
            orig_out = self.model(x, edge_index)
        orig_label = orig_out.argmax(dim=1).squeeze().item()

        perturb_matrix = np.zeros((self.n_samples, N), dtype=np.int32)
        pred_changes = np.zeros(self.n_samples, dtype=np.int32)

        for s in range(self.n_samples):
            # Randomly choose ~50% of nodes to perturb
            perturb_mask = np.random.rand(N) < 0.5
            perturb_matrix[s] = perturb_mask.astype(np.int32)

            x_perturbed = x.clone()
            perturbed_nodes = np.where(perturb_mask)[0]
            for node in perturbed_nodes:
                noise = torch.randn(F, device=self.device) * self.perturbation_std
                x_perturbed[node] += noise

            if self.task == "graph":
                out = self.model(x_perturbed, edge_index, batch)
            else:
                out = self.model(x_perturbed, edge_index)

            new_label = out.argmax(dim=1).squeeze().item()
            pred_changes[s] = int(new_label != orig_label)

        return perturb_matrix, pred_changes, int(orig_label)

    # ──────────────────────────────────────────────────────────
    # Chi-squared test for node importance
    # ──────────────────────────────────────────────────────────
    def _compute_node_importance(self,
                                  perturb_matrix: np.ndarray,
                                  pred_changes: np.ndarray) -> dict:
        """
        For each node, run chi-squared test between perturbation of that node
        and prediction change. Returns dict node_idx → p_value.
        """
        N = perturb_matrix.shape[1]
        node_pvals = {}

        for i in range(N):
            node_col = perturb_matrix[:, i]
            # Contingency table:
            # rows: node perturbed (0/1), cols: pred changed (0/1)
            tbl = np.zeros((2, 2), dtype=np.int32)
            for s in range(len(node_col)):
                tbl[node_col[s], pred_changes[s]] += 1

            # Skip if any marginal is zero
            if tbl.sum() == 0 or tbl[0].sum() == 0 or tbl[1].sum() == 0:
                node_pvals[i] = 1.0
                continue

            try:
                _, p, _, _ = chi2_contingency(tbl)
                node_pvals[i] = p
            except Exception:
                node_pvals[i] = 1.0

        return node_pvals

    # ──────────────────────────────────────────────────────────
    # Build dependency graph (Bayesian Network approximation)
    # ──────────────────────────────────────────────────────────
    def _build_pgm(self,
                   perturb_matrix: np.ndarray,
                   pred_changes: np.ndarray,
                   top_nodes: list) -> nx.DiGraph:
        """
        Build a simple BN: directed edges between top nodes if their joint
        perturbation is more informative than either alone.

        Uses conditional chi-squared as a proxy for dependency.
        """
        pgm = nx.DiGraph()
        pgm.add_nodes_from(top_nodes)

        for u, v in combinations(top_nodes, 2):
            col_u = perturb_matrix[:, u]
            col_v = perturb_matrix[:, v]

            # Joint contingency with prediction change
            # 4-cell table: (u_pert, v_pert) joint vs pred_change
            both_pert = ((col_u == 1) & (col_v == 1)).astype(int)
            tbl = np.zeros((2, 2), dtype=np.int32)
            for s in range(len(both_pert)):
                tbl[both_pert[s], pred_changes[s]] += 1

            try:
                _, p, _, _ = chi2_contingency(tbl)
            except Exception:
                p = 1.0

            if p < self.p_threshold:
                # Add directed edge from more important to less important node
                pgm.add_edge(u, v, weight=1 - p, p_value=p)

        return pgm

    # ──────────────────────────────────────────────────────────
    # Main explain() method
    # ──────────────────────────────────────────────────────────
    def explain(self, data: Data) -> dict:
        """
        Explain a single prediction using PGM-Explainer.

        Returns
        -------
        dict with:
            top_nodes       : list of most influential node indices
            node_pvals      : dict node → p-value
            pgm             : networkx DiGraph (Bayesian Network approximation)
            pred_label      : int
            perturb_matrix  : raw perturbation data
            pred_changes    : raw prediction change data
        """
        print("  PGM-Explainer: sampling perturbations...", end="\r")
        perturb_matrix, pred_changes, pred_label = self._sample_perturbations(data)
        print(f"  PGM-Explainer: {self.n_samples} samples done. "
              f"Pred={pred_label}, "
              f"Changes={pred_changes.sum()}/{self.n_samples}   ")

        node_pvals = self._compute_node_importance(perturb_matrix, pred_changes)

        # Select top_k nodes by smallest p-value
        sorted_nodes = sorted(node_pvals.items(), key=lambda kv: kv[1])
        top_nodes = [n for n, _ in sorted_nodes[:self.num_top_nodes]]

        pgm = self._build_pgm(perturb_matrix, pred_changes, top_nodes)

        return {
            "top_nodes": top_nodes,
            "node_pvals": node_pvals,
            "pgm": pgm,
            "pred_label": pred_label,
            "perturb_matrix": perturb_matrix,
            "pred_changes": pred_changes,
        }

    # ──────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────
    def visualize(self,
                  data: Data,
                  explanation: dict,
                  title: str = "PGM-Explainer",
                  save_path: str = None):
        """
        Two-panel: (a) original graph with top nodes highlighted,
                   (b) the Bayesian Network dependency structure.
        """
        G_orig = to_networkx(data, to_undirected=True)
        pgm = explanation["pgm"]
        top_nodes = set(explanation["top_nodes"])

        try:
            pos_orig = nx.kamada_kawai_layout(G_orig)
        except Exception:
            pos_orig = nx.spring_layout(G_orig, seed=42)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{title} | Predicted: Class {explanation['pred_label']}",
                     fontsize=13)

        # Panel A: original graph
        ax = axes[0]
        ax.set_title("Important Nodes (highlighted)")
        nc = ["red" if n in top_nodes else "steelblue" for n in G_orig.nodes()]
        ns = [200 if n in top_nodes else 60 for n in G_orig.nodes()]
        ec = ["red" if (u in top_nodes and v in top_nodes) else "lightgrey"
              for u, v in G_orig.edges()]
        nx.draw(G_orig, pos_orig, ax=ax, node_color=nc, node_size=ns,
                edge_color=ec, with_labels=False, arrows=False)
        ax.legend(handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="red", markersize=8, label="Top nodes"),
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="steelblue", markersize=8, label="Other"),
        ], loc="upper left", fontsize=9)

        # Panel B: PGM
        ax2 = axes[1]
        ax2.set_title("Dependency Structure (Bayesian Network Approx.)")
        if pgm.number_of_nodes() > 0:
            try:
                pos_pgm = nx.kamada_kawai_layout(pgm)
            except Exception:
                pos_pgm = nx.spring_layout(pgm, seed=0)

            edge_weights = [pgm[u][v].get("weight", 0.5) * 3
                            for u, v in pgm.edges()]
            nx.draw(pgm, pos_pgm, ax=ax2, node_size=500, node_color="salmon",
                    edge_color="crimson", width=edge_weights,
                    with_labels=True, font_size=8, arrows=True,
                    arrowsize=15, font_color="black")

            # p-value annotations
            node_pvals = explanation["node_pvals"]
            for node in pgm.nodes():
                x_pos, y_pos = pos_pgm[node]
                pv = node_pvals.get(node, 1.0)
                ax2.text(x_pos, y_pos - 0.12, f"p={pv:.3f}",
                         ha="center", fontsize=7, color="darkred")
        else:
            ax2.text(0.5, 0.5, "No significant\ndependencies found",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=12)
            ax2.set_axis_off()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_node_importance(self,
                              explanation: dict,
                              save_path: str = None):
        """Bar chart of -log(p-value) for each node (top 20)."""
        pvals = explanation["node_pvals"]
        sorted_nodes = sorted(pvals.items(), key=lambda kv: kv[1])[:20]
        nodes, pvs = zip(*sorted_nodes)
        importance = [-np.log10(max(p, 1e-10)) for p in pvs]

        plt.figure(figsize=(10, 4))
        colors = ["red" if n in explanation["top_nodes"] else "steelblue"
                  for n in nodes]
        plt.bar(range(len(nodes)), importance, color=colors, edgecolor="k")
        plt.xticks(range(len(nodes)), [f"N{n}" for n in nodes],
                   rotation=45, ha="right")
        plt.ylabel("-log10(p-value)")
        plt.title(f"Node Importance (chi-squared) | Class {explanation['pred_label']}")
        plt.axhline(-np.log10(self.p_threshold), color="r", linestyle="--",
                    label=f"p={self.p_threshold}")
        plt.legend()
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
