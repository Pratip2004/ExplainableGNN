"""
explainers/subgraphx.py
------------------------
SubgraphX (Yuan et al., ICML 2021) — Monte Carlo Tree Search + Shapley Values.

Searches for the most important *connected subgraph* using MCTS.
Uses Shapley values (cooperative game theory) as the principled
measure of subgraph importance, capturing interaction effects.

Especially effective for molecular graphs where functional groups
(connected subgraphs) are the meaningful explanation unit.

Simplified MCTS implementation compatible with any PyG GNN.
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph as pyg_subgraph
from torch_geometric.data import Data
from copy import deepcopy
import random


# ─────────────────────────────────────────────────────────────
# Shapley value computation (Monte Carlo sampling)
# ─────────────────────────────────────────────────────────────
class ShapleyEstimator:
    """
    Estimates Shapley values for subgraph nodes using
    Monte Carlo sampling over coalitions.
    """

    def __init__(self, model, task: str = "graph", n_samples: int = 50,
                 device: str = None):
        self.model = model
        self.task = task
        self.n_samples = n_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    @torch.no_grad()
    def score_coalition(self, data: Data, coalition: list,
                        target_class: int) -> float:
        """
        Score a coalition (subset of nodes) by running the GNN
        on the induced subgraph and returning target class probability.
        """
        if not coalition:
            return 0.0

        nodes = torch.tensor(coalition, dtype=torch.long)
        # Get induced subgraph
        edge_index_sub, _ = pyg_subgraph(
            nodes, data.edge_index, relabel_nodes=True,
            num_nodes=data.num_nodes
        )

        if edge_index_sub.size(1) == 0 and len(coalition) < 2:
            return 0.0

        x_sub = data.x[nodes].to(self.device)
        edge_index_sub = edge_index_sub.to(self.device)

        try:
            if self.task == "graph":
                batch = torch.zeros(len(coalition), dtype=torch.long,
                                    device=self.device)
                out = self.model(x_sub, edge_index_sub, batch)
            else:
                out = self.model(x_sub, edge_index_sub)
                if out.size(0) == 0:
                    return 0.0
                out = out.mean(0, keepdim=True)

            prob = F.softmax(out, dim=1)[0, target_class].item()
        except Exception:
            prob = 0.0

        return prob

    def shapley_value(self, data: Data, node_idx: int,
                      all_nodes: list, target_class: int) -> float:
        """
        Estimate Shapley value for node_idx within all_nodes using
        Monte Carlo sampling of orderings.
        """
        others = [n for n in all_nodes if n != node_idx]
        if not others:
            return self.score_coalition(data, [node_idx], target_class)

        shapley_sum = 0.0
        for _ in range(self.n_samples):
            # Random ordering of other nodes
            perm = random.sample(others, len(others))
            # Random split point
            k = random.randint(0, len(perm))
            coalition_without = perm[:k]
            coalition_with = perm[:k] + [node_idx]

            v_with = self.score_coalition(data, coalition_with, target_class)
            v_without = self.score_coalition(data, coalition_without, target_class)
            shapley_sum += (v_with - v_without)

        return shapley_sum / self.n_samples


# ─────────────────────────────────────────────────────────────
# MCTS Node
# ─────────────────────────────────────────────────────────────
class MCTSNode:
    def __init__(self, coalition: list, c_puct: float = 5.0):
        self.coalition = coalition          # current subset of nodes
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0
        self.c_puct = c_puct

    @property
    def Q(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def UCB(self, parent_visits: int) -> float:
        if self.visit_count == 0:
            return float("inf")
        exploit = self.Q
        explore = self.c_puct * math.sqrt(math.log(parent_visits + 1) /
                                           self.visit_count)
        return exploit + explore


# ─────────────────────────────────────────────────────────────
# SubgraphX main class
# ─────────────────────────────────────────────────────────────
class SubgraphX:
    """
    SubgraphX: MCTS-based subgraph explanation with Shapley values.

    At each MCTS step:
      - Selection: UCB-guided tree traversal
      - Expansion: try removing one node at a time
      - Evaluation: Shapley-based score of coalition
      - Backprop: update visit counts & values
    """

    def __init__(self,
                 model,
                 min_subgraph_size: int = 5,
                 max_subgraph_size: int = 15,
                 n_mcts_steps: int = 50,
                 n_shapley_samples: int = 20,
                 c_puct: float = 5.0,
                 task: str = "graph",
                 device: str = None):
        self.model = model
        self.min_subgraph_size = min_subgraph_size
        self.max_subgraph_size = max_subgraph_size
        self.n_mcts_steps = n_mcts_steps
        self.c_puct = c_puct
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.shapley = ShapleyEstimator(
            model, task, n_samples=n_shapley_samples, device=self.device
        )

    @torch.no_grad()
    def _get_pred_label(self, data: Data) -> int:
        data = data.to(self.device)
        if self.task == "graph":
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            out = self.model(data.x, data.edge_index, batch)
        else:
            out = self.model(data.x, data.edge_index)
        return out.argmax(dim=1).squeeze().item()

    # ── MCTS core ─────────────────────────────────────────────
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse to a leaf using UCB."""
        while node.children:
            best = max(node.children, key=lambda c: c.UCB(node.visit_count))
            node = best
        return node

    def _expand(self, node: MCTSNode, G: nx.Graph) -> list:
        """
        Expand by removing each node from coalition (if coalition > min_size).
        Also try adding neighboring nodes.
        """
        children = []
        coalition_set = set(node.coalition)

        # Shrink: remove one node
        if len(node.coalition) > self.min_subgraph_size:
            for n in node.coalition:
                new_coalition = [x for x in node.coalition if x != n]
                child = MCTSNode(new_coalition, self.c_puct)
                children.append(child)

        # Grow: add one neighbor node
        if len(node.coalition) < self.max_subgraph_size:
            boundary = set()
            for n in node.coalition:
                for nb in G.neighbors(n):
                    if nb not in coalition_set:
                        boundary.add(nb)
            for nb in list(boundary)[:5]:  # limit branching
                new_coalition = node.coalition + [nb]
                child = MCTSNode(new_coalition, self.c_puct)
                children.append(child)

        node.children = children
        return children

    def _evaluate(self, node: MCTSNode, data: Data, target_class: int) -> float:
        """Score node's coalition using Shapley estimator."""
        return self.shapley.score_coalition(data, node.coalition, target_class)

    def _backprop(self, path: list, value: float):
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value

    # ── Main search ───────────────────────────────────────────
    def explain(self, data: Data, target_class: int = None,
                top_k_nodes: int = 8) -> dict:
        """
        Run MCTS to find the best explanatory subgraph.

        Returns
        -------
        dict:
            best_coalition : list of node indices in best subgraph
            best_score     : Shapley score
            all_scores     : dict node → Shapley value
            pred_label     : int
            root           : MCTSNode (tree for inspection)
        """
        data_cpu = data.clone()
        data = data.to(self.device)

        if target_class is None:
            target_class = self._get_pred_label(data)

        G = to_networkx(data_cpu, to_undirected=True)
        all_nodes = list(G.nodes())

        if not all_nodes:
            return {"best_coalition": [], "best_score": 0.0,
                    "all_scores": {}, "pred_label": target_class}

        # Initial coalition: all nodes (then MCTS prunes it)
        init_size = min(self.max_subgraph_size, len(all_nodes))
        init_coalition = random.sample(all_nodes, init_size)

        root = MCTSNode(init_coalition, self.c_puct)
        best_score = -float("inf")
        best_coalition = init_coalition

        print(f"  SubgraphX: Running {self.n_mcts_steps} MCTS steps...", end="\r")

        for step in range(self.n_mcts_steps):
            # Selection
            path = [root]
            node = self._select(root)
            path.append(node)

            # Expansion
            children = self._expand(node, G)
            if children:
                child = random.choice(children)
                path.append(child)
                node = child

            # Evaluation
            score = self._evaluate(node, data_cpu, target_class)

            # Track best
            if score > best_score and len(node.coalition) >= self.min_subgraph_size:
                best_score = score
                best_coalition = node.coalition[:]

            # Backprop
            self._backprop(path, score)

        print(f"  SubgraphX: Done. Best score: {best_score:.4f}, "
              f"Coalition size: {len(best_coalition)}   ")

        # Individual Shapley values for top nodes
        print("  SubgraphX: Computing individual Shapley values...", end="\r")
        candidate_nodes = list(set(best_coalition + random.sample(
            all_nodes, min(top_k_nodes, len(all_nodes))
        )))
        all_scores = {}
        for n in candidate_nodes:
            sv = self.shapley.shapley_value(
                data_cpu, n, candidate_nodes, target_class
            )
            all_scores[n] = sv
        print("  SubgraphX: Shapley values computed.          ")

        return {
            "best_coalition": best_coalition,
            "best_score": best_score,
            "all_scores": all_scores,
            "pred_label": target_class,
            "root": root,
        }

    # ──────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────
    def visualize(self,
                  data: Data,
                  explanation: dict,
                  title: str = "SubgraphX",
                  save_path: str = None):
        G = to_networkx(data, to_undirected=True)
        coalition_set = set(explanation["best_coalition"])

        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{title} | Class {explanation['pred_label']} | "
                     f"Score: {explanation['best_score']:.4f}", fontsize=13)

        # Left: full graph
        ax = axes[0]
        ax.set_title("Best Subgraph (MCTS)")
        nc = ["orange" if n in coalition_set else "steelblue" for n in G.nodes()]
        ns = [250 if n in coalition_set else 50 for n in G.nodes()]
        ec = ["orange" if (u in coalition_set and v in coalition_set) else "lightgrey"
              for u, v in G.edges()]
        ew = [2.5 if (u in coalition_set and v in coalition_set) else 0.4
              for u, v in G.edges()]
        nx.draw(G, pos, ax=ax, node_color=nc, node_size=ns,
                edge_color=ec, width=ew, with_labels=False, arrows=False)

        # Right: Shapley value bar chart
        ax2 = axes[1]
        all_scores = explanation["all_scores"]
        if all_scores:
            sorted_items = sorted(all_scores.items(), key=lambda kv: kv[1],
                                  reverse=True)
            nodes, scores = zip(*sorted_items[:15])
            colors = ["orange" if n in coalition_set else "steelblue" for n in nodes]
            ax2.bar(range(len(nodes)), scores, color=colors, edgecolor="k")
            ax2.set_xticks(range(len(nodes)))
            ax2.set_xticklabels([f"N{n}" for n in nodes], rotation=45, ha="right")
            ax2.set_ylabel("Shapley Value")
            ax2.set_title("Node Shapley Values")
            ax2.axhline(0, color="k", linewidth=0.8)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
