"""
explainers/pgexplainer.py
--------------------------
PGExplainer (Luo et al., NeurIPS 2020) — Parameterized, generalizable explainer.

Unlike GNNExplainer (per-instance optimization), PGExplainer trains a small
neural network that LEARNS to predict edge masks from node embeddings.
This allows it to:
  - Generalize across instances (one trained explainer serves all)
  - Be fast at inference time (single forward pass)
  - Capture global patterns in what the GNN finds important
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class PGExplainerNet(nn.Module):
    """
    Small MLP that maps [emb_u || emb_v] → logit for edge (u,v) mask.
    Input: concatenation of embeddings of the two endpoint nodes.
    Output: scalar logit (passed through Sigmoid → edge probability).
    """
    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = emb_dim * 2   # concatenate both node embeddings
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, emb_u: torch.Tensor, emb_v: torch.Tensor) -> torch.Tensor:
        """
        emb_u, emb_v: [E, emb_dim] tensors for edge endpoints.
        Returns: [E, 1] logits.
        """
        x = torch.cat([emb_u, emb_v], dim=-1)
        return self.net(x).squeeze(-1)   # [E]


class PGExplainer:
    """
    Parameterized explainer for GNNs.

    Usage:
        pg = PGExplainer(gnn_model, emb_dim=64)
        pg.train_explainer(train_data_list)
        exp = pg.explain(data)
    """

    def __init__(self,
                 model: nn.Module,
                 emb_dim: int,
                 hidden_dim: int = 64,
                 epochs: int = 30,
                 lr: float = 3e-3,
                 temperature: float = 1.0,
                 coeff_size: float = 0.01,
                 coeff_ent: float = 0.5,
                 task: str = "graph",
                 device: str = None):
        self.model = model
        self.emb_dim = emb_dim
        self.task = task
        self.epochs = epochs
        self.lr = lr
        self.temperature = temperature
        self.coeff_size = coeff_size
        self.coeff_ent = coeff_ent
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.explainer_net = PGExplainerNet(emb_dim, hidden_dim).to(self.device)

    # ──────────────────────────────────────────────────────────
    # Get node embeddings from the frozen GNN
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def _get_embeddings(self, data: Data) -> torch.Tensor:
        """Extract node-level embeddings from the GNN backbone."""
        data = data.to(self.device)
        x, edge_index = data.x, data.edge_index

        # Try get_node_embeddings (GIN) or fall back to conv layers
        if hasattr(self.model, "get_node_embeddings"):
            emb = self.model.get_node_embeddings(x, edge_index)
        elif hasattr(self.model, "convs"):
            emb = x
            for i, conv in enumerate(self.model.convs):
                emb = conv(emb, edge_index)
                if hasattr(self.model, "bns"):
                    emb = self.model.bns[i](emb)
                emb = F.relu(emb)
        else:
            # Fallback: raw features
            emb = x

        return emb  # [N, emb_dim]

    # ──────────────────────────────────────────────────────────
    # Training the explainer network
    # ──────────────────────────────────────────────────────────
    def train_explainer(self,
                        data_list: list,
                        batch_size: int = 32,
                        save_path: str = None):
        """
        Train PGExplainerNet on a dataset.
        The loss encourages the predicted mask to reproduce the GNN's prediction.
        """
        optimizer = torch.optim.Adam(self.explainer_net.parameters(), lr=self.lr)
        self.explainer_net.train()

        print(f"\n{'=' * 55}")
        print(f"  Training PGExplainer | epochs={self.epochs}, lr={self.lr}")
        print(f"{'=' * 55}")
        loss_history = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            np.random.shuffle(data_list)

            for data in data_list:
                data = data.to(self.device)
                x, edge_index = data.x, data.edge_index
                optimizer.zero_grad()

                # 1. Get frozen node embeddings
                emb = self._get_embeddings(data)   # [N, D]

                # 2. Predict edge mask logits
                emb_u = emb[edge_index[0]]          # [E, D]
                emb_v = emb[edge_index[1]]          # [E, D]
                logits = self.explainer_net(emb_u, emb_v)   # [E]

                # 3. Gumbel-softmax reparameterisation for differentiable sampling
                edge_prob = self._gumbel_sigmoid(logits, self.temperature)  # [E]

                # 4. Forward pass with soft masked features (approximation)
                x_masked = x  # feature masking not used in PGExplainer
                with torch.no_grad():
                    if self.task == "graph":
                        batch = torch.zeros(x.size(0), dtype=torch.long,
                                            device=self.device)
                        orig_out = self.model(x_masked, edge_index, batch)
                    else:
                        orig_out = self.model(x_masked, edge_index)

                pred_label = orig_out.argmax(dim=1)

                # Use masked model output (scaled x by edge importance)
                # Approximate: weight node feature rows by incoming edge probs
                # mean over all edge probs affecting that node
                node_importance = torch.zeros(x.size(0), device=self.device)
                node_importance.scatter_add_(0, edge_index[1], edge_prob)
                counts = torch.zeros(x.size(0), device=self.device)
                counts.scatter_add_(0, edge_index[1],
                                    torch.ones(edge_index.size(1), device=self.device))
                counts = counts.clamp(min=1)
                node_importance = node_importance / counts
                x_reweighted = x * node_importance.unsqueeze(1)

                if self.task == "graph":
                    masked_out = self.model(x_reweighted, edge_index, batch)
                else:
                    masked_out = self.model(x_reweighted, edge_index)

                # 5. Loss: encourage masked output to match original prediction
                loss_pred = F.cross_entropy(masked_out, pred_label)

                # 6. Regularisation: sparsity + entropy
                eps = 1e-8
                loss_size = self.coeff_size * edge_prob.sum()
                ent = (-edge_prob * torch.log(edge_prob + eps)
                       - (1 - edge_prob) * torch.log(1 - edge_prob + eps))
                loss_ent = self.coeff_ent * ent.mean()

                loss = loss_pred + loss_size + loss_ent
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(data_list), 1)
            loss_history.append(avg_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.epochs} | Loss: {avg_loss:.4f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.explainer_net.state_dict(), save_path)
            print(f"  Saved PGExplainer → {save_path}")

        return loss_history

    def _gumbel_sigmoid(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Gumbel-sigmoid reparameterisation for binary variables."""
        u = torch.zeros_like(logits).uniform_().clamp(1e-8, 1 - 1e-8)
        gumbel = -torch.log(-torch.log(u))
        return torch.sigmoid((logits + gumbel) / temperature)

    # ──────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def explain(self, data: Data, top_k_edges: int = 10) -> dict:
        """
        Generate explanation for a single graph using the trained explainer net.
        """
        self.explainer_net.eval()
        data = data.to(self.device)
        x, edge_index = data.x, data.edge_index

        # Predicted label
        if self.task == "graph":
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
            orig_out = self.model(x, edge_index, batch)
        else:
            orig_out = self.model(x, edge_index)
        pred_label = orig_out.argmax(dim=1).item()

        # Embeddings → edge logits → mask
        emb = self._get_embeddings(data)
        emb_u = emb[edge_index[0]]
        emb_v = emb[edge_index[1]]
        logits = self.explainer_net(emb_u, emb_v)
        edge_mask = torch.sigmoid(logits).cpu()

        # Top-k edges
        topk_vals, topk_idx = edge_mask.topk(min(top_k_edges, edge_mask.size(0)))
        top_edges = []
        for idx, val in zip(topk_idx.tolist(), topk_vals.tolist()):
            src = edge_index[0, idx].item()
            dst = edge_index[1, idx].item()
            top_edges.append((src, dst, val))

        return {
            "edge_mask": edge_mask,
            "top_edges": top_edges,
            "pred_label": pred_label,
        }

    # ──────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────
    def visualize(self,
                  data: Data,
                  explanation: dict,
                  title: str = "PGExplainer",
                  save_path: str = None):
        """Draw explanation subgraph."""
        G = to_networkx(data, to_undirected=True)
        top_edges_set = {(s, d) for s, d, _ in explanation["top_edges"]}
        top_nodes = set([n for e in explanation["top_edges"] for n in e[:2]])

        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"{title} | Class {explanation['pred_label']}", fontsize=13)

        node_colors = ["red" if n in top_nodes else "steelblue" for n in G.nodes()]
        edge_colors = ["red" if (u, v) in top_edges_set or
                       (v, u) in top_edges_set else "lightgrey"
                       for u, v in G.edges()]
        edge_widths = [3.0 if (u, v) in top_edges_set or
                       (v, u) in top_edges_set else 0.4
                       for u, v in G.edges()]

        nx.draw(G, pos, ax=ax, node_size=100, node_color=node_colors,
                edge_color=edge_colors, width=edge_widths,
                with_labels=False, arrows=False)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
