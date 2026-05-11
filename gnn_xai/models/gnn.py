"""
models/gnn.py
-------------
GNN architectures used in the explainability framework.

Implements:
  - GCN  (Graph Convolutional Network)
  - GAT  (Graph Attention Network)
  - GIN  (Graph Isomorphism Network — stronger expressivity, preferred for XAI evals)
  - GraphSAGE

All models share a common interface:
    forward(x, edge_index, batch=None) → logits

For node-classification tasks, batch is None and the output shape is [N, num_classes].
For graph-classification tasks, global pooling is applied and shape is [B, num_classes].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    SAGEConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    BatchNorm,
)


# ─────────────────────────────────────────────────────────────
# Utility: build MLP (used inside GIN)
# ─────────────────────────────────────────────────────────────
def build_mlp(in_dim: int, hidden_dim: int, out_dim: int,
              num_layers: int = 2, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────
# 1. GCN
# ─────────────────────────────────────────────────────────────
class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).
    Supports both node- and graph-level tasks.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 task: str = "graph"):   # 'graph' | 'node'
        super().__init__()
        self.task = task
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * (num_layers - 1)
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(BatchNorm(dims[i + 1]))

        # Final conv layer → hidden (pooling applied separately)
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph":
            x = global_mean_pool(x, batch)

        return self.classifier(x)

    def get_embeddings(self, x, edge_index, batch=None):
        """Return pre-classifier embeddings for analysis."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        if self.task == "graph":
            x = global_mean_pool(x, batch)
        return x


# ─────────────────────────────────────────────────────────────
# 2. GAT
# ─────────────────────────────────────────────────────────────
class GAT(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2018).
    Multi-head attention at each layer.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.5,
                 task: str = "graph"):
        super().__init__()
        self.task = task
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads=heads, dropout=dropout, concat=True))
        self.bns.append(BatchNorm(hidden_channels * heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, dropout=dropout, concat=True))
            self.bns.append(BatchNorm(hidden_channels * heads))

        # Last layer: single head, no concat
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                  heads=1, dropout=dropout, concat=False))
        self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)

        if self.task == "graph":
            x = global_mean_pool(x, batch)

        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
# 3. GIN  (preferred for explainability evaluations)
# ─────────────────────────────────────────────────────────────
class GIN(nn.Module):
    """
    Graph Isomorphism Network (Xu et al., 2019).
    Maximally expressive among message-passing GNNs (Weisfeiler-Leman equivalent).
    Commonly used as the backbone for GNNExplainer benchmarks.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 pooling: str = "sum",   # 'sum' | 'mean' | 'max'
                 task: str = "graph"):
        super().__init__()
        self.task = task
        self.dropout = dropout

        pool_map = {"sum": global_add_pool,
                    "mean": global_mean_pool,
                    "max": global_max_pool}
        self.pool = pool_map[pooling]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * num_layers
        for i in range(num_layers):
            mlp = build_mlp(dims[i], hidden_channels, dims[i + 1],
                            num_layers=2, dropout=0.0)
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm(dims[i + 1]))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph":
            x = self.pool(x, batch)

        return self.classifier(x)

    def get_node_embeddings(self, x, edge_index):
        """Return node-level embeddings before pooling (used by explainers)."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return x


# ─────────────────────────────────────────────────────────────
# 4. GraphSAGE
# ─────────────────────────────────────────────────────────────
class GraphSAGE(nn.Module):
    """
    Inductive representation learning via GraphSAGE (Hamilton et al., 2017).
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 task: str = "graph"):
        super().__init__()
        self.task = task
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.bns.append(BatchNorm(dims[i + 1]))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph":
            x = global_mean_pool(x, batch)

        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────
def build_model(arch: str,
                in_channels: int,
                hidden_channels: int,
                out_channels: int,
                num_layers: int = 3,
                dropout: float = 0.5,
                task: str = "graph",
                **kwargs) -> nn.Module:
    """
    Factory to instantiate a GNN by name.
    arch: 'gcn' | 'gat' | 'gin' | 'sage'
    """
    arch = arch.lower()
    shared = dict(in_channels=in_channels,
                  hidden_channels=hidden_channels,
                  out_channels=out_channels,
                  num_layers=num_layers,
                  dropout=dropout,
                  task=task)
    if arch == "gcn":
        return GCN(**shared)
    elif arch == "gat":
        return GAT(**shared, heads=kwargs.get("heads", 4))
    elif arch == "gin":
        return GIN(**shared, pooling=kwargs.get("pooling", "sum"))
    elif arch == "sage":
        return GraphSAGE(**shared)
    else:
        raise ValueError(f"Unknown architecture: {arch}. "
                         f"Choose from 'gcn', 'gat', 'gin', 'sage'.")


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    x = torch.randn(20, 10)
    edge_index = torch.randint(0, 20, (2, 60))
    batch = torch.zeros(20, dtype=torch.long)

    for arch in ["gcn", "gat", "gin", "sage"]:
        model = build_model(arch, 10, 64, 2)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, batch)
        print(f"[{arch.upper():5s}] output shape: {out.shape}")
