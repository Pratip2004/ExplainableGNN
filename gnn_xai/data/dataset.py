"""
dataset.py
----------
Handles all dataset loading and graph construction for the GNN-XAI framework.

Supported datasets:
  - Mutagenicity (molecular graph classification — real benchmark)
  - BA-2Motifs   (synthetic, commonly used for GNN explainability evaluation)
  - BA-Shapes    (node classification benchmark)
  - Custom MRI   (illustrative; uses random graphs as placeholder)

Each dataset is returned as a list of torch_geometric.data.Data objects.
"""

import os
import random
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import TUDataset, ExplainerDataset
from torch_geometric.datasets import BAMultiShapesDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# Seed for reproducibility
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ─────────────────────────────────────────────────────────────
# 1. TU Datasets (Mutagenicity, MUTAG, etc.)
# ─────────────────────────────────────────────────────────────
def load_tu_dataset(name: str = "Mutagenicity", root: str = "data/TU"):
    """
    Load a TU graph-classification dataset.
    name: 'Mutagenicity' | 'MUTAG' | 'PROTEINS' | 'NCI1'
    """
    os.makedirs(root, exist_ok=True)
    dataset = TUDataset(root=root, name=name, use_node_attr=True)
    print(f"[Dataset] {name}: {len(dataset)} graphs, "
          f"{dataset.num_classes} classes, "
          f"{dataset.num_node_features} node features")
    return dataset


# ─────────────────────────────────────────────────────────────
# 2. BA-2Motifs  (synthetic — standard XAI benchmark)
# ─────────────────────────────────────────────────────────────
def _make_ba_graph(n_base: int, n_attach: int = 2) -> nx.Graph:
    """Generate a Barabási–Albert base graph."""
    G = nx.barabasi_albert_graph(n_base, n_attach, seed=random.randint(0, 9999))
    return G


def _attach_house_motif(G: nx.Graph) -> Tuple[nx.Graph, list]:
    """Attach a 'house' motif (cycle + roof) and return motif node indices."""
    start = G.number_of_nodes()
    # House: nodes 0-4 → bottom edge + two sides + roof edge
    house_edges = [(0,1),(1,2),(2,3),(3,0),(0,2),(1,3)]  # diamond as house base
    for u, v in house_edges:
        G.add_edge(start + u, start + v)
    # Connect motif to a random existing node
    anchor = random.randint(0, start - 1)
    G.add_edge(anchor, start)
    return G, list(range(start, start + 4))


def _attach_cycle_motif(G: nx.Graph, cycle_len: int = 6) -> Tuple[nx.Graph, list]:
    """Attach a cycle motif."""
    start = G.number_of_nodes()
    nodes = list(range(start, start + cycle_len))
    for i in range(cycle_len):
        G.add_edge(nodes[i], nodes[(i + 1) % cycle_len])
    anchor = random.randint(0, start - 1)
    G.add_edge(anchor, nodes[0])
    return G, nodes


def generate_ba2motifs(n_graphs: int = 500,
                       n_base: int = 20,
                       feat_dim: int = 10) -> list:
    """
    Generate BA-2Motifs dataset.
    Class 0 → BA + house motif
    Class 1 → BA + cycle motif
    Returns list of torch_geometric.data.Data
    """
    data_list = []
    for i in range(n_graphs):
        label = i % 2  # alternate classes
        G = _make_ba_graph(n_base)
        if label == 0:
            G, motif_nodes = _attach_house_motif(G)
        else:
            G, motif_nodes = _attach_cycle_motif(G)

        n = G.number_of_nodes()
        # Node features: random + motif indicator
        x = torch.randn(n, feat_dim)
        for mn in motif_nodes:
            x[mn, 0] = 2.0  # slight signal in feature 0

        edge_index = torch.tensor(
            [[u, v] for u, v in G.edges()] +
            [[v, u] for u, v in G.edges()],
            dtype=torch.long
        ).t().contiguous()

        # Ground-truth edge mask: 1 if both endpoints are motif nodes
        motif_set = set(motif_nodes)
        edge_mask_gt = torch.zeros(edge_index.size(1), dtype=torch.float)
        for j in range(edge_index.size(1)):
            if (edge_index[0, j].item() in motif_set and
                    edge_index[1, j].item() in motif_set):
                edge_mask_gt[j] = 1.0

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            motif_nodes=torch.tensor(motif_nodes, dtype=torch.long),
            edge_mask_gt=edge_mask_gt
        )
        data_list.append(data)

    print(f"[Dataset] BA-2Motifs: {len(data_list)} graphs, 2 classes, "
          f"{feat_dim} node features")
    return data_list


# ─────────────────────────────────────────────────────────────
# 3. BA-Shapes (node classification — synthetic)
# ─────────────────────────────────────────────────────────────
def generate_ba_shapes(n_base: int = 300,
                       n_motifs: int = 80,
                       feat_dim: int = 10) -> Data:
    """
    BA-Shapes: single large graph for node classification.
    Nodes in 'house' motifs are class 1/2/3 depending on position;
    all others are class 0.
    """
    G = _make_ba_graph(n_base)
    node_labels = {i: 0 for i in range(n_base)}

    for _ in range(n_motifs):
        G, motif_nodes = _attach_house_motif(G)
        for rank, mn in enumerate(motif_nodes):
            node_labels[mn] = min(rank + 1, 3)

    n = G.number_of_nodes()
    x = torch.randn(n, feat_dim)
    y = torch.tensor([node_labels.get(i, 0) for i in range(n)], dtype=torch.long)

    edge_index = torch.tensor(
        [[u, v] for u, v in G.edges()] +
        [[v, u] for u, v in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    print(f"[Dataset] BA-Shapes: {n} nodes, 4 classes, "
          f"{feat_dim} node features")
    return data


# ─────────────────────────────────────────────────────────────
# 4. Synthetic MRI-like graphs (illustrative)
# ─────────────────────────────────────────────────────────────
def generate_mri_synthetic(n_graphs: int = 200,
                            n_regions: int = 50,
                            feat_dim: int = 16) -> list:
    """
    Simulate MRI brain graphs.
    Nodes = brain regions; edges = spatial adjacency.
    Class 0 = no tumor; Class 1 = tumor present (dense sub-cluster injected).
    """
    data_list = []
    for i in range(n_graphs):
        label = i % 2
        # Base random geometric graph (simulates brain connectivity)
        pos = np.random.rand(n_regions, 2)
        G = nx.random_geometric_graph(n_regions, radius=0.3, pos={j: pos[j] for j in range(n_regions)})

        # Node features: intensity (ch0), texture (ch1-4), spatial (ch5-7), random rest
        x = np.random.randn(n_regions, feat_dim).astype(np.float32)
        x[:, 5] = pos[:, 0]  # x-coord
        x[:, 6] = pos[:, 1]  # y-coord

        tumor_nodes = []
        if label == 1:
            # Inject a dense tumor sub-cluster
            tumor_center = random.randint(0, n_regions - 1)
            tumor_nodes = [tumor_center]
            neighbors = list(G.neighbors(tumor_center))[:5]
            tumor_nodes.extend(neighbors)
            for tn in tumor_nodes:
                x[tn, 0] += 3.0   # high intensity
                x[tn, 1] += 2.0   # texture anomaly
                # add extra edges among tumor nodes
            for a in range(len(tumor_nodes)):
                for b in range(a + 1, len(tumor_nodes)):
                    G.add_edge(tumor_nodes[a], tumor_nodes[b])

        edge_index = torch.tensor(
            [[u, v] for u, v in G.edges()] +
            [[v, u] for u, v in G.edges()],
            dtype=torch.long
        ).t().contiguous()

        # handle disconnected graph edge case
        if edge_index.size(1) == 0:
            for j in range(n_regions - 1):
                G.add_edge(j, j + 1)
            edge_index = torch.tensor(
                [[u, v] for u, v in G.edges()] +
                [[v, u] for u, v in G.edges()],
                dtype=torch.long
            ).t().contiguous()

        data = Data(
            x=torch.tensor(x),
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            tumor_nodes=torch.tensor(tumor_nodes, dtype=torch.long)
        )
        data_list.append(data)

    print(f"[Dataset] MRI-Synthetic: {len(data_list)} graphs, 2 classes, "
          f"{feat_dim} node features, {n_regions} regions/scan")
    return data_list


# ─────────────────────────────────────────────────────────────
# 5. Train / Val / Test split helper
# ─────────────────────────────────────────────────────────────
def split_dataset(data_list: list,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  seed: int = 42) -> tuple:
    """
    Randomly split a list of Data objects into train/val/test.
    Returns (train_list, val_list, test_list).
    """
    random.seed(seed)
    indices = list(range(len(data_list)))
    random.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [data_list[i] for i in train_idx]
    val = [data_list[i] for i in val_idx]
    test = [data_list[i] for i in test_idx]

    print(f"[Split] Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing dataset generation...")
    print("=" * 60)

    ba2 = generate_ba2motifs(n_graphs=100)
    print(f"  Sample BA-2Motifs graph: nodes={ba2[0].num_nodes}, "
          f"edges={ba2[0].num_edges}, label={ba2[0].y.item()}")

    ba_shapes = generate_ba_shapes()
    print(f"  BA-Shapes: {ba_shapes.num_nodes} nodes")

    mri = generate_mri_synthetic(n_graphs=50)
    print(f"  MRI graph sample: nodes={mri[0].num_nodes}, label={mri[0].y.item()}")

    train, val, test = split_dataset(ba2)
    print("Dataset module OK.")
