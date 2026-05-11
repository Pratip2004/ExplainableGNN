"""
explainers/xnode.py
--------------------
X-Node: Self-Explanation is All We Need (Sengupta & Rekik, GRAIL@MICCAI 2025)
arXiv: 2508.10461   |   github.com/basiralab/X-Node

Architecture Overview
---------------------
For every node v, X-Node does three things *during* the forward pass
(NOT post-hoc):

1. Context Vector c_v
   Encodes interpretable topological cues from v's local neighborhood:
     - degree              (normalised)
     - clustering coefficient
     - eigenvector centrality (approximated via power iteration)
     - degree centrality   (= degree / (N-1))
     - 2-hop label agreement (fraction of 2-hop neighbours sharing predicted label)
     - average edge weight (mean cosine similarity to 1-hop neighbours)
     - feature saliency    (L2 norm of node features)
   Dimension: 7  (fixed)

2. Reasoner  r_θ : c_v  →  e_v   (explanation vector, dim = expl_dim)
   Lightweight MLP.  Serves three roles simultaneously:
     (a) Decoder enforces faithfulness:  decode(e_v) ≈ h_v  (node embedding)
     (b) Text injection: e_v is projected and added to h_v before
         the *next* GNN message-passing layer → explanation guides prediction
     (c) Natural-language generation (offline template when no LLM API key)

3. X-Node GNN backbone
   Wraps any standard GNN (GCN / GAT / GIN) and injects explanation
   vectors between layers via the text-injection mechanism.

Loss
----
  L = L_task  +  λ_faith * L_faithfulness
where
  L_faithfulness = MSE( decode(e_v), h_v )   ← reconstruction of embedding

Usage
-----
  model = XNodeGNN(in_channels=10, hidden=64, out_channels=2,
                   num_layers=3, backbone="gin", task="graph")
  out, node_expls = model(data.x, data.edge_index, batch=data.batch,
                          return_explanations=True)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool, BatchNorm
)
from torch_geometric.utils import to_networkx, degree as pyg_degree
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────
# 1.  Topological Context Builder
# ─────────────────────────────────────────────────────────────
class TopologicalContextBuilder:
    """
    Computes a 7-dimensional interpretable context vector for every node.

    Features (all normalised to [0, 1] or [-1, 1]):
      [0] degree_norm          — degree / max_degree
      [1] clustering_coeff     — local clustering coefficient
      [2] eigvec_centrality    — approximated via 10-step power iteration
      [3] degree_centrality    — degree / (N-1)
      [4] two_hop_agreement    — fraction of 2-hop neighbours with same label
      [5] avg_edge_weight      — mean cosine similarity to 1-hop neighbours
      [6] feature_saliency     — L2 norm of x_v (normalised by max over graph)

    All computations are done in NumPy/NetworkX so they are always
    available even without PyTorch Geometric extras.
    """

    CONTEXT_DIM = 7

    def __call__(self, data: Data, pred_labels: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        data        : PyG Data object (must have .x, .edge_index)
        pred_labels : [N] long tensor of predicted node labels (optional;
                      used for 2-hop agreement; zeros used if None)

        Returns
        -------
        context : [N, 7] float tensor
        """
        x = data.x.cpu().float()
        ei = data.edge_index.cpu()
        N = x.size(0)

        if pred_labels is None:
            pred_labels = torch.zeros(N, dtype=torch.long)
        else:
            pred_labels = pred_labels.cpu()

        G = to_networkx(data, to_undirected=True)
        # ensure all nodes present even if isolated
        G.add_nodes_from(range(N))

        context = torch.zeros(N, self.CONTEXT_DIM)

        # 0: degree (normalised)
        degs = torch.tensor([G.degree(i) for i in range(N)], dtype=torch.float)
        max_deg = degs.max().clamp(min=1)
        context[:, 0] = degs / max_deg

        # 1: clustering coefficient
        cc = nx.clustering(G)
        context[:, 1] = torch.tensor([cc.get(i, 0.0) for i in range(N)],
                                      dtype=torch.float)

        # 2: eigenvector centrality (power iteration, 10 steps, safe fallback)
        try:
            evc = nx.eigenvector_centrality_numpy(G)
            evc_vals = torch.tensor([evc.get(i, 0.0) for i in range(N)],
                                     dtype=torch.float)
            evc_max = evc_vals.max().clamp(min=1e-8)
            context[:, 2] = evc_vals / evc_max
        except Exception:
            context[:, 2] = context[:, 0]   # fallback to degree

        # 3: degree centrality = deg / (N-1)
        context[:, 3] = degs / max(N - 1, 1)

        # 4: 2-hop label agreement
        for v in range(N):
            two_hop = set()
            for u in G.neighbors(v):
                for w in G.neighbors(u):
                    if w != v:
                        two_hop.add(w)
            if two_hop:
                agree = sum(1 for w in two_hop
                            if pred_labels[w].item() == pred_labels[v].item())
                context[v, 4] = agree / len(two_hop)
            else:
                context[v, 4] = 0.0

        # 5: average edge weight (cosine similarity to neighbours)
        x_norm = F.normalize(x, dim=1)
        for v in range(N):
            nbrs = list(G.neighbors(v))
            if nbrs:
                sims = (x_norm[nbrs] * x_norm[v].unsqueeze(0)).sum(dim=1)
                context[v, 5] = sims.mean().clamp(-1, 1)

        # 6: feature saliency (L2 norm, normalised)
        norms = x.norm(dim=1)
        max_norm = norms.max().clamp(min=1e-8)
        context[:, 6] = norms / max_norm

        return context   # [N, 7]


# ─────────────────────────────────────────────────────────────
# 2.  Reasoner Module  (c_v → e_v)
# ─────────────────────────────────────────────────────────────
class Reasoner(nn.Module):
    """
    Lightweight MLP: context_dim → explanation_dim.

    Also contains:
      - decoder: explanation_dim → embedding_dim  (faithfulness loss)
      - text_injector: explanation_dim → embedding_dim
            (projects e_v so it can be added to h_v in message passing)
    """

    def __init__(self,
                 context_dim: int = 7,
                 explanation_dim: int = 32,
                 embedding_dim: int = 64):
        super().__init__()
        self.explanation_dim = explanation_dim

        # c_v  →  e_v
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, explanation_dim),
            nn.Tanh(),
        )

        # e_v  →  ĥ_v   (reconstruction for faithfulness loss)
        self.decoder = nn.Sequential(
            nn.Linear(explanation_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # e_v  →  injection vector (same dim as node embedding)
        self.injector = nn.Linear(explanation_dim, embedding_dim)

    def forward(self, context: torch.Tensor):
        """
        context: [N, context_dim]
        Returns:
            expl_vec   : [N, explanation_dim]  — compact explanation vector
            reconstructed : [N, embedding_dim] — for faithfulness loss
            injection  : [N, embedding_dim]    — to add into GNN message-passing
        """
        expl_vec = self.mlp(context)               # [N, expl_dim]
        reconstructed = self.decoder(expl_vec)     # [N, emb_dim]
        injection = self.injector(expl_vec)        # [N, emb_dim]
        return expl_vec, reconstructed, injection


# ─────────────────────────────────────────────────────────────
# 3.  X-Node GNN Backbone  (with text-injection between layers)
# ─────────────────────────────────────────────────────────────
class XNodeGNN(nn.Module):
    """
    Self-explaining GNN based on X-Node (Sengupta & Rekik, 2025).

    Forward pass:
      Layer 0: h^(0) = x
      For each GNN layer k:
          h^(k) = GNNLayer_k(h^(k-1), edge_index)
          if k < last_layer:
              context_k = TopologicalContext(h^(k), pred_labels_k)
              e_k, recon_k, inject_k = Reasoner(context_k)
              h^(k) = h^(k) + inject_k    ← TEXT INJECTION
      out = pool(h^(K)) → classifier

    Loss:
      L_total = L_CE + λ * Σ_k MSE(recon_k, h^(k-1))
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 backbone: str = "gin",   # 'gcn' | 'gat' | 'gin'
                 dropout: float = 0.3,
                 explanation_dim: int = 32,
                 task: str = "graph",
                 faith_weight: float = 0.1):
        super().__init__()

        self.task = task
        self.dropout = dropout
        self.faith_weight = faith_weight
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.context_builder = TopologicalContextBuilder()

        # ── GNN layers ──────────────────────────────────────
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * num_layers
        for i in range(num_layers):
            in_d, out_d = dims[i], dims[i + 1]
            if backbone == "gcn":
                self.convs.append(GCNConv(in_d, out_d))
            elif backbone == "gat":
                heads = 4
                if i < num_layers - 1:
                    self.convs.append(GATConv(in_d, out_d // heads,
                                              heads=heads, concat=True,
                                              dropout=dropout))
                    out_d_actual = (out_d // heads) * heads
                else:
                    self.convs.append(GATConv(in_d, out_d,
                                              heads=1, concat=False,
                                              dropout=dropout))
                    out_d_actual = out_d
                self.bns.append(BatchNorm(out_d_actual))
                continue
            elif backbone == "gin":
                mlp = nn.Sequential(
                    nn.Linear(in_d, out_d), nn.ReLU(),
                    nn.Linear(out_d, out_d)
                )
                self.convs.append(GINConv(mlp, train_eps=True))
            else:
                raise ValueError(f"Unknown backbone: {backbone}")
            self.bns.append(BatchNorm(out_d))

        # ── Reasoner modules (one per intermediate layer) ───
        # We inject AFTER each layer except the last
        self.reasoners = nn.ModuleList([
            Reasoner(
                context_dim=TopologicalContextBuilder.CONTEXT_DIM,
                explanation_dim=explanation_dim,
                embedding_dim=hidden_channels
            )
            for _ in range(num_layers - 1)
        ])

        # ── Classifier ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor = None,
                return_explanations: bool = False):
        """
        Parameters
        ----------
        x, edge_index, batch : standard PyG inputs
        return_explanations  : if True, also return per-node explanation dicts

        Returns
        -------
        logits : [B, num_classes] or [N, num_classes]
        explanations (optional) : list of dicts, one per intermediate layer,
            each with keys: 'context', 'expl_vec', 'reconstructed', 'injection'
        faith_loss (always computed internally, returned as attribute)
        """
        device = x.device
        faith_loss = torch.tensor(0.0, device=device)
        layer_explanations = []

        h = x
        for layer_idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_prev = h
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Text injection after every layer except the last
            if layer_idx < self.num_layers - 1:
                # Build context from current embeddings
                # (we use a temporary Data object for the context builder)
                tmp_data = Data(x=h.detach().cpu(),
                                edge_index=edge_index.cpu())

                # Soft predicted labels (argmax of linear probe on h)
                with torch.no_grad():
                    soft_labels = h.argmax(dim=1)

                ctx = self.context_builder(tmp_data, soft_labels)
                ctx = ctx.to(device)

                reasoner = self.reasoners[layer_idx]
                expl_vec, reconstructed, injection = reasoner(ctx)
                
                target = h_prev

                # First layer: input features may differ from hidden dim
                if reconstructed.size(1) != target.size(1):
                    target = h

                faith_loss = faith_loss + F.mse_loss(reconstructed, target)

                # TEXT INJECTION: add explanation into current embedding
                h = h + injection

                if return_explanations:
                    layer_explanations.append({
                        "layer": layer_idx,
                        "context": ctx.detach().cpu(),
                        "expl_vec": expl_vec.detach().cpu(),
                        "reconstructed": reconstructed.detach().cpu(),
                        "injection": injection.detach().cpu(),
                    })

        # Store for loss computation in trainer
        self._faith_loss = faith_loss

        # Pool for graph-level tasks
        if self.task == "graph" and batch is not None:
            h = global_add_pool(h, batch)

        logits = self.classifier(h)

        if return_explanations:
            return logits, layer_explanations
        return logits

    def total_loss(self, logits, targets):
        """Combined task loss + faithfulness loss."""
        ce = F.cross_entropy(logits, targets)
        return ce + self.faith_weight * self._faith_loss

    def get_node_embeddings(self, x, edge_index):
        """Return final node-level embeddings (for PGExplainer compatibility)."""
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        return h


# ─────────────────────────────────────────────────────────────
# 4.  X-Node Explainer wrapper (for explanation extraction)
# ─────────────────────────────────────────────────────────────
class XNodeExplainer:
    """
    Post-training interface for extracting and visualising
    X-Node explanations.

    Since X-Node is self-explaining, "explaining" just means
    running the forward pass with return_explanations=True and
    decoding the context vectors into human-readable form.
    """

    # Interpretable feature names (matches TopologicalContextBuilder)
    FEATURE_NAMES = [
        "Degree (norm)",
        "Clustering Coeff",
        "Eigvec Centrality",
        "Degree Centrality",
        "2-Hop Label Agree",
        "Avg Edge Weight",
        "Feature Saliency",
    ]

    def __init__(self, model: XNodeGNN, device: str = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    @torch.no_grad()
    def explain(self, data: Data) -> dict:
        """
        Run a single forward pass and extract per-node explanations.

        Returns
        -------
        dict:
            pred_label      : int — graph-level or dominant node-level prediction
            node_contexts   : [N, 7] — interpretable context vectors (last layer)
            node_expl_vecs  : [N, expl_dim] — compact explanation vectors
            top_nodes       : list of node indices ranked by explanation magnitude
            edge_mask       : [E] — derived importance score per edge
                              (avg of endpoint explanation magnitudes)
            layer_data      : full list of per-layer explanation dicts
            text_explanations : list of str — one natural-language explanation per node
        """
        data = data.to(self.device)
        x, edge_index = data.x, data.edge_index
        N = x.size(0)

        if self.model.task == "graph":
            batch = torch.zeros(N, dtype=torch.long, device=self.device)
            logits, layer_data = self.model(x, edge_index, batch,
                                            return_explanations=True)
        else:
            logits, layer_data = self.model(x, edge_index,
                                            return_explanations=True)

        pred_label = logits.argmax(dim=1).squeeze().item()

        # Use the LAST intermediate layer's explanations
        if not layer_data:
            # Edge case: single-layer model
            return {"pred_label": pred_label, "layer_data": [],
                    "node_contexts": None, "node_expl_vecs": None,
                    "top_nodes": [], "edge_mask": torch.zeros(edge_index.size(1)),
                    "text_explanations": []}

        last = layer_data[-1]
        ctx = last["context"]          # [N, 7]
        ev = last["expl_vec"]          # [N, expl_dim]

        # Node importance = L2 norm of explanation vector
        node_importance = ev.norm(dim=1)                    # [N]
        top_nodes = node_importance.argsort(descending=True).tolist()

        # Edge importance = average of endpoint node importances
        src = edge_index[0].cpu()
        dst = edge_index[1].cpu()
        edge_mask = (node_importance[src] + node_importance[dst]) / 2.0

        # Generate text explanations (offline, no API key needed)
        text_expls = self._generate_text_explanations(ctx, node_importance, pred_label)

        return {
            "pred_label": pred_label,
            "node_contexts": ctx,
            "node_expl_vecs": ev,
            "node_importance": node_importance,
            "top_nodes": top_nodes,
            "edge_mask": edge_mask,
            "layer_data": layer_data,
            "text_explanations": text_expls,
        }

    def _generate_text_explanations(self,
                                    ctx: torch.Tensor,
                                    importance: torch.Tensor,
                                    pred_label: int,
                                    top_n: int = 5) -> list:
        """
        Template-based natural language explanations (offline mode).
        In the original paper this calls Grok/Gemini; here we use
        rule-based templates so no API key is required.
        """
        N = ctx.size(0)
        top_indices = importance.argsort(descending=True)[:top_n].tolist()
        explanations = []

        for v in range(N):
            c = ctx[v]
            rank = (importance > importance[v]).sum().item() + 1
            deg_str = f"{c[0].item():.2f}"
            clust_str = f"{c[1].item():.2f}"
            agree_str = f"{c[4].item() * 100:.0f}%"
            sal_str = f"{c[6].item():.2f}"
            imp_str = f"{importance[v].item():.3f}"

            if v in top_indices:
                role = "HIGH-importance"
            elif rank > N * 0.8:
                role = "low-importance"
            else:
                role = "moderate-importance"

            text = (
                f"Node {v} [{role}] → Class {pred_label} | "
                f"Degree={deg_str}, Clustering={clust_str}, "
                f"2-hop agreement={agree_str}, "
                f"Feature saliency={sal_str} | "
                f"Explanation strength={imp_str}"
            )
            explanations.append(text)

        return explanations

    # ──────────────────────────────────────────────────────────
    # Batch explain
    # ──────────────────────────────────────────────────────────
    def explain_batch(self, data_list: list) -> list:
        results = []
        for i, data in enumerate(data_list):
            print(f"  X-Node: explaining {i+1}/{len(data_list)}...", end="\r")
            results.append(self.explain(data))
        print(f"  X-Node: {len(data_list)} explanations done.          ")
        return results

    # ──────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────
    def visualize(self,
                  data: Data,
                  explanation: dict,
                  title: str = "X-Node",
                  save_path: str = None,
                  top_k: int = 8):
        """
        3-panel figure:
          (a) Graph coloured by node explanation importance
          (b) Context vector heatmap (top-k nodes × 7 features)
          (c) Explanation strength bar chart
        """
        import networkx as nx
        from torch_geometric.utils import to_networkx

        G = to_networkx(data, to_undirected=True)
        imp = explanation["node_importance"]   # [N]
        ctx = explanation["node_contexts"]     # [N, 7]
        top_nodes = explanation["top_nodes"][:top_k]
        top_nodes_set = set(top_nodes)

        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        fig = plt.figure(figsize=(18, 5))
        fig.suptitle(f"{title}  |  Predicted Class: {explanation['pred_label']}",
                     fontsize=14, fontweight="bold")

        # ─ Panel A: graph ────────────────────────────────────
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title("Node Explanation Importance", fontsize=11)

        imp_np = imp.numpy()
        norm_imp = (imp_np - imp_np.min()) / (imp_np.max() - imp_np.min() + 1e-8)
        node_colors = [plt.cm.RdYlGn(v) for v in norm_imp]
        node_sizes = [50 + 400 * v for v in norm_imp]

        # Edge colours by importance
        ei_cpu = data.edge_index.cpu()
        em = explanation["edge_mask"].numpy()
        if em.max() > 0:
            em_norm = em / em.max()
        else:
            em_norm = em
        edge_colors = [plt.cm.Reds(v) for v in em_norm]
        edge_widths = [0.3 + 3.0 * v for v in em_norm]

        nx.draw(G, pos, ax=ax1,
                node_color=node_colors, node_size=node_sizes,
                edge_color=edge_colors, width=edge_widths,
                with_labels=False, arrows=False)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                                   norm=plt.Normalize(imp_np.min(), imp_np.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax1, fraction=0.03, pad=0.04,
                     label="Explanation importance")

        # ─ Panel B: context heatmap ───────────────────────────
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title(f"Context Vectors (top-{top_k} nodes)", fontsize=11)

        if ctx is not None and len(top_nodes) > 0:
            ctx_sub = ctx[top_nodes].numpy()   # [top_k, 7]
            im = ax2.imshow(ctx_sub.T, aspect="auto", cmap="viridis",
                            vmin=0, vmax=1)
            ax2.set_yticks(range(7))
            ax2.set_yticklabels(self.FEATURE_NAMES, fontsize=8)
            ax2.set_xticks(range(len(top_nodes)))
            ax2.set_xticklabels([f"N{n}" for n in top_nodes],
                                rotation=45, ha="right", fontsize=8)
            ax2.set_xlabel("Node index")
            plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)

        # ─ Panel C: importance bar chart ─────────────────────
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title("Explanation Strength per Node", fontsize=11)

        ranked_imp = imp[explanation["top_nodes"][:20]].numpy()
        ranked_labels = [f"N{n}" for n in explanation["top_nodes"][:20]]
        colors = ["crimson" if n in top_nodes_set else "steelblue"
                  for n in explanation["top_nodes"][:20]]
        ax3.bar(range(len(ranked_imp)), ranked_imp, color=colors, edgecolor="k")
        ax3.set_xticks(range(len(ranked_imp)))
        ax3.set_xticklabels(ranked_labels, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("||e_v||₂  (explanation strength)")
        ax3.set_xlabel("Node (ranked)")
        ax3.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved X-Node visualization → {save_path}")
        plt.close()

    def plot_context_radar(self, explanation: dict,
                           node_idx: int = 0,
                           save_path: str = None):
        """Radar (spider) chart of context vector for a single node."""
        ctx = explanation["node_contexts"]
        if ctx is None:
            return

        values = ctx[node_idx].numpy().tolist()
        values += values[:1]                  # close the loop
        labels = self.FEATURE_NAMES + [self.FEATURE_NAMES[0]]
        N = len(self.FEATURE_NAMES)
        angles = [n / float(N) * 2 * math.pi for n in range(N)] + [0]

        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(polar=True))
        ax.plot(angles, values, "o-", linewidth=2, color="crimson")
        ax.fill(angles, values, alpha=0.25, color="crimson")
        ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]],
                          self.FEATURE_NAMES, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"X-Node Context — Node {node_idx} | "
                     f"Pred: Class {explanation['pred_label']}",
                     fontsize=12, pad=20)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_text_explanations(self, explanation: dict,
                                save_path: str,
                                top_n: int = 10):
        """Write top-N text explanations to a .txt file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imp = explanation["node_importance"]
        top_nodes = explanation["top_nodes"][:top_n]
        texts = explanation["text_explanations"]
        with open(save_path, "w") as f:
            f.write(f"X-Node Text Explanations\n")
            f.write(f"Predicted Class: {explanation['pred_label']}\n")
            f.write("=" * 70 + "\n\n")
            for rank, node in enumerate(top_nodes, 1):
                f.write(f"Rank {rank}: {texts[node]}\n\n")
        print(f"  Saved text explanations → {save_path}")
