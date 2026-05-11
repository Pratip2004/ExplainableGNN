# GNN Explainability Framework — X-Node Edition

**Primary Method: X-Node: Self-Explanation is All We Need**
*(Sengupta & Rekik, GRAIL@MICCAI 2025 — arXiv:2508.10461)*

**Comparison: GNNExplainer, PGExplainer, PGM-Explainer, SubgraphX**

*Project Report — B.E. in Electronics & Telecommunication Engineering*
*Diptarshi Das | Pratip Modak | Supervisor: Dr. Ananda Shankar Chowdhury*

---

## What is X-Node?

X-Node is a **self-explaining GNN** — unlike all post-hoc methods, the explanation
is generated *as part of* each node's forward pass, not after the fact.

```
For every node v during the GNN forward pass:
  1.  c_v  = TopologicalContext(v)     ← 7 interpretable topological cues
  2.  e_v  = Reasoner(c_v)            ← compact explanation vector (MLP)
  3.  h_v  = h_v + Inject(e_v)        ← TEXT INJECTION into message-passing
  4.  ĥ_v  = Decode(e_v)              ← faithfulness reconstruction

Training loss: L = L_CE  +  λ × MSE(ĥ_v, h_v)
                            ────────────────────
                             faithfulness loss
```

**Seven topological context features per node:**
| # | Feature | What it captures |
|---|---------|-----------------|
| 0 | Degree (norm) | How connected is this node? |
| 1 | Clustering Coefficient | How tightly clustered are its neighbours? |
| 2 | Eigenvector Centrality | Global importance in the graph |
| 3 | Degree Centrality | Local importance (normalised degree) |
| 4 | 2-hop Label Agreement | Do neighbours agree on the prediction? |
| 5 | Avg Edge Weight | How similar are the node's features to its neighbours? |
| 6 | Feature Saliency | How strong are the node's own features? |

---

## Project Structure

```
gnn_xai/
├── main.py                         ← Full pipeline runner (start here)
├── setup.sh                        ← One-shot environment setup
├── requirements.txt
│
├── data/
│   └── dataset.py                  ← All dataset loaders & graph generators
│
├── models/
│   ├── gnn.py                      ← GCN, GAT, GIN, GraphSAGE architectures
│   └── trainer.py                  ← Training loop (X-Node + standard GNNs)
│
├── explainers/
│   ├── xnode.py                    ★ X-Node self-explaining GNN (PRIMARY)
│   │                                  - TopologicalContextBuilder (7 features)
│   │                                  - Reasoner (MLP: context → e_v)
│   │                                  - XNodeGNN (backbone + text injection)
│   │                                  - XNodeExplainer (extract + visualize)
│   ├── gnnexplainer.py             ← GNNExplainer (Ying et al., NeurIPS 2019)
│   ├── pgexplainer.py              ← PGExplainer  (Luo et al., NeurIPS 2020)
│   ├── pgm_explainer.py            ← PGM-Explainer (Vu & Thai, NeurIPS 2020)
│   └── subgraphx.py               ← SubgraphX    (Yuan et al., ICML 2021)
│
├── utils/
│   ├── metrics.py                  ← Fidelity+/-, Sparsity, AUC-ROC, Prec@K
│   └── report.py                   ← Auto-generate Markdown report
│
└── results/                        ← Created at runtime
    ├── plots/                      ← All generated figures
    ├── checkpoints/                ← Saved weights
    ├── text_explanations/          ← X-Node natural language outputs
    ├── results.json                ← All numerical results
    └── report.md                   ← Auto-generated report
```

---

## Setup

```bash
bash setup.sh
```

Or manually:
```bash
pip install torch torch-geometric numpy pandas matplotlib seaborn \
            scikit-learn networkx scipy tqdm
```

---

## Datasets

| Dataset | Task | Nodes/Graph | Classes | GT Edge Masks |
|---------|------|-------------|---------|---------------|
| **BA-2Motifs** *(recommended)* | Graph class. | ~25 | 2 | ✅ |
| BA-Shapes | Node class. | 300+ | 4 | ❌ |
| MRI-Synthetic *(brain tumour)* | Graph class. | 40 | 2 | ❌ |
| MUTAG *(molecular)* | Graph class. | ~17 | 2 | ❌ |
| PROTEINS | Graph class. | ~39 | 2 | ❌ |

---

## Running Experiments

### Fastest smoke-test (~2–3 min)
```bash
python main.py --arch xnode --dataset ba2motifs \
               --epochs 50 --explain_n 10 \
               --skip_posthoc
```

### X-Node only, full run (~8–10 min)
```bash
python main.py --arch xnode --dataset ba2motifs \
               --epochs 100 --explain_n 20 \
               --skip_subgraphx
```

### X-Node + all post-hoc comparison (~25 min)
```bash
python main.py --arch xnode --dataset ba2motifs \
               --epochs 100 --explain_n 20
```

### Standard GIN + post-hoc (original method, no X-Node primary)
```bash
python main.py --arch gin --dataset ba2motifs \
               --epochs 100 --explain_n 20
```

### MRI / medical imaging demo
```bash
python main.py --arch xnode --backbone gcn \
               --dataset mri --epochs 80 \
               --explain_n 15 --skip_subgraphx
```

### Real molecular dataset (MUTAG)
```bash
python main.py --arch xnode --dataset mutag \
               --epochs 150 --explain_n 15
```

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `xnode` | `xnode` = self-explaining; `gcn/gat/gin/sage` = standard |
| `--backbone` | `gin` | GNN backbone inside X-Node: `gcn`, `gat`, `gin` |
| `--dataset` | `ba2motifs` | ba2motifs, ba_shapes, mri, mutag, proteins |
| `--hidden` | `64` | Hidden channels |
| `--layers` | `3` | Number of GNN layers |
| `--expl_dim` | `32` | X-Node explanation vector dimension |
| `--faith_w` | `0.1` | X-Node faithfulness loss weight λ |
| `--dropout` | `0.3` | Dropout rate |
| `--epochs` | `100` | Training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--explain_n` | `20` | # graphs to explain |
| `--top_k` | `8` | Top-K nodes/edges in explanation |
| `--skip_posthoc` | False | Skip ALL post-hoc methods |
| `--skip_pgm` | False | Skip PGM-Explainer only |
| `--skip_subgraphx` | False | Skip SubgraphX only |

---

## Output Files

After running, `results/` contains:

| File | Description |
|------|-------------|
| `plots/training_curve.png` | Train/val loss & accuracy |
| `plots/confusion_matrix.png` | Prediction confusion matrix |
| `plots/xnode_viz.png` | **X-Node 3-panel: graph+heatmap+bar** |
| `plots/xnode_radar.png` | **X-Node radar chart for top node** |
| `text_explanations/xnode_graph0.txt` | **Natural language explanations** |
| `plots/gnnexplainer_viz.png` | GNNExplainer subgraph |
| `plots/pgexplainer_viz.png` | PGExplainer subgraph |
| `plots/pgm_explainer_viz.png` | PGM-Explainer BN structure |
| `plots/subgraphx_viz.png` | SubgraphX MCTS coalition |
| `plots/metrics_comparison.png` | Grouped bar chart comparison |
| `results.json` | All numerical results |
| `report.md` | Auto-generated experiment report |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Fidelity+** ↑ | Prediction maintained when keeping only explanation edges |
| **Fidelity-** ↑ | Prediction changes when removing explanation edges |
| **Sparsity** ↑ | Compactness (fraction of edges excluded) |
| **AUC-ROC** ↑ | Agreement with ground-truth motif masks (BA-2Motifs only) |
| **Precision@K** ↑ | Fraction of top-K edges in ground truth |

---

## References

1. **Sengupta & Rekik (2025). X-Node: Self-Explanation is All We Need.** *GRAIL@MICCAI*. arXiv:2508.10461
2. Ying et al. (2019). GNNExplainer. *NeurIPS*.
3. Luo et al. (2020). PGExplainer. *NeurIPS*.
4. Vu & Thai (2020). PGM-Explainer. *NeurIPS*.
5. Yuan et al. (2021). SubgraphX. *ICML*.
6. Xu et al. (2019). GIN. *ICLR*.
