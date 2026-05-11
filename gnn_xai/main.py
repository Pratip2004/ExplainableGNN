"""
main.py — GNN Explainability Framework (X-Node Edition)
=========================================================
Full pipeline implementing the methodology from:
  "A Comprehensive Survey of Existing XAI Methods for GNNs"
  Diptarshi Das & Pratip Modak, supervised by Dr. Ananda Shankar Chowdhury

PRIMARY method : X-Node (Sengupta & Rekik, GRAIL@MICCAI 2025)
                 Self-explaining GNN — explanation is part of the forward pass.
COMPARISON     : GNNExplainer, PGExplainer, PGM-Explainer, SubgraphX (post-hoc)

Pipeline
--------
  1. Load / generate dataset
  2. Build model  →  standard GNN  OR  X-Node (self-explaining)
  3. Train (X-Node uses CE + faithfulness loss)
  4. Evaluate predictive performance
  5. Generate explanations
       • X-Node     : built-in, zero extra cost
       • GNNExplainer, PGExplainer, PGM-Explainer, SubgraphX : post-hoc
  6. Compute XAI metrics (Fidelity+/-, Sparsity, AUC-ROC, Precision@K)
  7. Save plots, JSON results, Markdown report

Usage
-----
  # Fast smoke-test (≈3 min):
  python main.py --arch xnode --dataset ba2motifs --epochs 50 --explain_n 10 --skip_subgraphx

  # Full benchmark (≈20 min):
  python main.py --arch xnode --dataset ba2motifs --epochs 100 --explain_n 20

  # Standard GNN + post-hoc methods:
  python main.py --arch gin   --dataset ba2motifs --epochs 100 --explain_n 20

  # MRI demo:
  python main.py --arch xnode --dataset mri --epochs 80 --explain_n 15
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_geometric.loader import DataLoader

from data.dataset import (
    generate_ba2motifs, generate_ba_shapes,
    generate_mri_synthetic, load_tu_dataset,
    split_dataset, set_seed,
)
from models.gnn import build_model
from models.trainer import GNNTrainer

from explainers.xnode import XNodeGNN, XNodeExplainer
from explainers.gnnexplainer import GNNExplainer
from explainers.pgexplainer import PGExplainer
from explainers.pgm_explainer import PGMExplainer
from explainers.subgraphx import SubgraphX

from utils.metrics import (
    evaluate_explainer, print_comparison_table, plot_metrics_comparison,
)
from utils.report import save_results_json, generate_markdown_report


# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="GNN-XAI Framework (X-Node Edition)")
    p.add_argument("--dataset",  default="ba2motifs",
                   choices=["ba2motifs","ba_shapes","mri","mutag","proteins"])
    p.add_argument("--arch",     default="xnode",
                   choices=["xnode","gcn","gat","gin","sage"],
                   help="'xnode' = self-explaining; others = standard GNN + post-hoc")
    p.add_argument("--backbone", default="gin",
                   choices=["gcn","gat","gin"],
                   help="GNN backbone inside X-Node")
    p.add_argument("--hidden",   type=int, default=64)
    p.add_argument("--layers",   type=int, default=3)
    p.add_argument("--dropout",  type=float, default=0.3)
    p.add_argument("--expl_dim", type=int, default=32,
                   help="X-Node explanation vector dimension")
    p.add_argument("--faith_w",  type=float, default=0.1,
                   help="X-Node faithfulness loss weight λ")
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--explain_n", type=int, default=20,
                   help="# test graphs to explain per method")
    p.add_argument("--top_k",   type=int, default=8)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--skip_pgm",       action="store_true")
    p.add_argument("--skip_subgraphx", action="store_true")
    p.add_argument("--skip_posthoc",   action="store_true",
                   help="Skip ALL post-hoc methods (fastest; only X-Node)")
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
def load_data(args):
    name = args.dataset
    if name == "ba2motifs":
        dl = generate_ba2motifs(n_graphs=500, n_base=20, feat_dim=10)
        tr, va, te = split_dataset(dl, seed=args.seed)
        return tr, va, te, 10, 2, "graph", True
    elif name == "ba_shapes":
        data = generate_ba_shapes(n_base=300, n_motifs=80, feat_dim=10)
        n = data.num_nodes
        idx = list(range(n)); random.shuffle(idx)
        for mask_name, sl in [("train_mask",slice(None,int(.7*n))),
                               ("val_mask",  slice(int(.7*n),int(.85*n))),
                               ("test_mask", slice(int(.85*n),None))]:
            m = torch.zeros(n, dtype=torch.bool); m[idx[sl]] = True
            setattr(data, mask_name, m)
        return [data],[data],[data], 10, 4, "node", False
    elif name == "mri":
        dl = generate_mri_synthetic(n_graphs=300, n_regions=40, feat_dim=16)
        tr, va, te = split_dataset(dl, seed=args.seed)
        return tr, va, te, 16, 2, "graph", False
    elif name in ["mutag","proteins"]:
        dsn = {"mutag":"MUTAG","proteins":"PROTEINS"}[name]
        ds = load_tu_dataset(dsn, root=f"data/TU/{dsn}")
        dl = list(ds)
        if ds.num_node_features == 0:
            for d in dl: d.x = torch.ones((d.num_nodes,1), dtype=torch.float)
            nf = 1
        else:
            nf = ds.num_node_features
        tr, va, te = split_dataset(dl, seed=args.seed)
        return tr, va, te, nf, ds.num_classes, "graph", False
    raise ValueError(f"Unknown dataset: {name}")


# ─────────────────────────────────────────────────────────────
def build_xnode(args, num_feat, num_cls, task):
    return XNodeGNN(
        in_channels=num_feat,
        hidden_channels=args.hidden,
        out_channels=num_cls,
        num_layers=args.layers,
        backbone=args.backbone,
        dropout=args.dropout,
        explanation_dim=args.expl_dim,
        task=task,
        faith_weight=args.faith_w,
    )


# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'#'*62}")
    print(f"  GNN-XAI Framework  |  Primary: X-Node  |  Comparison: Post-hoc")
    print(f"  Dataset: {args.dataset}  |  Arch: {args.arch}  |  Device: {device}")
    print(f"{'#'*62}\n")

    for d in [args.results_dir,
              f"{args.results_dir}/checkpoints",
              f"{args.results_dir}/plots",
              f"{args.results_dir}/text_explanations"]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────
    print("━━━ Step 1: Loading dataset " + "─"*35)
    tr, va, te, num_feat, num_cls, task, has_gt = load_data(args)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(va, batch_size=args.batch_size)
    test_loader  = DataLoader(te, batch_size=args.batch_size)

    # ── 2. Build model ────────────────────────────────────────
    print("\n━━━ Step 2: Building model " + "─"*36)
    if args.arch == "xnode":
        model = build_xnode(args, num_feat, num_cls, task)
        print(f"  X-Node backbone={args.backbone} | expl_dim={args.expl_dim} | "
              f"faith_weight={args.faith_w}")
    else:
        model = build_model(args.arch, num_feat, args.hidden, num_cls,
                            args.layers, args.dropout, task)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # ── 3. Train ──────────────────────────────────────────────
    print("\n━━━ Step 3: Training " + "─"*41)
    trainer = GNNTrainer(model, {
        "lr": args.lr, "epochs": args.epochs, "patience": 25,
        "task": task, "save_dir": f"{args.results_dir}/checkpoints",
    }, device=device)

    history = trainer.train(train_loader, val_loader,
                            model_name=f"{args.arch}_{args.dataset}")
    trainer.plot_history(history,
                         save_path=f"{args.results_dir}/plots/training_curve.png")

    # ── 4. Evaluate GNN ───────────────────────────────────────
    print("\n━━━ Step 4: GNN evaluation " + "─"*35)
    class_names = [f"Class {i}" for i in range(num_cls)]
    gnn_full = trainer.evaluate(test_loader, class_names=class_names)
    gnn_metrics = {"accuracy": gnn_full["accuracy"], "f1_weighted": gnn_full["f1"]}
    trainer.plot_confusion_matrix(gnn_full["confusion_matrix"],
                                  class_names=class_names,
                                  save_path=f"{args.results_dir}/plots/confusion_matrix.png")

    # ── 5. Explanations ───────────────────────────────────────
    n_exp = min(args.explain_n, len(te))
    explain_data = te[:n_exp]
    all_exp_metrics = {}
    plot_paths = {
        "training_curve":   f"{args.results_dir}/plots/training_curve.png",
        "confusion_matrix": f"{args.results_dir}/plots/confusion_matrix.png",
    }

    # ─────────────────────────────────────────────────────────
    # 5a. X-NODE  (self-explaining — primary method)
    # ─────────────────────────────────────────────────────────
    print(f"\n━━━ Step 5a: X-Node (Self-Explaining) " + "─"*24)

    if args.arch == "xnode":
        xnode_model = model
    else:
        # If user chose standard arch, also train an X-Node for comparison
        print("  [X-Node] Training auxiliary X-Node model for comparison...")
        xnode_model = build_xnode(args, num_feat, num_cls, task)
        xn_trainer = GNNTrainer(xnode_model, {
            "lr": args.lr, "epochs": min(args.epochs, 80),
            "patience": 20, "task": task,
            "save_dir": f"{args.results_dir}/checkpoints",
        }, device=device)
        xn_trainer.train(train_loader, val_loader, model_name=f"xnode_{args.dataset}")

    xnode_explainer = XNodeExplainer(xnode_model, device=device)
    print(f"  Generating X-Node explanations for {n_exp} graphs...")
    xnode_exps = xnode_explainer.explain_batch(explain_data)

    # Visualise first example
    xn_viz = f"{args.results_dir}/plots/xnode_viz.png"
    xnode_explainer.visualize(explain_data[0], xnode_exps[0],
                               title="X-Node", save_path=xn_viz, top_k=args.top_k)
    plot_paths["xnode_viz"] = xn_viz

    # Radar chart for single node
    xn_radar = f"{args.results_dir}/plots/xnode_radar.png"
    xnode_explainer.plot_context_radar(xnode_exps[0], node_idx=xnode_exps[0]["top_nodes"][0],
                                        save_path=xn_radar)
    plot_paths["xnode_radar"] = xn_radar

    # Text explanations
    xn_txt = f"{args.results_dir}/text_explanations/xnode_graph0.txt"
    xnode_explainer.save_text_explanations(xnode_exps[0], save_path=xn_txt, top_n=10)

    # Metrics (X-Node edge_mask already computed from node importance)
    xn_metrics = evaluate_explainer(
        xnode_model, explain_data, xnode_exps,
        task=task, top_k=args.top_k, device=device, has_gt=has_gt
    )
    all_exp_metrics["X-Node"] = xn_metrics
    print(f"  X-Node Fidelity+: {xn_metrics['fidelity_plus']['mean']:.3f} | "
          f"Fidelity-: {xn_metrics['fidelity_minus']['mean']:.3f}")

    if not args.skip_posthoc:
        # ─────────────────────────────────────────────────────
        # 5b. GNNExplainer  (post-hoc baseline)
        # ─────────────────────────────────────────────────────
        print(f"\n━━━ Step 5b: GNNExplainer (post-hoc) " + "─"*25)
        posthoc_model = model   # use primary model for all post-hoc
        gnn_exp = GNNExplainer(posthoc_model, epochs=150, lr=0.01,
                               task=task, device=device)
        gnnexp_list = gnn_exp.explain_batch(explain_data, top_k_edges=args.top_k)

        viz = f"{args.results_dir}/plots/gnnexplainer_viz.png"
        gnn_exp.visualize(explain_data[0], gnnexp_list[0],
                          title="GNNExplainer", save_path=viz)
        plot_paths["gnnexplainer_viz"] = viz

        gnn_exp.plot_feature_importance(gnnexp_list[0],
            save_path=f"{args.results_dir}/plots/gnnexplainer_features.png")
        gnn_exp.plot_loss_curve(gnnexp_list[0],
            save_path=f"{args.results_dir}/plots/gnnexplainer_loss.png")

        gnnexp_metrics = evaluate_explainer(
            posthoc_model, explain_data, gnnexp_list,
            task=task, top_k=args.top_k, device=device, has_gt=has_gt)
        all_exp_metrics["GNNExplainer"] = gnnexp_metrics

        # ─────────────────────────────────────────────────────
        # 5c. PGExplainer  (post-hoc, parameterized)
        # ─────────────────────────────────────────────────────
        print(f"\n━━━ Step 5c: PGExplainer (post-hoc) " + "─"*26)
        pg_exp = PGExplainer(posthoc_model, emb_dim=args.hidden,
                             hidden_dim=64, epochs=20, lr=3e-3,
                             task=task, device=device)
        pg_ckpt = f"{args.results_dir}/checkpoints/pgexplainer.pt"
        pg_exp.train_explainer(tr[:min(200, len(tr))], save_path=pg_ckpt)
        pg_list = [pg_exp.explain(d, top_k_edges=args.top_k) for d in explain_data]

        pg_viz = f"{args.results_dir}/plots/pgexplainer_viz.png"
        pg_exp.visualize(explain_data[0], pg_list[0],
                         title="PGExplainer", save_path=pg_viz)
        plot_paths["pgexplainer_viz"] = pg_viz

        pg_metrics = evaluate_explainer(
            posthoc_model, explain_data, pg_list,
            task=task, top_k=args.top_k, device=device, has_gt=has_gt)
        all_exp_metrics["PGExplainer"] = pg_metrics

        # ─────────────────────────────────────────────────────
        # 5d. PGM-Explainer  (post-hoc, probabilistic)
        # ─────────────────────────────────────────────────────
        if not args.skip_pgm:
            print(f"\n━━━ Step 5d: PGM-Explainer (post-hoc) " + "─"*24)
            pgm_exp = PGMExplainer(posthoc_model, n_samples=60,
                                   perturbation_std=0.1, num_top_nodes=6,
                                   task=task, device=device)
            n_pgm = min(10, n_exp)
            pgm_list = [pgm_exp.explain(d) for d in explain_data[:n_pgm]]

            pgm_viz = f"{args.results_dir}/plots/pgm_explainer_viz.png"
            pgm_exp.visualize(explain_data[0], pgm_list[0],
                              title="PGM-Explainer", save_path=pgm_viz)
            plot_paths["pgm_explainer_viz"] = pgm_viz

            pgm_exp.plot_node_importance(pgm_list[0],
                save_path=f"{args.results_dir}/plots/pgm_node_importance.png")

            # Convert node importance to edge mask proxy
            pgm_edge = []
            for di, ei in zip(explain_data[:n_pgm], pgm_list):
                ts = set(ei["top_nodes"])
                em = torch.zeros(di.edge_index.size(1))
                for j in range(di.edge_index.size(1)):
                    u, v = di.edge_index[0,j].item(), di.edge_index[1,j].item()
                    if u in ts and v in ts: em[j] = 1.0
                pgm_edge.append({"edge_mask": em})
            pgm_metrics = evaluate_explainer(
                posthoc_model, explain_data[:n_pgm], pgm_edge,
                task=task, top_k=args.top_k, device=device, has_gt=has_gt)
            all_exp_metrics["PGM-Explainer"] = pgm_metrics
        else:
            print("  [PGM-Explainer] SKIPPED")

        # ─────────────────────────────────────────────────────
        # 5e. SubgraphX  (post-hoc, MCTS + Shapley)
        # ─────────────────────────────────────────────────────
        if not args.skip_subgraphx:
            print(f"\n━━━ Step 5e: SubgraphX (post-hoc) " + "─"*28)
            sgx = SubgraphX(posthoc_model, min_subgraph_size=3,
                            max_subgraph_size=12, n_mcts_steps=30,
                            n_shapley_samples=15, task=task, device=device)
            n_sgx = min(8, n_exp)
            sgx_list = []
            for i, d in enumerate(explain_data[:n_sgx]):
                print(f"  SubgraphX: {i+1}/{n_sgx}")
                sgx_list.append(sgx.explain(d, top_k_nodes=args.top_k))

            sgx_viz = f"{args.results_dir}/plots/subgraphx_viz.png"
            sgx.visualize(explain_data[0], sgx_list[0],
                          title="SubgraphX", save_path=sgx_viz)
            plot_paths["subgraphx_viz"] = sgx_viz

            # Convert to edge mask
            sgx_edge = []
            for di, ei in zip(explain_data[:n_sgx], sgx_list):
                cs = set(ei["best_coalition"]); scores = ei.get("all_scores", {})
                em = torch.zeros(di.edge_index.size(1))
                for j in range(di.edge_index.size(1)):
                    u, v = di.edge_index[0,j].item(), di.edge_index[1,j].item()
                    if u in cs and v in cs:
                        em[j] = max(0.0, (scores.get(u,0) + scores.get(v,0))/2)
                sgx_edge.append({"edge_mask": em})
            sgx_metrics = evaluate_explainer(
                posthoc_model, explain_data[:n_sgx], sgx_edge,
                task=task, top_k=args.top_k, device=device, has_gt=has_gt)
            all_exp_metrics["SubgraphX"] = sgx_metrics
        else:
            print("  [SubgraphX] SKIPPED")

    # ── 6. Metrics Summary ────────────────────────────────────
    print(f"\n━━━ Step 6: Metrics Comparison " + "─"*32)
    print_comparison_table(all_exp_metrics)
    cmp_path = f"{args.results_dir}/plots/metrics_comparison.png"
    plot_metrics_comparison(all_exp_metrics, save_path=cmp_path)
    plot_paths["metrics_comparison"] = cmp_path

    # ── 7. Save & Report ──────────────────────────────────────
    print(f"\n━━━ Step 7: Saving results " + "─"*36)
    save_results_json({
        "dataset": args.dataset, "arch": args.arch,
        "gnn_metrics": gnn_metrics,
        "explainer_metrics": all_exp_metrics,
        "config": vars(args),
    }, f"{args.results_dir}/results.json")

    generate_markdown_report(
        dataset_name=args.dataset, model_arch=args.arch,
        gnn_metrics=gnn_metrics,
        explainer_metrics=all_exp_metrics,
        config={"hidden_channels": args.hidden, "num_layers": args.layers,
                "dropout": args.dropout, "epochs": args.epochs,
                "lr": args.lr, "task": task,
                "backbone": args.backbone, "expl_dim": args.expl_dim,
                "faith_weight": args.faith_w},
        plot_paths=plot_paths,
        save_path=f"{args.results_dir}/report.md",
    )

    print(f"\n{'#'*62}")
    print(f"  DONE  |  Accuracy: {gnn_metrics['accuracy']:.4f}")
    print(f"  X-Node Fidelity+: {all_exp_metrics['X-Node']['fidelity_plus']['mean']:.3f}")
    print(f"  Results → {args.results_dir}/")
    print(f"{'#'*62}\n")


if __name__ == "__main__":
    main()
