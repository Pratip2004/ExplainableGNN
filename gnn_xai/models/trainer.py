"""
models/trainer.py
-----------------
Training loop, evaluation, and checkpointing.
Supports standard GNNs AND X-Node (CE + faithfulness loss).
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class GNNTrainer:
    def __init__(self, model: nn.Module, config: dict, device: str = None):
        self.config = {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 100,
                       "patience": 25, "save_dir": "results/checkpoints",
                       "task": "graph", **config}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.is_xnode = hasattr(model, "total_loss")   # X-Node detection
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=self.config["lr"],
                                    weight_decay=self.config["weight_decay"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10)
        os.makedirs(self.config["save_dir"], exist_ok=True)

    def _loss(self, out, targets):
        return self.model.total_loss(out, targets) if self.is_xnode \
               else self.criterion(out, targets)

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            if self.config["task"] == "graph":
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self._loss(out, batch.y)
                pred = out.argmax(1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
            else:
                out = self.model(batch.x, batch.edge_index)
                mask = getattr(batch, "train_mask", None)
                if mask is not None:
                    loss = self._loss(out[mask], batch.y[mask])
                    pred = out[mask].argmax(1)
                    correct += (pred == batch.y[mask]).sum().item()
                    total += mask.sum().item()
                else:
                    loss = self._loss(out, batch.y)
                    pred = out.argmax(1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1), correct / max(total, 1)

    @torch.no_grad()
    def eval_epoch(self, loader, mask_attr=None):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            batch = batch.to(self.device)
            if self.config["task"] == "graph":
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self._loss(out, batch.y)
                pred = out.argmax(1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
            else:
                out = self.model(batch.x, batch.edge_index)
                mask = getattr(batch, mask_attr, None) if mask_attr else None
                if mask is not None:
                    loss = self._loss(out[mask], batch.y[mask])
                    pred = out[mask].argmax(1)
                    correct += (pred == batch.y[mask]).sum().item()
                    total += mask.sum().item()
                else:
                    loss = self._loss(out, batch.y)
                    pred = out.argmax(1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            total_loss += loss.item()
        return total_loss / max(len(loader), 1), correct / max(total, 1)

    def train(self, train_loader, val_loader, model_name="gnn"):
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0
        ckpt = os.path.join(self.config["save_dir"], f"{model_name}_best.pt")
        label = "X-Node" if self.is_xnode else model_name.upper()
        print(f"\n{'='*60}\n  Training {label} | device={self.device}")
        if self.is_xnode:
            print(f"  L = CrossEntropy + {self.model.faith_weight:.2f}×Faithfulness")
        print(f"  epochs={self.config['epochs']}, lr={self.config['lr']}\n{'='*60}")
        t0 = time.time()
        for epoch in range(1, self.config["epochs"] + 1):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            va_loss, va_acc = self.eval_epoch(val_loader, mask_attr="val_mask")
            self.scheduler.step(va_loss)
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  [{epoch:4d}/{self.config['epochs']}] "
                      f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
                      f"Val {va_loss:.4f}/{va_acc:.4f} | {time.time()-t0:.0f}s")
            if va_loss < best_val_loss:
                best_val_loss, best_epoch, patience_counter = va_loss, epoch, 0
                torch.save(self.model.state_dict(), ckpt)
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    print(f"\n  Early stop @ {epoch} (best={best_epoch})")
                    break
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        print(f"\n  Best val loss: {best_val_loss:.4f} @ epoch {best_epoch}")
        return history

    @torch.no_grad()
    def evaluate(self, loader, class_names=None):
        self.model.eval()
        all_preds, all_labels = [], []
        for batch in loader:
            batch = batch.to(self.device)
            if self.config["task"] == "graph":
                out = self.model(batch.x, batch.edge_index, batch.batch)
                preds = out.argmax(1).cpu().numpy()
                labels = batch.y.cpu().numpy()
            else:
                out = self.model(batch.x, batch.edge_index)
                mask = getattr(batch, "test_mask", None)
                if mask is not None:
                    preds = out[mask].argmax(1).cpu().numpy()
                    labels = batch.y[mask].cpu().numpy()
                else:
                    preds = out.argmax(1).cpu().numpy()
                    labels = batch.y.cpu().numpy()
            all_preds.extend(preds); all_labels.extend(labels)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        report = classification_report(all_labels, all_preds,
                                       target_names=class_names, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\n{'─'*50}\n  Accuracy: {acc:.4f} | F1: {f1:.4f}\n{'─'*50}")
        print(report)
        return {"accuracy": acc, "f1": f1, "report": report,
                "confusion_matrix": cm, "preds": all_preds, "labels": all_labels}

    @staticmethod
    def plot_history(history, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (key_tr, key_va, ylabel, title) in zip(axes, [
            ("train_loss","val_loss","Loss","Loss Curve"),
            ("train_acc","val_acc","Accuracy","Accuracy Curve"),
        ]):
            ax.plot(history[key_tr], label="Train", lw=2)
            ax.plot(history[key_va], label="Val", lw=2)
            ax.set_title(title); ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm, class_names=None, save_path=None):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names or range(cm.shape[0]),
                    yticklabels=class_names or range(cm.shape[0]))
        plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
