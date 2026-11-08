"""Train the document classifier on data/images (see data_prep.py), then test it.

    python -m docsort.train                       # MobileNetV3-Large, 8 epochs, ~1 h on a laptop CPU
    python -m docsort.train --epochs 1 --limit 50 # quick smoke run

Writes models/docsort-classifier.pt, eval/classifier.md, eval/classifier.json and docs/confusion-matrix.png.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import date

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from docsort.classes import CLASSES, LABELS
from docsort.config import ROOT, get_settings
from docsort.model import build_model, eval_transform, head_parameters, train_transform


def loaders(size: int, batch: int, limit: int | None, workers: int):
    root = get_settings().data_dir / "images"
    sets = {
        "train": ImageFolder(root / "train", train_transform(size)),
        "val": ImageFolder(root / "val", eval_transform(size)),
        "test": ImageFolder(root / "test", eval_transform(size)),
    }
    for ds in sets.values():
        assert ds.classes == CLASSES, f"folders {ds.classes} != {CLASSES}"
    if limit:  # small random subset for smoke runs
        sets = {k: Subset(v, random.sample(range(len(v)), min(limit, len(v)))) for k, v in sets.items()}
    return {k: DataLoader(v, batch_size=batch, shuffle=(k == "train"), num_workers=workers,
                          persistent_workers=workers > 0) for k, v in sets.items()}


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x).argmax(1).numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def run_epoch(model, loader, optimizer, scheduler, loss_fn) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v3_large", choices=["efficientnet_b0", "mobilenet_v3_large"])
    ap.add_argument("--epochs", type=int, default=8, help="total epochs (the first 2 train only the new head)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(7); random.seed(7); np.random.seed(7)
    s = get_settings()
    data = loaders(s.image_size, args.batch, args.limit, args.workers)
    model = build_model(len(CLASSES), args.arch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    history, best = [], (-1.0, None)
    head_epochs = min(2, args.epochs)

    # Phase 1: freeze the pretrained backbone, train only the new head (fast, a strong baseline)
    for p in model.parameters():
        p.requires_grad = False
    for p in head_parameters(model):
        p.requires_grad = True
    opt = torch.optim.AdamW(head_parameters(model), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        if epoch == head_epochs + 1:
            # Phase 2: unfreeze everything and fine-tune with a small, decaying learning rate
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            steps = (args.epochs - head_epochs) * len(data["train"])
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.15)
        started = time.time()
        loss = run_epoch(model, data["train"], opt, sched if epoch > head_epochs else None, loss_fn)
        y, p = predict(model, data["val"])
        acc = float((y == p).mean())
        history.append({"epoch": epoch, "phase": "head" if epoch <= head_epochs else "fine-tune",
                        "train_loss": round(loss, 4), "val_acc": round(acc, 4), "minutes": round((time.time() - started) / 60, 1)})
        print(history[-1], flush=True)
        if acc > best[0]:
            best = (acc, {k: v.detach().clone().half() for k, v in model.state_dict().items()})  # fp16 = half the file size

    model.load_state_dict({k: v.float() for k, v in best[1].items()})
    y, p = predict(model, data["test"])
    report = classification_report(y, p, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=range(len(CLASSES)))
    metrics = {"date": date.today().isoformat(), "arch": args.arch, "image_size": s.image_size,
               "train_images": len(data["train"].dataset), "test_images": int(len(y)),
               "test_accuracy": round(float((y == p).mean()), 4), "test_macro_f1": round(float(f1_score(y, p, average="macro")), 4),
               "head_only_val_acc": history[head_epochs - 1]["val_acc"], "best_val_acc": round(best[0], 4),
               "history": history, "per_class": {c: {k: round(report[c][k], 3) for k in ("precision", "recall", "f1-score")} for c in CLASSES},
               "confusion_matrix": cm.tolist()}

    s.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best[1], "classes": CLASSES, "arch": args.arch, "image_size": s.image_size,
                "metrics": {k: metrics[k] for k in ("test_accuracy", "test_macro_f1", "date")}}, s.model_path)
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "classifier.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (ROOT / "eval" / "classifier.md").write_text(markdown(metrics), encoding="utf-8")
    plot_confusion(cm)
    print(f"test accuracy {metrics['test_accuracy']:.1%} · macro F1 {metrics['test_macro_f1']:.3f} → {s.model_path}")


def markdown(m: dict) -> str:
    lines = [f"# Classifier results\n",
             f"{m['date']} · `{m['arch']}` (ImageNet-pretrained) · {m['image_size']}×{m['image_size']} px · "
             f"{m['train_images']} training images · {m['test_images']} test images (never seen in training)\n",
             f"**Test accuracy {m['test_accuracy']:.1%} · macro F1 {m['test_macro_f1']:.3f}**\n",
             f"Training only the new head on frozen ImageNet features reached {m['head_only_val_acc']:.1%} on validation; "
             f"fine-tuning the whole network raised it to {m['best_val_acc']:.1%}.\n",
             "| Type | Precision | Recall | F1 |", "|---|---|---|---|"]
    for c, r in m["per_class"].items():
        lines.append(f"| {LABELS[c]} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1-score']:.2f} |")
    lines += ["\n| Epoch | Phase | Train loss | Val accuracy | Minutes |", "|---|---|---|---|---|"]
    lines += [f"| {h['epoch']} | {h['phase']} | {h['train_loss']} | {h['val_acc']:.1%} | {h['minutes']} |" for h in m["history"]]
    lines.append("\n![Confusion matrix](../docs/confusion-matrix.png)")
    return "\n".join(lines) + "\n"


def plot_confusion(cm: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [LABELS[c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=130)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="white" if norm[i, j] > 0.6 else "#1f2937")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True type"); ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "confusion-matrix.png")


if __name__ == "__main__":
    main()
