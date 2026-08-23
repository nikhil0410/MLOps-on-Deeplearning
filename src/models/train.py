"""Train the baseline CNN and track everything with MLflow (M1).

Responsibilities:
* build train/val/test dataloaders from the processed data
* train the model, logging params + per-epoch metrics to MLflow
* evaluate on the held-out test set
* log artifacts: confusion matrix + loss/accuracy curves
* save a serialized checkpoint (``models/model.pt``) and ``metrics.json``
  (used by DVC to track the pipeline output)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless backend (works in CI / containers)
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from src.config import load_params
from src.data.preprocess import build_dataloaders
from src.models.model import build_model


def _evaluate(model, loader, criterion, device) -> tuple[float, float, List[int], List[int]]:
    """Return (avg_loss, accuracy, y_true, y_pred) for a dataloader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, y_true, y_pred


def _plot_curves(history: Dict[str, List[float]], out_path: Path) -> None:
    """Save loss + accuracy curves as a single figure."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["val_loss"], "o-", label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], "o-", label="train")
    ax2.plot(epochs, history["val_acc"], "o-", label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_confusion(y_true, y_pred, class_names, out_path: Path) -> None:
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train() -> Dict[str, float]:
    params = load_params()
    d, t, m = params["data"], params["train"], params["mlflow"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(t["model_dir"]); model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(t["plots_dir"]); plots_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        processed_dir=d["processed_dir"],
        image_size=d["image_size"],
        batch_size=t["batch_size"],
        num_workers=t["num_workers"],
    )

    model = build_model(num_classes=t["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])

    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment_name"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    with mlflow.start_run(run_name=m["run_name"]):
        # --- log hyper-parameters ---
        mlflow.log_params(
            {
                "epochs": t["epochs"],
                "batch_size": t["batch_size"],
                "learning_rate": t["learning_rate"],
                "num_classes": t["num_classes"],
                "image_size": d["image_size"],
                "optimizer": "Adam",
                "architecture": "SimpleCNN",
            }
        )

        # --- training loop ---
        for epoch in range(1, t["epochs"] + 1):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            val_loss, val_acc, _, _ = _evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics(
                {"train_loss": train_loss, "train_acc": train_acc,
                 "val_loss": val_loss, "val_acc": val_acc},
                step=epoch,
            )
            print(f"Epoch {epoch}/{t['epochs']} | "
                  f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        # --- final test evaluation ---
        test_loss, test_acc, y_true, y_pred = _evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        print(f"Test | loss={test_loss:.4f} acc={test_acc:.4f}")

        # --- artifacts: curves + confusion matrix ---
        curves_path = plots_dir / "training_curves.png"
        cm_path = plots_dir / "confusion_matrix.png"
        _plot_curves(history, curves_path)
        _plot_confusion(y_true, y_pred, class_names, cm_path)
        mlflow.log_artifact(str(curves_path), artifact_path="plots")
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # --- save serialized checkpoint (.pt) ---
        model_path = model_dir / t["model_name"]
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": t["num_classes"],
                "class_names": class_names,
                "image_size": d["image_size"],
                "architecture": "SimpleCNN",
            },
            model_path,
        )
        mlflow.log_artifact(str(model_path), artifact_path="model")
        print(f"Saved model -> {model_path}")

        # --- metrics.json for DVC ---
        metrics = {
            "test_loss": round(test_loss, 4),
            "test_accuracy": round(test_acc, 4),
            "val_accuracy": round(history["val_acc"][-1], 4),
            "train_accuracy": round(history["train_acc"][-1], 4),
        }
        with open(t["metrics_file"], "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

    return {"test_accuracy": test_acc, "test_loss": test_loss}


if __name__ == "__main__":
    train()
