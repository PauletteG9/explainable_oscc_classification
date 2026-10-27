"""
MedKAN-inspired OSCC classifier
-------------------------------
A standalone, reproducible training/evaluation/heatmap pipeline for an OSCC
folder dataset. It implements a local RBF-KAN feature module (LIK-inspired)
and a global RBF-KAN classifier (GIK-inspired).

IMPORTANT:
This is a MedKAN-inspired implementation, NOT a verified line-by-line
reproduction of the authors' official implementation. Use it as an experimental
baseline and describe it accurately in research reporting.

Expected dataset layout:
dataset/
  train/Normal, train/OSCC
  val/Normal,   val/OSCC
  test/Normal,  test/OSCC

Commands:
  python medkan_oscc.py train --data_dir "C:\\path\\to\\dataset" --out_dir "C:\\path\\to\\output\\medkan_s1"
  python medkan_oscc.py predict --checkpoint "...\best_model.pt" --image "...\OSCC_400x_8.jpg"
  python medkan_oscc.py heatmap --checkpoint "...\best_model.pt" --image "...\OSCC_400x_8.jpg"
  python medkan_oscc.py evaluate --checkpoint "...\best_model.pt" --data_dir "...\dataset"
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_auc_score, precision_score,
    recall_score, accuracy_score, matthews_corrcoef
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RBFKANLinear(nn.Module):
    """KAN-style linear layer using learnable radial-basis edge functions."""
    def __init__(self, in_features, out_features, num_centers=8, sigma=0.5,
                 use_base=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = nn.Parameter(
            torch.linspace(-2.0, 2.0, num_centers).view(1, 1, num_centers)
        )
        self.rbf_weight = nn.Parameter(
            torch.empty(out_features, in_features, num_centers)
        )
        nn.init.normal_(self.rbf_weight, mean=0.0, std=0.025)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.use_base = use_base

    def forward(self, x):
        # x: [batch, in_features]
        x = x.clamp(-4, 4)
        distance = (x.unsqueeze(-1) - self.centers).pow(2)
        basis = torch.exp(-distance / (2.0 * self.sigma * self.sigma))
        spline_part = torch.einsum("bik,oik->bo", basis, self.rbf_weight)
        if self.use_base:
            base_part = F.silu(x) @ self.base_weight.t()
            out = spline_part + base_part
        else:
            out = spline_part
        return out + self.bias


class LocalRBFKANConv(nn.Module):
    """
    LIK-inspired local feature block.
    A conventional spatial convolution extracts local patterns, while a
    channel-wise RBF-KAN branch learns nonlinear transformations; both are fused.
    """
    def __init__(self, in_ch, out_ch, stride=1, groups=1, rbf_centers=5):
        super().__init__()
        self.spatial = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1,
            groups=groups, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                   stride=stride, bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)
        # RBF edge weights applied to each output channel at each pixel.
        self.centers = nn.Parameter(torch.linspace(-2, 2, rbf_centers))
        self.rbf_weight = nn.Parameter(torch.empty(out_ch, in_ch, rbf_centers))
        nn.init.normal_(self.rbf_weight, std=0.02)
        self.rbf_scale = nn.Parameter(torch.tensor(0.1))
        self.act = nn.GELU()
        self.skip = (nn.Identity() if in_ch == out_ch and stride == 1 else
                     nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride,
                                              bias=False), nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        local = self.norm1(self.spatial(x))
        b, c, h, w = x.shape
        # Compute RBF channel response at each pixel; reduce basis functions.
        x_clamped = x.clamp(-4, 4).permute(0, 2, 3, 1).unsqueeze(-1)
        centers = self.centers.view(1, 1, 1, 1, -1)
        basis = torch.exp(-((x_clamped - centers) ** 2) / 0.5)
        # [B,H,W,C,K] x [O,C,K] -> [B,H,W,O]
        rbf = torch.einsum("bhwck,ock->bhwo", basis, self.rbf_weight)
        rbf = rbf.permute(0, 3, 1, 2)
        if rbf.shape[-2:] != local.shape[-2:]:
            rbf = F.interpolate(rbf, size=local.shape[-2:], mode="bilinear",
                                align_corners=False)
        fused = local + self.norm2(self.pointwise(x)) + self.rbf_scale * rbf
        return self.act(fused + self.skip(x))


class MedKANInspired(nn.Module):
    """Compact local/global RBF-KAN image classifier."""
    def __init__(self, num_classes=2, width=24, global_hidden=128,
                 rbf_centers=8, dropout=0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.GELU()
        )
        self.lik1 = LocalRBFKANConv(width, width, stride=1, rbf_centers=5)
        self.lik2 = LocalRBFKANConv(width, width * 2, stride=2, rbf_centers=5)
        self.lik3 = LocalRBFKANConv(width * 2, width * 3, stride=2, rbf_centers=5)
        self.lik4 = LocalRBFKANConv(width * 3, width * 4, stride=2, rbf_centers=5)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gik = nn.Sequential(
            RBFKANLinear(width * 4, global_hidden, num_centers=rbf_centers),
            nn.GELU(),
            nn.Dropout(dropout),
            RBFKANLinear(global_hidden, num_classes, num_centers=rbf_centers)
        )

    def forward_features(self, x):
        x = self.stem(x)
        x = self.lik1(x)
        x = self.lik2(x)
        x = self.lik3(x)
        x = self.lik4(x)
        return x

    def forward(self, x):
        feature_map = self.forward_features(x)
        pooled = self.global_pool(feature_map).flatten(1)
        logits = self.gik(pooled)
        return logits


def build_transforms(img_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def get_datasets(data_dir, img_size):
    data_dir = Path(data_dir)
    required = [data_dir / x for x in ("train", "val", "test")]
    if not all(p.is_dir() for p in required):
        raise FileNotFoundError(
            "Expected presplit dataset folders: train/Normal, train/OSCC, "
            "val/Normal, val/OSCC, test/Normal, test/OSCC. "
            "Please point --data_dir to the dataset root."
        )
    train_ds = datasets.ImageFolder(required[0], transform=build_transforms(img_size, True))
    val_ds = datasets.ImageFolder(required[1], transform=build_transforms(img_size, False))
    test_ds = datasets.ImageFolder(required[2], transform=build_transforms(img_size, False))
    expected = train_ds.class_to_idx
    if val_ds.class_to_idx != expected or test_ds.class_to_idx != expected:
        raise ValueError(f"Class mapping differs between splits: train={expected}, "
                         f"val={val_ds.class_to_idx}, test={test_ds.class_to_idx}")
    if set(expected) != {"Normal", "OSCC"}:
        raise ValueError(f"Expected class folders Normal and OSCC; got {expected}")
    return train_ds, val_ds, test_ds, expected


def make_loader(ds, batch_size, shuffle, workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


def collect_predictions(model, loader, device):
    model.eval()
    ys, probs, preds, losses = [], [], [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]
            losses.append(criterion(logits, y).item())
            ys.extend(y.cpu().numpy().tolist())
            probs.extend(p.cpu().numpy().tolist())
            preds.extend((p >= 0.5).long().cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(probs), np.asarray(preds), float(np.mean(losses))


def metrics_dict(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "mcc": float(matthews_corrcoef(y, pred)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y))
    }


def save_evaluation(model, loader, device, out_dir, name):
    y, p, pred, loss = collect_predictions(model, loader, device)
    result = metrics_dict(y, p, 0.5)
    result["loss"] = loss
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(result, indent=2))
    with open(out_dir / f"{name}_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_label", "oscc_probability", "predicted_label"])
        for yi, pi, predi in zip(y, p, pred):
            w.writerow([int(yi), float(pi), int(predi)])
    return result


def train(args):
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds, val_ds, test_ds, class_to_idx = get_datasets(args.data_dir, args.img_size)
    print(f"Class mapping: {class_to_idx}")
    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Trainable parameters: {sum(p.numel() for p in MedKANInspired(width=args.width, global_hidden=args.hidden).parameters() if p.requires_grad):,}")

    train_loader = make_loader(train_ds, args.batch_size, True, args.workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.workers)
    model = MedKANInspired(width=args.width, global_hidden=args.hidden,
                           rbf_centers=args.rbf_centers, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_auc = -1.0
    best_loss = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        start = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        yv, pv, _, val_loss = collect_predictions(model, val_loader, device)
        val_auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) == 2 else 0.0
        val_metrics = metrics_dict(yv, pv, 0.5)
        scheduler.step(val_auc)
        epoch_row = {
            "epoch": epoch, "train_loss": running_loss / max(seen, 1),
            "val_loss": val_loss, "val_auc": val_auc,
            "val_accuracy": val_metrics["accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(epoch_row)
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={epoch_row['train_loss']:.4f} | val_loss={val_loss:.4f} | "
              f"val_auc={val_auc:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
              f"time={time.time()-start:.1f}s")

        # Prefer AUC, use validation loss as tie-breaker.
        improved = (val_auc > best_auc + 1e-4) or (abs(val_auc - best_auc) <= 1e-4 and val_loss < best_loss)
        if improved:
            best_auc, best_loss, bad_epochs = val_auc, val_loss, 0
            checkpoint = {
                "model_state": model.state_dict(), "class_to_idx": class_to_idx,
                "img_size": args.img_size, "width": args.width, "hidden": args.hidden,
                "rbf_centers": args.rbf_centers, "dropout": args.dropout,
                "seed": args.seed, "best_val_auc": best_auc,
                "architecture": "MedKAN-inspired local/global RBF-KAN (not official reproduction)"
            }
            torch.save(checkpoint, out_dir / "best_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    model, ckpt = load_model(out_dir / "best_model.pt", device)
    val_result = save_evaluation(model, val_loader, device, out_dir, "val")
    test_result = save_evaluation(model, test_loader, device, out_dir, "test")
    summary = {
        "device": str(device), "class_to_idx": class_to_idx,
        "best_validation_auc": best_auc, "validation_metrics": val_result,
        "test_metrics": test_result, "checkpoint": str(out_dir / "best_model.pt"),
        "note": "MedKAN-inspired implementation; not a verified official-code reproduction."
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== TEST METRICS (threshold=0.5) ===")
    for k, v in test_result.items():
        print(f"{k}: {v}")
    print(f"\nSaved all outputs to: {out_dir.resolve()}")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MedKANInspired(width=ckpt["width"], global_hidden=ckpt["hidden"],
                           rbf_centers=ckpt["rbf_centers"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def preprocess_image(image_path, img_size):
    img = Image.open(image_path).convert("RGB")
    tensor = build_transforms(img_size, False)(img).unsqueeze(0)
    return img, tensor


def predict_one(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    original, x = preprocess_image(args.image, ckpt["img_size"])
    x = x.to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    pred_idx = int(np.argmax(prob))
    confidence = float(prob[pred_idx])
    print(f"Image: {args.image}")
    print(f"Prediction: {idx_to_class[pred_idx]}")
    print(f"Normal probability: {prob[ckpt['class_to_idx']['Normal']]:.4f}")
    print(f"OSCC probability: {prob[ckpt['class_to_idx']['OSCC']]:.4f}")
    print(f"Predicted-class confidence: {confidence:.4f}")
    print("Reminder: this is an experimental research model, not a clinical diagnosis.")


def kan_roi_map(model, x, grid=14, chunk_size=32, target_class=1):
    """Perturbation-based ROI map from changes in the model's class-score margin.

    This is not Grad-CAM. Each grid cell is replaced with normalized zero
    (approximately the ImageNet mean colour), and the change in target-vs-other
    class logit margin is measured. Positive values indicate that masking that
    region reduced the target-class margin. It is model-based perturbation
    attribution, not a unique/parametric explanation of KAN spline functions.
    """
    model.eval()
    _, _, h, w = x.shape
    grid = max(1, min(int(grid), h, w))
    # Use integer boundaries so cells cover the image with no gaps.
    ys = [round(i * h / grid) for i in range(grid + 1)]
    xs = [round(i * w / grid) for i in range(grid + 1)]
    with torch.no_grad():
        base_logits = model(x)
        other_class = 1 - target_class
        base_margin = (base_logits[0, target_class] - base_logits[0, other_class]).item()
        variants, cells = [], []
        for gy in range(grid):
            for gx in range(grid):
                perturbed = x.clone()
                perturbed[:, :, ys[gy]:ys[gy+1], xs[gx]:xs[gx+1]] = 0.0
                variants.append(perturbed)
                cells.append((gy, gx))
        values = []
        for i in range(0, len(variants), chunk_size):
            batch = torch.cat(variants[i:i + chunk_size], dim=0)
            logits = model(batch)
            margins = logits[:, target_class] - logits[:, other_class]
            values.extend((base_margin - margins).detach().cpu().tolist())
    score_grid = np.zeros((grid, grid), dtype=np.float32)
    for (gy, gx), score in zip(cells, values):
        score_grid[gy, gx] = score
    # Preserve positive evidence for target class; negative values mean masking
    # the region increased target margin. Normalize symmetrically by max abs.
    denom = float(np.max(np.abs(score_grid))) + 1e-8
    normalized = score_grid / denom
    return normalized, base_logits.detach()


def _save_roi_figure(original, roi, pred_name, oscc_prob, out_path, target_name="OSCC"):
    from matplotlib.patches import Rectangle
    roi_up = np.asarray(Image.fromarray(roi.astype(np.float32), mode="F").resize(
        (original.width, original.height), Image.Resampling.BILINEAR))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original image")
    axes[1].imshow(roi_up, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title(f"Region ablation map ({target_name})")
    axes[2].imshow(original)
    axes[2].imshow(roi_up, cmap="coolwarm", vmin=-1, vmax=1, alpha=0.48)
    # Outline top 20% of positive cells as candidate ROIs, not clinical lesions.
    positive = roi[roi > 0]
    if positive.size:
        threshold = float(np.quantile(positive, 0.80))
        gh, gw = roi.shape
        cell_w, cell_h = original.width / gw, original.height / gh
        for gy in range(gh):
            for gx in range(gw):
                if roi[gy, gx] >= threshold and roi[gy, gx] > 0:
                    axes[2].add_patch(Rectangle((gx*cell_w, gy*cell_h), cell_w, cell_h,
                                                fill=False, edgecolor="white", linewidth=1.2))
    axes[2].set_title("Overlay + candidate high-impact regions")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Prediction: {pred_name} | OSCC probability: {oscc_prob:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def heatmap(args):
    """Generate a non-gradient, perturbation-based KAN-model ROI map for one image."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    original, x = preprocess_image(args.image, ckpt["img_size"])
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax().item())
    target_idx = ckpt["class_to_idx"].get("OSCC", 1) if args.target == "oscc" else pred_idx
    roi, base_logits = kan_roi_map(model, x, grid=args.grid, chunk_size=args.chunk_size,
                                   target_class=target_idx)
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    oscc_prob = float(probs[ckpt["class_to_idx"]["OSCC"]].cpu())
    target_name = idx_to_class[target_idx]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "kan_roi_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem}_kan_roi.png"
    _save_roi_figure(original, roi, idx_to_class[pred_idx], oscc_prob, out_path, target_name)
    np.save(out_dir / f"{Path(args.image).stem}_kan_roi_scores.npy", roi)
    print(f"Image: {args.image}")
    print(f"Prediction: {idx_to_class[pred_idx]}")
    print(f"Normal probability: {float(probs[ckpt['class_to_idx']['Normal']].cpu()):.4f}")
    print(f"OSCC probability: {oscc_prob:.4f}")
    print(f"ROI target class: {target_name}")
    print(f"Saved KAN-model ROI map: {out_path.resolve()}")
    print("Method: perturbation/occlusion score changes; no gradients/Grad-CAM used.")
    print("Highlighted regions are model-sensitive candidate ROIs, not clinically validated lesions.")

def batch_heatmaps(args):
    """Generate a Grad-CAM figure for every image in a split or image folder."""
    root = Path(args.data_dir) if args.data_dir else None
    if args.image_dir:
        image_root = Path(args.image_dir)
        if not image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_root}")
        images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
        split_name = image_root.name
        out_root = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "heatmaps" / split_name
        jobs = [(img, out_root) for img in images]
    else:
        if root is None or not root.is_dir():
            raise FileNotFoundError("Provide a valid --data_dir or --image_dir")
        splits = ["train", "val", "test"] if args.split == "all" else [args.split]
        jobs = []
        for split_name in splits:
            split_dir = root / split_name
            if not split_dir.is_dir():
                print(f"Skipping missing split folder: {split_dir}")
                continue
            images = sorted(p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
            base_out = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "heatmaps"
            jobs.extend((img, base_out / split_name) for img in images)

    if not jobs:
        raise FileNotFoundError("No supported images found. Check your image folder or dataset split.")

    total = len(jobs)
    print(f"Generating heatmaps for {total} image(s)...")
    manifest_path = (Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "heatmaps") / "heatmap_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (img_path, destination) in enumerate(jobs, start=1):
        print(f"\n[{i}/{total}] {img_path}")
        try:
            # Reuse the single-image ROI implementation for each image.
            heatmap(argparse.Namespace(checkpoint=args.checkpoint, image=str(img_path),
                                       out_dir=str(destination), cpu=args.cpu,
                                       target=args.target, grid=args.grid,
                                       chunk_size=args.chunk_size))
            rows.append([str(img_path), "success", ""])
        except Exception as exc:
            print(f"ERROR processing {img_path}: {exc}")
            rows.append([str(img_path), "failed", str(exc)])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "status", "error"])
        writer.writerows(rows)
    succeeded = sum(row[1] == "success" for row in rows)
    print(f"\nFinished: {succeeded}/{total} heatmaps generated.")
    print(f"Manifest: {manifest_path.resolve()}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    _, _, test_ds, class_to_idx = get_datasets(args.data_dir, ckpt["img_size"])
    loader = make_loader(test_ds, args.batch_size, False, args.workers)
    y, p, _, loss = collect_predictions(model, loader, device)
    result = metrics_dict(y, p, 0.5)
    result["loss"] = loss
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reevaluated_test_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="MedKAN-inspired OSCC pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rbf_centers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=train)

    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=predict_one)

    p = sub.add_parser("heatmap")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out_dir")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--target", choices=["oscc", "predicted"], default="oscc",
                   help="Class whose evidence to map; default is OSCC")
    p.add_argument("--grid", type=int, default=14,
                   help="Occlusion grid per side; higher values are more detailed but slower")
    p.add_argument("--chunk_size", type=int, default=32,
                   help="Number of masked image variants evaluated per batch")
    p.set_defaults(func=heatmap)

    p = sub.add_parser("heatmaps", help="Generate KAN-model ROI maps for all images in a dataset split or folder")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", help="Dataset root containing train/val/test folders")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--image_dir", help="Optional arbitrary folder to scan recursively; overrides --data_dir/--split")
    p.add_argument("--out_dir")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--target", choices=["oscc", "predicted"], default="oscc")
    p.add_argument("--grid", type=int, default=14)
    p.add_argument("--chunk_size", type=int, default=32)
    p.set_defaults(func=batch_heatmaps)

    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
