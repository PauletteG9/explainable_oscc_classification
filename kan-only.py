r"""
KAN-only OSCC classifier with a built-in region-of-interest map (no Swin Transformer, no pretrained weights).

How it works
  1. The image (224 x 224) is cut into a grid of non-overlapping 16 x 16 pixel patches (14 x 14 = 196 patches).
  2. A KAN (two spline layers, 768 -> 128 -> 2) scores EVERY patch from its own raw pixels: one Normal and one OSCC score.
  3. The image prediction is the MEAN of the 196 patch scores, so the patch scores ARE the region map
     (red = pushes towards the predicted class). Nothing post-hoc such as SHAP or LIME is needed to draw it.

Usage (PowerShell)
  python kan-only.py train   --data_dir "<archive>" --out_dir "<runs>\kanonly_s1" --seed 1
  python kan-only.py heatmap --out_dir "<runs>\kanonly_s1" --image "C:\path\image.jpg"
  python kan-only.py check   --data_dir "<archive>" --out_dir "<runs>\kanonly_s1"     (are the maps faithful?)
  quick test first: add --epochs 1 to train. If CUDA runs out of memory, add --batch_size 8.

Needs: torch, torchvision, numpy, scipy, scikit-learn, matplotlib, pillow. (No timm, shap or lime.)
The checkpoint (patch size 16) also loads in oscc_swin_kan_xai.py, so its explain/heatmap modes can compare SHAP and LIME.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_pos_idx(class_to_idx):
    """Index of the OSCC (positive) class."""
    for name, idx in class_to_idx.items():
        if any(k in name.lower() for k in ("oscc", "malig", "cancer", "carcinoma")):
            return idx
    return 1


def _stratified_split(targets, fracs, seed):
    """fracs e.g. (0.7, 0.15, 0.15) -> list of index arrays."""
    idx = np.arange(len(targets))
    if len(fracs) == 2:
        a, b = train_test_split(idx, test_size=fracs[1], stratify=targets, random_state=seed)
        return [a, b]
    a, tmp = train_test_split(idx, test_size=fracs[1] + fracs[2], stratify=targets, random_state=seed)
    b, c = train_test_split(tmp, test_size=fracs[2] / (fracs[1] + fracs[2]), stratify=targets[tmp], random_state=seed)
    return [a, b, c]


def make_data(data_dir, out_dir, batch_size, seed, workers, aug="basic"):
    """Two supported layouts:
    (a) pre-split:  data_dir/train/<class>/*, data_dir/test/<class>/*  (+ optional val|valid|validation/)
    (b) flat:       data_dir/<class>/*   -> stratified 70/15/15 split (saved to split.json)"""
    if aug == "strong":  # stronger stain/colour jitter, scale variation and mild blur
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    root = Path(data_dir)
    sub = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    pick = lambda *names: next((sub[n] for n in names if n in sub), None)
    d_train, d_val, d_test = pick("train"), pick("val", "valid", "validation"), pick("test")
    split_file = Path(out_dir) / "split.json"
    IF = datasets.ImageFolder

    if d_train is not None and d_test is not None:  # (a) dataset already split
        tr_aug, tr_eval, test_ds = IF(d_train, transform=train_tf), IF(d_train, transform=eval_tf), IF(d_test, transform=eval_tf)
        assert tr_aug.class_to_idx == test_ds.class_to_idx, "train/test class folders differ"
        tr_targets = np.array(tr_aug.targets)
        if d_val is not None:
            val_ds = IF(d_val, transform=eval_tf)
            assert val_ds.class_to_idx == tr_aug.class_to_idx, "train/val class folders differ"
            train_set, val_set, train_targets = tr_aug, val_ds, tr_targets
            train_eval_set = tr_eval
        else:  # no val folder: carve 15% of train (see note about augmented copies in the README)
            if split_file.exists():
                s = json.load(open(split_file)); tr_i, va_i = np.array(s["train"]), np.array(s["val"])
            else:
                tr_i, va_i = _stratified_split(tr_targets, (0.85, 0.15), seed)
                json.dump({"train": [int(i) for i in tr_i], "val": [int(i) for i in va_i]}, open(split_file, "w"))
            train_set, val_set, train_targets = Subset(tr_aug, tr_i), Subset(tr_eval, va_i), tr_targets[tr_i]
            train_eval_set = Subset(tr_eval, tr_i)
        test_set, test_idx, c2i = test_ds, np.arange(len(test_ds)), tr_aug.class_to_idx
    else:  # (b) flat class folders
        base = IF(data_dir)
        targets = np.array(base.targets)
        if split_file.exists():
            s = json.load(open(split_file)); tr_i, va_i, test_idx = (np.array(s[k]) for k in ("train", "val", "test"))
        else:
            tr_i, va_i, test_idx = _stratified_split(targets, (0.70, 0.15, 0.15), seed)
            json.dump({"train": [int(i) for i in tr_i], "val": [int(i) for i in va_i], "test": [int(i) for i in test_idx]},
                      open(split_file, "w"))
        tr_aug, ds_eval = IF(data_dir, transform=train_tf), IF(data_dir, transform=eval_tf)
        train_set, val_set, test_set = Subset(tr_aug, tr_i), Subset(ds_eval, va_i), Subset(ds_eval, test_idx)
        train_eval_set = Subset(ds_eval, tr_i)
        train_targets, test_ds, c2i = targets[tr_i], ds_eval, base.class_to_idx

    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    return {
        "class_to_idx": c2i,
        "train_targets": train_targets,
        "sizes": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "test_ds": test_ds,      # ImageFolder holding the test images (.samples -> file paths)
        "test_idx": test_idx,    # indices of the test images inside test_ds, in loader order
        "train_loader": DataLoader(train_set, shuffle=True, **kw),
        "train_eval_loader": DataLoader(train_eval_set, shuffle=False, **kw),  # training images, no augmentation
        "val_loader": DataLoader(val_set, shuffle=False, **kw),
        "test_loader": DataLoader(test_set, shuffle=False, **kw),
    }


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    """Temperature scaling: pick T minimizing validation NLL (grid search). Does not change predicted classes."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))
    nll = [-np.log(softmax_np(logits / t)[np.arange(len(y)), y] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def compute_metrics(probs, y, pos_idx, thr=None):
    yb = (y == pos_idx).astype(int)
    pb = ((probs[:, pos_idx] >= thr) if thr is not None else (probs.argmax(1) == pos_idx)).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    eps = 1e-12
    prec, sens = tp / (tp + fp + eps), tp / (tp + fn + eps)
    return {
        "accuracy": float((tp + tn) / len(yb)),
        "precision": float(prec),
        "sensitivity": float(sens),
        "specificity": float(tn / (tn + fp + eps)),
        "f1": float(2 * prec * sens / (prec + sens + eps)),
        "auc": float(roc_auc_score(yb, probs[:, pos_idx])),
        "confusion_matrix[[TN,FP],[FN,TP]]": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def youden_threshold(probs, y, pos_idx):
    """Threshold maximizing sensitivity + specificity - 1. Choose it on VALIDATION data only."""
    fpr, tpr, thr = roc_curve((y == pos_idx).astype(int), probs[:, pos_idx])
    return float(thr[np.argmax(tpr - fpr)])


def bootstrap_ci(probs, y, pos_idx, thr=None, n=1000, seed=0):
    """95% percentile bootstrap CIs over test images (image-level resampling)."""
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "sensitivity", "specificity", "f1", "auc"]
    vals = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = compute_metrics(probs[i], y[i], pos_idx, thr)
        for k in keys:
            vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(torch.softmax(model(x.to(device)), 1).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(device)).cpu())
        Y.append(y)
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def load_image(path):
    # same resize (bilinear) as the eval transform -> the explained input equals the model input
    im = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def overlay(ax, img, heat, title):
    m = float(np.abs(heat).max()) + 1e-8
    ax.imshow(img)
    ax.imshow(heat, cmap="bwr", vmin=-m, vmax=m, alpha=0.5)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _auc(x, y):
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def deletion_insertion(predict, img, heat, target, steps=25, blur_sigma=15, baseline="blur"):
    """Deletion: remove most-important pixels first (blur them) -> prob should drop fast (low AUC good).
    Insertion: reveal most-important pixels first over a blurred image -> prob should rise fast (high AUC good)."""
    H, W, _ = img.shape
    if baseline == "blur":
        base = gaussian_filter(img, sigma=(blur_sigma, blur_sigma, 0))
    else:  # "mean": fill with the image's mean colour (a different, non-blur perturbation)
        base = np.broadcast_to(img.mean(axis=(0, 1), keepdims=True), img.shape).copy()
    order = np.argsort(-heat.ravel(), kind="stable")
    fracs = np.linspace(0, 1, steps + 1)
    dele, ins = [], []
    for f in fracs:
        m = np.zeros(H * W, dtype=bool)
        m[order[:int(f * H * W)]] = True
        m = m.reshape(H, W, 1)
        dele.append(np.where(m, base, img))
        ins.append(np.where(m, img, base))
    pd = predict(np.stack(dele))[:, target]
    pi = predict(np.stack(ins))[:, target]
    return fracs, pd, pi, _auc(fracs, pd), _auc(fracs, pi)


def random_block_heat(shape_hw, block, rng):
    """Random reference at a coarse block granularity (comparable to SHAP's rectangular partitions)."""
    H, W = shape_hw
    g = rng.standard_normal((-(-H // block), -(-W // block)))
    return np.kron(g, np.ones((block, block)))[:H, :W]


def _margin_map(logits_t, target, h, w, size):
    """Per-token scores (n, K) -> (H, W) map of (target score - mean of the other scores), bilinearly upsampled."""
    other = [k for k in range(logits_t.shape[1]) if k != target]
    m = logits_t[:, target] - logits_t[:, other].mean(1)
    up = F.interpolate(m.view(1, 1, h, w), size=size, mode="bilinear", align_corners=False)
    return up[0, 0].cpu().numpy()


class Predictor:
    """Black-box f(images) -> class probabilities. images: (N,H,W,3) float in [0,1]."""

    def __init__(self, model, device, bs=32):
        self.model, self.device, self.bs = model.eval(), device, bs
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs):
        imgs = np.asarray(imgs, dtype=np.float32)
        out = []
        for s in range(0, len(imgs), self.bs):
            x = torch.from_numpy(imgs[s:s + self.bs]).permute(0, 3, 1, 2).to(self.device)
            out.append(torch.softmax(self.model((x - self.mean) / self.std), 1).cpu().numpy())
        return np.concatenate(out)


class KANLinear(nn.Module):
    """phi_{o,i}(x_i) = w_base * SiLU(x_i) + sum_k c_k * B_k(x_i);  y_o = sum_i phi_{o,i}(x_i).
    Inputs outside grid_range only pass through the SiLU base branch (splines are zero there)."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        self.grid_range = grid_range
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.act = nn.SiLU()
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.normal_(self.spline_weight, std=0.1 / (grid_size + spline_order) ** 0.5)

    def b_splines(self, x):  # x: (N, in) -> (N, in, grid_size + spline_order)
        g = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - g[:, :-(k + 1)]) / (g[:, k:-1] - g[:, :-(k + 1)]) * bases[:, :, :-1]
                     + (g[:, k + 1:] - x) / (g[:, k + 1:] - g[:, 1:-k]) * bases[:, :, 1:])
        return bases

    def forward(self, x):
        base = F.linear(self.act(x), self.base_weight)
        spline = F.linear(self.b_splines(x).flatten(1), self.spline_weight.flatten(1))
        return base + spline

    @torch.no_grad()
    def edge_activations(self, x):  # (N, in) -> (N, out, in): phi_{o,i}(x_i) for every edge
        base = torch.einsum("ni,oi->noi", self.act(x), self.base_weight)
        spline = torch.einsum("nik,oik->noi", self.b_splines(x), self.spline_weight)
        return base + spline

    @torch.no_grad()
    def edge_curve(self, xs):  # (G,) -> (G, out, in): each learned 1-D function on a grid
        return self.edge_activations(xs[:, None].expand(-1, self.in_features).contiguous())


class KANHead(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, grid_size=5, spline_order=3):
        super().__init__()
        self.norm0 = nn.LayerNorm(in_dim)
        self.kan1 = KANLinear(in_dim, hidden, grid_size, spline_order)
        self.norm1 = nn.LayerNorm(hidden)
        self.kan2 = KANLinear(hidden, num_classes, grid_size, spline_order)

    def hidden(self, f):  # input of the final KAN layer (used for spline interpretation)
        return self.norm1(self.kan1(self.norm0(f)))

    def forward(self, f):
        return self.kan2(self.hidden(f))


class KANTokenHead(nn.Module):
    """A KAN applied to EVERY final-stage Swin token; the image score is the MEAN of the token scores.
    So the per-token scores are a built-in class-evidence map: score(image) = mean over tokens of g(token).
    Nothing post-hoc is needed to draw the map; it is the head's own output."""

    def __init__(self, in_dim, hidden, num_classes, grid_size=5, spline_order=3):
        super().__init__()
        self.inner = KANHead(in_dim, hidden, num_classes, grid_size, spline_order)

    def token_logits(self, tokens):  # (B, n, C) -> (B, n, K)
        b, n, c = tokens.shape
        return self.inner(tokens.reshape(b * n, c)).view(b, n, -1)

    def forward(self, tokens):  # (B, n, C) -> (B, K)
        return self.token_logits(tokens).mean(1)


class PatchPixels(nn.Module):
    """'Backbone' for the KAN-only model: cuts the image into non-overlapping p x p patches and normalises each patch's raw
    pixel vector (3*p*p numbers). There are no learned spatial features and no pretrained weights; each patch is later
    scored by the KAN from its own pixels alone."""

    def __init__(self, patch=16):
        super().__init__()
        self.patch = patch
        self.num_features = 3 * patch * patch
        self.norm = nn.LayerNorm(self.num_features)

    def forward_features(self, x):  # (B, 3, H, W) -> (B, H/p, W/p, 3*p*p)
        b, c, H, W = x.shape
        p = self.patch
        t = x.unfold(2, p, p).unfold(3, p, p)  # (B, 3, H/p, W/p, p, p)
        t = t.permute(0, 2, 3, 1, 4, 5).reshape(b, H // p, W // p, c * p * p)
        return self.norm(t)

    def forward(self, x):
        return self.forward_features(x).mean(dim=(1, 2))


# ----------------------------------------------------------------------------
# The KAN-only model
# ----------------------------------------------------------------------------
class KANOnly(nn.Module):
    """Patches of raw pixels -> KAN score per patch -> mean = image score. No Swin, no pretrained weights."""

    def __init__(self, num_classes=2, patch=16, hidden=128, grid_size=5):
        super().__init__()
        self.backbone = PatchPixels(patch)  # named 'backbone' so the checkpoint also loads in oscc_swin_kan_xai.py
        self.head = KANTokenHead(self.backbone.num_features, hidden, num_classes, grid_size)

    def token_logits(self, x):  # x: (B, 3, 224, 224) -> per-patch scores (B, n, K), grid height, grid width
        tok = self.backbone.forward_features(x)
        b, h, w, c = tok.shape
        return self.head.token_logits(tok.reshape(b, h * w, c)), h, w

    def forward(self, x):
        return self.token_logits(x)[0].mean(1)


def with_mcc(m):
    (tn, fp), (fn, tp) = m["confusion_matrix[[TN,FP],[FN,TP]]"]
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m["mcc"] = float((tp * tn - fp * fn) / d) if d else 0.0
    return m


def load_model(out, device):
    ck = torch.load(Path(out) / "best.pt", map_location=device)
    model = KANOnly(len(ck["class_to_idx"]), ck.get("patch", 16), ck.get("hidden", 128), ck.get("grid_size", 5)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def patch_map(model, img, target, device):
    """(H, W) map of (patch score for `target` - mean of the other classes' patch scores). Its mean over patches equals
    the model's image-level margin for that class, so red patches really are what pushed the decision."""
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = (torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)[None].to(device) - mean) / std
    tl, h, w = model.token_logits(x)
    return _margin_map(tl[0], target, h, w, img.shape[:2])


# ----------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------
def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert IMG_SIZE % args.patch == 0, "--patch must divide 224 (use 16, or 8 for a finer but much heavier map)"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = make_data(args.data_dir, out, args.batch_size, args.seed, args.workers, aug=args.aug)
    c2i = data["class_to_idx"]
    assert len(c2i) == 2, "binary task expected (Normal vs OSCC); check that --data_dir points at the class folders"
    pos_idx = find_pos_idx(c2i)
    g = IMG_SIZE // args.patch
    print("classes:", c2i, "| positive (OSCC) idx:", pos_idx, "| split sizes:", data["sizes"], f"| patch grid: {g} x {g}")

    model = KANOnly(len(c2i), args.patch, args.hidden, args.grid_size).to(device)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    counts = np.bincount(data["train_targets"], minlength=len(c2i))
    weights = torch.tensor(counts.sum() / (len(c2i) * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    history, best, bad = [], -1.0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y in data["train_loader"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            n += len(y)
        sched.step()
        pv, yv = predict_loader(model, data["val_loader"], device)
        vm = compute_metrics(pv, yv, pos_idx)
        vloss = float(-np.log(pv[np.arange(len(yv)), yv] + 1e-12).mean())
        history.append({"epoch": ep, "train_loss": tl / n, "val_loss": vloss, "val_acc": vm["accuracy"], "val_auc": vm["auc"]})
        print(f"ep {ep:02d} | train {tl / n:.4f} | val loss {vloss:.4f} | acc {vm['accuracy']:.4f} | auc {vm['auc']:.4f}")
        if vm["auc"] + vm["accuracy"] > best:
            best, bad = vm["auc"] + vm["accuracy"], 0
            torch.save({"model": model.state_dict(), "class_to_idx": c2i, "backbone": "patchkan", "head_type": "patchkan",
                        "patch": args.patch, "hidden": args.hidden, "grid_size": args.grid_size}, out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
    json.dump(history, open(out / "history.json", "w"), indent=1)

    model, _ = load_model(out, device)
    pt, yt = predict_loader(model, data["test_loader"], device)
    pv, yv = predict_loader(model, data["val_loader"], device)
    lv, _ = predict_logits(model, data["val_loader"], device)
    lt, _ = predict_logits(model, data["test_loader"], device)
    T = fit_temperature(lv, yv)  # temperature fitted on validation only
    pt_cal = softmax_np(lt / T)
    thr = youden_threshold(pv, yv, pos_idx)  # chosen on validation only
    tm = {"threshold_0.5": with_mcc(compute_metrics(pt, yt, pos_idx)),
          "threshold_0.5_ci95": bootstrap_ci(pt, yt, pos_idx),
          "val_chosen_threshold": thr,
          "val_chosen_threshold_test_metrics": with_mcc(compute_metrics(pt, yt, pos_idx, thr)),
          "temperature": T,
          "setup": {"model": "kan-only", "patch": args.patch, "hidden": args.hidden, "grid_size": args.grid_size,
                    "seed": args.seed, "lr": args.lr, "wd": args.wd, "aug": args.aug}}
    json.dump(tm, open(out / "test_metrics.json", "w"), indent=1)
    json.dump({"temperature": T}, open(out / "calibration.json", "w"))
    pn = [k for k, v in c2i.items() if v == pos_idx][0]
    with open(out / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", f"p_{pn}_raw", f"p_{pn}_calibrated"])
        for j in range(len(yt)):
            w.writerow([Path(data["test_ds"].samples[int(data["test_idx"][j])][0]).name, int(yt[j]),
                        f"{pt[j, pos_idx]:.4f}", f"{pt_cal[j, pos_idx]:.4f}"])
    print("TEST:", json.dumps(tm, indent=1))


# ----------------------------------------------------------------------------
# Region maps for new images
# ----------------------------------------------------------------------------
def heatmap(args):
    """Prediction, calibrated confidence, a defer flag and the KAN patch map for one image or a folder of images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    model, ck = load_model(out, device)
    class_names = [n for n, _ in sorted(ck["class_to_idx"].items(), key=lambda t: t[1])]
    T = float(json.load(open(out / "calibration.json"))["temperature"]) if (out / "calibration.json").exists() else 1.0
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    if args.image:
        paths = [Path(args.image)]
    else:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        paths = sorted(p for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in exts)
    (out / "heatmaps").mkdir(exist_ok=True)
    for path in paths:
        img = load_image(path)
        with torch.no_grad():
            x = (torch.from_numpy(img).permute(2, 0, 1)[None].to(device) - mean) / std
            logits = model(x)[0].cpu().numpy()
        probs = softmax_np((logits / T)[None])[0]
        pred, conf = int(probs.argmax()), float(probs.max())
        defer = conf < args.defer_below
        h = patch_map(model, img, pred, device)
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.4))
        ax[0].imshow(img)
        ax[0].axis("off")
        ax[0].set_title(f"{path.name}\npred = {class_names[pred]} (confidence {conf:.2f})" +
                        ("\nLOW CONFIDENCE: defer to a pathologist" if defer else ""), fontsize=9,
                        color="tab:red" if defer else "black")
        overlay(ax[1], img, h, f"KAN patch map (red = supports {class_names[pred]})")
        plt.tight_layout()
        plt.savefig(out / "heatmaps" / f"{path.stem}_kanmap.png", dpi=150)
        plt.close()
        print(f"{path.name}: {class_names[pred]} ({conf:.2f}){'  DEFER' if defer else ''} -> heatmaps/{path.stem}_kanmap.png")


def check(args):
    """Are the maps faithful? Remove (blur) or restore the pixels the map ranks highest, and compare with random blocks of the
    same size. Deletion AUC: lower = better. Insertion AUC: higher = better. Runs on a random sample of test images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    model, ck = load_model(out, device)
    data = make_data(args.data_dir, out, 32, args.seed, args.workers)
    probs, y = predict_loader(model, data["test_loader"], device)
    predict = Predictor(model, device)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(y), size=min(args.n, len(y)), replace=False)
    res = {"KAN map": [], "random blocks": []}
    for i in idx:
        img = load_image(data["test_ds"].samples[int(data["test_idx"][int(i)])][0])
        pred = int(probs[int(i)].argmax())
        h = patch_map(model, img, pred, device)
        for name, m in [("KAN map", h), ("random blocks", random_block_heat(h.shape, ck.get("patch", 16), rng))]:
            _, _, _, d, ins = deletion_insertion(predict, img, m, pred)
            res[name].append((d, ins))
    print(f"Faithfulness on {len(idx)} random test images (blur baseline)")
    for name, v in res.items():
        v = np.array(v)
        sd = v.std(0, ddof=1) if len(v) > 1 else np.zeros(2)
        print(f"  {name:14s} deletion AUC {v[:, 0].mean():.3f} +- {sd[0]:.3f} | insertion AUC {v[:, 1].mean():.3f} +- {sd[1]:.3f}")
    json.dump({k: np.array(v).tolist() for k, v in res.items()}, open(out / "faithfulness_kan_map.json", "w"))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "heatmap", "check"])
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--aug", choices=["basic", "strong"], default="basic")
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--grid_size", type=int, default=5)
    p.add_argument("--image", default=None)
    p.add_argument("--image_dir", default=None)
    p.add_argument("--defer_below", type=float, default=0.75, help="confidence below which an image is flagged for human review")
    p.add_argument("--n", type=int, default=30, help="check mode: number of test images")
    a, _ = p.parse_known_args(argv)
    if a.mode in ("train", "check") and not a.data_dir:
        p.error("--data_dir is required for train and check")
    if a.mode == "heatmap" and not (a.image or a.image_dir):
        p.error("--image or --image_dir is required for heatmap")
    {"train": train, "heatmap": heatmap, "check": check}[a.mode](a)


if __name__ == "__main__":
    main()
