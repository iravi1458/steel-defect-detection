"""
Steel Surface Defect Detection — Severstal dataset.

Frozen DINOv2 features + logistic regression, compared against an
unsupervised PatchCore-style baseline on identical features.

Usage:
    python steel_defect_detection.py --data /path/to/severstal-steel-defect-detection

Notes:
    On Apple Silicon, PYTORCH_ENABLE_MPS_FALLBACK=1 is set below before torch
    is imported. DINOv2 interpolates its position embeddings with bicubic
    upsampling, which is not implemented for MPS.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader, Dataset

SEED = 42

# Aspect-preserving and divisible by the ViT patch size of 14.
# Native images are 1600x256; resizing to a square destroys thin
# longitudinal defects (AUROC 0.661 vs 0.754 on the unsupervised branch).
INPUT_SIZE = (252, 1596)

TRANSFORM = T.Compose([
    T.Resize(INPUT_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SteelDataset(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = os.path.join(self.root, "train_images", self.files[i])
        return TRANSFORM(Image.open(path).convert("RGB"))


def load_backbone(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    return model.eval().to(device)


@torch.no_grad()
def extract(model, root, files, device, batch_size=4, verbose=True):
    """Patch features for a list of images -> (N, 513, 768) float32."""
    loader = DataLoader(SteelDataset(root, files), batch_size=batch_size, num_workers=0)
    out = []
    for bi, batch in enumerate(loader):
        batch = batch.to(device)
        feats = model.get_intermediate_layers(batch, n=[3, 9], reshape=True)
        feats = torch.cat(feats, dim=1)                          # (B, 768, H, W)
        feats = torch.nn.functional.avg_pool2d(feats, 3, 2, 1)   # local context
        feats = feats.permute(0, 2, 3, 1).reshape(batch.size(0), -1, feats.size(1))
        out.append(feats.cpu().numpy().astype(np.float32))
        if verbose and bi % 25 == 0:
            print(f"    {bi * batch_size}/{len(files)}", flush=True)
    return np.concatenate(out)


def build_splits(root):
    """Disjoint train/test splits. Defect classes are drawn from images
    unused in training so per-class recall is not inflated."""
    df = pd.read_csv(os.path.join(root, "train.csv"))
    all_images = sorted(os.listdir(os.path.join(root, "train_images")))

    defective = set(df["ImageId"].unique())
    clean = [f for f in all_images if f not in defective]
    defective_list = sorted(defective)

    random.seed(SEED)
    random.shuffle(clean)
    random.shuffle(defective_list)

    splits = {
        "test_clean":     clean[500:700],
        "test_defect":    defective_list[300:400],
        "train_clean":    clean[900:1900],
        "train_defect":   defective_list[400:1400],
        "unsup_bank":     clean[:500],       # unsupervised branch only
    }

    train_ids = set(splits["train_clean"]) | set(splits["train_defect"])
    test_ids = set(splits["test_clean"]) | set(splits["test_defect"])
    assert not (train_ids & test_ids), "train/test overlap"

    for cid in [1, 2, 3, 4]:
        ids = sorted(df[df.ClassId == cid].ImageId.unique())
        random.seed(SEED)
        random.shuffle(ids)
        splits[f"class{cid}"] = [i for i in ids if i not in train_ids][:100]

    return df, splits


def supervised(features, splits):
    """Logistic regression on mean-pooled patch features."""
    X = np.r_[features["train_clean"].mean(1), features["train_defect"].mean(1)]
    y = np.r_[np.zeros(len(features["train_clean"])), np.ones(len(features["train_defect"]))]

    clf = LogisticRegression(max_iter=3000).fit(X, y)

    p_clean = clf.predict_proba(features["test_clean"].mean(1))[:, 1]
    p_defect = clf.predict_proba(features["test_defect"].mean(1))[:, 1]

    y_true = np.r_[np.zeros(len(p_clean)), np.ones(len(p_defect))]
    results = {
        "auroc_test": float(roc_auc_score(y_true, np.r_[p_clean, p_defect])),
        "operating": {},
        "per_class_5pct": {},
    }

    for fpr in [0.01, 0.05, 0.10, 0.20]:
        thr = np.quantile(p_clean, 1 - fpr)
        results["operating"][f"{fpr:.0%}"] = float((p_defect > thr).mean())

    thr5 = np.quantile(p_clean, 0.95)
    for cid in [1, 2, 3, 4]:
        p = clf.predict_proba(features[f"class{cid}"].mean(1))[:, 1]
        results["per_class_5pct"][str(cid)] = float((p > thr5).mean())

    cv = cross_val_score(LogisticRegression(max_iter=3000), X, y, cv=5, scoring="roc_auc")
    results["auroc_cv5"] = [float(s) for s in cv]
    results["auroc_cv5_mean"] = float(cv.mean())

    return clf, results


def unsupervised(features, device, bank_size=150_000, topk=10, chunk=25_000):
    """PatchCore-style memory bank. Trained on unannotated images only.

    Note these images are unlabelled, NOT verified defect-free — which is
    why this branch underperforms. See README.
    """
    patches = features["unsup_bank"].reshape(-1, features["unsup_bank"].shape[-1])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(patches), size=min(bank_size, len(patches)), replace=False)
    bank = torch.nn.functional.normalize(
        torch.tensor(patches[idx], device=device), dim=1
    )

    @torch.no_grad()
    def score(F, batch_size=2):
        out = []
        for i in range(0, len(F), batch_size):
            x = torch.nn.functional.normalize(
                torch.tensor(F[i:i + batch_size], device=device), dim=2
            )
            best = None
            for j in range(0, bank.shape[0], chunk):
                d = 1 - torch.einsum("bpd,md->bpm", x, bank[j:j + chunk])
                m = d.min(dim=2).values
                best = m if best is None else torch.minimum(best, m)
                del d
            out.append(best.topk(topk, dim=1).values.mean(dim=1).cpu().numpy())
            del x, best
        return np.concatenate(out)

    s_clean = score(features["test_clean"])
    s_defect = score(features["test_defect"])

    y_true = np.r_[np.zeros(len(s_clean)), np.ones(len(s_defect))]
    results = {
        "auroc": float(roc_auc_score(y_true, np.r_[s_clean, s_defect])),
        "operating": {},
    }
    for fpr in [0.01, 0.05, 0.10, 0.20]:
        thr = np.quantile(s_clean, 1 - fpr)
        results["operating"][f"{fpr:.0%}"] = float((s_defect > thr).mean())

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Severstal dataset root")
    ap.add_argument("--cache", default=None, help="feature cache dir (default: <data>/features)")
    ap.add_argument("--skip-unsupervised", action="store_true")
    args = ap.parse_args()

    root = os.path.expanduser(args.data)
    cache = args.cache or os.path.join(root, "features")
    os.makedirs(cache, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df, splits = build_splits(root)
    print(f"splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    model = load_backbone(device)

    features = {}
    for name, files in splits.items():
        if args.skip_unsupervised and name == "unsup_bank":
            continue
        path = os.path.join(cache, f"{name}.npy")
        if os.path.exists(path):
            features[name] = np.load(path)
            print(f"  {name}: cached {features[name].shape}")
        else:
            print(f"  {name}: extracting {len(files)} images")
            t0 = time.time()
            features[name] = extract(model, root, files, device)
            np.save(path, features[name])
            print(f"  {name}: {features[name].shape} in {time.time() - t0:.0f}s")

    print("\n=== supervised ===")
    clf, sup = supervised(features, splits)
    print(f"AUROC (test):    {sup['auroc_test']:.4f}")
    print(f"AUROC (5-fold):  {sup['auroc_cv5_mean']:.4f}")
    for fpr, recall in sup["operating"].items():
        print(f"  FPR {fpr:>4} -> recall {recall:.1%}")
    for cid, recall in sup["per_class_5pct"].items():
        print(f"  class {cid} @5% FPR: {recall:.1%}")

    out = {"supervised": sup}

    if not args.skip_unsupervised:
        print("\n=== unsupervised ===")
        unsup = unsupervised(features, device)
        print(f"AUROC: {unsup['auroc']:.4f}")
        for fpr, recall in unsup["operating"].items():
            print(f"  FPR {fpr:>4} -> recall {recall:.1%}")
        out["unsupervised"] = unsup

    with open(os.path.join(root, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> {os.path.join(root, 'results.json')}")


if __name__ == "__main__":
    main()
