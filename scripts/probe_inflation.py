#!/usr/bin/env python
"""Split inflation for a linear probe on frozen self-supervised features.

    python scripts/probe_inflation.py --features features_dinov2.npz --images images.csv \
        --main-split outputs/splits.csv --splits-dir outputs_inflation/splits \
        --out-dir outputs_probe --name DINOv2probe

Does the inflation measured by run_inflation.py depend on training a network on
the copies? A linear probe on frozen features can only memorise through its
weights on a fixed embedding, so if image-level splits still inflate its
accuracy, the effect is about the data, not about network training.

Uses the exact splits of run_inflation.py (<splits-dir>/<scheme>_seed<k>.csv):
the same train / val / internal-test / unseen partition for every image. A
multinomial logistic regression with balanced class weights is fitted on the
train part for each C in --cs; the C with the lowest validation log-loss (all
validation images, as a study using that split would choose) is kept and scored
on the internal test set (all images and originals only) and on the original
photographs of the unseen set. One summary.json per (scheme, seed), with the
keys of run_inflation.py, so analyze_inflation.py summarises both.

--features is an .npz with one array of shape (n_images, d), rows in the order
of --images (a CSV with a `path` column ending in the dataset-relative file
name used by the split CSVs). CPU only, seconds per run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def align(features_npz: str, images_csv: str, files: pd.Series) -> np.ndarray:
    """Feature rows reordered to match ``files`` (dataset-relative paths)."""
    d = np.load(features_npz)
    if len(d.files) != 1:
        raise ValueError(f"{features_npz}: expected one array, found {d.files}")
    X = d[d.files[0]].astype(np.float32)
    paths = pd.read_csv(images_csv)["path"]
    if len(paths) != len(X):
        raise ValueError(f"{len(X)} feature rows but {len(paths)} images")
    key = {}
    for i, p in enumerate(paths):
        p = p.replace(os.sep, "/")
        for f in (p, *(p.split("/", k)[-1] for k in range(1, p.count("/") + 1))):
            key.setdefault(f, i)
    try:
        rows = [key[f] for f in files]
    except KeyError as e:
        raise ValueError(f"no feature row for {e}") from None
    return X[rows]


def leakage(part: np.ndarray, photo: np.ndarray) -> float:
    train_photos = set(photo[part == "train"])
    return float(np.mean([p in train_photos for p in photo[part == "test"]]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--main-split", required=True, help="splits.csv of the main experiment")
    p.add_argument("--splits-dir", required=True, help="splits/ of run_inflation.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default="DINOv2probe", help="model name in the outputs")
    p.add_argument("--cs", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    args = p.parse_args()

    main_split = pd.read_csv(args.main_split)
    files = main_split["file"]
    X = align(args.features, args.images, files)
    classes = sorted(main_split["label"].unique())
    y = main_split["label"].map({c: k for k, c in enumerate(classes)}).to_numpy()
    photo = main_split["group"].to_numpy()
    original = main_split["original"].astype(bool).to_numpy()

    split_files = sorted(glob.glob(os.path.join(args.splits_dir, "*_seed*.csv")))
    if not split_files:
        raise SystemExit(f"no split CSVs in {args.splits_dir}")
    for sf in split_files:
        scheme, seed = re.match(r"(\w+?)_seed(\d+)\.csv", os.path.basename(sf)).groups()
        sp = pd.read_csv(sf)
        if list(sp["file"]) != list(files):
            raise RuntimeError(f"{sf} does not list the images of {args.main_split} in order")
        part = sp["part"].to_numpy()
        tr, va, te = (np.nonzero(part == k)[0] for k in ("train", "val", "test"))
        unseen_orig = np.nonzero((part == "unseen") & original)[0]
        te_orig = te[original[te]]

        t0 = time.time()
        fits = []
        for c in args.cs:
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
            clf.fit(X[tr], y[tr])
            fits.append((log_loss(y[va], clf.predict_proba(X[va]), labels=range(len(classes))), c, clf))
        val_loss, c, clf = min(fits, key=lambda f: f[0])
        acc = lambda idx: float((clf.predict(X[idx]) == y[idx]).mean())

        out = os.path.join(args.out_dir, "runs", f"{scheme}_seed{seed}_{args.name}")
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, "predictions.npz"), internal_idx=te,
                            internal_probs=clf.predict_proba(X[te]), unseen_idx=unseen_orig,
                            unseen_probs=clf.predict_proba(X[unseen_orig]))
        summary = {
            "scheme": scheme, "seed": int(seed), "model": args.name,
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_internal_test": int(len(te)),
            "n_internal_test_originals": int(len(te_orig)), "n_unseen_photos": int(len(unseen_orig)),
            "internal_test_images_with_photo_in_train": leakage(part, photo),
            "acc_internal": acc(te), "acc_internal_originals": acc(te_orig), "acc_unseen": acc(unseen_orig),
            "C": c, "val_log_loss": float(val_loss),
            "val_log_loss_by_C": {str(cc): float(l) for l, cc, _ in fits},
            "train_minutes": (time.time() - t0) / 60,
        }
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps({k: summary[k] for k in ("scheme", "seed", "C", "acc_internal",
                                                    "acc_internal_originals", "acc_unseen")}), flush=True)


if __name__ == "__main__":
    main()
