#!/usr/bin/env python
"""How much does splitting by image inflate accuracy on this dataset? (resumable)

    python scripts/run_inflation.py --data-roots <dataset dir> \
        --main-split outputs/splits.csv --out-dir outputs_inflation

The main experiment's held-out test photographs (fold -1 in --main-split) stay
untouched: no image of them, original or copy, is used for training or model
selection. They are the *unseen* test set. Everything else (the development
pool) is split again into train / validation / internal test, about
70/15/15 as in the published studies, under three schemes:

  image  stratified at random, image by image (what the published studies did)
  hash   grouped by perceptual hashes alone (difference AND DCT hash, eight
         flips/rotations, the same threshold as the main experiment)
  photo  grouped by source photograph (the main experiment's groups)

Each (scheme, seed, model) model is scored on its own internal test set, the
number a study would report, and on the original photographs of the unseen
test set. The gap between the two is the inflation. With the photo scheme the
two should agree.

Hyperparameters are those of the main experiment (skinclf.train.Config).
Monte Carlo dropout is skipped. Runs are seed-major, so a partial run still
gives every scheme and model for the first seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skinclf.data import build_groups, load_folder_dataset  # noqa: E402
from skinclf.train import Config, fit, predict  # noqa: E402

SCHEMES = ("image", "hash", "photo")


def internal_split(labels, groups, seed, n_test=7, n_val=6):
    """'train' / 'val' / 'test' per image: test 1/7, val 1/6 of the rest (~71/14/14).

    Stratified by class and grouped; with one group per image it is a plain
    stratified random split.
    """
    n = len(labels)
    part = np.full(n, "train", dtype=object)
    outer = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    rest, te = next(outer.split(np.zeros(n), labels, groups))
    part[te] = "test"
    inner = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed + 1)
    _, va = next(inner.split(np.zeros(len(rest)), labels[rest], groups[rest]))
    part[rest[va]] = "val"
    return part


def leakage(part, photo):
    """Share of internal-test images (and originals) whose source photograph is also in train."""
    train_photos = set(photo[part == "train"])
    te = part == "test"
    return float(np.mean([p in train_photos for p in photo[te]]))


def acc(probs, y):
    return float((probs.argmax(1) == y).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-roots", nargs="+", required=True)
    p.add_argument("--main-split", required=True, help="splits.csv of the main experiment")
    p.add_argument("--out-dir", default="outputs_inflation")
    p.add_argument("--models", nargs="+", default=["EfficientNetB0", "ResNet50", "CustomCNN"])
    p.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=SCHEMES)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--img-size", type=int, default=Config.img_size)
    p.add_argument("--epochs-custom", type=int, default=Config.epochs_custom)
    p.add_argument("--epochs-transfer", type=int, default=Config.epochs_transfer)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--hours", type=float, default=None, help="stop starting new runs after this")
    args = p.parse_args()
    cfg = Config(
        data_roots=tuple(args.data_roots), out_dir=args.out_dir, img_size=args.img_size,
        epochs_custom=args.epochs_custom, epochs_transfer=args.epochs_transfer,
        pretrained=args.pretrained, mixed_precision=args.mixed_precision,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**json.loads(cfg.to_json()), **vars(args)}, f, indent=2)

    import keras

    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"Keras {keras.__version__}, backend {keras.backend.backend()}", flush=True)

    ds = load_folder_dataset(list(args.data_roots), args.img_size)
    main_split = pd.read_csv(args.main_split)
    if list(main_split["file"]) != list(ds.files):
        raise RuntimeError(f"{args.main_split} does not match the images on disk")
    photo = main_split["group"].to_numpy()
    unseen = main_split["fold"].to_numpy() == -1
    unseen_orig = np.nonzero(unseen & ds.is_original)[0]
    dev = np.nonzero(~unseen)[0]

    hash_csv = os.path.join(args.out_dir, "hash_groups.csv")
    if os.path.exists(hash_csv):
        hash_groups = pd.read_csv(hash_csv)["hash_group"].to_numpy()
    else:
        hash_groups, stats = build_groups(ds, cfg.hash_threshold, use_filenames=False)
        pd.DataFrame({"file": ds.files, "hash_group": hash_groups}).to_csv(hash_csv, index=False)
        with open(os.path.join(args.out_dir, "hash_grouping_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print("Hash-only grouping:", json.dumps(stats), flush=True)
    scheme_groups = {"image": np.arange(len(ds.files)), "hash": hash_groups, "photo": photo}

    n_classes = len(ds.classes)
    start = time.time()
    for seed in args.seeds:
        for scheme in args.schemes:
            split_csv = os.path.join(args.out_dir, "splits", f"{scheme}_seed{seed}.csv")
            os.makedirs(os.path.dirname(split_csv), exist_ok=True)
            part = np.full(len(ds.files), "unseen", dtype=object)
            part[dev] = internal_split(ds.labels[dev], scheme_groups[scheme][dev], 1000 * seed + 17)
            if not os.path.exists(split_csv):
                pd.DataFrame({"file": ds.files, "part": part}).to_csv(split_csv, index=False)
            tr, va, te = (np.nonzero(part == k)[0] for k in ("train", "val", "test"))
            te_orig = te[ds.is_original[te]]
            leak = leakage(part, photo)
            for name in args.models:
                out = os.path.join(args.out_dir, "runs", f"{scheme}_seed{seed}_{name}")
                if os.path.exists(os.path.join(out, "summary.json")):
                    continue
                if args.hours is not None and (time.time() - start) / 3600 > args.hours:
                    print(f"Time budget of {args.hours} h reached; stopping.", flush=True)
                    return
                print(f"\n=== {scheme} | seed {seed} | {name} ===", flush=True)
                os.makedirs(out, exist_ok=True)
                model, history, best, minutes = fit(
                    cfg, name, 1000 * seed + 42, ds.images, ds.labels, tr, va, n_classes,
                    os.path.join(out, "history.csv"),
                )
                probs = {k: predict(model, ds.images[idx].astype("float32"), cfg.batch_size)
                         for k, idx in (("internal", te), ("unseen", unseen_orig))}
                np.savez_compressed(os.path.join(out, "predictions.npz"), internal_idx=te,
                                    internal_probs=probs["internal"], unseen_idx=unseen_orig,
                                    unseen_probs=probs["unseen"])
                summary = {
                    "scheme": scheme, "seed": seed, "model": name,
                    "n_train": int(len(tr)), "n_val": int(len(va)), "n_internal_test": int(len(te)),
                    "n_internal_test_originals": int(len(te_orig)),
                    "n_unseen_photos": int(len(unseen_orig)),
                    "internal_test_images_with_photo_in_train": leak,
                    "acc_internal": acc(probs["internal"], ds.labels[te]),
                    "acc_internal_originals": acc(probs["internal"][ds.is_original[te]], ds.labels[te_orig]),
                    "acc_unseen": acc(probs["unseen"], ds.labels[unseen_orig]),
                    "epochs_run": len(history.history["loss"]), "best_epoch": best.best_epoch + 1,
                    "train_minutes": minutes,
                }
                with open(os.path.join(out, "summary.json"), "w") as f:
                    json.dump(summary, f, indent=2)
                print(json.dumps(summary), flush=True)
                keras.backend.clear_session()
    print("All requested runs finished.", flush=True)


if __name__ == "__main__":
    main()
