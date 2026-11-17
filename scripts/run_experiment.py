#!/usr/bin/env python
"""Run the full cross-validated comparison (resumable).

    python scripts/run_experiment.py --data-roots /kaggle/input/skin-v1/train \
        /kaggle/input/skin-v1/valid --out-dir outputs

Steps: load images, group duplicates, build a grouped stratified split
(held-out test + K development folds), then train every model on every fold.
Finished (model, fold) runs are skipped, so re-running the same command after
an interrupted session picks up where it stopped. ``--hours`` stops cleanly
before a platform time limit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skinclf.data import (  # noqa: E402
    build_groups,
    check_no_group_leakage,
    load_folder_dataset,
    make_splits,
    split_table,
)
from skinclf.models import MODEL_NAMES  # noqa: E402
from skinclf.train import Config, train_one  # noqa: E402


def parse_args() -> tuple[Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    defaults = Config()
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, tuple):
            p.add_argument(flag, nargs="+", default=list(default))
        else:
            p.add_argument(flag, type=type(default), default=default)
    p.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--folds", nargs="+", type=int, default=None)
    p.add_argument("--hours", type=float, default=None, help="stop starting new runs after this")
    args = p.parse_args()
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    cfg.data_roots = tuple(cfg.data_roots)
    return cfg, args


def load_or_build_split(cfg: Config):
    ds = load_folder_dataset(list(cfg.data_roots), cfg.img_size)
    split_csv = os.path.join(cfg.out_dir, "splits.csv")
    if os.path.exists(split_csv):
        table = pd.read_csv(split_csv)
        if "phash" not in table.columns:
            raise RuntimeError(
                f"{split_csv} was made by the old difference-hash-only grouping, which missed "
                "mirrored and rotated copies; start from a fresh --out-dir"
            )
        if list(table["file"]) != list(ds.files):
            raise RuntimeError(
                f"{split_csv} does not match the images on disk; "
                "use a fresh --out-dir or delete it deliberately"
            )
        groups, fold = table["group"].to_numpy(), table["fold"].to_numpy()
        print(f"Reusing split from {split_csv}")
    else:
        groups, stats = build_groups(ds, cfg.hash_threshold)
        fold = make_splits(ds.labels, groups, cfg.test_fraction, cfg.n_folds, cfg.seed)
        split_table(ds, groups, fold).to_csv(split_csv, index=False)
        with open(os.path.join(cfg.out_dir, "grouping_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print("Grouping:", json.dumps(stats, indent=2))
    check_no_group_leakage(groups, fold)
    return ds, groups, fold


def describe_split(ds, fold) -> pd.DataFrame:
    part = np.where(fold == -1, "test", "fold" + fold.astype(str))
    df = pd.crosstab(pd.Series(part, name="partition"), pd.Series(np.array(ds.classes)[ds.labels], name="class"))
    df["total"] = df.sum(axis=1)
    return df


def main() -> None:
    cfg, args = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        f.write(cfg.to_json())

    import keras

    if cfg.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"Keras {keras.__version__}, backend {keras.backend.backend()}")

    ds, groups, fold = load_or_build_split(cfg)
    counts = describe_split(ds, fold)
    counts.to_csv(os.path.join(cfg.out_dir, "split_counts.csv"))
    print(counts.to_string())
    with open(os.path.join(cfg.out_dir, "classes.json"), "w") as f:
        json.dump(ds.classes, f)

    folds = args.folds if args.folds is not None else list(range(cfg.n_folds))
    start = time.time()
    # Fold-major order: every model gets fold 0 before any gets fold 1, so a
    # partial run still supports a like-for-like comparison.
    for k in folds:
        for name in args.models:
            if args.hours is not None and (time.time() - start) / 3600 > args.hours:
                print(f"Time budget of {args.hours} h reached; stopping before {name} fold {k}.")
                return
            print(f"\n=== {name} | fold {k} ===", flush=True)
            summary = train_one(cfg, name, k, ds.images, ds.labels, fold, len(ds.classes))
            print(json.dumps(summary), flush=True)
    print("All requested runs finished.")


if __name__ == "__main__":
    main()
