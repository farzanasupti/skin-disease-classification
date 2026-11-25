"""Tests for scripts/probe_inflation.py and analyze_inflation.py --extra-dirs.

Synthetic data only (no images, no Keras): 5 classes x 8 photographs, each
stored as an original plus 3 copies. Features are a strong per-photograph
vector plus a weak class vector, so a split that puts copies of a test
photograph into training lets the probe recognise it.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

from probe_inflation import align, leakage  # noqa: E402

CLASSES, PHOTOS, COPIES, DIM = 5, 8, 3, 32
DATA = "skin-disease-classification-dataset"


def _dataset(tmp_path):
    """Write splits.csv, features + images.csv (shuffled, absolute paths) and two split CSVs."""
    rng = np.random.default_rng(0)
    class_vec = rng.normal(size=(CLASSES, DIM))
    rows, feats = [], []
    for c in range(CLASSES):
        for p in range(PHOTOS):
            photo_vec = rng.normal(size=DIM)
            group = c * PHOTOS + p
            for k in range(COPIES + 1):
                name = f"{p}.jpg" if k == 0 else f"flip-{k}_{p}.jpg"
                rows.append({"file": f"{DATA}/class{c}/{name}", "label": f"class{c}", "group": group,
                             "fold": -1 if p == 0 else p % 5, "original": k == 0})
                f = photo_vec + 0.3 * class_vec[c] + 0.05 * rng.normal(size=DIM)
                feats.append(f / np.linalg.norm(f))
    main = pd.DataFrame(rows)
    main.to_csv(tmp_path / "splits.csv", index=False)

    # Features stored in a different order, under absolute paths, as the dedup benchmark writes them.
    order = rng.permutation(len(main))
    pd.DataFrame({"path": [f"/data/somewhere/{main.file[i]}" for i in order]}).to_csv(tmp_path / "images.csv", index=False)
    np.savez(tmp_path / "features.npz", dinov2=np.asarray(feats, dtype=np.float32)[order])

    splits = tmp_path / "splits"
    splits.mkdir()
    dev = np.nonzero(main.fold.to_numpy() != -1)[0]
    # image: every 7th dev image to test, every 6th to val, regardless of photograph
    part = np.full(len(main), "unseen", dtype=object)
    part[dev] = "train"
    part[dev[3::7]] = "test"
    part[dev[5::6]] = np.where(part[dev[5::6]] == "test", "test", "val")
    pd.DataFrame({"file": main.file, "part": part}).to_csv(splits / "image_seed0.csv", index=False)
    # photo: whole photographs, by the photograph index within the class
    p_idx = main.group.to_numpy() % PHOTOS
    part = np.where(main.fold.to_numpy() == -1, "unseen",
                    np.where(p_idx == 1, "test", np.where(p_idx == 2, "val", "train"))).astype(object)
    pd.DataFrame({"file": main.file, "part": part}).to_csv(splits / "photo_seed0.csv", index=False)
    return main, np.asarray(feats, dtype=np.float32)


def _probe(tmp_path, out):
    subprocess.run([sys.executable, os.path.join(SCRIPTS, "probe_inflation.py"),
                    "--features", str(tmp_path / "features.npz"), "--images", str(tmp_path / "images.csv"),
                    "--main-split", str(tmp_path / "splits.csv"), "--splits-dir", str(tmp_path / "splits"),
                    "--out-dir", str(out), "--cs", "0.1", "10"], check=True, capture_output=True)
    return {os.path.basename(os.path.dirname(f)): json.load(open(f))
            for f in glob.glob(os.path.join(out, "runs", "*", "summary.json"))}


def test_align_reorders_rows_by_relative_path(tmp_path):
    main, feats = _dataset(tmp_path)
    X = align(str(tmp_path / "features.npz"), str(tmp_path / "images.csv"), main.file)
    np.testing.assert_array_equal(X, feats)


def test_leakage_counts_photographs_not_images():
    photo = np.array([0, 0, 1, 1, 2])
    part = np.array(["train", "test", "train", "val", "test"], dtype=object)
    assert leakage(part, photo) == 0.5  # photo 0 has a training image, photo 2 does not


def test_probe_end_to_end(tmp_path):
    main, _ = _dataset(tmp_path)
    runs = _probe(tmp_path, tmp_path / "probe")
    assert set(runs) == {"image_seed0_DINOv2probe", "photo_seed0_DINOv2probe"}
    keys = {"scheme", "seed", "model", "n_train", "n_val", "n_internal_test", "n_internal_test_originals",
            "n_unseen_photos", "internal_test_images_with_photo_in_train", "acc_internal",
            "acc_internal_originals", "acc_unseen", "C", "val_log_loss"}
    for r in runs.values():
        assert keys <= set(r)
        assert r["n_unseen_photos"] == CLASSES  # one held-out photograph per class, originals only
        assert r["C"] in (0.1, 10.0)
        assert r["val_log_loss"] == min(r["val_log_loss_by_C"].values())
        assert all(0 <= r[k] <= 1 for k in ("acc_internal", "acc_internal_originals", "acc_unseen"))
    img, pho = runs["image_seed0_DINOv2probe"], runs["photo_seed0_DINOv2probe"]
    assert img["internal_test_images_with_photo_in_train"] == 1.0
    assert pho["internal_test_images_with_photo_in_train"] == 0.0
    # Copies of every internal-test photograph are in training: the probe recognises them.
    assert img["acc_internal_originals"] == 1.0
    assert img["acc_internal_originals"] > img["acc_unseen"]


def test_analyze_extra_dirs_keeps_pooled_numbers(tmp_path):
    _dataset(tmp_path)
    _probe(tmp_path, tmp_path / "probe")
    # Two fake network runs stand in for run_inflation.py output.
    base = {"seed": 0, "n_train": 1, "n_val": 1, "n_internal_test": 1, "n_internal_test_originals": 1,
            "n_unseen_photos": CLASSES, "acc_internal": 0.9, "acc_internal_originals": 0.95, "acc_unseen": 0.5}
    for scheme, leak in (("image", 1.0), ("photo", 0.0)):
        d = tmp_path / "infl" / "runs" / f"{scheme}_seed0_CNN"
        d.mkdir(parents=True)
        json.dump({**base, "scheme": scheme, "model": "CNN", "internal_test_images_with_photo_in_train": leak},
                  open(d / "summary.json", "w"))
    paper = tmp_path / "paper"
    subprocess.run([sys.executable, os.path.join(SCRIPTS, "analyze_inflation.py"), "--out-dir", str(tmp_path / "infl"),
                    "--extra-dirs", str(tmp_path / "probe"), "--paper-dir", str(paper)], check=True, capture_output=True)
    macros = (paper / "numbers_inflation.tex").read_text()
    table = (paper / "tables" / "inflation.tex").read_text()
    assert r"res@infl-image@acc_unseen\endcsname{50.0}" in macros  # pooled: the CNN run only
    assert r"res@infl@n_runs\endcsname{2}" in macros
    assert "res@infl-image-DINOv2probe@acc_unseen" in macros
    assert "res@infl-photo-CNN@acc_unseen" in macros
    assert table.count("DINOv2 probe$^a$") == 2 and "Logistic regression on frozen DINOv2" in table
