"""Checks for the duplicate-detection benchmark's scoring and grouping code."""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import dedup_benchmark as db  # noqa: E402
from skinclf.data import image_hashes  # noqa: E402


def _meta(n_fam=14, per_fam=4):
    rows = []
    for f in range(n_fam):
        for k in range(per_fam):
            rows.append({"family": f"c{f % 2}/{f}", "cls": f"c{f % 2}", "is_original": k == 0})
    return pd.DataFrame(rows)


def test_pr_from_hist_perfect_and_random():
    edges = db.score_bins("dinov2")
    pos = np.histogram(np.full(50, 0.9), edges)[0]
    neg = np.histogram(np.full(500, 0.1), edges)[0]
    r = db.pr_from_hist(pos, neg, edges)
    assert r["ap"] == 1.0 and r["roc_auc"] == 1.0 and r["recall_p99"] == 1.0
    assert 0.1 < r["thr_p99"] <= 0.9
    same = np.histogram(np.full(50, 0.5), edges)[0]
    r = db.pr_from_hist(same, same * 10, edges)
    assert abs(r["roc_auc"] - 0.5) < 1e-9 and r["recall_p99"] == 0.0


def test_hash_scores_find_flipped_copy():
    rng = np.random.default_rng(0)
    imgs = [cv2.GaussianBlur(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), (9, 9), 3) for _ in range(3)]
    imgs.append(cv2.flip(imgs[0], 1))
    h = [image_hashes(im) for im in imgs]
    d = np.stack([x[0] for x in h])
    p = np.stack([x[1] for x in h])
    sc = db.hash_scores(d, p, np.arange(4))
    for m in db.HASH_METHODS:
        assert sc[m][0, 0] == 0
        assert sc[m][0, 3] >= -2  # the mirror image is (almost) identical
        assert sc[m][0, 3] > sc[m][0, 1]


def test_grouping_metrics_reference_points():
    meta = _meta()
    fam = pd.factorize(meta.family)[0]
    by_family = db.grouping_metrics(fam, meta)
    assert by_family["families_intact"] == 1.0
    assert by_family["copies_cut_from_original"] == 0.0
    assert by_family["test_images_with_photo_in_train"] == 0.0
    by_image = db.grouping_metrics(np.arange(len(meta)), meta)
    assert by_image["families_intact"] == 0.0
    assert by_image["copies_cut_from_original"] == 1.0
    assert by_image["test_images_with_photo_in_train"] > 0.5


def test_analyze_end_to_end_on_planted_embeddings():
    meta = _meta()
    meta["aug"] = np.where(meta.is_original, "original", "Crop")
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(meta.family.nunique(), 16))
    fam = pd.factorize(meta.family)[0]
    x = centres[fam] + 0.05 * rng.normal(size=(len(meta), 16))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    results, retrieval, grouping = db.analyze(meta, {"dinov2": x}, chunk=7)
    assert results["pairs"]["dinov2"]["ap"] > 0.99
    assert retrieval.set_index("aug").loc["all copies", "top1"] == 1.0
    g = grouping.set_index("method").loc["dinov2"]
    assert g["families_intact"] == 1.0 and g["test_images_with_photo_in_train"] == 0.0
