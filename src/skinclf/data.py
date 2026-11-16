"""Image loading, duplicate-aware grouping, and leakage-free splits.

The dataset has no patient identifiers, so splits are made over *groups of
images that share a source*. Two images land in the same group when either

1. their filenames reduce to the same source stem (export tools such as
   Roboflow append ``_jpg.rf.<hash>`` to every copy of one original), or
2. their 64-bit difference hashes are within ``hash_threshold`` bits, which
   catches re-encoded or resized duplicates whatever their names.

A group never spans the test set and a training fold, so a near-copy of a
test image cannot be seen during training.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
_EXT_TOKEN = re.compile(r"[_.-](jpe?g|png|bmp|webp)$", re.IGNORECASE)


@dataclass
class Dataset:
    images: np.ndarray  # (N, H, W, 3) uint8, RGB
    labels: np.ndarray  # (N,) int class index
    classes: list[str]
    files: np.ndarray  # (N,) relative path, for traceability
    dhash: np.ndarray  # (N,) uint64


def source_stem(filename: str) -> str:
    """Strip extension and export-tool suffixes to recover the original name."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = _RF_SUFFIX.sub("", stem)
    return _EXT_TOKEN.sub("", stem)


def dhash64(img_rgb: np.ndarray) -> np.uint64:
    """64-bit difference hash of an RGB image."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def load_folder_dataset(roots: list[str], img_size: int) -> Dataset:
    """Load every image under ``root/<class>/`` for each root, merged.

    Images are converted to RGB (OpenCV reads BGR) and resized with area
    interpolation. Class names come from the sub-folder names, sorted.
    """
    classes = sorted(
        {d for r in roots for d in os.listdir(r) if os.path.isdir(os.path.join(r, d))}
    )
    cls_idx = {c: i for i, c in enumerate(classes)}
    images, labels, files, hashes = [], [], [], []
    for root in roots:
        for cls in classes:
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.lower().endswith(IMG_EXTS):
                    continue
                path = os.path.join(d, f)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hashes.append(dhash64(img))
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                images.append(img)
                labels.append(cls_idx[cls])
                files.append(os.path.join(os.path.basename(os.path.normpath(root)), cls, f))
    if not images:
        raise FileNotFoundError(f"no images found under {roots}")
    return Dataset(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        classes=classes,
        files=np.asarray(files),
        dhash=np.asarray(hashes, dtype=np.uint64),
    )


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_to_all(h: np.ndarray, i: int) -> np.ndarray:
    x = np.bitwise_xor(h[i + 1 :], h[i]).view(np.uint8).reshape(-1, 8)
    return _POPCOUNT8[x].sum(axis=1)


def build_groups(ds: Dataset, hash_threshold: int = 4) -> tuple[np.ndarray, dict]:
    """Assign a group id to every image; return ids and grouping statistics."""
    n = len(ds.labels)
    uf = _UnionFind(n)

    stems = [f"{ds.classes[l]}/{source_stem(f)}" for f, l in zip(ds.files, ds.labels)]
    first_seen: dict[str, int] = {}
    name_links = 0
    for i, s in enumerate(stems):
        if s in first_seen:
            uf.union(first_seen[s], i)
            name_links += 1
        else:
            first_seen[s] = i

    hash_links = 0
    if hash_threshold >= 0:
        for i in range(n - 1):
            close = np.nonzero(_hamming_to_all(ds.dhash, i) <= hash_threshold)[0]
            for j in close + i + 1:
                if uf.find(i) != uf.find(int(j)):
                    hash_links += 1
                uf.union(i, int(j))

    roots = np.array([uf.find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)

    sizes = np.bincount(groups)
    mixed = sum(
        1 for g in np.nonzero(sizes > 1)[0] if len(np.unique(ds.labels[groups == g])) > 1
    )
    stats = {
        "n_images": int(n),
        "n_groups": int(len(sizes)),
        "n_multi_image_groups": int((sizes > 1).sum()),
        "images_in_multi_image_groups": int(sizes[sizes > 1].sum()),
        "max_group_size": int(sizes.max()),
        "links_from_filenames": int(name_links),
        "links_from_hashes": int(hash_links),
        "groups_with_conflicting_labels": int(mixed),
        "hash_threshold_bits": int(hash_threshold),
    }
    return groups, stats


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def make_splits(
    labels: np.ndarray, groups: np.ndarray, test_fraction: float, n_folds: int, seed: int
) -> np.ndarray:
    """Return a fold id per image: -1 for the held-out test set, else 0..n_folds-1.

    Both the test split and the development folds are stratified by class and
    grouped, so no group appears in more than one partition.
    """
    n = len(labels)
    placeholder = np.zeros(n)
    outer = StratifiedGroupKFold(
        n_splits=max(2, round(1 / test_fraction)), shuffle=True, random_state=seed
    )
    dev_idx, test_idx = next(outer.split(placeholder, labels, groups))

    fold = np.full(n, -1, dtype=np.int64)
    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(inner.split(dev_idx, labels[dev_idx], groups[dev_idx])):
        fold[dev_idx[va]] = k
    assert (fold[test_idx] == -1).all()
    return fold


def check_no_group_leakage(groups: np.ndarray, fold: np.ndarray) -> None:
    """Raise if any group is spread over more than one partition."""
    df = pd.DataFrame({"g": groups, "f": fold})
    spread = df.groupby("g")["f"].nunique()
    bad = spread[spread > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} groups span several partitions")


def split_table(ds: Dataset, groups: np.ndarray, fold: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file": ds.files,
            "label": [ds.classes[i] for i in ds.labels],
            "group": groups,
            "fold": fold,
            "dhash": [f"{int(h):016x}" for h in ds.dhash],
        }
    )
