"""Image loading, duplicate-aware grouping, and leakage-free splits.

The dataset has no patient identifiers, so splits are made over *groups of
images that share a source*. Two images land in the same group when either

1. their filenames reduce to the same source stem. Export tools such as
   Roboflow append ``_jpg.rf.<hash>`` to every copy of one original, and the
   augmentation tool used for this dataset names each copy
   ``<operation>-<parameters><original stem>`` (``Crop-0350_03003227`` is a
   crop of ``3227``). The dataset is 341 photographs, each stored with 27 such
   crops, paddings, superpixel renderings, hue and saturation shifts, flips
   and resizes. Crops and superpixel copies do not look alike to a hash, so
   only the filename keeps the family together. Or
2. they look the same: after cropping any uniform border, a 64-bit difference
   hash *and* a 64-bit DCT hash both agree within ``hash_threshold`` bits, for
   the image as it is or for one of its flips or 90-degree rotations.

Rule 2 catches re-encoded, resized, recoloured, padded, mirrored and rotated
copies whatever their names. Each hash alone is not enough: difference hashes
of smooth or black-padded images collide, and chaining such collisions
through union-find merges unrelated images of every class into one group.
Requiring both hashes to agree prevents that.

A group never spans the test set and a training fold, so a near-copy of a
test image cannot be seen during training. Images that are not augmented
copies are flagged as originals; evaluation counts originals only, because the
copies of one photograph are not independent test cases.
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
# <operation>-<numeric parameters>, prefixed to the original stem by the tool
# that pre-augmented the dataset. The parameters run straight into the stem,
# so the stem is recovered by matching against the originals' names.
_AUG_PREFIX = re.compile(r"^(Crop|padd|superpixels|hue|saturation|flip|Resize)-[-\d_]*")


@dataclass
class Dataset:
    images: np.ndarray  # (N, H, W, 3) uint8, RGB
    labels: np.ndarray  # (N,) int class index
    classes: list[str]
    files: np.ndarray  # (N,) relative path, for traceability
    dhash: np.ndarray  # (N, 8) uint64 difference hash of each dihedral variant; [:, 0] = as stored
    phash: np.ndarray  # (N, 8) uint64 DCT hash of each dihedral variant
    border_cropped: np.ndarray  # (N,) bool, a uniform border was removed before hashing
    family: np.ndarray  # (N,) str, "<class>/<original stem>" shared by an original and its augmented copies
    is_original: np.ndarray  # (N,) bool, not an augmented copy


def source_stem(filename: str) -> str:
    """Strip extension and export-tool suffixes to recover the original name."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = _RF_SUFFIX.sub("", stem)
    return _EXT_TOKEN.sub("", stem)


def augmentation_families(stems: list[str], labels: list) -> tuple[list[str], np.ndarray]:
    """Map every image to the original it was made from; flag the originals.

    An image is an augmented copy when its stem is an augmentation prefix
    followed by the stem of an original in the same class (the longest such
    original wins). A prefixed name with no matching original keeps its own
    stem and is not counted as an original.
    """
    originals: dict = {}
    is_aug = [bool(_AUG_PREFIX.match(s)) for s in stems]
    for s, lab, aug in zip(stems, labels, is_aug):
        if not aug:
            originals.setdefault(lab, set()).add(s)
    family = []
    for s, lab, aug in zip(stems, labels, is_aug):
        base = s
        if aug:
            for start in range(len(s)):
                # Shortest prefix = longest original stem; the prefix must be
                # nothing but an operation name and its numeric parameters.
                if s[start:] in originals.get(lab, ()) and _AUG_PREFIX.fullmatch(s[:start]):
                    base = s[start:]
                    break
        family.append(f"{lab}/{base}")
    return family, ~np.asarray(is_aug, dtype=bool)


def _bits_to_u64(bits: np.ndarray) -> np.uint64:
    return np.uint64(int("".join("1" if b else "0" for b in bits.flatten()), 2))


def dhash64(gray: np.ndarray) -> np.uint64:
    """64-bit difference hash of a greyscale image."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return _bits_to_u64(small[:, 1:] > small[:, :-1])


def phash64(gray: np.ndarray) -> np.uint64:
    """64-bit DCT (perceptual) hash: low-frequency coefficients against their median."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coeffs = cv2.dct(small)[:8, :8].flatten()
    return _bits_to_u64(coeffs > np.median(coeffs[1:]))


def crop_uniform_border(gray: np.ndarray, tol: int = 12, min_frac: float = 0.2) -> tuple[np.ndarray, bool]:
    """Remove a uniform (e.g. black) frame around the content, if there is one.

    Copies that were shrunk onto a padded canvas otherwise hash like each other
    rather than like their original.
    """
    g = gray.astype(np.int16)
    edge = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = np.median(edge)
    if np.mean(np.abs(edge - bg) <= tol) < 0.9:
        return gray, False
    content = np.abs(g - bg) > tol
    rows, cols = np.nonzero(content.any(1))[0], np.nonzero(content.any(0))[0]
    if len(rows) == 0:
        return gray, False
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    if r1 - r0 < min_frac * g.shape[0] or c1 - c0 < min_frac * g.shape[1]:
        return gray, False
    return gray[r0:r1, c0:c1], (r1 - r0) * (c1 - c0) < 0.95 * g.size


# The eight symmetries of a square: identity, flips, rotations, transposes.
DIHEDRAL = (
    lambda x: x,
    lambda x: cv2.flip(x, 1),
    lambda x: cv2.flip(x, 0),
    lambda x: cv2.rotate(x, cv2.ROTATE_180),
    lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
    lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
    lambda x: cv2.transpose(x),
    lambda x: cv2.flip(cv2.transpose(x), -1),
)


def image_hashes(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Difference and DCT hashes of the border-cropped image under all eight symmetries."""
    gray, cropped = crop_uniform_border(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    variants = [t(gray) for t in DIHEDRAL]
    d = np.array([dhash64(v) for v in variants], dtype=np.uint64)
    p = np.array([phash64(v) for v in variants], dtype=np.uint64)
    return d, p, cropped


def load_folder_dataset(roots: list[str], img_size: int) -> Dataset:
    """Load every image under ``root/<class>/`` for each root, merged.

    Images are converted to RGB (OpenCV reads BGR) and resized with area
    interpolation. Class names come from the sub-folder names, sorted.
    """
    classes = sorted(
        {d for r in roots for d in os.listdir(r) if os.path.isdir(os.path.join(r, d))}
    )
    cls_idx = {c: i for i, c in enumerate(classes)}
    images, labels, files, dh, ph, cropped = [], [], [], [], [], []
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
                img_dh, img_ph, img_cropped = image_hashes(img)
                dh.append(img_dh)
                ph.append(img_ph)
                cropped.append(img_cropped)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                images.append(img)
                labels.append(cls_idx[cls])
                files.append(os.path.join(os.path.basename(os.path.normpath(root)), cls, f))
    if not images:
        raise FileNotFoundError(f"no images found under {roots}")
    family, is_original = augmentation_families(
        [source_stem(f) for f in files], [classes[l] for l in labels]
    )
    return Dataset(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        classes=classes,
        files=np.asarray(files),
        dhash=np.stack(dh),
        phash=np.stack(ph),
        border_cropped=np.asarray(cropped, dtype=bool),
        family=np.asarray(family),
        is_original=is_original,
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


def _hamming(a: np.uint64, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(b, a).view(np.uint8).reshape(-1, 8)
    return _POPCOUNT8[x].sum(axis=1)


def build_groups(ds: Dataset, hash_threshold: int = 8) -> tuple[np.ndarray, dict]:
    """Assign a group id to every image; return ids and grouping statistics."""
    n = len(ds.labels)
    uf = _UnionFind(n)

    stems = list(ds.family)
    first_seen: dict[str, int] = {}
    name_links = 0
    for i, s in enumerate(stems):
        if s in first_seen:
            uf.union(first_seen[s], i)
            name_links += 1
        else:
            first_seen[s] = i

    # Hashing a flipped image is not exactly flipping the hash (resizing and
    # cropping do not commute perfectly with symmetries), so each pair is tested
    # both ways: variant k of image i against image j as stored, and vice versa.
    def close_to_later(i: int, k: int) -> np.ndarray:
        fwd = (_hamming(ds.dhash[i, k], ds.dhash[i + 1 :, 0]) <= hash_threshold) & (
            _hamming(ds.phash[i, k], ds.phash[i + 1 :, 0]) <= hash_threshold
        )
        if k == 0:
            return fwd
        back = (_hamming(ds.dhash[i, 0], ds.dhash[i + 1 :, k]) <= hash_threshold) & (
            _hamming(ds.phash[i, 0], ds.phash[i + 1 :, k]) <= hash_threshold
        )
        return fwd | back

    hash_links = flip_rotation_links = 0
    if hash_threshold >= 0:
        for i in range(n - 1):
            for k in range(ds.dhash.shape[1]):
                close = close_to_later(i, k)
                for j in np.nonzero(close)[0] + i + 1:
                    if uf.find(i) != uf.find(int(j)):
                        hash_links += 1
                        flip_rotation_links += k > 0
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
        "links_from_flips_rotations": int(flip_rotation_links),
        "border_cropped_images": int(ds.border_cropped.sum()),
        "groups_with_conflicting_labels": int(mixed),
        "original_images": int(ds.is_original.sum()),
        "augmented_copies": int((~ds.is_original).sum()),
        "augmentation_families": int(len(np.unique(ds.family[~ds.is_original]))),
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
            "original": ds.is_original,
            "dhash": [f"{int(h):016x}" for h in ds.dhash[:, 0]],
            "phash": [f"{int(h):016x}" for h in ds.phash[:, 0]],
        }
    )
