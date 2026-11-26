#!/usr/bin/env python
"""Can duplicate detectors find the augmented copies in this dataset? (resumable)

    python scripts/dedup_benchmark.py --data-roots <dataset dir> --out-dir outputs_dedup

The dataset is 341 photographs, each stored with 27 augmented copies, and the
filenames say which photograph every copy was made from. That gives ground
truth for a question every dataset without patient identifiers raises: would
a perceptual hash, or a pretrained image embedding, have found the copies?

Detectors (each gives a similarity score for every pair of images):

  dhash, phash   64-bit difference / DCT hash after border cropping, best of
                 the eight flips and 90-degree rotations (Hamming distance)
  dhash+phash    the rule of the main experiment: both hashes agree, for the
                 same flip or rotation
  resnet50       ImageNet ResNet50 (the classifier backbone), pooled features
  dinov2         DINOv2 ViT-S/14, self-supervised, CLS token
  sscd           SSCD (ResNet50, disc_mixup), trained for copy detection

For each detector we report

  pairs      ROC AUC and average precision over all pairs, same photograph
             against not; also against same-class pairs only
  retrieval  for each copy, does its original beat every image of another
             photograph (top1; top10: at most 9 such images ahead), and is it
             above the detector's threshold, by augmentation type
  grouping   union-find groups at the threshold, as a splitting pipeline would
             form them: families kept whole, copies cut off from their
             original, the largest group, and the share of a 1/7 grouped test
             split whose photograph also has an image in training

Embedding thresholds are oracle thresholds, chosen with the ground truth for
99% pair precision, so the grouping results are the best case for each
detector. The hashes are also scored at the main experiment's 8 bits.

Stage 1 (features) caches hashes and embeddings in --out-dir; stage 2
(analyze) reads them. CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skinclf.data import (  # noqa: E402
    IMG_EXTS,
    augmentation_families,
    image_hashes,
    source_stem,
)

HASH_METHODS = ("dhash", "phash", "dhash+phash")
EMB_METHODS = ("resnet50", "dinov2", "sscd")
_OP = re.compile(r"^(Crop|padd|superpixels|hue|saturation|flip|Resize)-")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Stage 1: features
# ---------------------------------------------------------------------------
def list_images(roots: list[str]) -> pd.DataFrame:
    rows = []
    for root in roots:
        for cls in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
            for f in sorted(os.listdir(os.path.join(root, cls))):
                if f.lower().endswith(IMG_EXTS):
                    rows.append({"path": os.path.join(root, cls, f), "cls": cls, "file": f})
    meta = pd.DataFrame(rows)
    stems = [source_stem(f) for f in meta.file]
    family, is_orig = augmentation_families(stems, list(meta.cls))
    meta["family"] = family
    meta["is_original"] = is_orig
    meta["aug"] = [m.group(1) if (m := _OP.match(s)) else "original" for s in stems]
    return meta


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_short(img: np.ndarray, short: int, multiple: int = 1) -> np.ndarray:
    """Aspect-preserving resize to a short side of ``short`` (rounded to ``multiple``)."""
    h, w = img.shape[:2]
    s = short / min(h, w)
    nh = max(multiple, int(round(h * s / multiple)) * multiple)
    nw = max(multiple, int(round(w * s / multiple)) * multiple)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def to_tensor(img: np.ndarray):
    import torch

    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())[None]


def compute_hashes(meta: pd.DataFrame) -> dict:
    d, p = [], []
    for path in meta.path:
        dh, ph, _ = image_hashes(read_rgb(path))
        d.append(dh)
        p.append(ph)
    return {"dhash": np.stack(d), "phash": np.stack(p)}


def compute_embeddings(meta: pd.DataFrame, method: str, weights_dir: str) -> np.ndarray:
    feats = []
    if method == "resnet50":
        import keras

        model = keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg")
        pre = keras.applications.resnet50.preprocess_input
        batch = []
        for i, path in enumerate(meta.path):
            # Squashed to 224 x 224, as the classifiers see the images.
            batch.append(cv2.resize(read_rgb(path), (224, 224), interpolation=cv2.INTER_AREA))
            if len(batch) == 32 or i == len(meta) - 1:
                x = pre(np.stack(batch).astype(np.float32))
                feats.append(np.asarray(model.predict(x, verbose=0)))
                batch = []
        out = np.concatenate(feats)
    else:
        import torch

        if method == "dinov2":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            prep = lambda img: resize_short(img, 224, multiple=14)  # noqa: E731
        elif method == "sscd":
            # Short side 288, aspect kept, as in the SSCD reference code.
            model = torch.jit.load(os.path.join(weights_dir, "sscd_disc_mixup.torchscript.pt"))
            prep = lambda img: resize_short(img, 288)  # noqa: E731
        else:
            raise ValueError(method)
        model.eval()
        with torch.no_grad():
            for path in meta.path:
                feats.append(model(to_tensor(prep(read_rgb(path)))).numpy()[0])
        out = np.stack(feats)
    out = out.astype(np.float32)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Stage 2: pair scores
# ---------------------------------------------------------------------------
def _ham(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a[:, None] ^ b[None, :]).astype(np.int16)


def hash_scores(d: np.ndarray, p: np.ndarray, rows: np.ndarray) -> dict:
    """Similarity (minus Hamming distance) of ``rows`` against every image.

    Tested both ways, variant k of image i against image j as stored and vice
    versa, as in the main experiment's grouping.
    """
    best = {m: None for m in HASH_METHODS}
    for k in range(d.shape[1]):
        for dd, pp in (
            (_ham(d[rows, k], d[:, 0]), _ham(p[rows, k], p[:, 0])),
            (_ham(d[rows, 0], d[:, k]), _ham(p[rows, 0], p[:, k])),
        ):
            for m, v in (("dhash", dd), ("phash", pp), ("dhash+phash", np.maximum(dd, pp))):
                best[m] = v if best[m] is None else np.minimum(best[m], v)
    return {m: -v.astype(np.float32) for m, v in best.items()}


def score_bins(method: str) -> np.ndarray:
    """Histogram edges: one bin per Hamming distance, fine bins for cosine."""
    if method in HASH_METHODS:
        return np.arange(-64.5, 1.0, 1.0)
    return np.linspace(-1.0, 1.0 + 1e-6, 4001)


def pr_from_hist(pos: np.ndarray, neg: np.ndarray, edges: np.ndarray) -> dict:
    """Pair metrics from score histograms, sweeping the threshold from high to low."""
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    P, N = tp[-1], fp[-1]
    recall = tp / P
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    tpr = np.concatenate([[0.0], recall])
    fpr = np.concatenate([[0.0], fp / N])
    auc = float(np.trapezoid(tpr, fpr))
    lower = edges[:-1][::-1]  # threshold = lower edge of the lowest bin included

    def at_precision(target):
        ok = np.nonzero(precision >= target)[0]
        if len(ok) == 0:
            return None, 0.0
        i = ok[np.argmax(recall[ok])]
        return float(lower[i]), float(recall[i])

    t99, r99 = at_precision(0.99)
    t90, r90 = at_precision(0.90)
    return {
        "n_pos": int(P), "n_neg": int(N), "ap": ap, "roc_auc": auc,
        "thr_p99": t99, "recall_p99": r99, "thr_p90": t90, "recall_p90": r90,
    }


def components(edges_i: list, edges_j: list, n: int) -> np.ndarray:
    i = np.concatenate(edges_i) if edges_i else np.zeros(0, int)
    j = np.concatenate(edges_j) if edges_j else np.zeros(0, int)
    g = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def with_photo(meta: pd.DataFrame) -> pd.DataFrame:
    """Default the photograph to the filename family when no merge was given."""
    if "photo" not in meta:
        meta = meta.assign(photo=meta.family)
    return meta


def merge_same_photo(meta: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Put the families of originals listed as the same photograph together.

    ``pairs`` has columns ``file_a`` and ``file_b``, filenames of originals.
    """
    fam_of = dict(zip(meta.file[meta.is_original], meta.family[meta.is_original]))
    parent = {f: f for f in meta.family.unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs.file_a, pairs.file_b):
        parent[find(fam_of[a])] = find(fam_of[b])
    return meta.assign(photo=[find(f) for f in meta.family])


def grouping_metrics(groups: np.ndarray, meta: pd.DataFrame, seeds=range(5)) -> dict:
    meta = with_photo(meta)
    fam = meta.photo.to_numpy()
    src = meta.family.to_numpy()
    cls = meta.cls.to_numpy()
    orig_idx = {f: i for i, f in enumerate(src) if meta.is_original.iat[i]}
    copies = np.nonzero(~meta.is_original.to_numpy())[0]
    cut = np.mean([groups[c] != groups[orig_idx[src[c]]] for c in copies])
    df = pd.DataFrame({"fam": fam, "g": groups})
    intact = float((df.groupby("fam").g.nunique() == 1).mean())
    sizes = np.bincount(groups)
    mixed_cls = pd.DataFrame({"g": groups, "c": cls}).groupby("g").c.nunique()
    leak_all, leak_orig = [], []
    y = pd.factorize(cls)[0]
    for s in seeds:
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=s)
        _, te = next(sgkf.split(np.zeros(len(y)), y, groups))
        is_te = np.zeros(len(y), bool)
        is_te[te] = True
        train_fams = set(fam[~is_te])
        leaked = np.array([f in train_fams for f in fam[is_te]])
        leak_all.append(leaked.mean())
        leak_orig.append(leaked[meta.is_original.to_numpy()[is_te]].mean())
    return {
        "n_groups": int(len(sizes)),
        "largest_group": int(sizes.max()),
        "groups_with_several_classes": int((mixed_cls > 1).sum()),
        "families_intact": intact,
        "copies_cut_from_original": float(cut),
        "test_images_with_photo_in_train": float(np.mean(leak_all)),
        "test_originals_with_photo_in_train": float(np.mean(leak_orig)),
    }


def analyze(meta: pd.DataFrame, feats: dict, chunk: int = 256) -> dict:
    n = len(meta)
    meta = with_photo(meta)
    fam_codes = pd.factorize(meta.photo)[0]  # same photograph: the positives
    src_codes = pd.factorize(meta.family)[0]  # the original each copy was made from
    cls_codes = pd.factorize(meta.cls)[0]
    orig_of = np.full(n, -1)
    first = {}
    for i in np.nonzero(meta.is_original.to_numpy())[0]:
        first[src_codes[i]] = i
    for i in range(n):
        if not meta.is_original.iat[i]:
            orig_of[i] = first[src_codes[i]]
    methods = (list(HASH_METHODS) if "dhash" in feats else []) + [m for m in EMB_METHODS if m in feats]
    edges = {m: score_bins(m) for m in methods}
    hist = {m: {k: np.zeros(len(edges[m]) - 1, np.int64) for k in ("pos", "neg", "neg_cls")} for m in methods}
    copy_score = {m: np.full(n, np.nan, np.float32) for m in methods}
    copy_rank = {m: np.full(n, -1, np.int64) for m in methods}

    def scores_for(rows):
        out = {}
        if "dhash" in feats:
            out.update(hash_scores(feats["dhash"], feats["phash"], rows))
        for m in EMB_METHODS:
            if m in feats:
                out[m] = feats[m][rows] @ feats[m].T
        return out

    # Pass 1: histograms over pairs i < j, and each copy's rank of its original.
    for start in range(0, n, chunk):
        rows = np.arange(start, min(n, start + chunk))
        sc = scores_for(rows)
        upper = np.arange(n)[None, :] > rows[:, None]
        same = fam_codes[rows][:, None] == fam_codes[None, :]
        same_cls = cls_codes[rows][:, None] == cls_codes[None, :]
        for m in methods:
            s = sc[m]
            hist[m]["pos"] += np.histogram(s[upper & same], edges[m])[0]
            hist[m]["neg"] += np.histogram(s[upper & ~same], edges[m])[0]
            hist[m]["neg_cls"] += np.histogram(s[upper & ~same & same_cls], edges[m])[0]
            for r, i in enumerate(rows):
                o = orig_of[i]
                if o < 0:
                    continue
                v = s[r, o]
                copy_score[m][i] = v
                # Images of other photographs that score at least as high
                # (ties count against the detector). Siblings are not misses.
                copy_rank[m][i] = int(np.sum(s[r][~same[r]] >= v))

    results = {"pairs": {}, "pairs_same_class": {}, "thresholds": {}}
    for m in methods:
        results["pairs"][m] = pr_from_hist(hist[m]["pos"], hist[m]["neg"], edges[m])
        results["pairs_same_class"][m] = pr_from_hist(hist[m]["pos"], hist[m]["neg_cls"], edges[m])
        thr = {"p99": results["pairs"][m]["thr_p99"]}
        if m in HASH_METHODS:
            thr["8bits"] = -8.0
        results["thresholds"][m] = thr

    # Retrieval by augmentation type.
    rows = []
    copies = ~meta.is_original.to_numpy()
    for m in methods:
        for aug in sorted(meta.aug[copies].unique()) + ["all copies"]:
            sel = copies & ((meta.aug == aug).to_numpy() if aug != "all copies" else True)
            row = {"method": m, "aug": aug, "n": int(sel.sum()),
                   "top1": float(np.mean(copy_rank[m][sel] == 0)),
                   "top10": float(np.mean(copy_rank[m][sel] < 10))}
            for name, t in results["thresholds"][m].items():
                if t is not None:
                    row[f"above_{name}"] = float(np.mean(copy_score[m][sel] >= t))
            rows.append(row)
    retrieval = pd.DataFrame(rows)

    # Pass 2: groups at each threshold, all detectors in one sweep.
    todo = [(m, name, t) for m in methods for name, t in results["thresholds"][m].items() if t is not None]
    ei = {key[:2]: [] for key in todo}
    ej = {key[:2]: [] for key in todo}
    for start in range(0, n, chunk):
        r = np.arange(start, min(n, start + chunk))
        sc = scores_for(r)
        upper = np.arange(n)[None, :] > r[:, None]
        for m, name, t in todo:
            a, b = np.nonzero((sc[m] >= t) & upper)
            ei[m, name].append(r[a])
            ej[m, name].append(b)
    grouping = []
    for m, name, t in todo:
        g = components(ei[m, name], ej[m, name], n)
        grouping.append({"method": m, "threshold": name, "value": t, **grouping_metrics(g, meta)})
    # Reference points: one group per image (an image split) and per family.
    grouping.append({"method": "none (image split)", "threshold": "-", "value": None,
                     **grouping_metrics(np.arange(n), meta)})
    grouping.append({"method": "filenames", "threshold": "-", "value": None,
                     **grouping_metrics(src_codes, meta)})
    if (fam_codes != src_codes).any():
        grouping.append({"method": "filenames + same-photo list", "threshold": "-", "value": None,
                         **grouping_metrics(fam_codes, meta)})
    return results, retrieval, pd.DataFrame(grouping)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-roots", nargs="+", required=True)
    ap.add_argument("--out-dir", default="outputs_dedup")
    ap.add_argument("--weights-dir", default="weights", help="holds sscd_disc_mixup.torchscript.pt")
    ap.add_argument("--methods", nargs="+", default=["hashes", *EMB_METHODS])
    ap.add_argument("--stage", choices=["features", "analyze", "paper", "all"], default="all")
    ap.add_argument("--paper-dir", help="write numbers_dedup.tex and tables/dedup.tex here (stage paper or all)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--same-photo", help="CSV of originals (file_a, file_b) found by eye to be one photograph")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        import torch

        torch.set_num_threads(args.threads)
    except ImportError:
        pass

    meta_path = os.path.join(args.out_dir, "images.csv")
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
    else:
        meta = list_images(args.data_roots)
        meta.to_csv(meta_path, index=False)
    if args.same_photo:
        meta = merge_same_photo(meta, pd.read_csv(args.same_photo))
        print(f"{meta.family.nunique() - meta.photo.nunique()} families merged as the same photograph")
    print(f"{len(meta)} images, {meta.is_original.sum()} originals, "
          f"{meta.family.nunique()} families", flush=True)

    if args.stage in ("features", "all"):
        for m in args.methods:
            path = os.path.join(args.out_dir, f"features_{m}.npz")
            if os.path.exists(path):
                continue
            t = time.time()
            out = compute_hashes(meta) if m == "hashes" else {m: compute_embeddings(meta, m, args.weights_dir)}
            np.savez(path, **out)
            print(f"{m}: {time.time() - t:.0f} s", flush=True)

    if args.stage in ("analyze", "all"):
        feats = {}
        for m in args.methods:
            path = os.path.join(args.out_dir, f"features_{m}.npz")
            if os.path.exists(path):
                feats.update(dict(np.load(path)))
        results, retrieval, grouping = analyze(meta, feats)
        with open(os.path.join(args.out_dir, "pairs.json"), "w") as f:
            json.dump(results, f, indent=2)
        retrieval.to_csv(os.path.join(args.out_dir, "retrieval.csv"), index=False)
        grouping.to_csv(os.path.join(args.out_dir, "grouping.csv"), index=False)
        pd.set_option("display.width", 200)
        print(pd.DataFrame(results["pairs"]).T.round(4))
        print(pd.DataFrame(results["pairs_same_class"]).T[["ap", "roc_auc", "recall_p99"]].round(4))
        print(retrieval.round(3).to_string(index=False))
        print(grouping.round(3).to_string(index=False))

    if args.paper_dir and args.stage in ("paper", "all"):
        write_paper(args.out_dir, args.paper_dir)


# ---------------------------------------------------------------------------
# Paper output: \\res{dedup-<method>}{<metric>} and an S1 table
# ---------------------------------------------------------------------------
PAPER_METHODS = {"dhash": "dHash", "phash": "pHash", "dhash+phash": "dHash + pHash$^a$",
                 "resnet50": "ResNet50 (ImageNet)", "dinov2": "DINOv2 ViT-S/14", "sscd": "SSCD (copy detection)"}


def write_paper(out_dir, paper_dir):
    with open(os.path.join(out_dir, "pairs.json")) as f:
        pairs = json.load(f)["pairs"]
    retrieval = pd.read_csv(os.path.join(out_dir, "retrieval.csv"))
    grouping = pd.read_csv(os.path.join(out_dir, "grouping.csv"))
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    top1 = retrieval.pivot(index="method", columns="aug", values="top1")
    p99 = grouping[grouping.threshold == "p99"].set_index("method")
    lines = ["% Generated by scripts/dedup_benchmark.py --stage paper\n"]
    mac = lambda k, m, v: lines.append(f"\\expandafter\\def\\csname res@dedup-{k}@{m}\\endcsname{{{v}}}\n")  # noqa: E731
    rows = []
    for m, label in PAPER_METHODS.items():
        if m not in pairs:
            continue
        g = p99.loc[m]
        vals = {"ap": f"{pairs[m]['ap']:.2f}", "auc": f"{pairs[m]['roc_auc']:.2f}",
                "crop_top1": pct(top1.loc[m, "Crop"]), "superpixel_top1": pct(top1.loc[m, "superpixels"]),
                "families_intact": pct(g.families_intact),
                "test_leak": pct(g.test_originals_with_photo_in_train)}
        for k, v in vals.items():
            mac(m, k, v)
        rows.append(f"{label} & {vals['ap']} & {vals['auc']} & {vals['crop_top1']} & {vals['superpixel_top1']} & "
                    f"{vals['families_intact']} & {vals['test_leak']} \\\\\n")
    emb = [m for m in ("resnet50", "dinov2", "sscd") if m in pairs]
    mac("all", "max_superpixel_top1", pct(top1.loc[list(PAPER_METHODS)].reindex(columns=["superpixels"]).max().iloc[0]))
    mac("all", "min_test_leak", pct(p99.loc[[m for m in PAPER_METHODS if m in p99.index], "test_originals_with_photo_in_train"].min()))
    mac("all", "n_detectors", len(rows))
    mac("all", "n_embeddings", len(emb))
    fn = grouping.set_index("method").loc["filenames"]
    mac("filenames", "families_intact", pct(fn.families_intact))
    mac("filenames", "test_leak", pct(fn.test_originals_with_photo_in_train))
    os.makedirs(os.path.join(paper_dir, "tables"), exist_ok=True)
    with open(os.path.join(paper_dir, "numbers_dedup.tex"), "w") as f:
        f.writelines(lines)
    tex = ("% Generated by scripts/dedup_benchmark.py --stage paper\n"
           "\\begin{table}[!ht]\n\\centering\n"
           "\\caption{\\textbf{Could a duplicate detector have found the copies?} The filenames give the true "
           "photograph of every image, so each detector can be scored against them. Pairs: average precision (AP) and "
           "ROC AUC for telling pairs from the same photograph from all other pairs. Top-1: share of crops and of "
           "superpixel renderings whose own original is the detector's nearest image. Grouping: union--find groups at "
           "the threshold giving 99\\% pair precision (chosen with the true labels, so a best case), as a splitting "
           "pipeline would form them; share of photographs kept whole, and share of a one-seventh grouped test split "
           "whose photograph also has an image in training.}\n\\label{tab:dedup}\n\\small\n"
           "\\begin{tabular}{lrrrrrr}\n\\hline\n"
           "& \\multicolumn{2}{c}{\\textbf{Pairs}} & \\multicolumn{2}{c}{\\textbf{Top-1 (\\%)}} & "
           "\\multicolumn{2}{c}{\\textbf{Grouping (\\%)}} \\\\\n"
           "\\textbf{Detector} & \\textbf{AP} & \\textbf{AUC} & \\textbf{Crops} & \\textbf{Superpixels} & "
           "\\textbf{Kept whole} & \\textbf{Test leaked} \\\\\n\\hline\n" + "".join(rows) +
           "\\hline\n\\end{tabular}\n\\begin{flushleft}\n$^a$ The rule of the main pipeline's hash links "
           "(both hashes agree).\n\\end{flushleft}\n\\end{table}\n")
    with open(os.path.join(paper_dir, "tables", "dedup.tex"), "w") as f:
        f.write(tex)


if __name__ == "__main__":
    main()
