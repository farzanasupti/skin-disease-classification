"""Fast checks for grouping, splitting, preprocessing and the end-to-end pipeline.

Run with:  KERAS_BACKEND=torch pytest -q tests   (or the tensorflow backend)
The end-to-end test trains tiny randomly initialised models for one epoch on
synthetic images, so it needs no dataset and no GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from skinclf.data import (  # noqa: E402
    augmentation_families,
    build_groups,
    check_no_group_leakage,
    load_folder_dataset,
    make_splits,
    source_stem,
)

CLASSES = ["Nail_psoriasis", "SJS-TEN", "Vitiligo"]


def _make_dataset(root, per_class=24, seed=0):
    """Synthetic classes with distinct colours and textures, plus planted duplicates."""
    rng = np.random.default_rng(seed)
    planted = []
    for split in ("train", "valid"):
        for c, cls in enumerate(CLASSES):
            os.makedirs(os.path.join(root, split, cls), exist_ok=True)
    for c, cls in enumerate(CLASSES):
        for i in range(per_class):
            # Smooth, photo-like random field (low-res noise upsampled), tinted by class.
            field = rng.uniform(0, 90, (6, 6, 3)).astype(np.float32)
            img = cv2.resize(field, (64, 64), interpolation=cv2.INTER_CUBIC).clip(0, 90).astype(np.uint8)
            img[..., c] += 150
            cv2.circle(img, tuple(rng.integers(10, 54, 2).tolist()), int(rng.integers(5, 15)),
                       tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
            split = "valid" if i % 4 == 0 else "train"
            name = f"{cls}_{i}_jpg.rf.{rng.integers(1 << 30):08x}.jpg"
            cv2.imwrite(os.path.join(root, split, cls, name), img)
            if i % 6 == 0:
                # Same original exported twice by a tool, landing in the other split.
                other = "train" if split == "valid" else "valid"
                dup = f"{cls}_{i}_jpg.rf.{rng.integers(1 << 30):08x}.jpg"
                cv2.imwrite(os.path.join(root, other, cls, dup), img)
                planted.append(("name", cls, i))
            if i % 8 == 1:
                # Re-encoded and resized copy under an unrelated name.
                small = cv2.resize(img, (60, 60))
                cv2.imwrite(os.path.join(root, "train", cls, f"copy_{cls}_{i}.png"), small)
                planted.append(("hash", cls, i))
            if i % 8 == 3:
                # Copies the way pre-augmented datasets make them: shrunk onto a
                # black canvas, mirrored, turned upside down, brightened.
                canvas = np.zeros((96, 96, 3), np.uint8)
                canvas[24:72, 24:72] = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
                bright = np.clip(img.astype(np.int16) + 30, 0, 255).astype(np.uint8)
                for kind, variant in (("padded", canvas), ("mirror", cv2.flip(img, 1)),
                                      ("rot180", cv2.rotate(img, cv2.ROTATE_180)), ("bright", bright)):
                    cv2.imwrite(os.path.join(root, "valid", cls, f"copy_{cls}_{i}.{kind}.png"), variant)
                    planted.append((kind, cls, i))
            if i % 8 == 5:
                # Different images on identical black canvases. Each matches its own
                # original, but they must not match one another through the frame.
                canvas = np.zeros((96, 96, 3), np.uint8)
                canvas[24:72, 24:72] = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(root, "train", cls, f"framed_{cls}_{i}.png"), canvas)
                planted.append(("framed", cls, i))
            if i % 8 == 7:
                # Named the way the dataset's augmentation tool names copies: a
                # tight crop and a heavy blur, which a hash cannot tie to the
                # original. Only the filename keeps them in its group.
                crop = cv2.resize(img[8:40, 20:52], (64, 64), interpolation=cv2.INTER_CUBIC)
                blur = cv2.GaussianBlur(img, (0, 0), 6)
                for kind, variant in (("Crop-0350_0300", crop), ("superpixels-100", blur)):
                    cv2.imwrite(os.path.join(root, "train", cls, f"{kind}{cls}_{i}.jpg"), variant)
                    planted.append((kind, cls, i))
    return planted


def test_source_stem_strips_export_suffixes():
    assert source_stem("acne_12_jpg.rf.3f2a9c.jpg") == "acne_12"
    assert source_stem("IMG_0001.jpeg") == "IMG_0001"
    assert source_stem("x_png.rf.ab12.png") == "x"


def test_augmentation_families():
    stems = ["3227", "Crop-0350_03003227", "flip--13227", "padd-00_01003227", "Resize-450_4003227",
             "vitiligo-66", "hue-100vitiligo-66", "27", "superpixels-10027", "Crop-0350_0300999", "3227"]
    labels = ["a"] * 10 + ["b"]
    fam, orig = augmentation_families(stems, labels)
    assert fam[:5] == ["a/3227"] * 5
    assert fam[5:7] == ["a/vitiligo-66"] * 2
    assert fam[7:9] == ["a/27"] * 2  # longest matching original, not a shorter suffix
    assert fam[9] == "a/Crop-0350_0300999"  # prefixed but no original: kept apart
    assert fam[10] == "b/3227"  # families never cross classes
    assert orig.tolist() == [True, False, False, False, False, True, False, True, False, False, True]


def test_grouping_catches_planted_duplicates(tmp_path):
    planted = _make_dataset(tmp_path)
    ds = load_folder_dataset([str(tmp_path / "train"), str(tmp_path / "valid")], 32)
    groups, stats = build_groups(ds)
    # Every planted copy removes exactly one group; nothing else may merge,
    # in particular not the unrelated images framed on identical black canvases.
    assert stats["n_groups"] == len(ds.labels) - len(planted)
    assert stats["groups_with_conflicting_labels"] == 0
    assert stats["links_from_flips_rotations"] >= sum(p[0] in ("mirror", "rot180") for p in planted)
    assert stats["border_cropped_images"] > 0
    for kind, cls, i in planted:
        idx = [k for k, f in enumerate(ds.files)
               if f"{cls}_{i}_jpg" in f or f"copy_{cls}_{i}." in f or f"framed_{cls}_{i}." in f
               or f"0{cls}_{i}." in f]
        assert len(set(groups[idx])) == 1, (kind, cls, i)
    framed = [k for k, f in enumerate(ds.files) if "framed_" in f]
    assert len(framed) > 3 and len(set(groups[framed])) == len(framed)
    n_aug = sum(p[0].startswith(("Crop", "superpixels")) for p in planted)
    assert n_aug > 0 and stats["augmented_copies"] == n_aug
    assert stats["original_images"] == len(ds.labels) - n_aug


def test_hashes_survive_padding_flips_and_brightness():
    from skinclf.data import image_hashes

    rng = np.random.default_rng(3)
    field = rng.uniform(0, 255, (6, 6, 3)).astype(np.float32)
    img = cv2.resize(field, (64, 64), interpolation=cv2.INTER_CUBIC).clip(0, 255).astype(np.uint8)
    canvas = np.zeros((128, 128, 3), np.uint8)
    canvas[32:96, 32:96] = img
    d0, p0, c0 = image_hashes(img)
    d1, p1, c1 = image_hashes(canvas)
    assert not c0 and c1
    ham = lambda a, b: bin(int(a) ^ int(b)).count("1")  # noqa: E731
    assert ham(d0[0], d1[0]) <= 8 and ham(p0[0], p1[0]) <= 8
    dm, pm, _ = image_hashes(cv2.flip(img, 1))
    assert ham(d0[1], dm[0]) <= 8 and ham(p0[1], pm[0]) <= 8


def test_splits_are_grouped_and_cover_everything(tmp_path):
    _make_dataset(tmp_path)
    ds = load_folder_dataset([str(tmp_path / "train"), str(tmp_path / "valid")], 32)
    groups, _ = build_groups(ds)
    fold = make_splits(ds.labels, groups, test_fraction=0.2, n_folds=3, seed=1)
    check_no_group_leakage(groups, fold)
    assert set(np.unique(fold)) == {-1, 0, 1, 2}
    for part in (-1, 0, 1, 2):
        assert len(np.unique(ds.labels[fold == part])) == len(CLASSES)


def test_preprocessing_modes():
    keras = pytest.importorskip("keras")
    from skinclf.models import ImagenetPreprocess

    x = np.full((1, 2, 2, 3), 255.0, dtype="float32")
    x[..., 0] = 0.0  # R = 0, G = B = 255
    out = lambda mode: keras.ops.convert_to_numpy(ImagenetPreprocess(mode)(x))  # noqa: E731
    np.testing.assert_allclose(out("tf")[0, 0, 0], [-1, 1, 1], atol=1e-6)
    np.testing.assert_allclose(out("scale01")[0, 0, 0], [0, 1, 1], atol=1e-6)
    # caffe: BGR order, ImageNet means subtracted -> B=255-103.939, G=255-116.779, R=0-123.68
    np.testing.assert_allclose(out("caffe")[0, 0, 0], [151.061, 138.221, -123.68], atol=1e-3)
    np.testing.assert_allclose(out("none"), x)


def test_mc_dropout_varies_only_when_enabled():
    keras = pytest.importorskip("keras")
    from skinclf.models import build_model, set_mc_dropout

    model = build_model("CustomCNN", 3, img_size=32, pretrained=False)
    x = np.random.default_rng(0).uniform(0, 255, (4, 32, 32, 3)).astype("float32")
    a = keras.ops.convert_to_numpy(model(x, training=False))
    b = keras.ops.convert_to_numpy(model(x, training=False))
    np.testing.assert_allclose(a, b)
    set_mc_dropout(model, True)
    c = keras.ops.convert_to_numpy(model(x, training=False))
    d = keras.ops.convert_to_numpy(model(x, training=False))
    set_mc_dropout(model, False)
    assert not np.allclose(c, d)


@pytest.mark.slow
def test_end_to_end(tmp_path):
    pytest.importorskip("keras")
    _make_dataset(tmp_path / "data", per_class=30)
    out = tmp_path / "out"
    env = {**os.environ}
    common = [
        "--data-roots", str(tmp_path / "data" / "train"), str(tmp_path / "data" / "valid"),
        "--out-dir", str(out), "--img-size", "32", "--n-folds", "3", "--test-fraction", "0.2",
        "--epochs-custom", "2", "--epochs-transfer", "1", "--mc-samples", "3", "--no-pretrained",
    ]
    models = ["--models", "CustomCNN", "MobileNetV2", "EfficientNetB0"]
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_experiment.py"), *common, *models],
                   check=True, env=env)
    # A second invocation must resume (skip finished runs) rather than retrain.
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_experiment.py"), *common, *models],
                       check=True, env=env, capture_output=True, text=True)
    assert "Epoch 1/" not in r.stdout
    paper = tmp_path / "paper"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze.py"), "--out-dir", str(out),
                    "--paper-dir", str(paper), "--n-boot", "50", "--ensemble-size", "2"],
                   check=True, env=env)
    numbers = (paper / "numbers.tex").read_text()
    assert "res@Ensemble@test_accuracy" in numbers
    for t in ("cv", "test", "uncertainty", "efficiency", "mcnemar"):
        assert (paper / "tables" / f"{t}.tex").exists()
    for f in ("confusion", "reliability", "referral", "cv_folds"):
        assert (paper / "figures" / f"{f}.pdf").exists()
