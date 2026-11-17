"""Train one (model, fold) and save everything the analysis needs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import keras
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .models import build_model, count_params, set_mc_dropout


@dataclass
class Config:
    data_roots: tuple[str, ...] = ("/kaggle/input/skin-v1/train", "/kaggle/input/skin-v1/valid")
    out_dir: str = "outputs"
    img_size: int = 128
    seed: int = 42
    test_fraction: float = 0.15
    n_folds: int = 5
    hash_threshold: int = 8
    batch_size: int = 32
    lr: float = 1e-3
    l2: float = 1e-4
    epochs_custom: int = 150
    epochs_transfer: int = 50
    early_stop_patience: int = 20
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-7
    mc_samples: int = 30
    pretrained: bool = True
    save_models: bool = False
    mixed_precision: bool = False

    def epochs(self, model_name: str) -> int:
        return self.epochs_custom if model_name == "CustomCNN" else self.epochs_transfer

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class RestoreBestWeights(keras.callbacks.Callback):
    """Keep the weights of the epoch with the lowest validation loss.

    Restores at the end of training whether or not early stopping fired, so
    the saved model always matches ``best_epoch``.
    """

    def __init__(self):
        super().__init__()
        self.best = np.inf
        self.best_epoch = -1
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        loss = (logs or {}).get("val_loss", np.inf)
        if loss < self.best:
            self.best, self.best_epoch = loss, epoch
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def predict(model, x, batch_size):
    return model.predict(x, batch_size=batch_size, verbose=0)


def mc_predict(model, x, n_samples: int, batch_size: int, seed: int) -> np.ndarray:
    """(T, N, C) softmax samples with dropout active and everything else in inference mode."""
    keras.utils.set_random_seed(seed)
    set_mc_dropout(model, True)
    try:
        out = np.empty((n_samples, len(x), model.output_shape[-1]), dtype=np.float32)
        for t in range(n_samples):
            for s in range(0, len(x), batch_size):
                out[t, s : s + batch_size] = keras.ops.convert_to_numpy(
                    model(x[s : s + batch_size], training=False)
                )
        return out
    finally:
        set_mc_dropout(model, False)


def time_inference(model, x, batch_size: int, repeats: int = 3) -> dict[str, float]:
    predict(model, x[:batch_size], batch_size)  # warm-up / graph build
    n = min(len(x), 512)
    best_batched = min(_timed(lambda: predict(model, x[:n], batch_size)) for _ in range(repeats))
    single = x[:1]
    model(single, training=False)
    n1 = 50
    best_single = min(
        _timed(lambda: [model(single, training=False) for _ in range(n1)]) for _ in range(repeats)
    )
    return {
        "ms_per_image_batched": 1000 * best_batched / n,
        "ms_per_image_batch1": 1000 * best_single / n1,
    }


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run_dir(cfg: Config, model_name: str, fold: int) -> str:
    return os.path.join(cfg.out_dir, "runs", f"{model_name}_fold{fold}")


def train_one(
    cfg: Config,
    model_name: str,
    fold: int,
    images: np.ndarray,
    labels: np.ndarray,
    split_fold: np.ndarray,
    n_classes: int,
) -> dict:
    out = run_dir(cfg, model_name, fold)
    done_file = os.path.join(out, "summary.json")
    if os.path.exists(done_file):
        with open(done_file) as f:
            return json.load(f)
    os.makedirs(out, exist_ok=True)

    tr = np.nonzero((split_fold >= 0) & (split_fold != fold))[0]
    va = np.nonzero(split_fold == fold)[0]
    te = np.nonzero(split_fold == -1)[0]
    seed = cfg.seed + 1000 * fold
    keras.utils.set_random_seed(seed)

    model = build_model(
        model_name,
        n_classes,
        img_size=cfg.img_size,
        l2=cfg.l2,
        pretrained=cfg.pretrained,
        augment=True,
        seed=seed,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=labels[tr])
    y_tr = keras.utils.to_categorical(labels[tr], n_classes)
    y_va = keras.utils.to_categorical(labels[va], n_classes)
    x_tr = images[tr].astype("float32")
    x_va = images[va].astype("float32")
    x_te = images[te].astype("float32")

    best = RestoreBestWeights()
    callbacks = [
        best,
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.early_stop_patience, restore_best_weights=False
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr
        ),
        keras.callbacks.CSVLogger(os.path.join(out, "history.csv")),
    ]
    t0 = time.perf_counter()
    history = model.fit(
        x_tr,
        y_tr,
        validation_data=(x_va, y_va),
        epochs=cfg.epochs(model_name),
        batch_size=cfg.batch_size,
        class_weight={i: float(w) for i, w in enumerate(weights)},
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    train_minutes = (time.perf_counter() - t0) / 60

    np.savez_compressed(
        os.path.join(out, "predictions.npz"),
        val_idx=va,
        test_idx=te,
        val_probs=predict(model, x_va, cfg.batch_size),
        test_probs=predict(model, x_te, cfg.batch_size),
        val_mc=mc_predict(model, x_va, cfg.mc_samples, cfg.batch_size, seed + 1),
        test_mc=mc_predict(model, x_te, cfg.mc_samples, cfg.batch_size, seed + 2),
    )
    trainable, total = count_params(model)
    summary = {
        "model": model_name,
        "fold": fold,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "epochs_run": len(history.history["loss"]),
        "best_epoch": best.best_epoch + 1,
        "best_val_loss": float(best.best),
        "train_minutes": train_minutes,
        "params_trainable": trainable,
        "params_total": total,
        **time_inference(model, x_te, cfg.batch_size),
    }
    if cfg.save_models:
        model.save(os.path.join(out, "model.keras"))
    with open(done_file, "w") as f:
        json.dump(summary, f, indent=2)
    keras.backend.clear_session()
    return summary
