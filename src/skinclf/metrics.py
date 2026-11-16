"""Classification, calibration and uncertainty metrics, with bootstrap CIs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

EPS = 1e-12


def classification_metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(1)
    n_classes = probs.shape[1]
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", labels=range(n_classes), zero_division=0),
        "f1_weighted": f1_score(
            y, pred, average="weighted", labels=range(n_classes), zero_division=0
        ),
        "mcc": matthews_corrcoef(y, pred),
        "kappa": cohen_kappa_score(y, pred),
        "nll": float(-np.mean(np.log(probs[np.arange(len(y)), y] + EPS))),
        "brier": brier(y, probs),
        "ece": ece(y, probs),
    }
    if len(np.unique(y)) == n_classes:
        out["auc_macro_ovr"] = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    return {k: float(v) for k, v in out.items()}


def brier(y: np.ndarray, probs: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ece(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-class confidence (equal-width bins)."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_curve(y, probs, n_bins: int = 15):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    return np.array(rows)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(probs + EPS), axis=1)


def mutual_information(mc_samples: np.ndarray) -> np.ndarray:
    """Epistemic part of the entropy: H[mean p] - mean H[p]. samples: (T, N, C)."""
    mean = mc_samples.mean(0)
    return predictive_entropy(mean) - np.mean(
        [predictive_entropy(s) for s in mc_samples], axis=0
    )


def referral_curve(y, probs, uncertainty, rates=(0.0, 0.05, 0.1, 0.2, 0.3)):
    """Accuracy on the cases kept after referring the most uncertain fraction."""
    order = np.argsort(uncertainty)  # most certain first
    correct = probs.argmax(1)[order] == y[order]
    n = len(y)
    return {r: float(correct[: max(1, int(round(n * (1 - r))))].mean()) for r in rates}


def error_detection_auroc(y, probs, uncertainty) -> float:
    """AUROC of the uncertainty score for separating errors from correct cases."""
    errors = probs.argmax(1) != y
    if errors.all() or not errors.any():
        return float("nan")
    return float(roc_auc_score(errors, uncertainty))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def group_bootstrap_ci(
    y: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    metric_names=("accuracy", "balanced_accuracy", "f1_macro", "mcc", "kappa"),
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs from resampling *groups* (not images) with replacement."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == g)[0] for g in range(len(ug))]
    draws = {m: [] for m in metric_names}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([members[g] for g in pick])
        res = classification_metrics(y[idx], probs[idx])
        for m in metric_names:
            draws[m].append(res[m])
    return {
        m: (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))
        for m, v in draws.items()
    }
