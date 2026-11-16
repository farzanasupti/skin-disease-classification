"""Significance tests used to compare models.

* Across cross-validation folds: Friedman test on per-fold scores with mean
  ranks and the Nemenyi critical difference (Demšar, JMLR 2006).
* On the held-out test set: pairwise exact McNemar tests on per-image
  correctness, Holm-corrected for the number of pairs (Dietterich, 1998).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import binomtest, friedmanchisquare, rankdata

# Studentized range statistic / sqrt(2), alpha = 0.05, for k = 2..10 models.
_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def friedman(scores: np.ndarray, names: list[str]) -> dict:
    """scores: (n_folds, n_models), higher is better."""
    n, k = scores.shape
    stat, p = friedmanchisquare(*scores.T)
    ranks = np.vstack([rankdata(-row) for row in scores]).mean(0)
    cd = _NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * n)) if k in _NEMENYI_Q05 else float("nan")
    return {
        "chi2": float(stat),
        "df": k - 1,
        "p": float(p),
        "n_blocks": n,
        "mean_rank": {m: float(r) for m, r in zip(names, ranks)},
        "nemenyi_cd": float(cd),
    }


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, float(p)


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def pairwise_mcnemar(correct: dict[str, np.ndarray]) -> list[dict]:
    pairs = list(combinations(correct, 2))
    raw = [mcnemar_exact(correct[a], correct[b]) for a, b in pairs]
    adj = holm([r[2] for r in raw])
    return [
        {"a": a, "b": b, "a_only_correct": r[0], "b_only_correct": r[1], "p": r[2], "p_holm": pa}
        for (a, b), r, pa in zip(pairs, raw, adj)
    ]
