#!/usr/bin/env python
"""Turn saved predictions into every number, table and figure in the paper.

    python scripts/analyze.py --out-dir outputs --paper-dir paper/generated

Writes CSV/JSON to ``<out-dir>/results`` and, for the manuscript, LaTeX
tables plus ``numbers.tex``: a ``\\res{<key>}{<metric>}`` lookup so that no
result is typed by hand.

Everything that chooses something (the best model, the ensemble members) is
decided on cross-validation folds only; the held-out test set is used for
reporting, never for selection.

Validation and test metrics count original photographs only. Training and
early stopping use every image, but the 27 augmented copies of a photograph are
not independent cases, so scoring them would multiply one photograph's result
28-fold and narrow the confidence intervals without new evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skinclf import metrics as M  # noqa: E402
from skinclf import stats as S  # noqa: E402
from skinclf.models import MODEL_NAMES  # noqa: E402

ENSEMBLE = "Ensemble"
REFERRAL_RATES = (0.0, 0.05, 0.1, 0.2, 0.3)


def load_runs(out_dir: str, n_folds: int):
    runs = {}
    for name in MODEL_NAMES:
        per_fold = {}
        for k in range(n_folds):
            d = os.path.join(out_dir, "runs", f"{name}_fold{k}")
            if os.path.exists(os.path.join(d, "summary.json")):
                with open(os.path.join(d, "summary.json")) as f:
                    summary = json.load(f)
                pred = dict(np.load(os.path.join(d, "predictions.npz")))
                per_fold[k] = {"summary": summary, **pred}
        if per_fold:
            runs[name] = per_fold
    return runs


def fmt_pct(x):
    return f"{100 * x:.1f}"


def fmt3(x):
    return f"{x:.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--paper-dir", default="paper/generated")
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    res_dir = os.path.join(args.out_dir, "results")
    fig_dir = os.path.join(args.paper_dir, "figures")
    tab_dir = os.path.join(args.paper_dir, "tables")
    for d in (res_dir, fig_dir, tab_dir):
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(args.out_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(args.out_dir, "classes.json")) as f:
        classes = json.load(f)
    split = pd.read_csv(os.path.join(args.out_dir, "splits.csv"))
    labels = split["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = split["group"].to_numpy()
    if "original" not in split.columns:
        sys.exit("splits.csv has no 'original' column: it predates grouping by augmentation family.")
    original = split["original"].astype(bool).to_numpy()
    n_folds = cfg["n_folds"]

    runs = load_runs(args.out_dir, n_folds)
    complete = [m for m in runs if len(runs[m]) == n_folds]
    partial = {m: sorted(runs[m]) for m in runs if m not in complete}
    if partial:
        print(f"WARNING: incomplete models left out of comparisons: {partial}")
    if len(complete) < 2:
        sys.exit("Need at least two models with all folds finished.")

    # Score originals only: restrict saved predictions to them (val per fold, test once).
    for per_fold in runs.values():
        for r in per_fold.values():
            keep = original[r["val_idx"]]
            r["val_idx"], r["val_probs"] = r["val_idx"][keep], r["val_probs"][keep]
            if "val_mc" in r:
                r["val_mc"] = r["val_mc"][:, keep]
            keep = original[r["test_idx"]]
            r["test_idx"], r["test_probs"] = r["test_idx"][keep], r["test_probs"][keep]
            r["test_mc"] = r["test_mc"][:, keep]
    test_idx = runs[complete[0]][0]["test_idx"]
    y_test, g_test = labels[test_idx], groups[test_idx]

    # ------------------------------------------------------------------
    # Cross-validation (validation folds)
    # ------------------------------------------------------------------
    cv_rows = []
    for m in complete:
        for k, r in runs[m].items():
            yv = labels[r["val_idx"]]
            cv_rows.append({"model": m, "fold": k, **M.classification_metrics(yv, r["val_probs"])})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(os.path.join(res_dir, "cv_per_fold.csv"), index=False)
    cv_mean = cv.groupby("model").mean(numeric_only=True).drop(columns="fold")
    cv_std = cv.groupby("model").std(numeric_only=True).drop(columns="fold")
    order = cv_mean["f1_macro"].sort_values(ascending=False).index.tolist()
    best_single = order[0]
    members = order[: args.ensemble_size]

    fried = S.friedman(
        cv.pivot(index="fold", columns="model", values="f1_macro")[order].to_numpy(), order
    )

    # ------------------------------------------------------------------
    # Held-out test: per-model fold ensembles, MC dropout, cross-model ensemble
    # ------------------------------------------------------------------
    det, mc, mc_samples = {}, {}, {}
    single_fold_test = []
    for m in complete:
        assert all(np.array_equal(r["test_idx"], test_idx) for r in runs[m].values())
        det[m] = np.mean([r["test_probs"].astype(np.float64) for r in runs[m].values()], axis=0)
        det[m] /= det[m].sum(-1, keepdims=True)
        samples = np.concatenate([r["test_mc"].astype(np.float64) for r in runs[m].values()])
        samples /= samples.sum(-1, keepdims=True)
        mc_samples[m] = samples
        mc[m] = samples.mean(0)
        for k, r in runs[m].items():
            single_fold_test.append(
                {"model": m, "fold": k, **M.classification_metrics(y_test, r["test_probs"])}
            )
    det[ENSEMBLE] = np.mean([det[m] for m in members], axis=0)
    mc_samples[ENSEMBLE] = np.concatenate([mc_samples[m] for m in members])
    mc[ENSEMBLE] = np.mean([mc[m] for m in members], axis=0)
    single = pd.DataFrame(single_fold_test)
    single.to_csv(os.path.join(res_dir, "test_single_fold_models.csv"), index=False)

    report = order + [ENSEMBLE]
    test_rows, unc_rows = [], []
    for m in report:
        met = M.classification_metrics(y_test, det[m])
        ci = M.group_bootstrap_ci(y_test, det[m], g_test, n_boot=args.n_boot, seed=0)
        row = {"model": m, **met}
        for metric, (lo, hi) in ci.items():
            row[f"{metric}_lo"], row[f"{metric}_hi"] = lo, hi
        if m != ENSEMBLE:
            row["single_fold_accuracy_mean"] = single[single.model == m]["accuracy"].mean()
            row["single_fold_accuracy_std"] = single[single.model == m]["accuracy"].std()
        test_rows.append(row)

        mc_met = M.classification_metrics(y_test, mc[m])
        h_det = M.predictive_entropy(det[m])
        h_mc = M.predictive_entropy(mc[m])
        mi = M.mutual_information(mc_samples[m])
        unc = {
            "model": m,
            "accuracy_det": met["accuracy"],
            "accuracy_mc": mc_met["accuracy"],
            "ece_det": met["ece"],
            "ece_mc": mc_met["ece"],
            "nll_det": met["nll"],
            "nll_mc": mc_met["nll"],
            "brier_det": met["brier"],
            "brier_mc": mc_met["brier"],
            "err_auroc_softmax_entropy": M.error_detection_auroc(y_test, det[m], h_det),
            "err_auroc_mc_entropy": M.error_detection_auroc(y_test, mc[m], h_mc),
            "err_auroc_mutual_info": M.error_detection_auroc(y_test, mc[m], mi),
        }
        for rate, acc in M.referral_curve(y_test, mc[m], h_mc, REFERRAL_RATES).items():
            unc[f"acc_refer_{int(rate * 100)}"] = acc
        unc_rows.append(unc)
    test = pd.DataFrame(test_rows).set_index("model")
    unc = pd.DataFrame(unc_rows).set_index("model")
    test.to_csv(os.path.join(res_dir, "test_metrics.csv"))
    unc.to_csv(os.path.join(res_dir, "uncertainty.csv"))

    correct = {m: det[m].argmax(1) == y_test for m in report}
    mcn = pd.DataFrame(S.pairwise_mcnemar(correct))
    mcn.to_csv(os.path.join(res_dir, "mcnemar.csv"), index=False)

    eff = []
    for m in order:
        s = pd.DataFrame([r["summary"] for r in runs[m].values()])
        eff.append(
            {
                "model": m,
                "params_trainable": int(s.params_trainable.iloc[0]),
                "params_total": int(s.params_total.iloc[0]),
                "train_minutes_mean": s.train_minutes.mean(),
                "best_epoch_mean": s.best_epoch.mean(),
                "epochs_run_mean": s.epochs_run.mean(),
                "ms_per_image_batched": s.ms_per_image_batched.median(),
                "ms_per_image_batch1": s.ms_per_image_batch1.median(),
            }
        )
    eff = pd.DataFrame(eff).set_index("model")
    eff.to_csv(os.path.join(res_dir, "efficiency.csv"))

    per_class = {}
    for m in (best_single, ENSEMBLE):
        pr, rc, f1, sup = precision_recall_fscore_support(
            y_test, det[m].argmax(1), labels=range(len(classes)), zero_division=0
        )
        per_class[m] = pd.DataFrame(
            {"precision": pr, "recall": rc, "f1": f1, "support": sup}, index=classes
        )
        per_class[m].to_csv(os.path.join(res_dir, f"per_class_{m}.csv"))

    summary = {
        "models_complete": complete,
        "models_partial": partial,
        "cv_order_by_macro_f1": order,
        "best_single_model": best_single,
        "ensemble_members": members,
        "friedman": fried,
        "n_test_images": int(len(test_idx)),
        "n_test_groups": int(len(np.unique(g_test))),
    }
    with open(os.path.join(res_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Paper artefacts
    # ------------------------------------------------------------------
    write_numbers(
        os.path.join(args.paper_dir, "numbers.tex"),
        cfg, split, classes, test, unc, eff, cv_mean, cv_std, fried, mcn, summary, args.out_dir,
    )
    write_tables(tab_dir, order, test, unc, eff, cv_mean, cv_std, per_class, classes, mcn)
    make_figures(fig_dir, y_test, det, mc, best_single, classes, cv, order)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------
def _macro(key, metric, value):
    return f"\\expandafter\\def\\csname res@{key}@{metric}\\endcsname{{{value}}}\n"


def write_numbers(path, cfg, split, classes, test, unc, eff, cv_mean, cv_std, fried, mcn, summary, out_dir):
    lines = [
        "% Generated by scripts/analyze.py. Do not edit by hand.\n",
        "\\providecommand{\\res}[2]{\\csname res@#1@#2\\endcsname}\n",
    ]
    stats_path = os.path.join(out_dir, "grouping_stats.json")
    grouping = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    data = {
        "n_images": len(split),
        "n_classes": len(classes),
        "n_groups": split["group"].nunique(),
        "n_test_images": int(((split.fold == -1) & split.original).sum()),
        "n_dev_images": int(((split.fold >= 0) & split.original).sum()),
        "n_test_images_with_copies": int((split.fold == -1).sum()),
        "n_dev_images_with_copies": int((split.fold >= 0).sum()),
        "n_folds": cfg["n_folds"],
        "mc_samples": cfg["mc_samples"],
        **{k: v for k, v in grouping.items() if isinstance(v, int)},
    }
    for k, v in data.items():
        lines.append(_macro("data", k, f"{v:,}".replace(",", "{,}")))

    for m in test.index:
        key = m
        t = test.loc[m]
        for metric in ("accuracy", "balanced_accuracy", "f1_macro", "auc_macro_ovr"):
            if metric in t:
                lines.append(_macro(key, f"test_{metric}", fmt_pct(t[metric])))
        for metric in ("mcc", "kappa"):
            lines.append(_macro(key, f"test_{metric}", fmt3(t[metric])))
        for metric in ("accuracy", "f1_macro", "balanced_accuracy"):
            lines.append(_macro(key, f"test_{metric}_ci", f"{fmt_pct(t[metric + '_lo'])}--{fmt_pct(t[metric + '_hi'])}"))
        lines.append(_macro(key, "test_mcc_ci", f"{fmt3(t['mcc_lo'])}--{fmt3(t['mcc_hi'])}"))
        u = unc.loc[m]
        for metric in ("ece_det", "ece_mc"):
            lines.append(_macro(key, metric, fmt_pct(u[metric])))
        for metric in ("err_auroc_softmax_entropy", "err_auroc_mc_entropy", "err_auroc_mutual_info"):
            lines.append(_macro(key, metric, fmt3(u[metric])))
        for rate in REFERRAL_RATES:
            lines.append(_macro(key, f"acc_refer_{int(rate * 100)}", fmt_pct(u[f"acc_refer_{int(rate * 100)}"])))
        if m in cv_mean.index:
            for metric in ("accuracy", "f1_macro", "balanced_accuracy"):
                lines.append(_macro(key, f"cv_{metric}", fmt_pct(cv_mean.loc[m, metric])))
                lines.append(_macro(key, f"cv_{metric}_sd", fmt_pct(cv_std.loc[m, metric])))
            e = eff.loc[m]
            lines.append(_macro(key, "params_trainable", f"{int(e.params_trainable):,}".replace(",", "{,}")))
            lines.append(_macro(key, "params_total_m", f"{e.params_total / 1e6:.2f}"))
            lines.append(_macro(key, "ms_batched", f"{e.ms_per_image_batched:.1f}"))

    order = summary["cv_order_by_macro_f1"]
    lines.append(f"\\def\\bestmodel{{{summary['best_single_model']}}}\n")
    lines.append(f"\\def\\worstmodel{{{order[-1]}}}\n")
    lines.append(_macro("summary", "best", summary["best_single_model"]))
    lines.append(_macro("summary", "worst", order[-1]))
    lines.append(_macro("summary", "members", ", ".join(summary["ensemble_members"])))
    lines.append(_macro("summary", "n_members", len(summary["ensemble_members"])))
    lines.append(_macro("summary", "friedman_chi2", f"{fried['chi2']:.2f}"))
    lines.append(_macro("summary", "friedman_df", fried["df"]))
    lines.append(_macro("summary", "friedman_p", _pfmt(fried["p"])))
    lines.append(_macro("summary", "nemenyi_cd", f"{fried['nemenyi_cd']:.2f}"))
    ens = mcn[(mcn.a == ENSEMBLE) | (mcn.b == ENSEMBLE)]
    ens_best = ens[(ens.a == summary["best_single_model"]) | (ens.b == summary["best_single_model"])]
    if len(ens_best):
        lines.append(_macro("summary", "mcnemar_ens_vs_best_p", _pfmt(ens_best.p_holm.iloc[0])))
    lines.append(_macro("summary", "n_sig_pairs", int((mcn.p_holm < 0.05).sum())))
    lines.append(_macro("summary", "n_pairs", len(mcn)))
    with open(path, "w") as f:
        f.writelines(lines)


def _pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _tex(df: pd.DataFrame, path: str, caption: str, label: str, colfmt: str):
    body = df.to_latex(index=False, escape=False, column_format=colfmt)
    with open(path, "w") as f:
        f.write(
            "% Generated by scripts/analyze.py. Do not edit by hand.\n"
            "\\begin{table}[ht]\n\\centering\n\\small\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n{body}\\end{{table}}\n"
        )


def write_tables(tab_dir, order, test, unc, eff, cv_mean, cv_std, per_class, classes, mcn):
    pm = lambda m, s: f"{fmt_pct(m)} $\\pm$ {fmt_pct(s)}"  # noqa: E731
    cv_tab = pd.DataFrame(
        {
            "Model": order,
            "Accuracy (\\%)": [pm(cv_mean.loc[m, "accuracy"], cv_std.loc[m, "accuracy"]) for m in order],
            "Balanced acc. (\\%)": [pm(cv_mean.loc[m, "balanced_accuracy"], cv_std.loc[m, "balanced_accuracy"]) for m in order],
            "Macro-F1 (\\%)": [pm(cv_mean.loc[m, "f1_macro"], cv_std.loc[m, "f1_macro"]) for m in order],
            "MCC": [f"{cv_mean.loc[m, 'mcc']:.3f} $\\pm$ {cv_std.loc[m, 'mcc']:.3f}" for m in order],
        }
    )
    _tex(cv_tab, os.path.join(tab_dir, "cv.tex"),
         "Cross-validation performance on the development folds (mean $\\pm$ SD over folds). Models are ordered by mean macro-F1, the selection criterion.",
         "tab:cv", "lcccc")

    rows = []
    for m in test.index:
        t = test.loc[m]
        rows.append({
            "Model": m,
            "Accuracy (\\%)": f"{fmt_pct(t.accuracy)} [{fmt_pct(t.accuracy_lo)}, {fmt_pct(t.accuracy_hi)}]",
            "Balanced acc. (\\%)": fmt_pct(t.balanced_accuracy),
            "Macro-F1 (\\%)": f"{fmt_pct(t.f1_macro)} [{fmt_pct(t.f1_macro_lo)}, {fmt_pct(t.f1_macro_hi)}]",
            "MCC": f"{fmt3(t.mcc)} [{fmt3(t.mcc_lo)}, {fmt3(t.mcc_hi)}]",
            "AUC (\\%)": fmt_pct(t.auc_macro_ovr) if "auc_macro_ovr" in t else "--",
        })
    _tex(pd.DataFrame(rows), os.path.join(tab_dir, "test.tex"),
         "Held-out test performance. Each model's fold models are averaged; Ensemble averages the top models chosen on cross-validation. Brackets: 95\\% group-bootstrap confidence intervals.",
         "tab:test", "lccccc")

    u = unc.loc[test.index]
    unc_tab = pd.DataFrame({
        "Model": u.index,
        "ECE det.": [fmt_pct(v) for v in u.ece_det],
        "ECE MC": [fmt_pct(v) for v in u.ece_mc],
        "NLL det.": [fmt3(v) for v in u.nll_det],
        "NLL MC": [fmt3(v) for v in u.nll_mc],
        "AUROC$_\\text{err}$ softmax H": [fmt3(v) for v in u.err_auroc_softmax_entropy],
        "AUROC$_\\text{err}$ MC H": [fmt3(v) for v in u.err_auroc_mc_entropy],
        "AUROC$_\\text{err}$ MI": [fmt3(v) for v in u.err_auroc_mutual_info],
        "Acc. @20\\% referred": [fmt_pct(v) for v in u.acc_refer_20],
    })
    _tex(unc_tab, os.path.join(tab_dir, "uncertainty.tex"),
         "Calibration and uncertainty on the held-out test set. ECE in \\%. AUROC$_\\text{err}$: how well each uncertainty score separates wrong from correct predictions. Last column: accuracy on the 80\\% of cases kept after referring the 20\\% with the highest MC-dropout entropy.",
         "tab:uncertainty", "lcccccccc")

    eff_tab = pd.DataFrame({
        "Model": eff.index,
        "Trainable params": [f"{v:,}" for v in eff.params_trainable],
        "Total params (M)": [f"{v / 1e6:.2f}" for v in eff.params_total],
        "Train time (min)": [f"{v:.1f}" for v in eff.train_minutes_mean],
        "Best epoch": [f"{v:.0f}" for v in eff.best_epoch_mean],
        "Inference (ms/img)": [f"{v:.2f}" for v in eff.ms_per_image_batched],
    })
    _tex(eff_tab, os.path.join(tab_dir, "efficiency.tex"),
         "Model size and cost. Training time and best epoch are means over folds; inference time is the median over folds of batched prediction (batch 32) on the training GPU.",
         "tab:efficiency", "lrrrrr")

    for m, df in per_class.items():
        t = df.copy()
        t.insert(0, "Class", [c.replace("_", " ") for c in t.index])
        for c in ("precision", "recall", "f1"):
            t[c] = [fmt_pct(v) for v in t[c]]
        t.columns = ["Class", "Precision (\\%)", "Recall (\\%)", "F1 (\\%)", "Support"]
        _tex(t, os.path.join(tab_dir, f"per_class_{m}.tex"),
             f"Per-class held-out test performance of {m}.", f"tab:perclass-{m}", "lcccr")

    mt = mcn.copy()
    mt["p (Holm)"] = [_pfmt(p) for p in mt.p_holm]
    mt = mt.rename(columns={"a": "Model A", "b": "Model B", "a_only_correct": "A only correct", "b_only_correct": "B only correct"})
    _tex(mt[["Model A", "Model B", "A only correct", "B only correct", "p (Holm)"]],
         os.path.join(tab_dir, "mcnemar.tex"),
         "Pairwise exact McNemar tests on held-out test predictions, Holm-corrected.",
         "tab:mcnemar", "llrrr")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(fig_dir, y, det, mc, best, classes, cv, order):
    short = [c.replace("_", " ") for c in classes]
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight"})

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, m in zip(axes, (best, ENSEMBLE)):
        cm = confusion_matrix(y, det[m].argmax(1), labels=range(len(classes)))
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks(range(len(classes)), short, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), short)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(m)
    fig.savefig(os.path.join(fig_dir, "confusion.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    for m, probs, style in ((best, det[best], "-"), (best, mc[best], "--"), (ENSEMBLE, mc[ENSEMBLE], ":")):
        rc = M.reliability_curve(y, probs)
        tag = "MC dropout" if probs is not det[m] else "deterministic"
        ax.plot(rc[:, 0], rc[:, 1], style, marker="o", ms=3, label=f"{m}, {tag}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(fig_dir, "reliability.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    rates = np.linspace(0, 0.5, 26)
    for m in (best, ENSEMBLE):
        h = M.predictive_entropy(mc[m])
        curve = M.referral_curve(y, mc[m], h, rates)
        ax.plot(100 * rates, [100 * curve[r] for r in rates], label=f"{m}, MC entropy")
        rng = np.random.default_rng(0)
        rand = M.referral_curve(y, mc[m], rng.random(len(y)), rates)
        ax.plot(100 * rates, [100 * rand[r] for r in rates], ls=":", c="grey",
                label="random referral" if m == best else None)
    ax.set_xlabel("Cases referred to a clinician (%)")
    ax.set_ylabel("Accuracy on retained cases (%)")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(fig_dir, "referral.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, m in enumerate(order):
        v = 100 * cv[cv.model == m]["f1_macro"].to_numpy()
        ax.scatter(np.full(len(v), i), v, s=12, c="tab:blue")
        ax.hlines(v.mean(), i - 0.3, i + 0.3, colors="k")
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
    ax.set_ylabel("Validation macro-F1 (%)")
    fig.savefig(os.path.join(fig_dir, "cv_folds.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
