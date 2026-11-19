# Counting photographs, not images: duplicate-aware evaluation of skin disease classifiers

This is the code for our paper *Counting Photographs, Not Images: Duplicate-Aware Evaluation of Deep Learning for Skin Disease Classification* (with MD Mahi Sarkar and Farhana Rahman Elma, in preparation).

I compare a small custom CNN with seven frozen ImageNet backbones (VGG16, ResNet50, MobileNetV2, EfficientNetB0, DenseNet121, InceptionV3 and Xception) on five skin conditions: acne, hyperpigmentation, nail psoriasis, Stevens-Johnson syndrome/toxic epidermal necrolysis (SJS-TEN) and vitiligo.

The project started as a simple benchmark. Then, while auditing the dataset, I noticed that its 9,548 images are really 341 photographs, and each photograph comes with 27 copies that were already cropped, padded, recoloured, flipped or resized before release. Split that by image, and nearly every test photograph has a copy sitting in the training set. So most of the work here went into making sure the numbers count photographs, not copies:

- **Grouping by photograph.** There are no patient IDs, so I group images by the photograph they came from. I use the filenames, since every copy is named after its original, and two perceptual hashes that both have to agree, after cropping off any frame and checking flips and rotations. A group never crosses the test set or a cross-validation fold.
- **Scoring originals only.** Models train on every image, but validation and test metrics count each original photograph once. Its copies aren't independent cases.
- **Choosing on validation only.** Five-fold grouped cross-validation picks the best model and the ensemble members. I only look at the held-out test set for the final report.
- **Checking the uncertainty.** Monte Carlo dropout runs at inference. I report calibration (ECE, NLL, Brier), how well uncertainty picks out errors, and accuracy after referring the least confident cases to a person.

Every number, table and figure in the manuscript comes from `scripts/analyze.py`, run on saved predictions. I don't type any of them in by hand. The manuscript itself will be added here once it's published.

## Data

*Skin Disease Classification Dataset*, Mendeley Data, [doi:10.17632/3hckgznc67.1](https://doi.org/10.17632/3hckgznc67.1) (9,548 images, five classes). Download it and point `--data-roots` at the folder (or folders) with one sub-folder per class. I don't redistribute the images here.

## Running

**On Kaggle.** This is how I produced the paper's results. Open [`notebooks/kaggle_run.ipynb`](notebooks/kaggle_run.ipynb) and follow the first cell. A full run is 8 models × 5 folds, which doesn't fit in one 12-hour Kaggle session, so the notebook can pick up where the last session stopped.

**Anywhere else:**

```bash
pip install -r requirements.txt
export KERAS_BACKEND=tensorflow
python scripts/run_experiment.py --data-roots /path/to/dataset --out-dir outputs --mixed-precision
python scripts/analyze.py --out-dir outputs --paper-dir paper/generated
```

`python scripts/run_experiment.py --help` lists every setting (image size, folds, epochs, L2 strength, MC samples and so on). The defaults are the ones I used in the paper.

## Outputs

| Path | Contents |
|---|---|
| `outputs/splits.csv` | Every image with its group, partition, original/copy flag and perceptual hash |
| `outputs/grouping_stats.json` | How many duplicates were found and how |
| `outputs/runs/<model>_fold<k>/` | Training history, per-run summary, validation/test probabilities and MC-dropout samples |
| `outputs/results/` | Aggregated metrics, confidence intervals, significance tests (CSV/JSON) |
| `paper/generated/` | LaTeX tables, figures and `numbers.tex` for the manuscript (written locally, not committed) |

## Tests

```bash
pip install pytest
KERAS_BACKEND=tensorflow pytest -q tests   # torch backend also works
```

The tests cover duplicate grouping, leak-free splitting, the preprocessing each backbone expects, and MC dropout. One of them runs the whole pipeline on synthetic images on a CPU, which takes about two minutes.

## Layout

```
src/skinclf/data.py      loading, duplicate grouping, grouped stratified splits
src/skinclf/models.py    CustomCNN, frozen backbones with correct preprocessing, MC dropout
src/skinclf/train.py     training one (model, fold) and saving predictions
src/skinclf/metrics.py   classification, calibration, uncertainty, group bootstrap
src/skinclf/stats.py     Friedman/Nemenyi across folds, Holm-corrected McNemar on test
scripts/                 experiment runner and analysis
notebooks/               Kaggle runner
```

## Licence

MIT. See [LICENSE](LICENSE).
