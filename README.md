# Explainable Multimodal Neurocomputing for Severe Hypoglycemia Risk Prediction in Type 1 Diabetes Using Static and Dynamic Ensemble Selection

Code and data of the paper by C. Pompéi, T. Abuhmed and S. El-Sappagh.

The framework predicts severe hypoglycemia (SH) in people with Type 1 Diabetes from three modalities: structured
clinical, cognitive and psychological assessments (tabular), medical text describing conditions and medications
(Clinical BioBERT embeddings), and continuous glucose monitoring (TS2Vec representations). It compares classical
machine-learning models, static ensembles and dynamic ensemble selection (DES) methods, under Early Fusion
(concatenation + neural feature selection) and Late Fusion (one model per modality + logistic-regression
meta-classifier), with Friedman–Nemenyi tests over 10 random 70/30 splits.

### Main results (198 participants, mean ± std over 10 random splits; Table 25 of the paper)

| Fusion | Category | Best model (by AUC-ROC) | Accuracy | AUC-ROC |
|---|---|---|---|---|
| Early | Classical ML | LR | 0.7617 ± 0.0373 | 0.8626 ± 0.0258 |
| Early | Static ensemble | CatBoost | **0.8167 ± 0.0394** | 0.8846 ± 0.0253 |
| Early | DES (Pool 2) | DES-P / FIRE-DES-P | 0.7950 ± 0.0289 | 0.8676 ± 0.0241 |
| Late | Classical ML | SVC | 0.7883 ± 0.0325 | 0.8842 ± 0.0207 |
| Late | Static ensemble | CatBoost | 0.8150 ± 0.0311 | **0.9128 ± 0.0258** |
| Late | DES (Pool 2) | DES-P / FIRE-DES-P | 0.7983 ± 0.0353 | 0.8819 ± 0.0274 |

DES methods are competitive but do not outperform the best static ensemble. The modality ablation (Table 20) shows
that most of the predictive information is carried by the tabular modality.

---

## Repository structure

```
data/
  labels.csv                     participant IDs and labels (SH case / control) of the 198 participants
  encoded_features/              encoded tabular, text and CGM representations of each participant
src/                             shared code: paths, data preparation, neural feature selection, statistics
experiments/
  early_fusion.py                Tables 8-13, 22, 24 and Figures 6-11
  additional_analyses.py         Tables 20, 21, 23 and the analysis of the DES selection (Section 4.15)
  late_fusion.py                 Tables 14-19 and Figures 12-17
analysis/
  make_tables.py                 rebuilds Tables 8-25 from the result files  -> results/tables.md
  make_figures.py                rebuilds Figures 6-17 (TikZ/pgfplots)       -> results/figures/
results/                         result files of the paper (JSON), tables.md, figures/
preprocessing/
  processing_tables.py           encodes the raw data (BioBERT, TS2Vec, tabular encoder) into data/
```

## Installation

Python 3.13, CPU only (tested on Linux):

```bash
git clone https://github.com/InfoLab-SKKU/Explainable-Ensemble-Hypoglycemia.git
cd Explainable-Ensemble-Hypoglycemia
pip install -r requirements.txt          # or: conda env create -f environment.yml
```

`requirements.txt` gives the exact versions used for the paper. DESlib 0.3.7 is not compatible with
recent scikit-learn versions out of the box; the scripts apply a one-line compatibility patch (`src/common.py`,
`experiments/late_fusion.py`).

## Reproducing the results

All commands are run from the repository root. The encoded features of the 198 participants are included in
`data/`, so no download is needed. All experiments use the random states 0-9 (10 stratified 70/30 splits).

**Step 1 - Run the experiments** (about 1.5 h in total on 2 CPU cores; each script writes its result file to
`results/` and overwrites the provided one):

```bash
python experiments/early_fusion.py            # ~15 min -> results/early_fusion.json
python experiments/additional_analyses.py     # ~15 min -> results/additional_analyses.json
python experiments/late_fusion.py base        # classical models and static ensembles -> results/late_classical.json, late_static.json
python experiments/late_fusion.py pool1       # DES, Pool 1 -> results/late_pool1.json
python experiments/late_fusion.py pool2       # DES, Pool 2 -> results/late_pool2.json
python experiments/late_fusion.py pool3       # DES, Pool 3 -> results/late_pool3.json
```

The three DES pools are independent and can be run in parallel (about 1 h in total).

**Step 2 - Rebuild the tables and figures of the paper** from the result files:

```bash
python analysis/make_tables.py                # -> results/tables.md (Tables 8-25 and Friedman-Nemenyi tests)
python analysis/make_figures.py               # -> results/figures/*.tex (+ figures.pdf if pdflatex is installed)
python analysis/compare_splits.py             # -> results/splits_early.csv, results/splits_late.csv
```

**Step 3 - Check the results.** `results/tables.md` contains the values of Tables 8-25 of the paper. With the
package versions of `requirements.txt`, re-running the experiments reproduces the provided result files exactly,
except the training and inference times of Table 24, which depend on the machine. To check a re-run against the
provided files, keep a copy of `results/` before Step 1 and compare the JSON files.

Step 2 can also be run directly on the provided result files, without re-running the experiments.

| Script | Paper |
|---|---|
| `experiments/early_fusion.py` | Tables 8-13 (classical, static, DES Pools 1-3, ranking), Table 22 (sensitivity to *k*), Table 24 (computational cost), Figures 6-11 |
| `experiments/additional_analyses.py` | Table 20 (modality ablation), Table 21 (class imbalance), Table 23 (modality-level Shapley values), DES selection analysis |
| `experiments/late_fusion.py` | Tables 14-19, Figures 12-17 |
| `analysis/make_tables.py` | Tables 8-25 |
| `analysis/make_figures.py` | Figures 6-17 |
| `analysis/compare_splits.py` | training and test participant IDs of each run, for Early and Late Fusion (Section 5) |

### Regenerating the encoded features (optional)

The raw data come from the Jaeb Center for Health Research and are available on
[Figshare](https://figshare.com/articles/dataset/Multimodal-fusion-severe-hypo-data/25136942/2). To regenerate the
files of `data/`, copy the `preprocessed` folder of the Figshare package to `data/preprocessed/`, install
`transformers`, and run:

```bash
python preprocessing/processing_tables.py
```

This writes `data/encoded_features/*_encoded.csv` and `data/labels.csv` (see `preprocessing/README.md`).

## Protocol notes

- **Neural feature selection (Early Fusion).** The MLP-based feature selection is fitted on the training split of
  each run only, and then applied to the corresponding test split (`src/common.py`, `src/nnsel.py`). Feature
  ranking, outlier removal, SMOTE, scaling and the selection of the DES hyperparameters are also fitted on the
  training split of each run.
- **Tabular encoder.** The ANOVA ranking of the tabular encoder (`SelectKBest` with `f_classif`) is configured with
  k = 112, the number of encoded tabular columns, so all columns are retained. The encoders only compute
  unsupervised statistics on all participants (label encoding, median imputation, standardization, BioBERT and
  TS2Vec embeddings).
- **Cohort and splits.** Early and Late Fusion are evaluated on the same 198 participants (`data/labels.csv`) and
  with the same random states. Early Fusion keeps the participants in the order of `data/labels.csv`, whereas Late
  Fusion sorts them by `PtID` after merging the three modalities; the same random state therefore gives different
  partitions in the two strategies. `analysis/compare_splits.py` writes the training and test IDs of each run.

## License

MIT, see `LICENSE`.
