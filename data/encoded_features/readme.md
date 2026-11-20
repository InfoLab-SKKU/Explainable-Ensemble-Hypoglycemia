# Encoded features

Encoded representations of the three modalities, one row per participant (`PtID` + features), produced by
`preprocessing/processing_tables.py`:

| File | Modality | Encoder | Dimensions |
|---|---|---|---|
| `tabular_encoded.csv` | demographic variables, blood tests, psychological, behavioral and cognitive assessments, medication chart | imputation, encoding, scaling and `SelectKBest` | 112 |
| `text_encoded.csv` | medical conditions and medication free text | Clinical BioBERT | 768 |
| `timeseries_encoded.csv` | continuous glucose monitoring (CGM) sequences and CGM-derived metrics | TS2Vec | 128 |

The labels and the list of the 198 participants used in the experiments are in `data/labels.csv`.

- **Early Fusion** (`experiments/early_fusion.py`): an MLP-based neural feature selection, fitted on the training split
  of each random split, reduces the three representations to 50, 100 and 32 features, which are concatenated
  (182 features).
- **Late Fusion** (`experiments/late_fusion.py`): one model is trained per modality and a logistic-regression
  meta-classifier combines their predicted probabilities.
