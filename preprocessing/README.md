# Preprocessing

`processing_tables.py` encodes the raw multimodal data of the Jaeb Center for Health Research
([Figshare](https://figshare.com/articles/dataset/Multimodal-fusion-severe-hypo-data/25136942/2)) into the files of
`data/`:

- **tabular data** (clinical, cognitive, psychological and behavioral assessments): encoding of categorical
  variables, imputation, scaling and statistical feature selection (`SelectKBest`, 112 features) ->
  `data/encoded_features/tabular_encoded.csv`;
- **medical text** (conditions and medications): Clinical BioBERT embeddings (768 dimensions) ->
  `data/encoded_features/text_encoded.csv`;
- **CGM time series**: TS2Vec representations and glucose variability metrics (128 dimensions) ->
  `data/encoded_features/timeseries_encoded.csv`;
- the IDs and labels of the participants with the three modalities -> `data/labels.csv`.

Usage, from the repository root, after copying the `preprocessed` folder of the Figshare package to `data/preprocessed/`:

```bash
pip install transformers
python preprocessing/processing_tables.py
```
