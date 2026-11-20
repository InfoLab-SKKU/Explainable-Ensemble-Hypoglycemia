"""Save the training and test participant IDs of each random split, for Early and Late Fusion, and compare them.

Both strategies use the same 198 participants and the random states 0-9 with a stratified 70/30 split
(sklearn.model_selection.train_test_split). The split is drawn on the row order of each pipeline: Early Fusion keeps
the order of data/labels.csv, whereas Late Fusion sorts the participants by PtID after merging the three modalities.
The same random state therefore produces different partitions in the two strategies.

Usage (from the repository root):
    python analysis/compare_splits.py     # writes results/splits_early.csv, results/splits_late.csv, prints the overlap
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA, RESULTS, SEEDS


def splits(labels, y):
    rows = []
    for rs in SEEDS:
        tr, te = train_test_split(np.arange(len(labels)), test_size=0.3, stratify=y, random_state=rs)
        rows += [dict(random_state=rs, PtID=int(p), set='train') for p in labels['PtID'].values[tr]]
        rows += [dict(random_state=rs, PtID=int(p), set='test') for p in labels['PtID'].values[te]]
    return pd.DataFrame(rows)


def main():
    labels = pd.read_csv(DATA / 'labels.csv')
    # Early Fusion (src/common.py): row order of data/labels.csv
    early = splits(labels, (labels['BCaseControlStatus'] == 'Control').astype(int).values)
    # Late Fusion (experiments/late_fusion.py): participants sorted by PtID
    srt = labels.sort_values('PtID').reset_index(drop=True)
    late = splits(srt, srt['BCaseControlStatus'])
    early.to_csv(RESULTS / 'splits_early.csv', index=False)
    late.to_csv(RESULTS / 'splits_late.csv', index=False)
    print('random state | identical test sets | test participants in common (of 60)')
    for rs in SEEDS:
        a = set(early.query('random_state == @rs and set == "test"').PtID)
        b = set(late.query('random_state == @rs and set == "test"').PtID)
        print(f'{rs:12d} | {str(a == b):19s} | {len(a & b)}')


if __name__ == '__main__':
    main()
