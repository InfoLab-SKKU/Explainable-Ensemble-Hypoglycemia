"""Loading and summarising the result files written by the experiment scripts.

Early Fusion results (results/early_fusion.json) are a list with one dict per random split; Late Fusion results
(results/late_*.json) are one dict per experiment, mapping each model to its per-split scores. Both are converted to the
same structure: FAMILIES[fusion][family][model] = {'acc': [...], 'prec': [...], 'rec': [...], 'f1': [...], 'auc': [...],
'roc': [(fpr, tpr), ...]} with one value per split.
"""
import json
import numpy as np
from scipy.stats import friedmanchisquare, rankdata
from src.config import RESULTS
from src.cdplot import compute_CD

METRICS = ['acc', 'prec', 'rec', 'f1', 'auc']
CLASSICAL = ['DT', 'LR', 'NB', 'KN', 'MLP', 'SVC']
STATIC = ['RF', 'XGB', 'GB', 'AB', 'CB']
FAMILY_NAMES = {'classical': 'Classical ML', 'static': 'Static ensembles', 'pool1': 'DES (Pool 1)',
                'pool2': 'DES (Pool 2)', 'pool3': 'DES (Pool 3)'}


def load_json(name):
    return json.load(open(RESULTS / name))


def load_early():
    runs = load_json('early_fusion.json')
    out = {}
    for fam in ['classical', 'static', 'pool1', 'pool2', 'pool3']:
        get = (lambda r: r[fam]) if fam in ('classical', 'static') else (lambda r: r['des'][fam[-1]])
        out[fam] = {n: {**{m: [get(r)[n][m] for r in runs] for m in METRICS},
                        'roc': [(get(r)[n]['fpr'], get(r)[n]['tpr']) for r in runs]} for n in get(runs[0])}
    return out


def load_late():
    keys = {'acc': 'accuracy', 'prec': 'precision', 'rec': 'recall', 'f1': 'f1', 'auc': 'auc'}
    out = {}
    for fam in ['classical', 'static', 'pool1', 'pool2', 'pool3']:
        d = load_json(f'late_{fam}.json')
        out[fam] = {n: {**{m: v[k] for m, k in keys.items()}, 'roc': [tuple(c) for c in v['roc']]} for n, v in d.items()}
    return out


def load_all():
    return {'early': load_early(), 'late': load_late()}


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def friedman_nemenyi(scores):
    """scores: dict model -> list of per-split scores (higher is better).
    Returns the Friedman statistic, p-value, Nemenyi critical difference (alpha = 0.05) and average ranks."""
    names = list(scores)
    M = np.array([scores[n] for n in names]).T  # (splits, models)
    stat, p = friedmanchisquare(*M.T)
    ranks = np.array([rankdata(-row, method='average') for row in M]).mean(0)
    cd = compute_CD(list(ranks), M.shape[0])
    return dict(stat=float(stat), p=float(p), cd=float(cd), ranks={n: float(r) for n, r in zip(names, ranks)})


def ranking_candidates(F, best_pool='pool2'):
    """DES methods of the best pool and static ensembles, sorted by mean AUC (Tables 13 and 19)."""
    cand = [(f'{n} (DES)', F[best_pool][n]) for n in F[best_pool]] + [(f'{n} (Static)', F['static'][n]) for n in STATIC]
    return sorted(cand, key=lambda c: -np.mean(c[1]['auc']))
