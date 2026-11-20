"""Additional analyses of Section 4.15 (Early Fusion).

Produces results/additional_analyses.json, used for Tables 20, 21 and 23 and the analysis of the DES selection:
- modality ablation, with and without the neural feature selection (Table 20),
- class-imbalance strategies: SMOTE, class weights, random undersampling, none (Table 21),
- modality-level Shapley values for static and dynamic models (Table 23),
- number of classifiers selected by DES-P per test patient (main protocol and held-out DSEL).

Usage (from the repository root):  python experiments/additional_analyses.py [--jobs 2]
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import *
from src.config import RESULTS, SEEDS
import json, itertools, math
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from deslib.des import DESP
from experiments.early_fusion import base_pool, POOLS, C_LR

MOD_DIM = {'tabular': 50, 'text': 100, 'timeseries': 32}

def lr_model(Xr, yr, Xte, yte, rs, cw=None):
    nf = min(100, Xr.shape[1])
    corr = np.nan_to_num(np.abs([np.corrcoef(Xr[:, j], yr)[0, 1] for j in range(Xr.shape[1])]))
    top = np.argsort(corr)[::-1][:nf]
    c = LogisticRegression(C=C_LR, max_iter=200, random_state=rs, class_weight=cw).fit(Xr[:, top], yr)
    return c, top

def cb_cell18(Xtr_raw, ytr_raw, Xte_raw, rs, balance='smote'):
    sc = MinMaxScaler().fit(Xtr_raw); Xt, Xe = sc.transform(Xtr_raw), sc.transform(Xte_raw)
    lr = LogisticRegression(C=C_LR, max_iter=200, random_state=rs).fit(Xt, ytr_raw)
    top = np.argsort(np.abs(lr.coef_[0]))[::-1][:min(30, Xt.shape[1])]
    Xs, ys = Xt[:, top], ytr_raw
    kw = {}
    if balance == 'smote':
        Xs, ys = SMOTE(random_state=rs).fit_resample(Xs, ys)
    elif balance == 'rus':
        Xs, ys = RandomUnderSampler(random_state=rs).fit_resample(Xs, ys)
    elif balance == 'weights':
        kw = dict(auto_class_weights='Balanced')
    c = CatBoostClassifier(random_state=rs, verbose=0, thread_count=1, **kw).fit(Xs, ys)
    return c, sc, top

def pool2(rs, weighted=False, ratio=1.0):
    return [RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=7, random_state=rs, bootstrap=False, n_jobs=1,
                                   class_weight='balanced' if weighted else None),
            CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=rs, verbose=0, thread_count=1,
                               auto_class_weights='Balanced' if weighted else None),
            XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric='mlogloss', random_state=rs, n_jobs=1,
                          scale_pos_weight=ratio if weighted else 1.0)]

def m_short(mm):
    return {k: mm[k] for k in ['acc', 'auc', 'sens_case', 'spec']}

def run_split(rs):
    out = {'rs': rs, 'imb': {}, 'abl': {}, 'shap': {}, 'sel': {}}
    # ---------------- class imbalance
    for strat in ['smote', 'weights', 'rus', 'none']:
        D = prep(rs, balance='smote' if strat == 'smote' else ('rus' if strat == 'rus' else 'none'))
        cw = 'balanced' if strat == 'weights' else None
        c, top = lr_model(D['Xr'], D['yr'], D['Xte'], D['yte'], rs, cw)
        r = {'LR': m_short(metrics(c, D['Xte'][:, top], D['yte']))}
        c, sc, top = cb_cell18(D['Xtr_raw'], D['ytr_raw'], D['Xte_raw'], rs, balance=strat)
        r['CB'] = m_short(metrics(c, sc.transform(D['Xte_raw'])[:, top], D['yte']))
        ratio = np.sum(D['yr'] == 0) / max(1, np.sum(D['yr'] == 1))
        P = [m.fit(D['Xr'], D['yr']) for m in pool2(rs, weighted=(strat == 'weights'), ratio=ratio)]
        r['DESP'] = m_short(metrics(DESP(P, k=7, random_state=rs).fit(D['Xr'], D['yr']), D['Xte'], D['yte']))
        r['n_train_case'], r['n_train_control'] = int(np.sum(D['ytr_raw'] == 0)), int(np.sum(D['ytr_raw'] == 1))
        out['imb'][strat] = r
    # ---------------- modality ablation
    combos = [c for n in (1, 2, 3) for c in itertools.combinations(MODS, n)]
    for raw in (False, True):
        for mods in combos:
            if raw and len(mods) < 3:
                continue
            D = prep(rs, mods=list(mods), raw=raw)
            c, top = lr_model(D['Xr'], D['yr'], D['Xte'], D['yte'], rs)
            r = {'LR': m_short(metrics(c, D['Xte'][:, top], D['yte']))}
            c, sc, top = cb_cell18(D['Xtr_raw'], D['ytr_raw'], D['Xte_raw'], rs)
            r['CB'] = m_short(metrics(c, sc.transform(D['Xte_raw'])[:, top], D['yte']))
            P = [m.fit(D['Xr'], D['yr']) for m in pool2(rs)]
            r['DESP'] = m_short(metrics(DESP(P, k=7, random_state=rs).fit(D['Xr'], D['yr']), D['Xte'], D['yte']))
            out['abl'][('raw:' if raw else '') + '+'.join(mods)] = r
    # ---------------- modality-level Shapley values & DES selection analysis
    D = prep(rs)
    Xr, yr, Xte, yte = D['Xr'], D['yr'], D['Xte'], D['yte']
    lr, top = lr_model(Xr, yr, Xte, yte, rs)
    cb, sc, top30 = cb_cell18(D['Xtr_raw'], D['ytr_raw'], D['Xte_raw'], rs)
    # map: our common space = Xr scaling. For CB we need its own scaling of the raw train/test -> keep raw arrays
    P = [c.fit(Xr, yr) for n, c in base_pool(rs)]
    desp = DESP(P, k=7, random_state=rs).fit(Xr, yr)
    # held-out DSEL variant
    Xa, Xd, ya, yd = train_test_split(D['Xtr_raw'], D['ytr_raw'], test_size=0.3, stratify=D['ytr_raw'], random_state=rs)
    Xa, ya = SMOTE(random_state=rs).fit_resample(Xa, ya)
    s3 = MinMaxScaler().fit(Xa)
    P3 = [c.fit(s3.transform(Xa), ya) for n, c in base_pool(rs)]
    desp_ho = DESP(P3, k=7, random_state=rs).fit(s3.transform(Xd), yd)
    # all models are evaluated on the raw (unscaled) feature space through wrappers
    sc_main = MinMaxScaler().fit(SMOTE(random_state=rs).fit_resample(D['Xtr_raw'], D['ytr_raw'])[0])
    F = {
        'LR': lambda X: lr.predict_proba(sc_main.transform(X)[:, top])[:, 1],
        'CB': lambda X: cb.predict_proba(sc.transform(X)[:, top30])[:, 1],
        'DESP': lambda X: desp.predict_proba(sc_main.transform(X))[:, 1],
        'DESP-heldout': lambda X: desp_ho.predict_proba(s3.transform(X))[:, 1],
    }
    bounds, s = {}, 0
    for m in MODS:
        bounds[m] = (s, s + MOD_DIM[m]); s += MOD_DIM[m]
    rng = np.random.RandomState(rs)
    bg = D['Xtr_raw'][rng.choice(len(D['Xtr_raw']), 30, replace=False)]
    Xq = D['Xte_raw']
    for name, f in F.items():
        v = {}
        for S in itertools.chain.from_iterable(itertools.combinations(MODS, n) for n in range(4)):
            Xs = np.repeat(Xq[:, None, :], len(bg), axis=1).copy()  # (n, b, d)
            for m in MODS:
                if m not in S:
                    a, b = bounds[m]; Xs[:, :, a:b] = bg[None, :, a:b]
            v[S] = f(Xs.reshape(-1, Xq.shape[1])).reshape(len(Xq), len(bg)).mean(1)
        phi = {}
        for m in MODS:
            others = [o for o in MODS if o != m]; tot = 0
            for n in range(3):
                for S in itertools.combinations(others, n):
                    w = math.factorial(len(S)) * math.factorial(2 - len(S)) / 6
                    S2 = tuple(o for o in MODS if o in S or o == m)
                    tot = tot + w * (v[S2] - v[tuple(o for o in MODS if o in S)])
            phi[m] = float(np.mean(np.abs(tot)))
        out['shap'][name] = phi
    # selection analysis (DES-P rule: competence = local accuracy on k neighbours > 1/2)
    for name, des, Xq_s in [('main', desp, sc_main.transform(Xq)), ('heldout', desp_ho, s3.transform(Xq))]:
        _, nn_idx = des.roc_algorithm_.kneighbors(Xq_s, n_neighbors=7)
        comp = des.DSEL_processed_[nn_idx, :].mean(axis=1)  # (n, L)
        sel = comp > 0.5
        sel[~sel.any(1)] = True
        out['sel'][name] = {'mean_selected': float(sel.sum(1).mean()), 'frac_all_selected': float(sel.all(1).mean()),
                            'freq': dict(zip(POOLS[3], sel.mean(0).round(4).tolist())),
                            'mean_competence': float(comp.mean())}
    print('done', rs, flush=True)
    return out

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=2); a = ap.parse_args()
    t0 = time.time()
    for rs in SEEDS:
        split_features(rs)
    res = Parallel(n_jobs=a.jobs)(delayed(run_split)(rs) for rs in SEEDS)
    json.dump(res, open(RESULTS / 'additional_analyses.json', 'w'))
    print('total', round(time.time() - t0), 's')
