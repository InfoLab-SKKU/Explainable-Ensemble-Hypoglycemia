"""Early Fusion experiments of the paper (neural feature selection fitted on the training split of each random split).

Produces results/early_fusion.json, used for Tables 8-13, 22 and 24 and Figures 6-11:
- classical classifiers (Table 8), static ensembles (Table 9), DES with Pools 1-3 (Tables 10-12),
- DES hyperparameters (k, with_IH, Hc) selected on a validation subset of the training data,
- sensitivity to k with the main protocol and with a held-out DSEL (Table 22),
- training time, inference time and model size (Table 24).

Usage (from the repository root):  python experiments/early_fusion.py [--jobs 2]
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import *
from src.config import RESULTS, SEEDS
import json, itertools
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.base import clone
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from deslib.des import KNORAE, KNORAU, KNOP, DESMI, METADES, DESKNN, DESP

C_LR = 0.09659168435718246
KS = [3, 5, 7, 9, 11, 13, 15]
DES = {  # name: (class, DFP, extra grid)
    'KNORAE': (KNORAE, False, {'with_IH': [True, False]}), 'KNORAU': (KNORAU, False, {'with_IH': [True, False]}),
    'KNOP': (KNOP, False, {}), 'DESMI': (DESMI, False, {}), 'METADES': (METADES, False, {'Hc': [0.5, 0.7, 0.9]}),
    'DESKNN': (DESKNN, False, {}), 'DESP': (DESP, False, {}),
    'FIRE-KNORA-U': (KNORAU, True, {}), 'FIRE-KNORA-E': (KNORAE, True, {'with_IH': [True, False]}),
    'FIRE-METADES': (METADES, True, {'Hc': [0.5, 0.7, 0.9]}), 'FIRE-DESKNN': (DESKNN, True, {}),
    'FIRE-DESP': (DESP, True, {}), 'FIRE-KNOP': (KNOP, True, {}),
}
POOL_DES = {1: list(DES), 2: ['KNORAE', 'DESP', 'FIRE-KNORA-E', 'FIRE-DESKNN', 'FIRE-DESP'],
            3: ['KNORAE', 'DESP', 'FIRE-KNORA-E', 'FIRE-DESKNN', 'FIRE-DESP']}

def base_pool(rs):
    return [
        ('RF', RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=7, random_state=rs, bootstrap=False, n_jobs=1)),
        ('CB', CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=rs, verbose=0, thread_count=1)),
        ('AB', AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=rs)),
        ('XGB', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric='mlogloss', random_state=rs, n_jobs=1)),
        ('GB', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=rs)),
        ('Voting', VotingClassifier([('rf', RandomForestClassifier(random_state=rs, n_jobs=1)), ('xgb', XGBClassifier(random_state=rs, eval_metric='mlogloss', n_jobs=1)),
                                     ('cat', CatBoostClassifier(verbose=0, random_state=rs, thread_count=1))], voting='soft')),
        ('Stacking', StackingClassifier([('rf', RandomForestClassifier(random_state=rs, n_jobs=1)), ('xgb', XGBClassifier(random_state=rs, eval_metric='mlogloss', n_jobs=1)),
                                         ('cat', CatBoostClassifier(verbose=0, random_state=rs, thread_count=1))], final_estimator=LogisticRegression(random_state=rs), cv=3)),
    ]
POOLS = {1: ['RF', 'CB', 'AB', 'XGB', 'GB'], 2: ['RF', 'CB', 'XGB'], 3: ['RF', 'CB', 'AB', 'XGB', 'GB', 'Voting', 'Stacking']}

def fit_timed(clf, X, y):
    t = time.perf_counter(); clf.fit(X, y); return clf, time.perf_counter() - t

def run_split(rs):
    D = prep(rs)
    Xr, yr, Xte, yte = D['Xr'], D['yr'], D['Xte'], D['yte']
    out = {'rs': rs, 'classical': {}, 'static': {}, 'des': {}, 'kabl': {}}
    # ---- classical: top-100 |corr| on SMOTE train
    corr = np.nan_to_num(np.abs([np.corrcoef(Xr[:, j], yr)[0, 1] for j in range(Xr.shape[1])]))
    top = np.argsort(corr)[::-1][:100]
    clfs = {'DT': DecisionTreeClassifier(random_state=rs, criterion='gini', max_depth=6, min_samples_leaf=10, min_samples_split=9),
            'LR': LogisticRegression(random_state=rs, C=C_LR, max_iter=100), 'NB': GaussianNB(var_smoothing=0.0058873326349240295),
            'KN': KNeighborsClassifier(metric='manhattan', n_neighbors=8), 'MLP': MLPClassifier(random_state=rs, max_iter=1000, alpha=0.0003079393718075164,
            hidden_layer_sizes=195, learning_rate_init=0.0001675266159417717), 'SVC': SVC(probability=True, C=0.95, random_state=rs)}
    for n, c in clfs.items():
        c, tf = fit_timed(c, Xr[:, top], yr)
        out['classical'][n] = {**metrics(c, Xte[:, top], yte), 't_fit_s': tf, 'size_kb': size_kb(c)}
    # ---- static (cell 18): MinMax on non-oversampled train, LR-coef ranking, top 30, SMOTE
    sc = MinMaxScaler().fit(D['Xtr_raw']); Xt2, Xte2 = sc.transform(D['Xtr_raw']), sc.transform(D['Xte_raw'])
    lr = LogisticRegression(C=C_LR, max_iter=100, random_state=rs).fit(Xt2, D['ytr_raw'])
    top30 = np.argsort(np.abs(lr.coef_[0]))[::-1][:30]
    Xs, ys = SMOTE(random_state=rs).fit_resample(Xt2[:, top30], D['ytr_raw'])
    st = {'RF': RandomForestClassifier(random_state=rs, n_jobs=1), 'XGB': XGBClassifier(random_state=rs, eval_metric='logloss', n_jobs=1),
          'GB': GradientBoostingClassifier(random_state=rs), 'AB': AdaBoostClassifier(random_state=rs),
          'CB': CatBoostClassifier(random_state=rs, verbose=0, thread_count=1)}
    for n, c in st.items():
        c, tf = fit_timed(c, Xs, ys)
        out['static'][n] = {**metrics(c, Xte2[:, top30], yte), 't_fit_s': tf, 'size_kb': size_kb(c)}
    # ---- DES: pool fitted on SMOTE train (all features), tuned on a validation split of it (as in cell 25)
    fitted, tfit = {}, {}
    for n, c in base_pool(rs):
        fitted[n], tfit[n] = fit_timed(c, Xr, yr)
    Xtv, Xval, ytv, yval = train_test_split(Xr, yr, test_size=0.2, stratify=yr, random_state=rs)
    for p, names in POOLS.items():
        P = [fitted[n] for n in names]; tpool = sum(tfit[n] for n in names)
        out['des'][p] = {}
        for dn in POOL_DES[p]:
            cls, dfp, extra = DES[dn]
            grid = [dict(zip(extra, v)) for v in itertools.product(*extra.values())] or [{}]
            best, best_acc = None, -1
            for k in KS:
                for g in grid:
                    try:
                        m = cls(P, k=k, DFP=dfp, random_state=rs, **g).fit(Xtv, ytv)
                        a = accuracy_score(yval, m.predict(Xval))
                    except Exception:
                        continue
                    if a > best_acc:
                        best, best_acc = dict(k=k, **g), a
                    if p == 3 and dn in POOL_DES[3] and (not g or g == grid[0]):  # test metrics vs k (Table 22)
                        mf = cls(P, k=k, DFP=dfp, random_state=rs, **g).fit(Xr, yr)
                        mm = metrics(mf, Xte, yte)
                        out['kabl'].setdefault(dn, {})[k] = {'acc': mm['acc'], 'auc': mm['auc']}
            if best is None:
                continue
            m, tf = fit_timed(cls(P, DFP=dfp, random_state=rs, **best), Xr, yr)
            out['des'][p][dn] = {**metrics(m, Xte, yte), 'params': best, 't_fit_s': tf + tpool, 'size_kb': size_kb(m)}
    # ---- DES with a held-out DSEL (proper protocol): pool trained on 70% of the real train split (SMOTE there),
    #      regions of competence defined on the remaining 30% (real, non-oversampled patients)
    Xa, Xd, ya, yd = train_test_split(D['Xtr_raw'], D['ytr_raw'], test_size=0.3, stratify=D['ytr_raw'], random_state=rs)
    Xa, ya = SMOTE(random_state=rs).fit_resample(Xa, ya)
    sc3 = MinMaxScaler().fit(Xa); Xa, Xd, Xte3 = sc3.transform(Xa), sc3.transform(Xd), sc3.transform(D['Xte_raw'])
    P3 = [c.fit(Xa, ya) for n, c in base_pool(rs)]
    out['des_ho'] = {}
    for dn in POOL_DES[3]:
        cls, dfp, extra = DES[dn]
        for k in KS:
            try:
                m = cls(P3, k=k, DFP=dfp, random_state=rs).fit(Xd, yd)
                mm = metrics(m, Xte3, yte)
                out['des_ho'].setdefault(dn, {})[k] = {'acc': mm['acc'], 'auc': mm['auc']}
            except Exception as e:
                pass
    for n, c in zip(POOLS[3], P3):  # static members of the same pool, for reference
        mm = metrics(c, Xte3, yte); out['des_ho'].setdefault('member_' + n, {'acc': mm['acc'], 'auc': mm['auc']})
    print('done', rs, flush=True)
    return out

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=2); a = ap.parse_args()
    t0 = time.time()
    for rs in SEEDS:  # feature selection is cached per split (sequential: torch)
        split_features(rs)
    print('features ready', round(time.time() - t0), 's', flush=True)
    res = Parallel(n_jobs=a.jobs)(delayed(run_split)(rs) for rs in SEEDS)
    json.dump(res, open(RESULTS / 'early_fusion.json', 'w'))
    print('total', round(time.time() - t0), 's')
