"""Data loading, per-split neural feature selection and the evaluation protocol of the paper (Early Fusion).

Every data-dependent step is fitted on the training split only: neural feature selection,
Local Outlier Factor, SMOTE and Min-Max scaling. The test split is balanced by random undersampling."""
import warnings, os, pickle, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import deslib.base as _db  # DESlib 0.3.7 compatibility with recent scikit-learn versions
from sklearn.utils.validation import validate_data as _vd
_db.BaseDS._validate_data = lambda self, *a, **k: _vd(self, *a, **k)
from src.nnsel import NeuralFeatureSelector
from src.config import DATA, ENCODED, CACHE

cf = pd.read_csv(DATA / 'labels.csv')
Y = (cf['BCaseControlStatus'] == 'Control').astype(int).values  # Case=0, Control=1 (LabelEncoder order)
PID = cf['PtID'].values
MODS = ['tabular', 'text', 'timeseries']
ENC = {m: pd.read_csv(ENCODED / f'{m}_encoded.csv').set_index('PtID').loc[PID].values for m in MODS}
SEL = {
    'tabular': dict(target_features=50, hidden_dims=[256, 128, 64], epochs=80, learning_rate=0.001, batch_size=16, patience=10),
    'text': dict(target_features=100, hidden_dims=[512, 256, 128], epochs=60, learning_rate=0.0005, batch_size=16, patience=8),
    'timeseries': dict(target_features=32, hidden_dims=[128, 64, 32], epochs=100, learning_rate=0.001, batch_size=16, patience=15),
}

def split_features(rs):
    """Neural feature selection fitted on the training split only. Cached per split."""
    f = f'{CACHE}/feat_{rs}.pkl'
    if os.path.exists(f):
        return pickle.load(open(f, 'rb'))
    idx = np.arange(len(Y))
    tr, te = train_test_split(idx, test_size=0.3, stratify=Y, random_state=rs)
    torch.manual_seed(rs); np.random.seed(rs)
    out = {'tr': tr, 'te': te, 'mods': {}}
    for m, kw in SEL.items():
        s = NeuralFeatureSelector(**kw).fit(ENC[m][tr], Y[tr])
        out['mods'][m] = (s.transform(ENC[m][tr]), s.transform(ENC[m][te]))
    pickle.dump(out, open(f, 'wb'))
    return out

def prep(rs, mods=MODS, raw=False, balance='smote'):
    """Paper protocol: split -> LOF(train) -> balance(train) -> undersample test -> MinMax(train).
    Returns dict with SMOTE+scaled train/test and the un-oversampled scaled train."""
    F = split_features(rs)
    if raw:
        Xtr = np.hstack([ENC[m][F['tr']] for m in mods]); Xte = np.hstack([ENC[m][F['te']] for m in mods])
    else:
        Xtr = np.hstack([F['mods'][m][0] for m in mods]); Xte = np.hstack([F['mods'][m][1] for m in mods])
    ytr, yte = Y[F['tr']], Y[F['te']]
    mask = LocalOutlierFactor().fit_predict(Xtr) != -1
    Xtr, ytr = Xtr[mask], ytr[mask]
    mc = np.bincount(yte).min()
    Xte, yte = RandomUnderSampler(sampling_strategy={0: mc, 1: mc}, random_state=rs).fit_resample(Xte, yte)
    if balance == 'smote':
        Xr, yr = SMOTE(random_state=rs).fit_resample(Xtr, ytr)
    elif balance == 'rus':
        Xr, yr = RandomUnderSampler(random_state=rs).fit_resample(Xtr, ytr)
    else:
        Xr, yr = Xtr, ytr
    sc = MinMaxScaler().fit(Xr)
    return dict(Xr=sc.transform(Xr), yr=yr, Xte=sc.transform(Xte), yte=yte, Xtr_raw=Xtr, ytr_raw=ytr, Xte_raw=Xte)

def metrics(clf, Xte, yte):
    t = time.perf_counter(); p = clf.predict_proba(Xte)[:, 1]; t_inf = (time.perf_counter() - t) / len(yte)
    yp = (p >= 0.5).astype(int) if not hasattr(clf, 'predict') else clf.predict(Xte)
    fpr, tpr, _ = roc_curve(yte, p)
    return dict(acc=accuracy_score(yte, yp), prec=precision_score(yte, yp, average='weighted'),
                rec=recall_score(yte, yp, average='weighted'), f1=f1_score(yte, yp, average='weighted'),
                auc=auc(fpr, tpr), sens_case=float(np.mean(yp[yte == 0] == 0)), spec=float(np.mean(yp[yte == 1] == 1)),
                fpr=fpr.tolist(), tpr=tpr.tolist(), t_inf_ms=1000 * t_inf)

def size_kb(obj):
    try:
        return len(pickle.dumps(obj)) / 1024
    except Exception:
        return float('nan')
