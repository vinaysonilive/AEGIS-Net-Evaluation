"""Data loading, split, scaling and selective SMOTE (no TensorFlow dependency)."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, LabelEncoder
from . import config as C


def load_ciciot2023(train_csv, test_csv, label_encoder=None):
    """Loads the predefined train.csv / test.csv, shuffles both with random_state=SPLIT_SEED (as in the reference notebook) and
    returns RAW float64 arrays. float64 matters: casting to float32 BEFORE the quantile transformer collapses nearly equal
    feature values onto the same float32 number and lowered the saved model's macro-F1 from 0.7431 to 0.6434 in our audit."""
    tr = pd.read_csv(train_csv).sample(frac=1.0, random_state=C.SPLIT_SEED).reset_index(drop=True)
    te = pd.read_csv(test_csv).sample(frac=1.0, random_state=C.SPLIT_SEED).reset_index(drop=True)
    feats = [c for c in tr.columns if c != C.LABEL_COLUMN]
    le = label_encoder or LabelEncoder().fit(sorted(tr[C.LABEL_COLUMN].unique()))
    X = tr[feats].values.astype(np.float64); y = le.transform(tr[C.LABEL_COLUMN].values).astype(np.int32)
    Xte = te[feats].values.astype(np.float64); yte = le.transform(te[C.LABEL_COLUMN].values).astype(np.int32)
    return X, y, Xte, yte, feats, le


def split_indices(y):
    """80/20 stratified split of train.csv (64 % / 16 % of the data after the official test split is removed)."""
    return train_test_split(np.arange(len(y)), test_size=C.VAL_FRACTION, random_state=C.SPLIT_SEED, stratify=y)


def fit_scaler(kind, X_fit):
    """kind: 'quantile' | 'standard'. Fit on the rows you pass (train split only for the reported method)."""
    if kind == "quantile":
        return QuantileTransformer(n_quantiles=C.N_QUANTILES, output_distribution="normal",
                                   subsample=C.QUANTILE_SUBSAMPLE, random_state=C.SPLIT_SEED).fit(X_fit)
    if kind == "standard":
        return StandardScaler().fit(X_fit)
    raise ValueError(kind)


def selective_smote(X, y, floor=C.T_FLOOR, seed=42):
    """Selective SMOTE: classes with fewer than `floor` samples are raised to `floor`; larger classes are untouched.
    NOTE: all 46 (quantile-normalised) features are interpolated - there is NO discrete/binary feature mask in the reported experiments."""
    from imblearn.over_sampling import SMOTE
    cnt = pd.Series(y).value_counts()
    strat = {int(c): int(floor) for c, n in cnt.items() if n < floor}
    if not strat:
        return X, y, 0
    k = min(5, max(1, int(min(cnt[c] for c in strat)) - 1))
    Xf, yf = SMOTE(sampling_strategy=strat, random_state=seed, k_neighbors=k).fit_resample(X, y)
    return Xf, yf, len(yf) - len(y)


def log_class_weights(y, n_classes=34):
    """W_c = ln(1 + N / (N_c + 1)), normalised to mean 1 (computed on the post-SMOTE training labels)."""
    cls, c = np.unique(y, return_counts=True)
    w = np.log(1.0 + len(y) / (c + 1.0)); w = w / w.mean()
    full = np.ones(n_classes, np.float32); full[cls] = w
    return full
