"""All reported metrics, confidence intervals and the alert-precision calculation (no TensorFlow dependency)."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, confusion_matrix)
from . import config as C


def full_metrics(y_true, y_pred, n_classes, macro_map=None):
    """Majority/minority groups use TEST support N_c >= MINORITY_THRESHOLD (Section 3.7.2)."""
    labels = np.arange(n_classes); cnt = np.bincount(y_true, minlength=n_classes)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    maj, mino = (cnt >= C.MINORITY_THRESHOLD), (cnt < C.MINORITY_THRESHOLD) & (cnt > 0)
    out = dict(acc=accuracy_score(y_true, y_pred), bal_acc=balanced_accuracy_score(y_true, y_pred),
               macro_f1=f[cnt > 0].mean(), weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
               weighted_recall=(r * cnt).sum() / cnt.sum(), mcc=matthews_corrcoef(y_true, y_pred),
               macro_f1_major=f[maj].mean(), macro_f1_minor=f[mino].mean(), n_major=int(maj.sum()), n_minor=int(mino.sum()))
    if macro_map is not None:
        out["macro8_f1"] = f1_score(macro_map[y_true], macro_map[y_pred], average="macro", zero_division=0)
        out["macro8_acc"] = accuracy_score(macro_map[y_true], macro_map[y_pred])
    return {k: float(v) for k, v in out.items()}


def t_ci(x, level=0.95):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return float(x.mean()), np.nan, np.nan
    h = stats.t.ppf((1 + level) / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def paired_delta(df, metric, ref_tag, tag):
    """Seed-wise paired difference (tag - ref) with a 95 % t-interval (needs per-seed results from scripts/run_ablation.py)."""
    a = df[df.tag == tag].set_index("seed")[metric]; b = df[df.tag == ref_tag].set_index("seed")[metric]
    d = (a[a.index.intersection(b.index)] - b[a.index.intersection(b.index)]).values
    m, lo, hi = t_ci(d)
    return dict(tag=tag, ref=ref_tag, metric=metric, n=len(d), delta=m, ci_lo=lo, ci_hi=hi)


def welch_delta(mean_a, sd_a, mean_b, sd_b, n=5, level=0.95):
    """UNPAIRED difference a - b with a Welch interval from mean +/- SD of n seeds each (used when only summary rows exist)."""
    va, vb = sd_a ** 2 / n, sd_b ** 2 / n; d = mean_a - mean_b; se = np.sqrt(va + vb)
    dfw = (va + vb) ** 2 / (va ** 2 / (n - 1) + vb ** 2 / (n - 1)); h = stats.t.ppf((1 + level) / 2, dfw) * se
    return d, d - h, d + h


def per_class_ci(y_true, y_pred, n_classes, B=1000, seed=2026):
    """Per-class precision/recall/F1 with stratified multinomial bootstrap CIs on the confusion matrix and Clopper-Pearson recall CIs."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    def prf(c):
        tp = np.diag(c).astype(float); fp = c.sum(0) - tp; fn = c.sum(1) - tp
        p = np.divide(tp, tp + fp, out=np.zeros(n_classes), where=(tp + fp) > 0)
        r = np.divide(tp, tp + fn, out=np.zeros(n_classes), where=(tp + fn) > 0)
        f = np.divide(2 * p * r, p + r, out=np.zeros(n_classes), where=(p + r) > 0); return p, r, f
    rng = np.random.default_rng(seed); probs = cm / np.maximum(cm.sum(1, keepdims=True), 1); boots = np.zeros((B, 3, n_classes))
    for b in range(B):
        cb = np.stack([rng.multinomial(cm[i].sum(), probs[i]) if cm[i].sum() else np.zeros(n_classes, int) for i in range(n_classes)])
        boots[b] = np.stack(prf(cb))
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0); p0, r0, f0 = prf(cm)
    def cp(k, n, a=0.05):
        if n == 0: return (np.nan, np.nan)
        return (stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0, stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0)
    cps = [cp(int(cm[i, i]), int(cm[i].sum())) for i in range(n_classes)]
    return pd.DataFrame(dict(n_test=cm.sum(1), precision=p0, p_lo=lo[0], p_hi=hi[0], recall=r0, r_lo=lo[1], r_hi=hi[1],
                             recall_CP_lo=[c[0] for c in cps], recall_CP_hi=[c[1] for c in cps], f1=f0, f1_lo=lo[2], f1_hi=hi[2])), boots


def alert_precision(y_true, y_pred, benign_idx, prevalences=(0.001, 0.01, 0.10), B=1000, seed=1):
    """Attack-vs-benign TPR/FPR and alert precision prev*TPR / (prev*TPR + (1-prev)*FPR) with bootstrap intervals."""
    is_att, pred_att = y_true != benign_idx, y_pred != benign_idx
    tpr = (pred_att & is_att).sum() / is_att.sum(); fpr = (pred_att & ~is_att).sum() / (~is_att).sum()
    rng = np.random.default_rng(seed); n = len(y_true); tb, fb = [], []
    for _ in range(B):
        ii = rng.integers(0, n, n); ia, pa = is_att[ii], pred_att[ii]
        tb.append((pa & ia).sum() / max(ia.sum(), 1)); fb.append((pa & ~ia).sum() / max((~ia).sum(), 1))
    rows = [dict(prevalence="test set", TPR=tpr, FPR=fpr, test_attack_share=is_att.mean())]
    for prev in prevalences:
        f = lambda t, q: prev * t / (prev * t + (1 - prev) * q)
        rows.append(dict(prevalence=f"{prev:.1%}", TPR=tpr, FPR=fpr, alert_precision=f(tpr, fpr),
                         ci_lo=f(np.percentile(tb, 2.5), np.percentile(fb, 97.5)), ci_hi=f(np.percentile(tb, 97.5), np.percentile(fb, 2.5)),
                         false_alerts_per_1M_flows=(1 - prev) * fpr * 1e6))
    return pd.DataFrame(rows)
