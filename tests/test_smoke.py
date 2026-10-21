"""Smoke tests on synthetic data (no dataset needed):  pytest -q"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pytest
from aegis_net.data import selective_smote, log_class_weights, fit_scaler
from aegis_net.metrics import full_metrics, welch_delta, alert_precision
from aegis_net.taxonomy import macro_vector, MACRO_ORDER

def test_selective_smote_only_raises_small_classes():
    rng = np.random.RandomState(0); y = np.r_[np.zeros(5000, int), np.ones(40, int), np.full(10, 2)]
    X = rng.randn(len(y), 46); Xf, yf, n = selective_smote(X, y, floor=100, seed=1)
    c = np.bincount(yf); assert c[0] == 5000 and c[1] == 100 and c[2] == 100 and n == 150

def test_class_weights_mean_one():
    y = np.r_[np.zeros(100, int), np.ones(10, int)]; w = log_class_weights(y, 2)
    assert abs(w[[0, 1]].mean() - 1.0) < 1e-6 and w[1] > w[0]

def test_sqlinjection_is_web_not_benign():
    assert MACRO_ORDER[macro_vector(["SqlInjection"])[0]] == "Web"
    with pytest.raises(KeyError): macro_vector(["NotAClass"])

def test_metrics_and_welch():
    y = np.array([0, 0, 1, 1, 2, 2]); p = np.array([0, 0, 1, 0, 2, 2]); m = full_metrics(y, p, 3)
    assert abs(m["acc"] - 5 / 6) < 1e-9 and abs(m["acc"] - m["weighted_recall"]) < 1e-9
    d, lo, hi = welch_delta(0.7481, 0.0060, 0.7010, 0.0072); assert lo < d < hi and lo > 0

def test_alert_precision_decreases_with_prevalence():
    rng = np.random.RandomState(0); y = rng.randint(0, 3, 5000); p = np.where(rng.rand(5000) < 0.9, y, rng.randint(0, 3, 5000))
    t = alert_precision(y, p, 0, B=50); ap = t.alert_precision.dropna().values; assert ap[0] < ap[1] < ap[2]

def test_quantile_scaler_float64_roundtrip():
    X = np.random.RandomState(0).lognormal(size=(2000, 46)); sc = fit_scaler("quantile", X); Z = sc.transform(X)
    assert Z.shape == X.shape and np.isfinite(Z).all()

def test_architecture_parameter_count():
    tf = pytest.importorskip("tensorflow")
    from aegis_net.model import build_model, REF_PARAMS, build_mlp, mlp_hidden_for
    assert build_model().count_params() == REF_PARAMS == 210978
    assert build_model(attn=False).count_params() == 144674 and build_model(rnn="gap").count_params() == 112162
    assert build_mlp(hidden=(mlp_hidden_for(REF_PARAMS),) * 3).count_params() == 211399
