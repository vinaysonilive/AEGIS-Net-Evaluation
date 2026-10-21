"""Evaluate a saved model (best_model.keras + scaler.pkl + label_encoder.pkl) on the predefined test file: headline metrics,
per-class CIs, alert precision, duplicate audit and de-duplicated scoring (no training)."""
from _common import *
import joblib, hashlib, pandas as pd, tensorflow as tf
from aegis_net import config as C
from aegis_net.metrics import full_metrics, per_class_ci, alert_precision
from aegis_net.model import CategoricalFocalLoss
from aegis_net.data import load_ciciot2023

ap = add_data_args(argparse.ArgumentParser())
ap.add_argument("--artifacts", required=True, help="folder with best_model.keras, scaler.pkl, label_encoder.pkl")
a = ap.parse_args(); os.makedirs(f"{a.out}/tables", exist_ok=True)
le = joblib.load(f"{a.artifacts}/label_encoder.pkl"); sc = joblib.load(f"{a.artifacts}/scaler.pkl")
X, y, Xte, yte, feats, _ = load_ciciot2023(a.train_csv, a.test_csv, label_encoder=le)
nc = len(le.classes_); mm = macro_vector(list(le.classes_))
model = tf.keras.models.load_model(f"{a.artifacts}/best_model.keras", compile=False, custom_objects={"CategoricalFocalLoss": CategoricalFocalLoss})
yp = model.predict(sc.transform(Xte).astype("float32")[..., None], batch_size=2048, verbose=1).argmax(1)   # float64 -> quantile -> float32
M = full_metrics(yte, yp, nc, mm); print(json.dumps(M, indent=1)); json.dump(M, open(f"{a.out}/tables/saved_model_metrics.json", "w"), indent=1)
pc, boots = per_class_ci(yte, yp, nc); pc.insert(0, "cls", list(le.classes_)); pc.to_csv(f"{a.out}/tables/per_class_ci.csv", index=False)
alert_precision(yte, yp, list(le.classes_).index(C.BENIGN)).to_csv(f"{a.out}/tables/alert_precision.csv", index=False)
# exact-duplicate audit on RAW features (train split vs test)
itr, iva = split_indices(y)
rh = lambda Z: np.concatenate([pd.util.hash_pandas_object(pd.DataFrame(Z[i:i + 500000]), index=False).values for i in range(0, len(Z), 500000)])
htr, hte = rh(X[itr]), rh(Xte); in_tr = np.isin(hte, htr); _, first = np.unique(hte, return_index=True); isf = np.zeros(len(hte), bool); isf[first] = True
rows = []
for nm, mk in [("all test rows", np.ones(len(yte), bool)), ("exclude rows seen in train", ~in_tr), ("exclude train-seen and test-internal duplicates", ~in_tr & isf)]:
    r = full_metrics(yte[mk], yp[mk], nc, mm); r.update(subset=nm, n=int(mk.sum())); rows.append(r)
pd.DataFrame(rows).to_csv(f"{a.out}/tables/dedup_scoring.csv", index=False)
print(f"test rows with an exact training duplicate: {in_tr.sum():,} ({100 * in_tr.mean():.2f} %)")
