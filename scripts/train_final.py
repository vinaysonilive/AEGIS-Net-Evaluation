"""Train the proposed final model (M6: quantile transformer + selective SMOTE + focal loss) and export deployable artifacts:
best_model.keras, scaler.pkl, label_encoder.pkl, config.json. One seed per call (default 42)."""
from _common import *
import joblib, tensorflow as tf
from aegis_net import config as C
from aegis_net.train import run_experiment
from aegis_net.model import build_model, REF_PARAMS

ap = add_data_args(argparse.ArgumentParser())
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=C.EPOCHS)
a = ap.parse_args()
assert build_model().count_params() == REF_PARAMS, "architecture differs from the manuscript (210,978 parameters)"
data, feats, le = load_data(a)
m = run_experiment("M6", a.seed, data, a.out, epochs=a.epochs, keep_weights=True)
model = build_model(len(feats), len(le.classes_)); model.load_weights(f"{a.out}/runs/M6__s{a.seed}.weights.h5")
art = f"{a.out}/release"; model.save(f"{art}/best_model.keras")
joblib.dump(joblib.load(f"{a.out}/runs/M6__s{a.seed}.scaler.pkl"), f"{art}/scaler.pkl"); joblib.dump(le, f"{art}/label_encoder.pkl")
json.dump(dict(seed=a.seed, hyper=dict(batch=C.BATCH_SIZE, epochs=a.epochs, patience=C.PATIENCE, lr=C.LR, gamma=C.GAMMA, alpha=C.ALPHA,
          t_floor=C.T_FLOOR), tensorflow=tf.__version__, test_metrics=m), open(f"{art}/config.json", "w"), indent=2)
print("artifacts written to", art)
