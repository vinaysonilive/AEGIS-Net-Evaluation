"""Train any of the 14 ablation configurations (M1-M14) for several seeds. Per-seed results enable PAIRED confidence intervals.
Example (final model, five seeds, full protocol):  python scripts/run_ablation.py --train-csv train.csv --test-csv test.csv --configs M6
Reduced protocol used for the re-runs in the revision:  --epochs 5 --train-frac 0.25 --seeds 42"""
from _common import *
from aegis_net import config as C
from aegis_net.train import run_experiment, CONFIGS

ap = add_data_args(argparse.ArgumentParser())
ap.add_argument("--configs", nargs="+", default=["M6"], choices=list(CONFIGS))
ap.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
ap.add_argument("--epochs", type=int, default=C.EPOCHS)
ap.add_argument("--train-frac", type=float, default=1.0, help="random fraction of the training split (0.25 = reduced protocol)")
ap.add_argument("--scaler-fit", choices=["train_split", "train_csv"], default="train_split",
                help="train_split: scaler fitted on the 64 %% training split only (manuscript claim); train_csv: whole train.csv (saved-model pipeline)")
ap.add_argument("--keep-weights", action="store_true")
a = ap.parse_args()
data, feats, le = load_data(a)
json.dump(dict(features=feats, classes=list(le.classes_)), open(f"{a.out}/release/schema.json", "w"), indent=1)
for tag in a.configs:
    for s in a.seeds:
        run_experiment(tag, s, data, a.out, scaler_fit=a.scaler_fit, epochs=a.epochs, train_frac=a.train_frac, keep_weights=a.keep_weights)
