import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from aegis_net.data import load_ciciot2023, split_indices
from aegis_net.taxonomy import macro_vector

def add_data_args(ap):
    ap.add_argument("--train-csv", required=True, help="CICIoT2023 train.csv (predefined training file)")
    ap.add_argument("--test-csv", required=True, help="CICIoT2023 test.csv (predefined test file)")
    ap.add_argument("--out", default="./results", help="output directory (runs/, preds/, tables/, release/)")
    return ap

def load_data(a):
    X, y, Xte, yte, feats, le = load_ciciot2023(a.train_csv, a.test_csv)
    itr, iva = split_indices(y)
    os.makedirs(f"{a.out}/release", exist_ok=True)
    np.savez_compressed(f"{a.out}/release/split_indices.npz", idx_train=itr, idx_val=iva)
    print(f"train.csv={len(y):,} train={len(itr):,} val={len(iva):,} test={len(yte):,} features={len(feats)} classes={len(le.classes_)}")
    return dict(X=X, y=y, Xte=Xte, yte=yte, idx_tr=itr, idx_val=iva, n_classes=len(le.classes_),
                macro_map=macro_vector(list(le.classes_))), feats, le
