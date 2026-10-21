"""Seeded, resumable training of the 14 ablation configurations (M1-M14) of the manuscript."""
import json, os, time, gc, random
import numpy as np, pandas as pd, tensorflow as tf
from . import config as C
from .data import split_indices, fit_scaler, selective_smote, log_class_weights
from .metrics import full_metrics
from .model import build_model, build_mlp, match_params, mlp_hidden_for, SparseFocalLoss, REF_PARAMS

# loss, class weights, SMOTE, scaler, architecture  (identical to the M1-M14 rows of Table 2)
CONFIGS = {
 "M1":  dict(loss="ce",    cw=False, smote=False, scaler="standard", arch="full"),
 "M2":  dict(loss="ce",    cw=False, smote=True,  scaler="standard", arch="full"),
 "M3":  dict(loss="ce",    cw=True,  smote=False, scaler="standard", arch="full"),
 "M4":  dict(loss="ce",    cw=True,  smote=True,  scaler="standard", arch="full"),
 "M5":  dict(loss="focal", cw=False, smote=False, scaler="standard", arch="full"),
 "M6":  dict(loss="focal", cw=False, smote=True,  scaler="quantile", arch="full"),   # proposed final model
 "M7":  dict(loss="focal", cw=True,  smote=False, scaler="standard", arch="full"),
 "M8":  dict(loss="focal", cw=True,  smote=True,  scaler="standard", arch="full"),
 "M9":  dict(loss="focal", cw=False, smote=True,  scaler="standard", arch="full"),
 "M10": dict(loss="focal", cw=False, smote=True,  scaler="quantile", arch="no_attention"),
 "M11": dict(loss="focal", cw=False, smote=True,  scaler="quantile", arch="no_bilstm"),
 "M12": dict(loss="focal", cw=False, smote=True,  scaler="quantile", arch="no_attention_matched"),
 "M13": dict(loss="focal", cw=False, smote=True,  scaler="quantile", arch="no_bilstm_matched"),
 "M14": dict(loss="focal", cw=False, smote=True,  scaler="quantile", arch="mlp_matched"),
}


def make_model(arch, n_feat, n_cls):
    if arch == "full":                 return build_model(n_feat, n_cls)
    if arch == "no_attention":         return build_model(n_feat, n_cls, attn=False)
    if arch == "no_bilstm":            return build_model(n_feat, n_cls, rnn="gap")
    if arch == "no_attention_matched":
        u, _ = match_params(REF_PARAMS, "lstm_units", 16, 200, attn=False); return build_model(n_feat, n_cls, attn=False, lstm_units=u)
    if arch == "no_bilstm_matched":
        d, _ = match_params(REF_PARAMS, "dense", 16, 400, attn=True, rnn="flatten"); return build_model(n_feat, n_cls, rnn="flatten", dense=d)
    if arch == "mlp_matched":          return build_mlp(n_feat, n_cls, (mlp_hidden_for(REF_PARAMS, n_feat, n_cls),) * 3)
    raise ValueError(arch)


def set_seed(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    if deterministic: tf.config.experimental.enable_op_determinism()


def run_experiment(tag, seed, data, out_dir, scaler_fit="train_split", epochs=C.EPOCHS, train_frac=1.0, keep_weights=False):
    """data = dict(X, y, Xte, yte, idx_tr, idx_val, n_classes, macro_map). Writes runs/<tag>__s<seed>.json and preds/...npz; skips finished runs."""
    cfg = CONFIGS[tag]; os.makedirs(f"{out_dir}/runs", exist_ok=True); os.makedirs(f"{out_dir}/preds", exist_ok=True)
    jpath = f"{out_dir}/runs/{tag}__s{seed}.json"
    if os.path.exists(jpath): return json.load(open(jpath))
    set_seed(seed); tf.keras.backend.clear_session(); t0 = time.time()
    X, y, Xte, yte, itr, iva, nc = (data[k] for k in ("X", "y", "Xte", "yte", "idx_tr", "idx_val", "n_classes"))
    fit_rows = X[itr] if scaler_fit == "train_split" else X                       # 'train_csv' reproduces the saved-scaler pipeline
    sc = fit_scaler(cfg["scaler"], fit_rows)
    Xtr, ytr = sc.transform(X[itr]).astype("float32"), y[itr]; Xva, yva = sc.transform(X[iva]).astype("float32"), y[iva]
    Xt = sc.transform(Xte).astype("float32")
    if train_frac < 1.0:
        keep = np.random.RandomState(seed).rand(len(ytr)) < train_frac; Xtr, ytr = Xtr[keep], ytr[keep]
    n_syn = 0
    if cfg["smote"]: Xtr, ytr, n_syn = selective_smote(Xtr, ytr, C.T_FLOOR, seed)
    sw = log_class_weights(ytr, nc)[ytr] if cfg["cw"] else None
    model = make_model(cfg["arch"], Xtr.shape[1], nc)
    loss = tf.keras.losses.SparseCategoricalCrossentropy() if cfg["loss"] == "ce" else SparseFocalLoss(C.GAMMA, C.ALPHA, nc)
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LR), loss=loss, metrics=["sparse_categorical_accuracy"])
    hist = model.fit(Xtr[..., None], ytr, sample_weight=sw, validation_data=(Xva[..., None], yva), epochs=epochs, batch_size=C.BATCH_SIZE,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.PATIENCE, restore_best_weights=True)], verbose=2)
    proba = model.predict(Xt[..., None], batch_size=1024, verbose=0); yp = proba.argmax(1)
    m = full_metrics(yte, yp, nc, data.get("macro_map"))
    m.update(tag=tag, seed=seed, params=int(model.count_params()), n_train=int(len(ytr)), n_synthetic=int(n_syn), scaler_fit=scaler_fit,
             epochs_run=len(hist.history["loss"]), minutes=round((time.time() - t0) / 60, 2), **cfg)
    np.savez_compressed(f"{out_dir}/preds/{tag}__s{seed}.npz", y_true=yte, y_pred=yp.astype(np.int16), proba=proba.astype(np.float16))
    if keep_weights:
        model.save_weights(f"{out_dir}/runs/{tag}__s{seed}.weights.h5"); import joblib; joblib.dump(sc, f"{out_dir}/runs/{tag}__s{seed}.scaler.pkl")
    json.dump(m, open(jpath, "w"))
    print(f"[{tag} s{seed}] macroF1={m['macro_f1']:.4f} minorF1={m['macro_f1_minor']:.4f} acc={m['acc']:.4f} ({m['minutes']} min)")
    del Xtr, Xva, model; gc.collect(); return m
