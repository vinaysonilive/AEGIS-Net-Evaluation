"""Model, losses and parameter-matched ablation variants (reference architecture: 210,978 parameters)."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L


@tf.keras.utils.register_keras_serializable(package="CustomLosses")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Loss of the reference notebook (alpha is a scalar multiplier). Needed to deserialise best_model.keras."""
    def __init__(self, gamma=2.0, alpha=0.25, name="categorical_focal_loss", **kw):
        super().__init__(name=name, **kw); self.gamma, self.alpha = gamma, alpha
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(self.alpha * tf.pow(1.0 - y_pred, self.gamma) * ce, axis=-1)
    def get_config(self):
        c = super().get_config(); c.update(gamma=self.gamma, alpha=self.alpha); return c


class SparseFocalLoss(tf.keras.losses.Loss):
    """Same maths as CategoricalFocalLoss for integer labels (no one-hot matrix in memory). gamma=0 gives alpha*CE."""
    def __init__(self, gamma=2.0, alpha=0.25, n_classes=34, name="sparse_focal", **kw):
        super().__init__(name=name, **kw); self.gamma, self.alpha, self.n = gamma, alpha, n_classes
    def call(self, y_true, y_pred):
        y = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), self.n)
        p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return tf.reduce_sum(self.alpha * tf.pow(1.0 - p, self.gamma) * (-y * tf.math.log(p)), axis=-1)


def build_model(n_feat=46, n_cls=34, attn=True, rnn="bilstm", lstm_units=64, dense=128, conv=(64, 128),
                heads=4, key_dim=32, name="aegis"):
    """Conv64-Pool-Drop-Conv128-Pool-Drop-[MHA+Add+LN]-BiLSTM64-Drop-Dense128-Drop-Softmax.
    rnn: 'bilstm' (reference, last state) | 'flatten' (Flatten replaces the BiLSTM) | 'gap' (GlobalAveragePooling1D)."""
    inp = L.Input((n_feat, 1)); x = inp
    x = L.Conv1D(conv[0], 3, activation="relu", padding="same")(x); x = L.MaxPooling1D(2)(x); x = L.Dropout(0.3)(x)
    x = L.Conv1D(conv[1], 3, activation="relu", padding="same")(x); x = L.MaxPooling1D(2)(x); x = L.Dropout(0.3)(x)
    if attn:
        a = L.MultiHeadAttention(num_heads=heads, key_dim=key_dim)(x, x)
        x = L.LayerNormalization()(L.Add()([x, a]))
    if rnn == "bilstm":
        x = L.Bidirectional(L.LSTM(lstm_units, return_sequences=False))(x); x = L.Dropout(0.3)(x)
    elif rnn == "flatten":
        x = L.Flatten()(x); x = L.Dropout(0.3)(x)
    else:
        x = L.GlobalAveragePooling1D()(x); x = L.Dropout(0.3)(x)
    x = L.Dense(dense, activation="relu")(x); x = L.Dropout(0.4)(x)
    return tf.keras.Model(inp, L.Dense(n_cls, activation="softmax")(x), name=name)


def build_mlp(n_feat=46, n_cls=34, hidden=(305, 305, 305)):
    inp = L.Input((n_feat, 1)); x = L.Flatten()(inp)
    for h in hidden:
        x = L.Dense(h, activation="relu")(x); x = L.Dropout(0.3)(x)
    return tf.keras.Model(inp, L.Dense(n_cls, activation="softmax")(x), name="mlp")


REF_PARAMS = 210_978   # asserted by tests/test_smoke.py and scripts/train.py


def match_params(target, key, lo, hi, **spec):
    """Grid-search the integer `key` (lstm_units / dense) so the parameter count is closest to `target`."""
    best = min(range(lo, hi + 1), key=lambda v: abs(build_model(**{**spec, key: v}).count_params() - target))
    return best, build_model(**{**spec, key: best}).count_params()


def mlp_hidden_for(target, n_feat=46, n_cls=34):
    return min(range(32, 700), key=lambda h: abs(build_mlp(n_feat, n_cls, (h,) * 3).count_params() - target))


def layer_shape_table(model):
    import pandas as pd
    return pd.DataFrame([dict(layer=l.name, type=l.__class__.__name__, output_shape=str(l.output.shape),
                              params=l.count_params()) for l in model.layers])
