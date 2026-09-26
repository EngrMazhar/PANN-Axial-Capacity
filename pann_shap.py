"""Shapley attribution of the learned correction delta (manuscript Fig. 15).

Supplemental code for "Physics-anchored neural network for the axial capacity of
FRP-strengthened hollow steel columns".

For each leave-one-source-out fold the PANN is refitted on the training programs and the
correction branch delta(x), Eq. (42), is explained on the withheld program with a
KernelExplainer. The anchor is not explained, because it carries no trainable weight.
Attribution is therefore out of sample for every record.

The refit uses the configuration selected most often across the LOSO folds (Table 4):
hidden layers (96, 48), L2 penalty 0.01, learning rate 0.003, correction bound a = 1.2,
one seed per fold. KernelExplainer uses a 15-cluster k-means background drawn from up to
100 training records and 200 coalition samples per record.

    python pann_shap.py                  # all 19 folds, resumable through shap_ckpt/
    python pann_shap.py --only "Teng & Hu 2007 (CBM)"   # a single fold, for a quick check
"""
import os, sys, time, argparse, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import shap

DATA_S2 = "PANN_Data_S2_Derived_Quantities_and_Model_Inputs.xlsx"
FEATURES = ["ln_A_es", "fy_over_235", "Deq_over_teq", "lambda_bar",
            "shape_CHS", "rho_L", "rho_H", "Lf_over_L"]
BOUND, HIDDEN, L2, LR = 1.2, (96, 48), 1e-2, 3e-3
MAX_EPOCHS, PATIENCE, VAL_FRACTION = 2000, 60, 0.15
CKPT = "shap_ckpt"


class BoundedCorrection(layers.Layer):
    """delta = a tanh(z / a), Eq. (42). No trainable weight."""
    def __init__(self, bound=1.2, **kw):
        super().__init__(**kw)
        self.bound = float(bound)

    def call(self, z):
        return self.bound*tf.tanh(z/self.bound)


def build(n_in, seed=0):
    """Returns the full PANN, Eq. (41), and the correction branch alone."""
    keras.utils.set_random_seed(seed)
    x = keras.Input((n_in,), name="inputs")
    h = x
    for i, u in enumerate(HIDDEN, 1):
        h = layers.Dense(u, activation="tanh", kernel_regularizer=keras.regularizers.l2(L2),
                         name=f"hidden_{i}")(h)
    z = layers.Dense(1, kernel_regularizer=keras.regularizers.l2(L2), name="raw_output")(h)
    a = keras.Input((1,), name="ln_N_phys")
    d = BoundedCorrection(BOUND, name="bounded_correction")(z)
    full = keras.Model([x, a], layers.Add(name="anchor_offset")([a, d]))
    return full, keras.Model(x, d)


def train(model, tin, ytr, vin, yval):
    """Full-batch Adam on the squared error of ln P_u, Eq. (43), with early stopping."""
    opt = keras.optimizers.Adam(LR)

    @tf.function(reduce_retracing=True)
    def step(inp, t):
        with tf.GradientTape() as tape:
            p = tf.squeeze(model(inp, training=True), -1)
            loss = tf.reduce_mean(tf.square(p - t))
            if model.losses:
                loss += tf.add_n(model.losses)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

    @tf.function(reduce_retracing=True)
    def val_loss(inp, t):
        return tf.reduce_mean(tf.square(tf.squeeze(model(inp, training=False), -1) - t))

    best, weights, stale = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        step(tin, ytr)
        v = float(val_loss(vin, yval))
        if v < best - 1e-7:
            best, stale, weights = v, 0, [w.numpy() for w in model.weights]
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if weights:
        for w, x in zip(model.weights, weights):
            w.assign(x)


def run(only=None):
    ml = pd.read_excel(DATA_S2, sheet_name="Model_inputs")
    X = ml[FEATURES].to_numpy(np.float32)
    y = ml["ln_P_u"].to_numpy(np.float32)
    anchor = ml["ln_N_phys"].to_numpy(np.float32)
    grp = ml["loso_group"].to_numpy()
    os.makedirs(CKPT, exist_ok=True)
    groups = sorted(np.unique(grp)) if only is None else [only]
    for k, g in enumerate(groups):
        ck = os.path.join(CKPT, f"fold_{sorted(np.unique(grp)).index(g):02d}.npz")
        if os.path.exists(ck):
            continue
        te, tr = np.where(grp == g)[0], np.where(grp != g)[0]
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]).astype(np.float32), sc.transform(X[te]).astype(np.float32)
        idx = np.random.default_rng(0).permutation(len(tr))
        nv = max(2, int(VAL_FRACTION*len(tr)))
        va, tt = idx[:nv], idx[nv:]
        a_tr = anchor[tr].reshape(-1, 1)
        keras.backend.clear_session()
        full, branch = build(X.shape[1])
        train(full, [Xtr[tt], a_tr[tt]], y[tr][tt], [Xtr[va], a_tr[va]], y[tr][va])
        bg = Xtr[np.random.default_rng(0).choice(len(Xtr), min(100, len(Xtr)), replace=False)]
        f = lambda q: branch(q.astype(np.float32), training=False).numpy().ravel()
        ex = shap.KernelExplainer(f, shap.kmeans(bg, 15))
        sv = np.asarray(ex.shap_values(Xte, nsamples=200, silent=True))
        np.savez(ck, idx=te, delta=f(Xte), shap=sv)
        print(f"fold {g}: {len(te)} records done", flush=True)
    return ml


def summarize(ml):
    """Pool the folds and report share and direction per input."""
    files = sorted(f for f in os.listdir(CKPT) if f.endswith(".npz"))
    SV = np.full((len(ml), len(FEATURES)), np.nan)
    delta = np.full(len(ml), np.nan)
    for f in files:
        z = np.load(os.path.join(CKPT, f))
        SV[z["idx"]] = z["shap"]
        delta[z["idx"]] = z["delta"]
    done = ~np.isnan(delta)
    X = ml[FEATURES].to_numpy(float)
    mabs = np.abs(SV[done]).mean(0)
    share = 100*mabs/mabs.sum()
    rows = []
    for j, f in enumerate(FEATURES):
        r = np.corrcoef(X[done, j], SV[done, j])[0, 1] if X[done, j].std() > 0 else np.nan
        rows.append(dict(input=f, mean_abs_shap=mabs[j], share_pct=share[j], corr_value_shap=r))
    T = pd.DataFrame(rows).sort_values("share_pct", ascending=False)
    out = pd.DataFrame(SV, columns=[f"shap_{f}" for f in FEATURES])
    out.insert(0, "ID", ml["ID"].values)
    out["delta"] = delta
    out.to_csv("shap_values.csv", index=False)
    return T, int(done.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single loso_group")
    args = ap.parse_args()
    ml = run(args.only)
    T, n = summarize(ml)
    print(f"\nrecords explained: {n} of {len(ml)}")
    print(T.round(3).to_string(index=False))
    top4 = T["share_pct"].head(4).sum()
    print(f"four largest inputs carry {top4:.1f}% of the attribution")
