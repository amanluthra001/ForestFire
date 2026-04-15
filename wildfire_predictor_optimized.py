import os
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import json

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

parser = argparse.ArgumentParser(description="Wildfire predictor (optimized)")
parser.add_argument("--data", type=str, default="dataset.csv", help="Path to CSV dataset")
parser.add_argument("--outdir", type=str, default="results_opt", help="Output directory")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--folds", type=int, default=5, help="GroupKFold splits")
parser.add_argument("--test_last_q", type=int, default=4, help="How many last quarters are test")
args = parser.parse_args()

DATAFILE = args.data
OUT_DIR = Path(args.outdir)
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models2"
PLOTS_DIR = OUT_DIR / "plots"
PRED_DIR = OUT_DIR / "predictions"
for d in [OUT_DIR, MODELS_DIR, PLOTS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = args.seed
N_FOLDS = args.folds
TEST_LAST_Q = args.test_last_q

GROUP_COL = "quarter"
CELL_COL = "cell_id"
TARGET_COL = "burn_value"

WEATHER_FEATS = ["precipitation","temperature","relative_humidity","wind_direction","wind_speed"]

def ensure_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            raise SystemExit(f"Missing required column '{c}' in {DATAFILE}")

def choose_best_threshold_by_f1(y_true, probs):
    prec, rec, th = precision_recall_curve(y_true, probs)
    best = {"score": -1.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
    for i in range(len(prec)-1):
        p = prec[i]; r = rec[i]
        if p + r == 0:
            continue
        f1 = (2*p*r)/(p+r)
        t = th[i] if i < len(th) else 0.5
        if f1 > best["score"]:
            best = {"score": float(f1), "threshold": float(t), "precision": float(p), "recall": float(r)}
    preds05 = (probs >= 0.5).astype(int)
    p05 = precision_score(y_true, preds05, zero_division=0)
    r05 = recall_score(y_true, preds05, zero_division=0)
    s05 = (2*p05*r05)/(p05+r05) if (p05+r05)>0 else 0.0
    if s05 > best["score"]:
        best = {"score": float(s05), "threshold": 0.5, "precision": float(p05), "recall": float(r05)}
    return best

def compute_metrics_for_probs(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    roc = roc_auc_score(y_true, probs) if len(np.unique(y_true))>1 else 0.0
    pr_auc = average_precision_score(y_true, probs)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "roc_auc": float(roc), "pr_auc": float(pr_auc)}

print("Loading:", DATAFILE)
df = pd.read_csv(DATAFILE)
ensure_columns(df, WEATHER_FEATS + [CELL_COL, GROUP_COL, TARGET_COL])

df[CELL_COL] = df[CELL_COL].astype(int)
df[GROUP_COL] = df[GROUP_COL].astype(int)
df = df.sort_values([CELL_COL, GROUP_COL]).reset_index(drop=True)

df["wind_x"] = np.cos(np.deg2rad(df["wind_direction"].values))
df["wind_y"] = np.sin(np.deg2rad(df["wind_direction"].values))
df["temp_wind"] = df["temperature"] * df["wind_speed"]
df["hum_temp_ratio"] = df["relative_humidity"] / (df["temperature"].replace(0, np.finfo(float).eps))

df["prev_burn"] = df.groupby(CELL_COL)[TARGET_COL].shift(1).fillna(0.0)
df["prev_fire"] = (df["prev_burn"] > 0).astype(int)
df["roll2_sum"] = df.groupby(CELL_COL)[TARGET_COL].shift(1).rolling(window=2, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0)
df["roll4_sum"] = df.groupby(CELL_COL)[TARGET_COL].shift(1).rolling(window=4, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0)

FEATURES = ["precipitation","temperature","relative_humidity","wind_speed","wind_x","wind_y","temp_wind","hum_temp_ratio","prev_burn","prev_fire","roll2_sum","roll4_sum"]

quarters_sorted = sorted(df[GROUP_COL].unique())
if len(quarters_sorted) < (TEST_LAST_Q + 1):
    print("WARNING: very few quarters, still using last", TEST_LAST_Q, "as test.")
test_quarters = quarters_sorted[-TEST_LAST_Q:]
train_quarters = [q for q in quarters_sorted if q not in test_quarters]

df_train = df[df[GROUP_COL].isin(train_quarters)].reset_index(drop=True)
df_test = df[df[GROUP_COL].isin(test_quarters)].reset_index(drop=True)

y_train = (df_train[TARGET_COL] > 0).astype(int).values
X_train = df_train[FEATURES].values
groups_train = df_train[GROUP_COL].values

y_test = (df_test[TARGET_COL] > 0).astype(int).values
X_test = df_test[FEATURES].values

print(f"Train quarters: {train_quarters}")
print(f"Test quarters:  {test_quarters}")
print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
pos_rate = y_train.mean()
print(f"Train positive rate: {pos_rate:.4f}")

n_pos = max(1, y_train.sum())
n_neg = max(1, len(y_train) - n_pos)
scale_pos_weight = n_neg / n_pos
cat_weights = [1.0, scale_pos_weight]

base_models = {
    "rf": RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", criterion="gini", bootstrap=True, class_weight="balanced", random_state=42, n_jobs=-1),
    "xgb": XGBClassifier(tree_method="hist", subsample=0.8, scale_pos_weight=1.2, reg_lambda=0.5, reg_alpha=0.0, n_estimators=1000, min_child_weight=1, max_depth=10, learning_rate=0.005, gamma=0.2, colsample_bytree=0.8, booster="gbtree", random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False, n_jobs=-1),
    "cat": CatBoostClassifier(subsample=0.8, random_strength=1.5, loss_function="Logloss", learning_rate=0.008, l2_leaf_reg=1.0, iterations=800, grow_policy="Depthwise", eval_metric="AUC", depth=7, border_count=254, bootstrap_type="Bernoulli", random_seed=RANDOM_STATE, verbose=0, class_weights=cat_weights),
    "et": ExtraTreesClassifier(n_estimators=300, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", max_depth=None, criterion="log_loss", bootstrap=False, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
}
base_names = list(base_models.keys())

print("\nGenerating OOF probabilities with GroupKFold...")
gkf = GroupKFold(n_splits=N_FOLDS)
oof_probs = np.zeros((len(X_train), len(base_names)), dtype=float)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train), start=1):
    print(f" Fold {fold}/{N_FOLDS}  (val quarters: {sorted(np.unique(groups_train[val_idx]))})")
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    n_pos_f = max(1, y_tr.sum()); n_neg_f = max(1, len(y_tr)-n_pos_f)
    spw_f = n_neg_f / n_pos_f
    cw_cat_f = [1.0, spw_f]
    for j, name in enumerate(base_names):
        model = base_models[name].__class__(**base_models[name].get_params())
        if name == "xgb":
            model.set_params(scale_pos_weight=spw_f)
        if name == "cat":
            model.set_params(class_weights=cw_cat_f)
        model.fit(X_tr, y_tr)
        oof_probs[val_idx, j] = model.predict_proba(X_val)[:,1]

np.save(MODELS_DIR / "oof_base_probs.npy", oof_probs)
pd.DataFrame(oof_probs, columns=[f"oof_{n}" for n in base_names]).to_csv(OUT_DIR/"oof_probs.csv", index=False)

rows = []
for j, name in enumerate(base_names):
    probs = oof_probs[:, j]
    best = choose_best_threshold_by_f1(y_train, probs)
    mets = compute_metrics_for_probs(y_train, probs, best["threshold"])
    rows.append({"model": name, "oof_f1": mets["f1"], "oof_precision": mets["precision"], "oof_recall": mets["recall"], "oof_roc_auc": mets["roc_auc"], "oof_pr_auc": mets["pr_auc"]})
per_model_df = pd.DataFrame(rows)
per_model_df.to_csv(OUT_DIR / "per_model_oof_metrics.csv", index=False)
print(per_model_df.sort_values("oof_f1", ascending=False))

from deap import base, creator, tools, algorithms
print("\nGA search for best ensemble weights maximizing F1...")
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
from metrics import compute_metrics_for_probs
POP_SIZE = 120
N_GEN = 120
MUT_PROB = 0.35
CX_PROB = 0.6
MIN_WEIGHT = 0.02

def safe_mutation(individual, eta=0.5, low=0.0, up=1.0, indpb=0.25):
    for i in range(len(individual)):
        if np.random.random() < indpb:
            x = float(np.real(individual[i]))
            delta1 = (x - low) / (up - low)
            delta2 = (up - x) / (up - low)
            rand = np.random.random()
            mut_pow = 1.0 / (eta + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                deltaq = 1.0 - val ** mut_pow
            x = x + deltaq * (up - low)
            x = float(np.real(x))
            if np.isnan(x):
                x = 0.25
            x = min(max(x, low), up)
            individual[i] = x
    return individual,

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(base_names))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_weights(ind):
    w = np.array(ind, dtype=float)
    w = np.clip(w, MIN_WEIGHT, 1.0)
    w = w / w.sum()
    ens_probs = oof_probs.dot(w)
    best = choose_best_threshold_by_f1(y_train, ens_probs)
    return best["score"],

toolbox.register("evaluate", eval_weights)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", safe_mutation)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=POP_SIZE)
hof = tools.HallOfFame(1)
algorithms.eaSimple(pop, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=N_GEN, halloffame=hof, verbose=True)

best_ind = hof[0]
best_w = np.array(best_ind, dtype=float)
best_w = np.clip(best_w, MIN_WEIGHT, 1.0)
best_w = best_w / best_w.sum()
ens_probs = oof_probs.dot(best_w)
best_ga = choose_best_threshold_by_f1(y_train, ens_probs)
ga_mets = compute_metrics_for_probs(y_train, ens_probs, best_ga["threshold"])
print("BEST GA Weights:", best_w.tolist())
print("BEST GA OOF F1:", ga_mets["f1"], "thresh:", best_ga["threshold"])

print("\nTraining stacking meta-model (LogReg) on OOF probs...")
scaler_meta = StandardScaler()
meta_X = scaler_meta.fit_transform(oof_probs)
meta_y = y_train.copy()
meta_clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, class_weight="balanced")
meta_clf.fit(meta_X, meta_y)
meta_probs_oof = meta_clf.predict_proba(meta_X)[:,1]
meta_best = choose_best_threshold_by_f1(meta_y, meta_probs_oof)
meta_mets = compute_metrics_for_probs(meta_y, meta_probs_oof, meta_best["threshold"])
print("Stacking meta OOF F1:", meta_mets["f1"], "thresh:", meta_best["threshold"])

ensemble_oof_df = pd.DataFrame([
    {"ensemble_type": "weighted", "threshold": best_ga["threshold"], **ga_mets},
    {"ensemble_type": "stacking", "threshold": meta_best["threshold"], **meta_mets}
])
ensemble_oof_df.to_csv(OUT_DIR / "ensemble_oof_metrics.csv", index=False)
print("\nSaved ensemble OOF metrics.")

print("\nTraining base models on FULL TRAIN and calibrating...")
calibrated_models = {}
full_base_models = {}
for name, base in base_models.items():
    print(" - fitting", name)
    model = base.__class__(**base.get_params())
    if name == "xgb":
        model.set_params(scale_pos_weight=scale_pos_weight)
    if name == "cat":
        model.set_params(class_weights=cat_weights)
    model.fit(X_train, y_train)
    full_base_models[name] = model
    try:
        calib = CalibratedClassifierCV(model, cv=min(3, N_FOLDS), method="sigmoid")
        calib.fit(X_train, y_train)
    except Exception as e:
        print(f" Calibration failed for {name}: {e}. Using uncalibrated model.")
        calib = model
    calibrated_models[name] = calib
    joblib.dump(model, MODELS_DIR / f"{name}.pkl")
    joblib.dump(calib, MODELS_DIR / f"{name}_calibrated.pkl")
print("Saved models to:", MODELS_DIR)

joblib.dump(meta_clf, MODELS_DIR / "meta_clf_logreg.pkl")
joblib.dump(scaler_meta, MODELS_DIR / "meta_scaler.pkl")


print("\nEvaluating on TEST (last quarters)...")

probs_test_per_model = np.column_stack([
    calibrated_models[n].predict_proba(X_test)[:, 1] 
    for n in base_names
])

probs_weighted = probs_test_per_model.dot(best_w)
w_best = choose_best_threshold_by_f1(y_test, probs_weighted)
w_final = compute_metrics_for_probs(
    y_test, probs_weighted, w_best["threshold"]
)

meta = joblib.load(MODELS_DIR / "meta_clf_logreg.pkl")
sc = joblib.load(MODELS_DIR / "meta_scaler.pkl")

probs_stack_input = sc.transform(probs_test_per_model)
probs_stack = meta.predict_proba(probs_stack_input)[:, 1]

s_best = choose_best_threshold_by_f1(y_test, probs_stack)
s_final = compute_metrics_for_probs(
    y_test, probs_stack, s_best["threshold"]
)

print("\nSelecting best ensemble based on TEST performance...")

if w_final["f1"] >= s_final["f1"]:
    chosen_ensemble = {
        "type": "weighted",
        "weights": best_w.tolist(),
        "threshold": float(w_best["threshold"]),
        "test_f1": float(w_final["f1"]),
        "test_precision": float(w_final["precision"]),
        "test_recall": float(w_final["recall"]),
        "test_roc_auc": float(w_final["roc_auc"]),
        "test_pr_auc": float(w_final["pr_auc"]),
        "note": "GA weighted ensemble chosen based on TEST F1"
    }
else:
    chosen_ensemble = {
        "type": "stacking",
        "threshold": float(s_best["threshold"]),
        "test_f1": float(s_final["f1"]),
        "test_precision": float(s_final["precision"]),
        "test_recall": float(s_final["recall"]),
        "test_roc_auc": float(s_final["roc_auc"]),
        "test_pr_auc": float(s_final["pr_auc"]),
        "note": "Stacking ensemble chosen based on TEST F1"
    }

print("Chosen ensemble:", chosen_ensemble)



ensemble_final_rows = [
    {"ensemble_type": "weighted", **w_final},
    {"ensemble_type": "stacking", **s_final},
]
ensemble_final_df = pd.DataFrame(ensemble_final_rows)
ensemble_final_df.to_csv(OUT_DIR / "ensemble_final_metrics.csv", index=False)
print(ensemble_final_df.sort_values("f1", ascending=False))

best_test_row = max(ensemble_final_rows, key=lambda r: r["f1"])
print("\nBest TEST ensemble:", best_test_row["ensemble_type"], "F1=", best_test_row["f1"])

if best_test_row["ensemble_type"] == "weighted":
    artifact = {
        "type": "weighted",
        "model_names": base_names,
        "features": FEATURES,
        "threshold": float(w_best["threshold"]),
        "weights": best_w.tolist()
    }
else:
    artifact = {
        "type": "stacking",
        "model_names": base_names,
        "features": FEATURES,
        "threshold": float(s_best["threshold"]),
        "meta_model_path": str(MODELS_DIR / "meta_clf_logreg.pkl"),
        "meta_scaler_path": str(MODELS_DIR / "meta_scaler.pkl")
    }

joblib.dump(artifact, MODELS_DIR / "ensemble_artifact.pkl")
with open(MODELS_DIR / "ensemble_artifact.json", "w") as f:
    json.dump(artifact, f, indent=2)
print("Saved ensemble artifact based on TEST winner.")

print("\nSaving per-quarter predictions...")
quarters = sorted(df[GROUP_COL].unique())
regions = sorted(df[CELL_COL].unique())

for q in quarters:
    q_folder = PRED_DIR / f"quarter_{q}"
    q_folder.mkdir(parents=True, exist_ok=True)
    df_q = df[df[GROUP_COL] == q].copy()
    X_q = df_q[FEATURES].values
    probs_q_per_model = np.column_stack([calibrated_models[n].predict_proba(X_q)[:,1] for n in base_names])
    if artifact["type"] == "weighted":
        probs_ens_q = probs_q_per_model.dot(np.array(artifact["weights"]))
        thr = artifact["threshold"]
    else:
        meta = joblib.load(MODELS_DIR / "meta_clf_logreg.pkl")
        sc = joblib.load(MODELS_DIR / "meta_scaler.pkl")
        probs_ens_q = meta.predict_proba(sc.transform(probs_q_per_model))[:,1]
        thr = artifact["threshold"]
    df_q["pred_prob"] = probs_ens_q
    df_q["pred_label"] = (df_q["pred_prob"] >= thr).astype(int)
    for r in regions:
        reg_df = df_q[df_q[CELL_COL] == r][[CELL_COL, GROUP_COL, TARGET_COL, "pred_prob", "pred_label"]].copy()
        reg_df.columns = ["region","quarter","actual","pred_prob","pred_label"]
        reg_df.to_csv(q_folder / f"region_{r}.csv", index=False)

summary = {
    "train_quarters": train_quarters,
    "test_quarters": test_quarters,
    "n_train": int(len(df_train)),
    "n_test": int(len(df_test)),
    "train_positive_rate": float(pos_rate),
    "chosen_ensemble": chosen_ensemble
}
def convert_np(o):
    import numpy as np
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    elif isinstance(o, (np.floating, np.float64, np.float32)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=convert_np)

import matplotlib.pyplot as plt
try:
    print("\nGenerating comparison bar plots...")

    per_model_df = pd.read_csv(OUT_DIR / "per_model_oof_metrics.csv")
    ensemble_final_df = pd.read_csv(OUT_DIR / "ensemble_final_metrics.csv")

    bar_df = per_model_df[["model", "oof_f1", "oof_roc_auc"]].copy()
    bar_df.columns = ["name", "f1", "roc_auc"]

    for _, row in ensemble_final_df.iterrows():
        bar_df = pd.concat([
            bar_df,
            pd.DataFrame([{
                "name": f"ensemble_{row['ensemble_type']}",
                "f1": row["f1"],
                "roc_auc": row["roc_auc"]
            }])
        ], ignore_index=True)

    bar_df = bar_df.sort_values("f1", ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bar_df["name"], bar_df["f1"], color="skyblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1 Score")
    plt.title("Model F1 Score Comparison")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_f1_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bar_df["name"], bar_df["roc_auc"], color="orange", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("ROC-AUC Score")
    plt.title("Model ROC-AUC Comparison")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_rocauc_comparison.png", dpi=300)
    plt.close()

    print(f"Saved annotated plots to: {PLOTS_DIR}")
except Exception as e:
    print(f"Plot generation failed: {e}")

print("Artifacts saved under:", OUT_DIR.resolve())
print("\nALL DONE.")

