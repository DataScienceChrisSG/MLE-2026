"""
Train, evaluate and select a model for one slot at one run date.

Split (all data has a label observed on or before the run date):
  - out-of-time (OOT): the latest OOT_MONTHS application months  -> never touched in training;
                       used for the promotion decision and as the monitoring reference
  - development: older months, split 80/20 stratified into train / test
      train -> hyper-parameter search with 3-fold cross-validation, scored on AUC of probabilities
      test  -> pick the algorithm (ties go to the simpler logistic regression) and the cut-off
  The chosen configuration is then refit on train + test (all development data) for deployment.

Every transformation (imputation, scaling) lives inside the sklearn pipeline, so it is fitted on
training data only and travels with the model to inference (no train-test contamination, no skew).
"""
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import config as C
from utils import ml_data
from utils import metrics as M
from utils import model_registry as R


def candidate_algorithms():
    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(solver="liblinear", max_iter=5000, random_state=C.RANDOM_SEED)),
    ])
    lr_space = {"model__C": loguniform(1e-3, 10), "model__penalty": ["l1", "l2"]}

    # XGBoost handles missing values natively, so no imputer is needed
    gb = Pipeline([
        ("model", xgb.XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=2, random_state=C.RANDOM_SEED)),
    ])
    gb_space = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.5, 0.7, 0.9],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__reg_lambda": [1, 2, 5],
        "model__gamma": [0, 0.1, 0.5],
    }
    return {"logistic_regression": (lr, lr_space), "xgboost": (gb, gb_space)}


def feature_importance(pipeline, features):
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = np.abs(model.coef_[0])   # standardised inputs, so |coefficient| is comparable
    s = pd.Series(imp, index=features).sort_values(ascending=False)
    return {k: float(v) for k, v in s.items()}


def reference_profile(pipeline, X, features, importance):
    """Distributions at training time, used later to measure drift (PSI / CSI / missing rates)."""
    scores = pipeline.predict_proba(X)[:, 1]
    score_edges = M.make_bins(scores)
    top = [f for f in importance if f != "has_clickstream"][: C.TOP_FEATURES_MONITORED]
    feats = {}
    for f in top:
        edges = M.make_bins(X[f].values)
        feats[f] = {"edges": edges, "props": M.bin_proportions(X[f].values, edges),
                    "missing_rate": float(X[f].isna().mean())}
    return {
        "n": int(len(X)),
        "score_edges": score_edges,
        "score_props": M.bin_proportions(scores, score_edges),
        "mean_score": float(scores.mean()),
        "features": feats,
    }


def train_slot(slot, run_date):
    """Returns a model artefact (dict) for the best algorithm, trained on labels known at run_date."""
    data = ml_data.slot_population(ml_data.labelled_data(run_date), slot)
    features = ml_data.feature_columns(data, C.MODEL_SLOTS[slot]["feature_set"])
    months = sorted(data["snapshot_date"].unique())
    oot_months = months[-C.OOT_MONTHS:]
    dev = data[~data["snapshot_date"].isin(oot_months)]
    oot = data[data["snapshot_date"].isin(oot_months)]

    X_dev, y_dev = dev[features].astype(float), dev["label"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X_dev, y_dev, test_size=C.TEST_SIZE, stratify=y_dev,
                                              random_state=C.RANDOM_SEED, shuffle=True)
    X_oot, y_oot = oot[features].astype(float), oot["label"].astype(int)
    print(f"[{slot}] run {run_date}: train {len(X_tr)} | test {len(X_te)} | oot {len(X_oot)} "
          f"(dev {months[0]}..{dev['snapshot_date'].max()}, oot {oot_months[0]}..{oot_months[-1]}) | {len(features)} features")

    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_SEED)
    results = {}
    for name, (pipe, space) in candidate_algorithms().items():
        search = RandomizedSearchCV(pipe, space, n_iter=C.SEARCH_ITERATIONS, scoring="roc_auc", cv=cv,
                                    random_state=C.RANDOM_SEED, n_jobs=1, refit=True)
        search.fit(X_tr, y_tr)
        p_tr = search.best_estimator_.predict_proba(X_tr)[:, 1]
        p_te = search.best_estimator_.predict_proba(X_te)[:, 1]
        results[name] = {
            "search": search,
            "cv_auc": float(search.best_score_),
            "train_auc": M.safe_auc(y_tr, p_tr),
            "test_auc": M.safe_auc(y_te, p_te),
            "threshold": M.ks_threshold(y_te, p_te),
        }
        print(f"   {name:20s} cv {results[name]['cv_auc']:.3f} | train {results[name]['train_auc']:.3f} | test {results[name]['test_auc']:.3f}")

    # algorithm choice on the test split; prefer the simpler model when they are practically tied
    best = max(results, key=lambda k: results[k]["test_auc"])
    lr = results["logistic_regression"]["test_auc"]
    if best != "logistic_regression" and results[best]["test_auc"] - lr <= C.ALGORITHM_TIE_TOLERANCE:
        best = "logistic_regression"
    chosen = results[best]

    # deploy: same configuration refit on all development data (train + test)
    final = clone(chosen["search"].best_estimator_).fit(X_dev, y_dev)
    p_oot = final.predict_proba(X_oot)[:, 1]
    threshold = chosen["threshold"]
    oot_auc = M.safe_auc(y_oot, p_oot)
    importance = feature_importance(final, features)
    reference = reference_profile(final, pd.concat([X_dev, X_oot]), features, importance)

    version = R.version_name(slot, run_date)
    metrics = {
        "cv_auc": chosen["cv_auc"], "train_auc": chosen["train_auc"], "test_auc": chosen["test_auc"],
        "oot_auc": oot_auc, "oot_gini": 2 * oot_auc - 1, "oot_ks": M.ks_statistic(y_oot, p_oot),
        "oot_default_rate": float(y_oot.mean()), "oot_mean_pd": float(p_oot.mean()),
        **{f"oot_{k}": float(v) for k, v in M.classification_summary(y_oot, p_oot, threshold).items()},
    }
    card = {
        "version": version,
        "slot": slot,
        "slot_label": C.MODEL_SLOTS[slot]["label"],
        "population": C.MODEL_SLOTS[slot]["population"],
        "algorithm": best,
        "trained_on": run_date,
        "label_definition": C.LABEL_DEF,
        "data_window": {
            "development_months": [months[0], dev["snapshot_date"].max()],
            "oot_months": [oot_months[0], oot_months[-1]],
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "n_dev": int(len(X_dev)), "n_oot": int(len(X_oot)),
            "default_rate_dev": float(y_dev.mean()), "default_rate_oot": float(y_oot.mean()),
        },
        "features": features,
        "n_features": len(features),
        "hyperparameters": {k.replace("model__", ""): (v if isinstance(v, (int, float, str)) else str(v))
                            for k, v in chosen["search"].best_params_.items()},
        "decision_threshold": threshold,
        "metrics": metrics,
        "algorithm_comparison": {k: {m: v[m] for m in ["cv_auc", "train_auc", "test_auc"]} for k, v in results.items()},
        "top_features": dict(list(importance.items())[:15]),
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                        "xgboost": xgb.__version__, "pandas": pd.__version__},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[{slot}] chosen {best}: OOT AUC {oot_auc:.3f} (Gini {2*oot_auc-1:.3f}), cut-off {threshold:.3f}")
    return {
        "slot": slot, "version": version, "model": final, "features": features,
        "threshold": threshold, "reference": reference, "card": card,
        "oot_months": oot_months,
    }
