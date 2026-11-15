"""Train-only preprocessing, validation selection, frozen test evaluation."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_score, recall_score, f1_score, confusion_matrix)
from .features import NUMERIC, CATEGORICAL, FEATURES, AUDIT


def pipeline(model):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return Pipeline([("preprocess", ColumnTransformer([
        ("numeric", numeric, NUMERIC), ("category", categorical, CATEGORICAL)])),
        ("model", model)])


def metrics(y, p, threshold):
    y = np.asarray(y)
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(y), "prevalence": float(y.mean()),
            "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "brier": float(brier_score_loss(y, p)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "selection_rate": float(pred.mean()),
            "fpr": float(fp / (fp + tn)) if fp + tn else None,
            "fnr": float(fn / (fn + tp)) if fn + tp else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def fit(train, val):
    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_c0.1": LogisticRegression(C=0.1, max_iter=2000, random_state=42),
        "logistic_c1": LogisticRegression(C=1, max_iter=2000, random_state=42),
        "forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                         max_depth=12, n_jobs=-1, random_state=42),
    }
    models, scores = {}, {}
    for name, estimator in candidates.items():
        model = pipeline(estimator).fit(train[FEATURES], train.target)
        models[name] = model
        scores[name] = float(average_precision_score(val.target, model.predict_proba(val[FEATURES])[:, 1]))
    name = max(scores, key=scores.get)
    selected = models[name]
    prob = selected.predict_proba(val[FEATURES])[:, 1]
    grid = np.linspace(0.05, 0.95, 181)
    # Threshold maximizes validation F1; first maximum gives deterministic ties.
    threshold = float(max(grid, key=lambda x: f1_score(val.target, prob >= x, zero_division=0)))
    return selected, models["dummy"], {"selected": name, "validation_ap": scores,
                                      "threshold": threshold,
                                      "validation": metrics(val.target, prob, threshold)}


def bootstrap(test, p, threshold, repetitions=300):
    """Percentile CIs resample students, keeping concurrent enrolments together."""
    rng = np.random.default_rng(42)
    indices = test.groupby("id_student", sort=True).indices
    ids = np.array(list(indices))
    draws = {key: [] for key in ["average_precision", "roc_auc", "recall", "precision", "brier"]}
    for _ in range(repetitions):
        ix = np.concatenate([indices[s] for s in rng.choice(ids, len(ids), replace=True)])
        values = metrics(test.target.to_numpy()[ix], np.asarray(p)[ix], threshold)
        for key in draws:
            if values[key] is not None:
                draws[key].append(values[key])
    return {key: [float(x) for x in np.quantile(vals, [0.025, 0.975])] for key, vals in draws.items()}


def fairness(test, p, threshold, min_n=30):
    rows = []
    for attribute in AUDIT:
        groups = test[attribute].fillna("Unknown").astype(str)
        for group in sorted(groups.unique()):
            mask = groups.eq(group).to_numpy()
            y = test.target.to_numpy()[mask]
            row = {"attribute": attribute, "group": group, "n": int(mask.sum()),
                   "positives": int(y.sum()), "negatives": int((1-y).sum())}
            # Do not publish unstable tiny-group rate estimates.
            suppressed = len(y) < min_n or y.sum() < 5 or (1-y).sum() < 5
            row["suppressed"] = suppressed
            if not suppressed:
                row.update(metrics(y, np.asarray(p)[mask], threshold))
            rows.append(row)
    result = pd.DataFrame(rows)
    gaps = {}
    for attr in AUDIT:
        sub = result.loc[result.attribute.eq(attr) & ~result.suppressed]
        gaps[attr] = {key: float(sub[key].max() - sub[key].min()) if len(sub) >= 2 else None
                      for key in ["selection_rate", "recall", "fpr"]}
    return result, gaps
