"""Obligor probability-of-default model with an internal rating-grade map.

Trains on the UCI Statlog German Credit dataset (1,000 obligors).
Grades are an internal research scale for analyst review. They are not agency ratings.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "german.data"
OUT = ROOT / "reports" / "metrics.json"

COLS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment_since",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "label",
]

NOTCHES = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]


def grade_from_pd(p, cuts):
    for cut, grade in zip(cuts, NOTCHES[:-1]):
        if p < cut:
            return grade
    return "D"


def ks_statistic(y, p):
    order = np.argsort(p)
    y = np.asarray(y)[order]
    n_bad = y.sum()
    n_good = len(y) - n_bad
    if n_bad == 0 or n_good == 0:
        return 0.0
    cdf_bad = np.cumsum(y) / n_bad
    cdf_good = np.cumsum(1 - y) / n_good
    return float(np.max(np.abs(cdf_bad - cdf_good)))


def main():
    df = pd.read_csv(DATA, sep=r"\s+", header=None, names=COLS)
    # UCI coding: 1 = good obligor, 2 = bad obligor. Model the bad class as default.
    df["default"] = (df["label"] == 2).astype(int)
    y = df["default"].to_numpy()
    X = df.drop(columns=["label", "default"])

    num = ["duration_months", "credit_amount", "installment_rate", "residence_since", "age_years", "existing_credits", "dependents"]
    cat = [c for c in X.columns if c not in num]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )
    base = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=4,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=7,
        n_jobs=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=7, stratify=y_train
    )
    pipe = Pipeline([("pre", pre), ("model", base)])
    pipe.fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    cal_pd = calibrated.predict_proba(X_cal)[:, 1]
    cuts = [float(x) for x in np.quantile(cal_pd, np.linspace(0.125, 0.875, 7))]
    pd_hat = calibrated.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, pd_hat))
    ks = ks_statistic(y_test, pd_hat)
    brier = float(brier_score_loss(y_test, pd_hat))
    gini = 2 * auc - 1

    grades = [grade_from_pd(float(p), cuts) for p in pd_hat]
    dist = pd.Series(grades).value_counts().to_dict()
    dist = {g: int(dist.get(g, 0)) for g in NOTCHES}

    # Permutation importance on the uncalibrated pipeline, holdout, default-class AUC drop.
    rng = np.random.default_rng(7)
    base_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    drops = {}
    for col in ["checking_status", "duration_months", "credit_history", "credit_amount", "savings", "employment_since", "installment_rate", "property", "age_years"]:
        Xp = X_test.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        auc_p = roc_auc_score(y_test, pipe.predict_proba(Xp)[:, 1])
        drops[col] = round(float(base_auc - auc_p), 4)
    drivers = sorted(drops.items(), key=lambda kv: kv[1], reverse=True)

    scored = X_test.copy()
    scored["probability_of_default"] = np.round(pd_hat, 4)
    scored["internal_grade"] = grades
    scored["actual_default"] = y_test
    scored.to_csv(ROOT / "reports" / "holdout_grades.csv", index=False)

    metrics = {
        "dataset": "UCI Statlog German Credit (1,000 obligors)",
        "holdout": "25% stratified",
        "model": "XGBoost + Platt (sigmoid) calibration",
        "notch_cuts_from_calibration_pd": [round(c, 4) for c in cuts],
        "auc": round(auc, 4),
        "gini": round(gini, 4),
        "ks": round(ks, 4),
        "brier": round(brier, 4),
        "default_rate_holdout": round(float(y_test.mean()), 4),
        "grade_distribution_holdout": dist,
        "top_drivers_auc_drop": drivers[:5],
        "note": "Internal grades are PD bands for research triage. They are not a credit rating.",
    }
    OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
