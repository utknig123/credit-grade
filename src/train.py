"""Issuer financial-risk model for a ratings-style research desk.

447 corporate issuers. Features follow the financial ratios CRISIL Ratings
names in its public criteria (gearing, interest coverage, profitability,
liquidity, scale), plus sector as the industry-risk input.

The grade is a financial-risk notch only. It is not a CRISIL, S&P or Moody's rating.
Parent support, management and project risk are stated as gaps in every note.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "credit_rating_training_data.csv"
REPORTS = ROOT / "reports"

# Major notches. Plus/minus modifiers are folded in so a 447-row file can support a split.
NOTCHES = ["AA_up", "A", "BBB", "BB", "B", "CCC_down"]
NOTCH_LABEL = {
    "AA_up": "AA and above",
    "A": "A",
    "BBB": "BBB",
    "BB": "BB",
    "B": "B",
    "CCC_down": "CCC and below",
}

RATIO_COLS = [
    "debt_to_equity",
    "interest_coverage",
    "debt_to_ebitda",
    "current_ratio",
    "net_margin",
    "roa",
    "log_assets",
]

# How each column maps onto CRISIL's published financial-risk parameters.
# DSCR, NCATD and net worth are not in this file. RoCE is approximated by ROA.
RATIO_ROLE = {
    "debt_to_equity": "capital structure (gearing)",
    "interest_coverage": "interest coverage",
    "debt_to_ebitda": "debt burden versus operating earnings",
    "current_ratio": "liquidity (current ratio)",
    "net_margin": "profitability",
    "roa": "return on assets (stand-in for RoCE)",
    "log_assets": "scale",
}

GAPS = (
    "Debt service coverage, net cash accrual to total debt, and net worth are not in the file. "
    "Parent or group support, management risk, project risk and accounting quality are outside this model. "
    "CRISIL Ratings does not assign a rating from ratios alone."
)


def to_major(rating: str) -> str:
    r = str(rating).strip().upper().replace(" ", "")
    if r in {"AAA", "AA+", "AA", "AA-"}:
        return "AA_up"
    if r in {"A+", "A", "A-"}:
        return "A"
    if r in {"BBB+", "BBB", "BBB-"}:
        return "BBB"
    if r in {"BB+", "BB", "BB-"}:
        return "BB"
    if r in {"B+", "B", "B-"}:
        return "B"
    return "CCC_down"


def winsorize_fit(train: pd.DataFrame):
    bounds = {}
    for col in RATIO_COLS:
        lo, hi = np.nanpercentile(train[col].astype(float), [1, 99])
        bounds[col] = (float(lo), float(hi))
    return bounds


def winsorize_apply(frame: pd.DataFrame, bounds):
    out = frame.copy()
    for col, (lo, hi) in bounds.items():
        out[col] = out[col].astype(float).clip(lo, hi)
    return out


def adjacent_accuracy(y_true, y_pred):
    idx = {n: i for i, n in enumerate(NOTCHES)}
    yt = np.array([idx[v] for v in y_true])
    yp = np.array([idx[v] for v in y_pred])
    return float(np.mean(np.abs(yt - yp) <= 1))


def direction(value, median, higher_is_stronger):
    if pd.isna(value) or pd.isna(median):
        return "in line with"
    if abs(value - median) / (abs(median) + 1e-6) < 0.08:
        return "in line with"
    stronger = value > median if higher_is_stronger else value < median
    return "stronger than" if stronger else "weaker than"


HIGHER_IS_STRONGER = {
    "debt_to_equity": False,
    "interest_coverage": True,
    "debt_to_ebitda": False,
    "current_ratio": True,
    "net_margin": True,
    "roa": True,
    "log_assets": True,
}


def committee_note(row, grade, proba, sector_medians, drivers):
    lines = [
        f"{row['name']} ({row['ticker']}), {row['sector']}.",
        f"Financial-risk notch: {NOTCH_LABEL[grade]} (model confidence {proba:.0%}).",
        "This is not a credit rating. It is a triage grade from published financial ratios.",
        "Versus the sector median:",
    ]
    for col in drivers:
        role = RATIO_ROLE[col]
        val = float(row[col])
        med = float(sector_medians.loc[row["sector"], col])
        d = direction(val, med, HIGHER_IS_STRONGER[col])
        lines.append(f"- {role}: {val:.2f}, {d} the {row['sector']} median of {med:.2f}.")
    lines.append(GAPS)
    return "\n".join(lines)


def main():
    df = pd.read_csv(DATA)
    df["notch"] = df["rating"].map(to_major)
    y = df["notch"]
    X = df[["sector"] + RATIO_COLS].copy()

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.25, random_state=7, stratify=y
    )
    bounds = winsorize_fit(X_train)
    X_train = winsorize_apply(X_train, bounds)
    X_test = winsorize_apply(X_test, bounds)

    pre = ColumnTransformer(
        [
            ("sector", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
            ("ratios", "passthrough", RATIO_COLS),
        ]
    )
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(NOTCHES),
        eval_metric="mlogloss",
        random_state=7,
        n_jobs=1,
    )
    pipe = Pipeline([("pre", pre), ("model", clf)])
    label_to_i = {n: i for i, n in enumerate(NOTCHES)}
    i_to_label = {i: n for n, i in label_to_i.items()}
    pipe.fit(X_train, y_train.map(label_to_i))

    proba = pipe.predict_proba(X_test)
    pred_i = proba.argmax(axis=1)
    pred = [i_to_label[i] for i in pred_i]
    y_true = y_test.tolist()
    conf = proba.max(axis=1)

    exact = float(accuracy_score(y_true, pred))
    adjacent = adjacent_accuracy(y_true, pred)
    bal = float(balanced_accuracy_score(y_true, pred))
    macro_f1 = float(f1_score(y_true, pred, average="macro", labels=NOTCHES, zero_division=0))

    rng = np.random.default_rng(7)
    base = exact
    drops = {}
    for col in RATIO_COLS + ["sector"]:
        Xp = X_test.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        pred_p = [i_to_label[i] for i in pipe.predict(Xp)]
        drops[col] = round(float(base - accuracy_score(y_true, pred_p)), 4)
    drivers = [c for c, _ in sorted(drops.items(), key=lambda kv: kv[1], reverse=True) if c in RATIO_COLS][:4]

    medians = X_train.groupby(df_train["sector"])[RATIO_COLS].median()

    notes = []
    holdout = df_test.copy()
    holdout["predicted_notch"] = pred
    holdout["confidence"] = np.round(conf, 4)
    holdout["actual_notch"] = y_true
    # One note per predicted notch, preferring a correct call so the sample is readable.
    for notch in NOTCHES:
        pool = holdout[(holdout["predicted_notch"] == notch) & (holdout["actual_notch"] == notch)]
        if pool.empty:
            pool = holdout[holdout["predicted_notch"] == notch]
        if pool.empty:
            continue
        row = pool.iloc[0]
        src = X_test.loc[row.name]
        src = src.copy()
        src["name"] = row["name"]
        src["ticker"] = row["ticker"]
        src["sector"] = row["sector"]
        notes.append(committee_note(src, notch, float(row["confidence"]), medians, drivers[:3]))

    (REPORTS / "sample_committee_notes.txt").write_text("\n\n".join(notes), encoding="utf-8")
    holdout.to_csv(REPORTS / "holdout_issuer_grades.csv", index=False)

    metrics = {
        "dataset": "447 corporate issuers, one row each, public letter rating plus financial ratios",
        "source_file": "data/credit_rating_training_data.csv",
        "what_it_covers": [
            "capital structure (debt/equity)",
            "interest coverage",
            "debt/EBITDA",
            "current ratio",
            "profitability (net margin)",
            "return (ROA as a RoCE stand-in)",
            "scale (log assets)",
            "industry via sector",
        ],
        "what_it_does_not_cover": [
            "debt service coverage",
            "net cash accrual to total debt",
            "net worth",
            "parent or group support",
            "management risk",
            "project risk",
            "accounting quality",
        ],
        "holdout": "25% stratified",
        "model": "XGBoost multiclass on six major notches",
        "exact_accuracy": round(exact, 4),
        "adjacent_notch_accuracy": round(adjacent, 4),
        "balanced_accuracy": round(bal, 4),
        "macro_f1": round(macro_f1, 4),
        "driver_exact_accuracy_drop": drops,
        "note": "Financial-risk triage only. Not an agency rating.",
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
