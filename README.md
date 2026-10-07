# Credit Grade

Probability-of-default model with an eight-notch internal grade scale for research triage.

The grades (AAA through D) are rank bands on calibrated default probability. They are an internal score for an analyst queue. They are not a credit rating and they are not a CRISIL, S&P, or Moody's opinion.

## Data

UCI Statlog German Credit: 1,000 obligors, 20 account and credit attributes, good/bad outcome.
Source: [UCI ML Repository, dataset 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
The file `data/german.data` is the original UCI extract.

Bad obligors (UCI label 2) are the default class. The published base default rate is 30%.

## Model

- Stratified holdout, 25%.
- Fit XGBoost on 75% of the training split.
- Platt (sigmoid) calibration on the remaining 25% of the training split (`CalibratedClassifierCV`, `cv="prefit"`).
- Holdout is untouched until scoring.
- Notch cuts are the calibration-set PD quantiles, frozen before the holdout is graded.
- Drivers are holdout AUC drop under permutation of one column at a time.

## Run

```bash
pip install -r requirements.txt
python src/train.py
```

Writes `reports/metrics.json` and `reports/holdout_grades.csv`.

## Latest holdout (seed 7)

| Metric | Value |
| --- | --- |
| AUC | 0.780 |
| Gini | 0.560 |
| KS | 0.444 |
| Brier | 0.173 |

Strongest drivers on this split: checking-account status, loan duration, credit amount, property, credit history.
