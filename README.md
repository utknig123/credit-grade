# Credit Grade

Financial-risk triage for a ratings-style research desk.

CRISIL Ratings scores an issuer on business risk, financial risk, management risk and project risk, then adjusts for parent or group support. The financial-risk leg is ratio work, not a mechanical score. This project covers that leg only.

## What it uses

447 corporate issuers. Each row has a public letter rating and:

| Column | Ratings-desk meaning |
| --- | --- |
| debt/equity | Capital structure (gearing) |
| interest coverage | Ability to pay interest from earnings |
| debt/EBITDA | Debt burden against operating earnings |
| current ratio | Liquidity |
| net margin | Profitability |
| ROA | Return. A stand-in for RoCE, which this file does not contain |
| log assets | Scale |
| sector | Industry risk |

## What it refuses to pretend

Debt service coverage, net cash accrual to total debt, and net worth are not in the file. Neither are parent support, management, project risk or accounting quality. Every committee note says so. The output is not a CRISIL, S&P or Moody's rating.

## Model

XGBoost, six major notches (plus and minus modifiers folded in). Extremes are grouped: AAA with AA, and CCC with D, because those buckets are too thin to split.

Holdout is 25% and stratified. Ratios are winsorised at the 1st and 99th percentile of the training fold only.

The note generator does not call an LLM. It writes from the model's notch, its confidence, and the issuer's ratios against the sector median. A ratings desk would rather have a sourced sentence than a fluent one.

## Latest holdout (seed 7)

| Metric | Value |
| --- | --- |
| Exact notch | 60.7% |
| Within one notch | 88.4% |
| Balanced accuracy | 58.8% |
| Macro F1 | 0.62 |

Largest accuracy drops when a column is shuffled: scale, sector, profitability, return, gearing.

## Run

```bash
pip install -r requirements.txt
python src/train.py
```

Writes `reports/metrics.json`, `reports/holdout_issuer_grades.csv` and `reports/sample_committee_notes.txt`.
