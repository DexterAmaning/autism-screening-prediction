# Data

## `raw/autism_screening_adult_uci426.zip`

| | |
|---|---|
| Dataset | Autism Screening Adult |
| Creator | Fadi Thabtah (donated 2017-12-23) |
| Source | UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/426/autism+screening+adult |
| Direct file | https://archive.ics.uci.edu/static/public/426/autism+screening+adult.zip |
| DOI | https://doi.org/10.24432/C5F019 |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0). Redistributed here unchanged with attribution. |
| Downloaded | 2026-10-24 |
| SHA-256 | `c57b29866c9715ff49ff4cce52228e6c460bdce3f615000c33b4d53df99d6d09` |
| Contents | `Autism-Adult-Data.arff` (704 rows, 21 attributes) and the creator's variable description |

**Citation:** Thabtah, F. (2017). *Autism Screening Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F019

Data notes, verified in the notebook and tests:
- `result` equals the sum of `A1_Score`–`A10_Score` for every row, and `Class/ASD` = YES exactly when `result` ≥ 7. The label is the AQ-10 screening rule, not a clinical diagnosis.
- Missing values are coded `?` (age: 2, ethnicity: 95, relation: 95). One age of 383 is treated as missing.
- The data is anonymous, with no identifiers.

### Why not the earlier `autism.csv`?

The first version of this project used an 800-row `autism.csv` from a Kaggle competition. That file has fractional ages, and its labels don't follow the AQ-10 rule, which suggests a synthetic derivative of this UCI data. Its competition terms don't clearly allow redistribution, so it isn't used or included here.
