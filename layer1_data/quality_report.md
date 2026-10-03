# Vaidya — Layer 1 Data Quality Report

## Dataset: MedMCQA (train split)

| Metric | Value |
|--------|-------|
| Raw rows (all types) | 182,822 |
| After single-choice filter | 120,765 |
| After null drop | 120,765 |
| After deduplication | 120,705 |
| Duplicate rate | 0.05% |
| Explanation present (train) | 88.1% |
| Unique subjects | 21 |

## Split sizes

| Split | Rows |
|-------|------|
| train | 96,564 |
| val   | 12,070 |
| test  | 12,071 |

## Text length (characters)

| Stat | Value |
|------|-------|
| Mean | 902 |
| p50  | 741 |
| p95  | 1944 |
| p99  | 3244 |
| Max  | 22792 |

## Gate check

- [ ] train.parquet ≥ 140k rows
- [ ] val.parquet ≥ 17k rows
- [ ] test.parquet ≥ 17k rows
- [ ] Duplicate rate < 1%
- [ ] S3 sync: s3://vaidya-artifacts/data/

_Token length analysis (actual BPE tokens) is in notebook 03._
