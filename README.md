# fhi-sme-challenge

Predicting the Financial Health Index (FHI) of small and medium-sized enterprises
across Eswatini, Lesotho, Malawi, and Zimbabwe. Built for the data.org Financial
Health Prediction Challenge hosted on Zindi.

## Clean Pipeline (Ordinal-Only)

This repository is now centered on one canonical pipeline:

1. `python src/preprocess.py`
2. `python src/features.py`
3. `python src/ordinal_train.py`
4. `python src/ordinal_predict.py`

`ordinal_train.py` and `ordinal_predict.py` are designed to run consecutively with
no intermediate scripts required.

## Project Structure

```text
fhi-sme-challenge/
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- submissions/
|-- src/
|   |-- preprocess.py
|   |-- features.py
|   |-- target_encoding.py
|   |-- ordinal_train.py
|   `-- ordinal_predict.py
|-- tests/
|-- environment.yml
`-- README.md
```

## Notes

- `src/target_encoding.py` contains shared target-encoding logic used by both
  ordinal training and prediction.
- `models/ordinal_artifacts.pkl` is the trained artifact used by
  `src/ordinal_predict.py`.
- `submissions/submission_ordinal_oof*.csv` files are generated submissions.

## Setup

```bash
conda env create -f environment.yml
conda activate fhi-sme
```
