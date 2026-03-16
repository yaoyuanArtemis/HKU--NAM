# Dataset Notes

This directory is intended for local dataset storage. Large files are excluded
from version control and should therefore be downloaded on demand.

Expected sources include:

- **Credit Card Fraud Detection**: `creditcard.csv`, from [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **California Housing**: either via `sklearn.datasets.fetch_california_housing` or a local `california_housing.csv`
- **COMPAS**: `compas-analysis-master/compas-scores-two-years.csv`, from [ProPublica](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)
- **FICO**: `FICO-Explainable-ML-Challenge-HELOC-Dataset-master/HelocData.csv`, from the [FICO Challenge](https://community.fico.com/s/explainable-machine-learning-challenge)

Refer to `data_utils.py` for the authoritative path conventions and loading logic.
