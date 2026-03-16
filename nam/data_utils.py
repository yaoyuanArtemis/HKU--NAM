# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data readers for regression/ binary classification datasets."""

import gzip
import os
import os.path as osp
import tarfile
from typing import Tuple, Dict, Union, Iterator, List

os.environ.setdefault('KMP_USE_SHM', '0')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = osp.join(osp.dirname(__file__), 'data')
DatasetType = Tuple[np.ndarray, np.ndarray]


def _resolve_dataset_path(relative_path):
  """Builds an absolute path under local DATA_PATH."""
  return osp.join(DATA_PATH, relative_path)


def _resolve_existing_dataset_path(*relative_paths):
  """Return the first existing dataset path under ``DATA_PATH``."""
  resolved_paths = [_resolve_dataset_path(path) for path in relative_paths]
  for path in resolved_paths:
    if osp.exists(path):
      return path
  return resolved_paths[0]


def save_array_to_disk(filename,
                       np_arr,
                       allow_pickle = False):
  """Saves a np.ndarray to a specified file on disk."""
  with open(filename, 'wb') as f:
    with gzip.GzipFile(fileobj=f) as outfile:
      np.save(outfile, np_arr, allow_pickle=allow_pickle)


def read_dataset(dataset_name,
                 header = 'infer',
                 names = None,
                 delim_whitespace = False):
  dataset_path = _resolve_dataset_path(dataset_name)
  with open(dataset_path, 'r', encoding='utf-8') as f:
    df = pd.read_csv(
        f, header=header, names=names, delim_whitespace=delim_whitespace)
  return df


def load_breast_data():
  """Load and return the Breast Cancer Wisconsin dataset (classification)."""

  breast = load_breast_cancer()
  feature_names = list(breast.feature_names)
  return {
      'problem': 'classification',
      'X': pd.DataFrame(breast.data, columns=feature_names),
      'y': breast.target,
  }


def load_adult_data():
  """Loads the Adult Income dataset.

  Predict whether income exceeds $50K/yr based on census data. Also known as
  "Census Income" dataset. For more info, see
  https://archive.ics.uci.edu/ml/datasets/Adult.

  Returns:
    A dict containing the `problem` type (regression or classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """
  df = read_dataset('adult.data', header=None)
  df.columns = [
      'Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum',
      'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Gender',
      'CapitalGain', 'CapitalLoss', 'HoursPerWeek', 'NativeCountry', 'Income'
  ]
  train_cols = df.columns[0:-1]
  label = df.columns[-1]
  x_df = df[train_cols]
  y_df = df[label]
  return {'problem': 'classification', 'X': x_df, 'y': y_df}


def load_heart_data():
  """Loads the Heart Disease dataset.

  The Cleveland Heart Disease Data found in the UCI machine learning repository
  consists of 14 variables measured on 303 individuals who have heart disease.
  See https://www.kaggle.com/sonumj/heart-disease-dataset-from-uci for more
  info.

  Returns:
    A dict containing the `problem` type (regression or classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """
  df = read_dataset('HeartDisease.csv')
  train_cols = df.columns[0:-2]
  label = df.columns[-2]
  x_df = df[train_cols]
  y_df = df[label]
  # Replace NaN values with the mode value in the column.
  for col_name in x_df.columns:
    x_df[col_name].fillna(x_df[col_name].mode()[0], inplace=True)
  return {
      'problem': 'classification',
      'X': x_df,
      'y': y_df,
  }


def load_credit_data():
  """Loads the Credit Fraud Detection dataset.

  This dataset contains transactions made by credit cards in September 2013 by
  european cardholders. It presents transactions that occurred in 2 days, where
  we have 492 frauds out of 284,807 transactions. It is highly unbalanced, the
  positive class (frauds) account for 0.172% of all transactions.
  See https://www.kaggle.com/mlg-ulb/creditcardfraud for more info.

  Returns:
    A dict containing the `problem` type (i.e. classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """
  credit_path = _resolve_existing_dataset_path(
      'creditcard.csv',
      'Credit Card Fraud Detection/creditcard.csv')
  df = pd.read_csv(credit_path)
  df = df.dropna()
  train_cols = df.columns[0:-1]
  label = df.columns[-1]
  x_df = df[train_cols]
  y_df = df[label]
  return {
      'problem': 'classification',
      'X': x_df,
      'y': y_df,
  }


def load_telco_churn_data():
  """Loads Telco Customer Churn dataset.

  Predict behavior to retain customers based on relevant customer data.
  See https://www.kaggle.com/blastchar/telco-customer-churn/ for more info.

  Returns:
    A dict containing the `problem` type (i.e. classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """
  df = read_dataset('WA_Fn-UseC_-Telco-Customer-Churn.csv')
  train_cols = df.columns[1:-1]  # First column is an ID
  label = df.columns[-1]
  x_df = df[train_cols]
  # Impute missing values
  x_df['TotalCharges'] = x_df['TotalCharges'].replace(' ', 0).astype('float64')
  y_df = df[label]  # 'Yes', 'No'.
  return {
      'problem': 'classification',
      'X': x_df,
      'y': y_df,
  }


def load_mimic2_data():
  """Loads the preprocessed Mimic-II ICU Mortality prediction dataset.

  The task is to predict mortality rate in Intensive Care Units (ICUs) based on
  using data from the first 48 hours of the ICU stay. See
  https://mimic.physionet.org/ for more info.

  Returns:
    A dict containing the `problem` type (i.e. classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """

  # Create column names
  attr_dict_path = _resolve_dataset_path('mimic2/mimic2.dict')
  with open(attr_dict_path, 'r', encoding='utf-8') as f:
    attributes = f.readlines()
  column_names = [x.split(' ,')[0] for x in attributes]

  df = read_dataset(
      'mimic2/mimic2.data',
      header=None,
      names=column_names,
      delim_whitespace=True)
  train_cols = column_names[:-1]
  label = column_names[-1]
  x_df = df[train_cols]
  y_df = df[label]
  return {
      'problem': 'classification',
      'X': x_df,
      'y': y_df,
  }


def load_recidivism_data():
  """Loads the ProPublica COMPAS recidivism dataset.

  COMPAS is a proprietary score developed to predict re-cidivism risk, which is
  used to inform bail, sentencing and parole decisions. In 2016, ProPublica
  released recidivism data on defendants in Broward County, Florida. See
  https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis
  for more info.

  Returns:
    A dict containing the `problem` type (i.e. classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """

  df = load_recidivism_dataframe()
  feature_cols = [
      'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
      'priors_count', 'c_charge_degree', 'race', 'sex'
  ]
  x_df = df[feature_cols]
  y_df = df['two_year_recid']
  return {
      'problem': 'classification',
      'X': x_df,
      'y': y_df,
  }


def load_recidivism_dataframe():
  """Load the locally available COMPAS dataframe with project filtering."""
  recid_path = _resolve_existing_dataset_path(
      'compas-scores-two-years.csv',
      'compas-analysis-master/compas-scores-two-years.csv')
  df = pd.read_csv(recid_path)
  df = df[df['sex'].isin(['Male', 'Female'])]
  df = df[df['two_year_recid'].isin([0, 1])]
  return df.reset_index(drop=True)


def load_recidivism_multitask_data(
    include_sex_feature: bool = True,
) -> Dict[str, Union[str, np.ndarray, List[str]]]:
  """Load COMPAS as a two-task gender-conditioned classification problem."""
  df = load_recidivism_dataframe()
  feature_cols = [
      'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
      'priors_count', 'c_charge_degree', 'race'
  ]
  if include_sex_feature:
    feature_cols.append('sex')
  x_df = df[feature_cols]
  transformed_x, column_names = transform_data(x_df)
  transformed_x = transformed_x.astype('float32')

  task_names = ['Female', 'Male']
  targets = np.zeros((len(df), len(task_names)), dtype=np.float32)
  masks = np.zeros_like(targets)
  task_index = np.where(df['sex'].to_numpy() == 'Female', 0, 1).astype(np.int64)
  row_index = np.arange(len(df))
  labels = df['two_year_recid'].to_numpy(dtype=np.float32)
  targets[row_index, task_index] = labels
  masks[row_index, task_index] = 1.0

  return {
      'problem': 'multitask_classification',
      'X': transformed_x,
      'y': targets,
      'mask': masks,
      'task_index': task_index,
      'task_names': task_names,
      'column_names': column_names,
      'feature_columns': feature_cols,
      'raw_frame': df,
  }


def load_fico_score_data():
  """Loads the FICO Score dataset.

  The FICO score is a widely used proprietary credit score todetermine credit
  worthiness for loans in the United States. The FICO dataset is comprised of
  real-world anonymized credit applications made by customers and their assigned
  FICO Score, based on their credit report information. For more info, refer to
  https://community.fico.com/s/explainable-machine-learning-challenge.

  Returns:
    A dict containing the `problem` type (i.e. regression) and the
    input features `X` as a pandas.Dataframe and the FICO scores `y` as
    np.ndarray.
  """

  fico_path = _resolve_existing_dataset_path(
      'HelocData.csv',
      'FICO-Explainable-ML-Challenge-HELOC-Dataset-master/HelocData.csv')
  df = pd.read_csv(fico_path)
  df = df.replace([-9, -8, -7], np.nan).dropna()
  if 'ExternalRiskEstimate' in df.columns:
    label = 'ExternalRiskEstimate'
  elif 'x1' in df.columns:
    # Local HELOC fallback: x1 is the first score-like numeric feature.
    label = 'x1'
  else:
    candidate_cols = [c for c in df.columns if c.lower().startswith('x')]
    if not candidate_cols:
      raise ValueError('No suitable numeric target column found for FICO data.')
    label = sorted(candidate_cols)[0]
  drop_cols = [label]
  if 'RiskPerformance' in df.columns:
    drop_cols.append('RiskPerformance')
  if 'RiskFlag' in df.columns:
    drop_cols.append('RiskFlag')
  x_df = df.drop(columns=drop_cols)
  y_df = df[label]
  return {
      'problem': 'regression',
      'X': x_df,
      'y': y_df.values,
  }


def load_california_housing_data(
):
  """Loads the California Housing dataset.

  California  Housing  dataset is a canonical machine learning dataset derived
  from the 1990 U.S. census to understand the influence of community
  characteristics on housing prices. The task is regression to predict the
  median price of houses (in million dollars) in each district in California.
  For more info, refer to
  https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset.

  Returns:
    A dict containing the `problem` type (i.e. regression) and the
    input features `X` as a pandas.Dataframe and the regression targets `y` as
    np.ndarray.
  """
  # Local project CSV (if target exists) then sklearn fallback.
  local_csv_path = osp.join(
      osp.dirname(__file__), 'data', 'california_housing.csv')
  if not osp.exists(local_csv_path):
    local_csv_path = osp.join(
        osp.dirname(__file__), 'data', 'California Housing',
        'california_housing.csv')
  if osp.exists(local_csv_path):
    local_df = pd.read_csv(local_csv_path)
    if 'MedHouseVal' in local_df.columns:
      x_df = local_df.drop(columns=['MedHouseVal'])
      y = local_df['MedHouseVal'].values
      return {'problem': 'regression', 'X': x_df, 'y': y}
    if 'median_house_value' in local_df.columns:
      x_df = local_df.drop(columns=['median_house_value'])
      y = local_df['median_house_value'].values
      return {'problem': 'regression', 'X': x_df, 'y': y}
    if 'target' in local_df.columns:
      x_df = local_df.drop(columns=['target'])
      y = local_df['target'].values
      return {'problem': 'regression', 'X': x_df, 'y': y}
    raise ValueError(
        'Local California Housing CSV must include one of '
        '`MedHouseVal`, `median_house_value`, or `target` columns: '
        f'{local_csv_path}')

  # Sklearn fallback always has target.
  housing = fetch_california_housing(
      as_frame=True,
      data_home=osp.join(DATA_PATH, 'sklearn_cache'))
  return {
      'problem': 'regression',
      'X': housing.data.copy(),
      'y': housing.target.values,
  }


class CustomPipeline(Pipeline):
  """Custom sklearn Pipeline to transform data."""

  def apply_transformation(self, x):
    """Applies all transforms to the data, without applying last estimator.

    Args:
      x: Iterable data to predict on. Must fulfill input requirements of first
        step of the pipeline.

    Returns:
      xt: Transformed data.
    """
    xt = x
    for _, transform in self.steps[:-1]:
      xt = transform.fit_transform(xt)
    return xt


def transform_data(df):
  """Apply a fixed set of transformations to the pd.Dataframe `df`.

  Args:
    df: Input dataframe containing features.

  Returns:
    Transformed dataframe and corresponding column names. The transformations
    include (1) encoding categorical features as a one-hot numeric array, (2)
    identity `FunctionTransformer` for numerical variables. This is followed by
    scaling all features to the range (-1, 1) using min-max scaling.
  """
  is_categorical = np.array([dt.kind == 'O' for dt in df.dtypes])
  categorical_cols = df.columns.values[is_categorical].tolist()
  numerical_cols = df.columns.values[~is_categorical].tolist()

  # Use pandas one-hot encoding to avoid sklearn API compatibility issues
  # across versions while keeping the same transformed representation.
  if categorical_cols:
    transformed_df = pd.get_dummies(
        df,
        columns=categorical_cols,
        prefix_sep=': ',
        dtype=np.float32)
  else:
    transformed_df = df.copy()
  if numerical_cols:
    transformed_df[numerical_cols] = transformed_df[numerical_cols].astype(
        np.float32)

  scaler = MinMaxScaler(feature_range=(-1, 1))
  transformed = scaler.fit_transform(transformed_df)
  return transformed, list(transformed_df.columns)


def load_dataset(dataset_name):
  """Loads the dataset according to the `dataset_name` passed.

  Args:
    dataset_name: Name of the dataset to be loaded.

  Returns:
    data_x: np.ndarray of size (n_examples, n_features) containining the
      features per input data point where n_examples is the number of examples
      and n_features is the number of features.
    data_y: np.ndarray of size (n_examples, ) containing the label/target
      for each example where n_examples is the number of examples.
    column_names: A list containing the feature names.

  Raises:
    ValueError: If the `dataset_name` is not in ('Telco', 'BreastCancer',
    'Adult', 'Credit', 'Heart', 'Mimic2', 'Recidivism', 'Fico', Housing').
  """
  if dataset_name == 'Telco':
    dataset = load_telco_churn_data()
  elif dataset_name == 'BreastCancer':
    dataset = load_breast_data()
  elif dataset_name == 'Adult':
    dataset = load_adult_data()
  elif dataset_name == 'Credit':
    dataset = load_credit_data()
  elif dataset_name == 'Heart':
    dataset = load_heart_data()
  elif dataset_name == 'Mimic2':
    dataset = load_mimic2_data()
  elif dataset_name == 'Recidivism':
    dataset = load_recidivism_data()
  elif dataset_name == 'Fico':
    dataset = load_fico_score_data()
  elif dataset_name == 'Housing':
    dataset = load_california_housing_data()
  else:
    raise ValueError('{} not found!'.format(dataset_name))

  data_x, data_y = dataset['X'].copy(), dataset['y'].copy()
  problem_type = dataset['problem']
  data_x, column_names = transform_data(data_x)
  data_x = data_x.astype('float32')
  if (problem_type == 'classification') and \
      (not isinstance(data_y, np.ndarray)):
    data_y = pd.get_dummies(data_y).values
    data_y = np.argmax(data_y, axis=-1)
  data_y = data_y.astype('float32')
  return data_x, data_y, column_names


def get_train_test_fold(
    data_x,
    data_y,
    fold_num,
    num_folds,
    stratified = True,
    random_state = 42):
  """Returns a specific fold split for K-Fold cross validation.

  Randomly split dataset into `num_folds` consecutive folds and returns the fold
  with index `fold_index` for testing while the `num_folds` - 1 remaining folds
  form the training set.

  Args:
    data_x: Training data, with shape (n_samples, n_features), where n_samples
      is the number of samples and n_features is the number of features.
    data_y: The target variable, with shape (n_samples), for supervised learning
      problems.  Stratification is done based on the y labels.
    fold_num: Index of fold used for testing.
    num_folds: Number of folds.
    stratified: Whether to preserve the percentage of samples for each class in
      the different folds (only applicable for classification).
    random_state: Seed used by the random number generator.

  Returns:
    (x_train, y_train): Training folds containing 1 - (1/`num_folds`) fraction
      of entire data.
    (x_test, y_test): Test fold containing 1/`num_folds` fraction of data.
  """
  if stratified:
    stratified_k_fold = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=random_state)
  else:
    stratified_k_fold = KFold(
        n_splits=num_folds, shuffle=True, random_state=random_state)
  assert fold_num <= num_folds and fold_num > 0, 'Pass a valid fold number.'
  for train_index, test_index in stratified_k_fold.split(data_x, data_y):
    if fold_num == 1:
      x_train, x_test = data_x[train_index], data_x[test_index]
      y_train, y_test = data_y[train_index], data_y[test_index]
      return (x_train, y_train), (x_test, y_test)
    else:
      fold_num -= 1


def split_training_dataset(
    data_x,
    data_y,
    n_splits,
    stratified = True,
    test_size = 0.125,
    random_state = 1337):
  """Yields a generator that randomly splits data into (train, validation) set.

  The train set is used for fitting the DNNs/NAMs while the validation set is
  used for early stopping.

  Args:
    data_x: Training data, with shape (n_samples, n_features), where n_samples
      is the number of samples and n_features is the number of features.
    data_y: The target variable, with shape (n_samples), for supervised learning
      problems.  Stratification is done based on the y labels.
    n_splits: Number of re-shuffling & splitting iterations.
    stratified: Whether to preserve the percentage of samples for each class in
      the (train, validation) splits. (only applicable for classification).
    test_size: The proportion of the dataset to include in the validation split.
    random_state: Seed used by the random number generator.

  Yields:
    (x_train, y_train): The training data split.
    (x_validation, y_validation): The validation data split.
  """
  if stratified:
    stratified_shuffle_split = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state)
  else:
    stratified_shuffle_split = ShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state)
  split_gen = stratified_shuffle_split.split(data_x, data_y)

  for train_index, validation_index in split_gen:
    x_train, x_validation = data_x[train_index], data_x[validation_index]
    y_train, y_validation = data_y[train_index], data_y[validation_index]
    assert x_train.shape[0] == y_train.shape[0]
    yield (x_train, y_train), (x_validation, y_validation)
