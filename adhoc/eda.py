# %%
%%time
# Standard library imports
import sys
from dataclasses import dataclass
from pathlib import Path

# Third party imports
import numpy as np
import polars as pl
from pandera.typing.polars import LazyFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Local imports
from src.data_processor.process import (
    one_hot_encoding,
    replace_null_data_with_default,
    select_cols,
)
from src.models.lightgbm.model import LightGBM
from src.models.lightgbm.preprocess import create_input_data_for_lightgbm

# Magic commands
%matplotlib inline
%load_ext autoreload
%autoreload 2


@dataclass
class Config:
    project_dir: Path = Path("../")
    data_dir: Path = project_dir / "data"
    raw_dir: Path = data_dir / "raw"
    interim_dir: Path = data_dir / "interim"
    processed_dir: Path = data_dir / "processed"


config = Config()
sys.path.append(str(config.project_dir.resolve()))

# Load data
df_train = pl.read_csv(config.raw_dir / "train.csv")
df_test = pl.read_csv(config.raw_dir / "test.csv")

from src.data_processor.process import (
    replace_null_data_with_default,
    select_cols,
    one_hot_encoding,
)

# 必要な特徴量のみを選択
feature_name_list = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target_name_list = ["Survived"]

# PassengerIdを除外してデータを選択
df_train_x = select_cols(df_train, feature_name_list)
df_train_y = select_cols(df_train, target_name_list)

# 欠損値の処理
df_train_x_processed = replace_null_data_with_default(
    df_train_x,
    replace_dict={
        "Age": df_train_x["Age"].mean(),
        "Embarked": "Unknown",
    },
)

# テストデータの処理（PassengerIdを除外）
df_test_x = select_cols(df_test, feature_name_list)
df_test_x_processed = replace_null_data_with_default(
    df_test_x,
    replace_dict={
        "Age": df_train_x["Age"].mean(),
        "Fare": df_train_x["Fare"].mean(),
        "Embarked": "Unknown",
    },
)

# One-hotエンコーディング
df_train_x_processed = one_hot_encoding(df_train_x_processed, ["Sex", "Embarked"])
df_test_x_processed = one_hot_encoding(df_test_x_processed, ["Sex", "Embarked"])

# %% 

from sklearn.model_selection import StratifiedKFold
from src.models.lightgbm.preprocess import create_input_data_for_lightgbm

from src.models.lightgbm.model import LightGBM
import numpy as np

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
from sklearn.metrics import accuracy_score

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "min_data_in_leaf": 20,
    "min_gain_to_split": 0.01,
}
oof_predictions = np.zeros(len(df_train_x_processed))
scores = []
for i, (train_, val_) in enumerate(kf.split(df_train_x_processed, df_train_y)):
    x_train, x_val = df_train_x_processed[train_], df_train_x_processed[val_]
    y_train = df_train_y[train_]
    y_val = df_train_y[val_]

    lgb_train = create_input_data_for_lightgbm(x_train, y_train)
    lgb_eval = create_input_data_for_lightgbm(x_val, y_val, reference=lgb_train)

    model = LightGBM()
    model.train(
        params=params,
        train_set=lgb_train,
        valid_sets=lgb_eval,
    )
    y_val_pred = model.predict(x_val)
    y_val_pred_binary = (y_val_pred > 0.5).astype(int)
    oof_predictions[val_] = y_val_pred_binary
    score = accuracy_score(y_val, y_val_pred_binary)
    scores.append(score)
print(f"LightGBM CV scores: {scores}")
print(f"LightGBM Mean CV score: {np.mean(scores)}")
# %%
# XGBoostでの実装
from src.models.xgboost.preprocess import create_input_data_for_xgboost
from src.models.xgboost.model import XGBoost

# XGBoostのパラメータ設定
# XGBoostでの実装
from src.models.xgboost.preprocess import create_input_data_for_xgboost
from src.models.xgboost.model import XGBoost

# XGBoostのパラメータ設定
xgb_params = {
    "objective": "binary:logistic",
    "max_depth": 5,
    "eta": 0.1,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
}

oof_predictions_xgb = np.zeros(len(df_train_x_processed))
scores_xgb = []

for i, (train_, val_) in enumerate(kf.split(df_train_x_processed, df_train_y)):
    x_train, x_val = df_train_x_processed[train_], df_train_x_processed[val_]
    y_train = df_train_y[train_]
    y_val = df_train_y[val_]

    xgb_train = create_input_data_for_xgboost(x_train, y_train)
    xgb_eval = create_input_data_for_xgboost(x_val, y_val)

    model = XGBoost()
    model.train(
        params=xgb_params,
        train_set=xgb_train,
        valid_sets=xgb_eval,
    )

    y_val_pred = model.predict(x_val)
    y_val_pred_binary = (y_val_pred > 0.5).astype(int)
    oof_predictions_xgb[val_] = y_val_pred_binary
    score = accuracy_score(y_val, y_val_pred_binary)
    scores_xgb.append(score)

print(f"XGBoost CV scores: {scores_xgb}")
print(f"XGBoost Mean CV score: {np.mean(scores_xgb)}")
# %%
# CatBoostでの実装
from src.models.catboost.preprocess import create_input_data_for_catboost
from src.models.catboost.model import CatBoost

# CatBoostのパラメータ設定
cb_params = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "random_seed": 42,
}

oof_predictions_cb = np.zeros(len(df_train_x_processed))
scores_cb = []

for i, (train_, val_) in enumerate(kf.split(df_train_x_processed, df_train_y)):
    x_train, x_val = df_train_x_processed[train_], df_train_x_processed[val_]
    y_train = df_train_y[train_]
    y_val = df_train_y[val_]

    cb_train = create_input_data_for_catboost(x_train, y_train)
    cb_eval = create_input_data_for_catboost(x_val, y_val)

    model = CatBoost()
    model.train(
        params=cb_params,
        train_set=cb_train,
        valid_sets=cb_eval,
    )

    y_val_pred = model.predict(x_val)
    y_val_pred_binary = (y_val_pred > 0.5).astype(int)
    oof_predictions_cb[val_] = y_val_pred_binary
    score = accuracy_score(y_val, y_val_pred_binary)
    scores_cb.append(score)

print(f"CatBoost CV scores: {scores_cb}")
print(f"CatBoost Mean CV score: {np.mean(scores_cb)}")
# %%
