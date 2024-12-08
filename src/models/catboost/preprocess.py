import numpy as np
import polars as pl

import catboost as cb


class CatBoostDataset:
    def __init__(self, features, label=None):
        self.features = features
        self.label = label

    def get_features(self):
        return self.features

    def get_label(self):
        return self.label


def create_input_data_for_catboost(
    df_x: pl.DataFrame, df_y: pl.DataFrame, reference=None
):
    features = df_x.to_numpy()
    if df_y is not None:
        label = df_y.to_numpy().ravel()
    else:
        label = None
    return CatBoostDataset(features, label)
