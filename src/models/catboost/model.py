import polars as pl

import catboost as cb
from src.models.base.model import BaseModel


class CatBoost(BaseModel):
    def __init__(self):
        self.model = None

    def train(self, params, train_set, valid_sets):
        self.model = cb.CatBoostClassifier(**params)
        self.model.fit(
            train_set.get_features(),
            train_set.get_label(),
            eval_set=(valid_sets.get_features(), valid_sets.get_label()),
            verbose=False,
        )

    def predict(self, x_for_predict: pl.DataFrame):
        return self.model.predict_proba(x_for_predict.to_numpy())[:, 1]
