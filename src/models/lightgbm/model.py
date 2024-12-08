import polars as pl

import lightgbm as lgb
from src.models.base.model import BaseModel


class LightGBM(BaseModel):
    def __init__(self):
        self.model = None

    def train(
        self,
        params,
        train_set,
        valid_sets,
    ):
        self.model = lgb.train(
            params,
            train_set,
            valid_sets=valid_sets,
            num_boost_round=1000,
        )

    def predict(self, x_for_predict: pl.DataFrame):
        return self.model.predict(x_for_predict.to_numpy())
