import polars as pl

import xgboost as xgb
from src.models.base.model import BaseModel


class XGBoost(BaseModel):
    def __init__(self):
        self.model = None

    def train(self, params, train_set, valid_sets):
        evals = [(train_set, "train")]
        if valid_sets is not None:
            evals.append((valid_sets, "valid"))

        self.model = xgb.train(
            params,
            train_set,
            num_boost_round=1000,
            evals=evals,
            verbose_eval=False,
        )

    def predict(self, x_for_predict: pl.DataFrame):
        return self.model.predict(
            xgb.DMatrix(x_for_predict.to_numpy()), output_margin=False
        )
