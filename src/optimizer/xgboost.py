import json

import numpy as np
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.models.xgboost.model import XGBoost
from src.models.xgboost.preprocess import create_input_data_for_xgboost

from .base import BaseOptimizer

with open("../configs/default.json", "r") as f:
    config = json.load(f)


class XGBoostOptimizer(BaseOptimizer):
    def _get_model_params(self, trial):
        """XGBoost固有のパラメータ探索空間を定義"""
        return {
            **config["default_xgb_params"],
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 0.1, 10.0, log=True
            ),
        }

    def _create_model(self):
        """XGBoostモデルのインスタンスを作成"""
        return XGBoost()

    def _create_dataset(self, X, y, reference=None):
        """XGBoost用のデータセット形式に変換"""
        return create_input_data_for_xgboost(X, y)

    def _objective(self, trial):
        """最適化の目的関数"""
        params = self._get_model_params(trial)
        kf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        scores = []

        for train_idx, val_idx in kf.split(self.X_train, self.y_train):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]

            train_data = self._create_dataset(X_train_fold, y_train_fold)
            val_data = self._create_dataset(X_val_fold, y_val_fold)

            model = self._create_model()
            model.train(
                params=params,
                train_set=train_data,
                valid_sets=val_data,
            )

            y_val_pred = model.predict(X_val_fold)
            y_val_pred_binary = (y_val_pred > 0.5).astype(int)
            score = self.score_function(y_val_fold, y_val_pred_binary)
            scores.append(score)

        return np.mean(scores)

    def optimize(self, direction="maximize"):
        """パラメータの最適化を実行"""
        study = optuna.create_study(direction=direction)
        study.optimize(self._objective, n_trials=self.n_trials)

        self.best_params = study.best_params
        self.best_score = study.best_value

        return self.best_params, self.best_score
