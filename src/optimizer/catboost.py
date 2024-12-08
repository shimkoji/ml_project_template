import json

import numpy as np
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from src.models.catboost.model import CatBoost
from src.models.catboost.preprocess import create_input_data_for_catboost

from .base import BaseOptimizer

# Load configuration from JSON file
with open("../configs/default.json", "r") as f:
    config = json.load(f)


class CatBoostOptimizer(BaseOptimizer):
    def _get_model_params(self, trial):
        """CatBoostRegressor固有のパラメータ探索空間を定義"""
        return {
            **config["default_catboost_params"],
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-8, 10.0, log=True
            ),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        }

    def _create_model(self):
        """CatBoostモデルのインスタンスを作成"""
        return CatBoost()

    def _create_dataset(self, X, y, reference=None):
        """CatBoost用のデータセット形式に変換"""
        return create_input_data_for_catboost(X, y)

    def _objective(self, trial):
        """最適化の目的関数"""
        params = self._get_model_params(trial)
        kf = KFold(
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
