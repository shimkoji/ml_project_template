import json

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from src.models.lightgbm.model import LightGBM
from src.models.lightgbm.preprocess import create_input_data_for_lightgbm

from .base import BaseOptimizer

with open("../configs/default.json", "r") as f:
    config = json.load(f)


class LightGBMOptimizer(BaseOptimizer):
    def _get_model_params(self, trial):
        """LightGBM固有のパラメータ探索空間を定義"""
        return {
            **config["default_lgbm_params"],
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "min_gain_to_split": trial.suggest_float(
                "min_gain_to_split", 0.001, 0.1, log=True
            ),
        }

    def _create_model(self):
        """LightGBMモデルのインスタンスを作成"""
        return LightGBM()

    def _create_dataset(self, X, y, reference=None):
        """LightGBM用のデータセット形式に変換"""
        return create_input_data_for_lightgbm(X, y, reference=reference)

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
            val_data = self._create_dataset(
                X_val_fold, y_val_fold, reference=train_data
            )

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
