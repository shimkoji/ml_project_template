from abc import ABC, abstractmethod

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold


class BaseOptimizer(ABC):
    def __init__(
        self,
        X_train,
        y_train,
        score_function,
        n_trials=100,
        n_splits=5,
        random_state=42,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.score_function = score_function
        self.random_state = random_state
        self.best_params = None
        self.best_score = None

    @abstractmethod
    def _get_model_params(self, trial):
        """モデル固有のパラメータ探索空間を定義"""
        pass

    @abstractmethod
    def _create_model(self):
        """モデルのインスタンスを作成"""
        pass

    @abstractmethod
    def _create_dataset(self, X, y, reference=None):
        """モデル固有のデータセット形式に変換"""
        pass

    @abstractmethod
    def _objective(self, trial):
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

    @abstractmethod
    def optimize(self, direction="maximize"):
        study = optuna.create_study(direction=direction)
        study.optimize(self._objective, n_trials=self.n_trials)

        self.best_params = study.best_params
        self.best_score = study.best_value

        return self.best_params, self.best_score
