from .base_model import BaseModel
import lightgbm as lgb
import numpy as np

class LightGBMModel(BaseModel):
    def __init__(self, num_leaves=31, max_depth=-1, learning_rate=0.05, n_estimators=50, n_jobs=-1):
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X, y):
        self.model = lgb.LGBMClassifier(
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs
        )
        self.model.fit(X, y)

    def predict(self, X):
        return np.round(self.model.predict(X))

    def get_params(self, deep=True):
        return {
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'n_jobs': self.n_jobs
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
