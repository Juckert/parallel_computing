from .base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForest(BaseModel):
    def __init__(self, n_trees=100, max_depth=None, n_jobs=-1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X, y):
        self.model = RandomForestClassifier(
            n_estimators=self.n_trees,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs
        )
        self.model.fit(X, y)

    def predict(self, X):
        return np.round(self.model.predict(X))

    def get_params(self, deep=True):
        return {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'n_jobs': self.n_jobs
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
