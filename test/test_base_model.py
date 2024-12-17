import pytest
from models.base_model import BaseModel

class DummyModel(BaseModel):
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return [1] * len(X)
    
    def get_params(self, deep=True):
        return {}
    
    def set_params(self, **params):
        return self

def test_base_model():
    model = DummyModel()
    assert model is not None
    assert model.predict([1, 2, 3]) == [1, 1, 1]
