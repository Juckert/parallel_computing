import pytest
import numpy as np
from models.gradient_boosting_model import GradientBoostingModel
from sklearn.datasets import make_classification

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y

def test_gradient_boosting_training(sample_data):
    X, y = sample_data
    model = GradientBoostingModel(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert np.unique(predictions).size <= 2
