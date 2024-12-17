import pytest
import numpy as np
from models.random_forest import RandomForest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@pytest.fixture
def data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_random_forest_fit(data):
    X_train, X_test, y_train, y_test = data
    model = RandomForest(n_trees=100, max_depth=10)
    model.fit(X_train, y_train)
    assert model.model is not None

def test_random_forest_predict(data):
    X_train, X_test, y_train, y_test = data
    model = RandomForest(n_trees=100, max_depth=10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)

def test_random_forest_accuracy(data):
    X_train, X_test, y_train, y_test = data
    model = RandomForest(n_trees=100, max_depth=10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    assert acc > 0.7
