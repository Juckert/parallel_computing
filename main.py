import time
from models.random_forest import RandomForest
from models.lightgbm_model import LightGBMModel
from models.gradient_boosting_model import GradientBoostingModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from joblib import Parallel, delayed

# Функция для обучения и оценки модели
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    elapsed_time = time.time() - start_time
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}, Time taken: {elapsed_time:.4f} seconds")
    return accuracy, elapsed_time

# Главная функция
def main():
    # Генерация искусственного набора данных
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Инициализация моделей
    rf_model = RandomForest(n_trees=100, max_depth=10)
    lgb_model = LightGBMModel(num_leaves=31, n_estimators=100)
    gb_model = GradientBoostingModel(n_estimators=100, learning_rate=0.1)

    # Параллельное обучение моделей
    models = [rf_model, lgb_model, gb_model]
    
    results = Parallel(n_jobs=-1)(delayed(train_and_evaluate)(model, X_train, X_test, y_train, y_test) for model in models)
    
    # Сортировка результатов по точности
    results_sorted = sorted(zip(models, results), key=lambda x: x[1][0], reverse=True)

    print("\nModel Performance Summary:")
    for model, (accuracy, time_taken) in results_sorted:
        print(f"{model.__class__.__name__} - Accuracy: {accuracy:.4f}, Time Taken: {time_taken:.4f} seconds")

if __name__ == "__main__":
    main()
