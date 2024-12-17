from sklearn.ensemble import HistGradientBoostingClassifier

class GradientBoostingModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=None):
        """
        Инициализация HistGradientBoostingClassifier.
        Параметры:
            - n_estimators: Количество деревьев (итераций бустинга).
            - learning_rate: Скорость обучения.
            - max_depth: Максимальная глубина деревьев.
        """
        self.model = HistGradientBoostingClassifier(
            max_iter=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth
        )
    
    def fit(self, X, y):
        """Обучение модели на данных."""
        self.model.fit(X, y)
    
    def predict(self, X):
        """Предсказание меток для входных данных."""
        return self.model.predict(X)
