from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Абстрактный базовый класс для всех моделей.
    Все модели (например, RandomForest, LightGBM, GradientBoosting) должны наследовать этот класс
    и реализовывать методы fit и predict.
    """
    
    @abstractmethod
    def fit(self, X, y):
        """
        Обучение модели на данных.
        
        :param X: Данные для обучения (матрица признаков)
        :param y: Метки классов (целевая переменная)
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Предсказание с использованием обученной модели.
        
        :param X: Данные для предсказания (матрица признаков)
        :return: Предсказанные метки классов
        """
        pass

    @abstractmethod
    def get_params(self, deep=True):
        """
        Получить параметры модели.
        
        :param deep: Если True, параметры будут рекурсивно собраны из всех компонентов модели.
        :return: Словарь параметров модели
        """
        pass

    @abstractmethod
    def set_params(self, **params):
        """
        Установить параметры модели.
        
        :param params: Параметры для настройки модели
        :return: Обновленная модель с новыми параметрами
        """
        pass
