# **Parallel Tree Ensemble Training** 🚀

Этот проект демонстрирует обучение ансамблей деревьев решений (**Random Forest**, **HistGradientBoosting**, **LightGBM**) с использованием **параллельных вычислений** для ускорения процесса. Проект разработан на **Python** с использованием библиотек `scikit-learn`, `lightgbm` и `joblib`.

---

## **Содержание**
- [Описание проекта](#описание-проекта)
- [Параллельные вычисления](#параллельные-вычисления)
- [Тестирование](#тестирование)
- [Результаты и выводы](#результаты-и-выводы)
- [Скринкаст эталонного решения своей практической работы](#Скринкаст-эталонного-решения-своей-практической-работы)

---

## **Описание проекта**

Цель проекта — продемонстрировать возможности параллельного обучения ансамблей деревьев решений для задач классификации. Поддерживаются следующие модели:
- **Random Forest** (многопоточное обучение с `n_jobs`).
- **HistGradientBoosting** (оптимизированный бустинг на гистограммах).
- **LightGBM** (быстрый бустинг с поддержкой многопоточности).

Проект также реализует модульное тестирование через **pytest** и замеры времени обучения моделей.

---

### **Параллельные вычисления**

В проекте используются следующие механизмы для реализации параллельных вычислений:

#### **1. n_jobs в моделях**  
- Параметр **`n_jobs`** используется в **RandomForestClassifier** и **LightGBM** для параллельного обучения деревьев решений.  
- Он позволяет автоматически распределять задачи по доступным ядрам процессора.  
- **Преимущества**: ускоряет обучение за счёт использования нескольких потоков или процессов.

#### **2. joblib.Parallel для параллельного обучения моделей**  
- Библиотека **Joblib** позволяет выполнять параллельное выполнение задач с помощью функции **`Parallel`**.  
- В проекте **`joblib.Parallel`** используется для параллельного обучения нескольких моделей одновременно.  
- **Преимущества**: эффективно масштабирует выполнение задач на нескольких процессорах.

#### **3. Многопроцессорность с распределением задач**  
- **Joblib** автоматически распределяет задачи по доступным ядрам процессора.  
- Использование **`n_jobs=-1`** означает, что все доступные ядра будут задействованы для выполнения задач.  
- **Преимущества**: позволяет максимально эффективно использовать все вычислительные ресурсы системы.

---

### **Тестирование**

В проекте используется библиотека **pytest** для написания и выполнения тестов, обеспечивая проверку правильности работы моделей и алгоритмов. Основные аспекты тестирования включают:

1. **Юнит-тесты**:
   - Тестируются основные компоненты проекта, такие как модели, функции и методы.
   - Проверяется корректность работы каждой отдельной части системы, например, обучение моделей и предсказания.

2. **Параллельное выполнение тестов**:
   - Для ускорения процесса тестирования используется параллельное выполнение тестов с помощью библиотеки **pytest-xdist**.

3. **Проверка времени выполнения**:
   - Время обучения и предсказания моделей замеряется для проверки производительности и эффективности параллельных вычислений.

---

### **Результаты и выводы**

В ходе реализации проекта были обучены три модели деревьев решений: **Random Forest**, **GradientBoosting** и **LightGBM**. Результаты показали следующие ключевые моменты:

1. **Ускорение обучения**:
   - Время обучения моделей с использованием параллельных вычислений:
     - **LightGBMModel**: 0.2672 секунд
     - **RandomForest**: 0.3060 секунд
     - **GradientBoostingModel**: 0.3847 секунд
   - Использование параллельных вычислений с параметром **`n_jobs=-1`** и **joblib.Parallel** позволило ускорить обучение моделей, что заметно улучшило производительность.

2. **Точность моделей**:
   - **GradientBoostingModel** показал наилучшую точность:
     - **Accuracy**: 0.9150
     - **Precision**: 0.9207
     - **Recall**: 0.9150
     - **F1 Score**: 0.9151
     - **Confusion Matrix**:
       ```
       [[90  3]
        [14 93]]
       ```
   - **RandomForest** продемонстрировал следующие результаты:
     - **Accuracy**: 0.9000
     - **Precision**: 0.9047
     - **Recall**: 0.9000
     - **F1 Score**: 0.9001
     - **Confusion Matrix**:
       ```
       [[88  5]
        [15 92]]
       ```
   - **LightGBMModel** показал немного меньшую точность:
     - **Accuracy**: 0.8950
     - **Precision**: 0.9006
     - **Recall**: 0.8950
     - **F1 Score**: 0.8951
     - **Confusion Matrix**:
       ```
       [[88  5]
        [16 91]]
       ```

3. **Резюме по производительности моделей**:
   - **GradientBoostingModel** показал лучшую точность, однако, потребовал больше времени для обучения, чем другие модели.
   - **RandomForest** и **LightGBMModel** показали схожие результаты по точности, но **RandomForest** работал немного быстрее.

4. **Выводы**:
   - Параллельные вычисления значительно ускоряют обучение моделей, особенно на многозадачных системах.
   - Выбор модели зависит от задачи: **GradientBoosting** может быть предпочтительнее для задач с высокой точностью, в то время как **RandomForest** и **LightGBM** могут быть более быстрыми и эффективными для других типов задач.
   - Использование параллельных вычислений в моделях с **`n_jobs=-1`** и с **joblib.Parallel** улучшает производительность, что особенно важно для работы с большими объемами данных.

---

### **Скринкаст эталонного решения своей практической работы**

[Посмотреть скринкаст](https://drive.google.com/file/d/1AE3GRyFZJzifsoN-FdNXLALd76Pib5Sr/view?usp=sharing)
