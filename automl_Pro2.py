
"""# Задание 2

1. Скачайте и распакуйте "акционный" датасет (ссылка: https://storage.yandexcloud.net/terraai/sources/shares.zip)
2. Выберите любой файл с датасетом, разделите датасет на тренировочную и проверочную выборки, найдите лучшую модель, визуализируйте график предикта.
3. Проведите серию обучений. На каждом шаге:
  а). добавляете к тренировочной выборке по N экземпляров из проверочной выборки,
  б). уменьшаете проверочную выборку на соответсвующее количество экземпляров,
  в). инициализируете новую модель и обучаете ее на тренировочной выборке из пункта а),  
  г). сохраняете N последующих значений предикта на проверочной выборке из пункта б).
  Выберите число N экспериментально, исходя из размера проверочной выборки и адекватного времени на обучение.
4. "Склейте" все полученные интервалы предикта в один массив.
5. На одном графике визуализируйте проверочную выборку, результат одной модели из п.2 и результат составного массива.  
6. Сделайте вывод
"""

# Импортируем необходимые библиотеки
import pandas as pd
import gdown
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

"""## Загрузка и анализ данных"""

# Загрузка обучающих данных (скачивание при отсутствии локального файла)
URL = 'https://storage.yandexcloud.net/terraai/sources/shares.zip'
DATASET_CSV_DIR = 'data/GAZP_1d.csv'
if not os.path.exists(DATASET_CSV_DIR):
    download_filename = gdown.download(URL, None, quiet=True)
    with zipfile.ZipFile(download_filename, 'r') as zip_ref:
        zip_ref.extractall('data')
    os.remove(download_filename)

# Возьмем "дневные" акции ГАЗПРОМа
DATASET_CSV_DIR = 'data/GAZP_1d.csv'


df = pd.read_csv(DATASET_CSV_DIR)
print(df.shape)
df

# Переформатирование датасета, исключение ненужных колонок

df = df[['<DATE>', '<OPEN>', '<HIGH>','<LOW>','<CLOSE>']]
df['<DATE>'] = pd.to_datetime(df['<DATE>'], format='%Y%m%d')
df.head()

# Ваше решение

# Сортировка по дате и подготовка признаков/таргета
df = df.sort_values('<DATE>')
X = df[['<OPEN>', '<HIGH>', '<LOW>']].values
y = df['<CLOSE>'].values
dates = df['<DATE>'].values

# Деление на тренировочную и проверочную выборки
X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(
    X, y, dates, test_size=0.2, shuffle=False
)

# Подбор лучшей модели среди нескольких вариантов
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42)
}
best_model = None
best_name = None
best_score = float('inf')
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    if mse < best_score:
        best_score = mse
        best_model = model
        best_name = name

# Предикт лучшей модели на проверочной выборке
y_pred = best_model.predict(X_val)

plt.figure(figsize=(12, 6))
plt.plot(dates_val, y_val, label='Проверочная выборка')
plt.plot(dates_val, y_pred, label=f'Предикт {best_name}')
plt.legend()
plt.title('Сравнение модели на проверочной выборке')
plt.xlabel('Дата')
plt.ylabel('Цена закрытия')
plt.tight_layout()
plt.show()

# Пошаговый перенос данных из проверки в обучение
N = 20
X_train_dyn = X_train.copy()
y_train_dyn = y_train.copy()
X_val_dyn = X_val.copy()
y_val_dyn = y_val.copy()
pred_intervals = []

while len(X_val_dyn) > 0:
    step = min(N, len(X_val_dyn))
    model = clone(best_model)
    model.fit(X_train_dyn, y_train_dyn)
    preds = model.predict(X_val_dyn[:step])
    pred_intervals.extend(preds)
    X_train_dyn = np.vstack([X_train_dyn, X_val_dyn[:step]])
    y_train_dyn = np.concatenate([y_train_dyn, y_val_dyn[:step]])
    X_val_dyn = X_val_dyn[step:]
    y_val_dyn = y_val_dyn[step:]

pred_intervals = np.array(pred_intervals)

# Итоговый график
plt.figure(figsize=(12, 6))
plt.plot(dates_val, y_val, label='Проверочная выборка')
plt.plot(dates_val, y_pred, label=f'Предикт {best_name}')
plt.plot(dates_val, pred_intervals, label='Пошаговый предикт', linestyle='--')
plt.legend()
plt.title('Итоговый график предсказаний')
plt.xlabel('Дата')
plt.ylabel('Цена закрытия')
plt.tight_layout()
plt.show()

print('Начальный MSE лучшей модели:', best_score)
print('MSE пошагового предсказания:', mean_squared_error(y_val, pred_intervals))
