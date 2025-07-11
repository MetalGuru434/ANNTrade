# Работа с массивами
import numpy as np

# Работа с таблицами
import pandas as pd

# Классы-конструкторы моделей нейронных сетей
from tensorflow.keras.models import Sequential, Model

# Основные слои
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, LSTM, MaxPooling1D

# Оптимизаторы
from tensorflow.keras.optimizers import Adam

# Генератор выборки временных рядов
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Нормировщики
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Коллбэк
from tensorflow.keras.callbacks import LearningRateScheduler

# Загрузка датасетов из облака google
import gdown

# Отрисовка графиков
import matplotlib.pyplot as plt

# Отрисовка графики в ячейке colab
#%matplotlib inline

# Отключение предупреждений
import warnings
warnings.filterwarnings('ignore')

# Назначение размера и стиля графиков по умолчанию

from pylab import rcParams
plt.style.use('ggplot')
rcParams['figure.figsize'] = (14, 7)

# Загрузка датасетов из облака

gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l11/16_17.csv', None, quiet=True)
gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l11/18_19.csv', None, quiet=True)


# Чтение данных в таблицы с удалением ненужных столбцов

data16_17 = pd.read_csv('16_17.csv', sep=';').drop(columns=['DATE', 'TIME'])
data18_19 = pd.read_csv('18_19.csv', sep=';').drop(columns=['DATE', 'TIME'])

# Просмотр размерностей получившихся таблиц

print(data16_17.shape)
print(data18_19.shape)


# Создание общего набора данных из двух датасетов

data = pd.concat([data16_17,data18_19])   # Объединение датасетов
data = data.reset_index(drop = True)      # Обнуление индексов

# Проверка формы данных
print(data.shape)


# Получение названий столбцов

col = data.columns
print(col)


# Задание циклов для столбцов таким образом, чтобы происходил перебор всех возможных пар:
# Перебор индексов массива с именами столбцов 'OPEN', 'MAX', 'MIN', 'CLOSE', 'VOLUME',
# получая пары 'OPEN' и 'MAX', 'OPEN' и 'MIN', 'OPEN' и 'CLOSE' ..., 'MAX' и 'MIN', 'MAX' и 'CLOSE' и т.д

for i in range(col.shape[0]): # Для всех пар
    for j in range(i + 1 , col.shape[0]): # Расчет
        data[col[i] + '-' + col[j]] = data[col[i]] - data[col[j]] # Разности
        data['|' + col[i] + '-' + col[j] + '|'] = abs(data[col[i]] - data[col[j]]) # Модулей разностей
        data[col[i] + '*' + col[j]] = data[col[i]] * data[col[j]] # Произведения

# Для каждого столбца 'OPEN', 'MAX', 'MIN', 'CLOSE', 'VOLUME' расчет:
for i in col:
    # Обратные значения. 1e-3 в формуле нужно, чтобы случайно не разделить на 0
    data['Обратный ' + i] = 1 / (data[i] + 1e-3)
    # Создание пустого столбца
    data['Производная от ' + i] = np.nan
    # При помощи срезов расчет первых производных, .reset_index(drop=True) нужен для корректных расчетов
    data['Производная от ' + i][1:] = data[i][1:].reset_index(drop=True) - data[i][:-1].reset_index(drop=True)
    # Создание пустого столбца
    data['Вторая производная от ' + i] = np.nan
    # При помощи срезов расчет вторых производных
    data['Вторая производная от ' + i][2:] = data[i][2:].reset_index(drop=True) - 2 * data[i][1:-1].reset_index(drop=True) + data[i][:-2].reset_index(drop=True)


# Просмотр результатов
data


# Использование всех столбцов, кроме первых двух
data = np.array(data.iloc[2:])

# Перевод в numpy
data = np.array(data)

# Переменная, для использования одной и той же архитектуры под разные матрицы
columnsamount = data.shape[1]

# ---------------------------------------------------------------------------
# Новый вариант подготовки выборки и обучения модели. Для прогноза будем
# использовать только исходные 5 столбцов ("OPEN", "MAX", "MIN",
# "CLOSE", "VOLUME") без дополнительных признаков. В качестве входа сети
# сформируем последовательность длиной 200: 100 последних точек с шагом 1 и
# 100 точек с шагом 10. На каждой итерации будем предсказывать значение
# следующего временного шага столбца CLOSE.

# Получение исходных данных без расширения признаков
base_data = pd.concat([data16_17, data18_19]).reset_index(drop=True)
base_data = np.array(base_data)
columnsamount = base_data.shape[1]

# Индекс столбца, который будем предсказывать
target_idx = list(data16_17.columns).index('CLOSE') if 'CLOSE' in data16_17.columns else 0

# Масштабирование признаков
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(base_data)

# Параметры формирования последовательностей
step_one = 100
step_ten = 100
lookback = step_ten * 10  # самый дальний шаг
seq_len = step_one + step_ten

# Подготовка обучающих примеров
X, y = [], []
for i in range(lookback, len(data_scaled)):
    # 100 точек с шагом 10 (отдалённое прошлое)
    idx10 = [i - 10 * j for j in range(step_ten, 0, -1)]
    seq10 = data_scaled[idx10]
    # 100 последних точек подряд
    seq1 = data_scaled[i - step_one:i]
    seq = np.vstack([seq10, seq1])
    X.append(seq)
    y.append(data_scaled[i, target_idx])

X = np.array(X)
y = np.array(y)

# Разделение на обучающую и тестовую выборки
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Модель LSTM с увеличенным окном просмотра
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_len, columnsamount)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(0.001), loss='mse')

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=2,
)

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

test_mse = model.evaluate(X_test, y_test, verbose=0)
print('Test MSE:', test_mse)


