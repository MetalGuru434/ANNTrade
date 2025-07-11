# Работа с массивами данных
import numpy as np

# Работа с таблицами
import pandas as pd

# Отрисовка графиков
import matplotlib.pyplot as plt

# Функции-утилиты для работы с категориальными данными
from tensorflow.keras import utils

# Класс для конструирования последовательной модели нейронной сети
from tensorflow.keras.models import Sequential

# Основные слои
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    SpatialDropout1D,
    BatchNormalization,
    Embedding,
    GlobalAveragePooling1D,
    Activation,
)
from tensorflow.keras.callbacks import EarlyStopping

# Токенизатор для преобразование текстов в последовательности
from tensorflow.keras.preprocessing.text import Tokenizer

# Заполнение последовательностей до определенной длины
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка датасетов из облака google
# import gdown

# Для работы с файлами в Colaboratory
import os

# Отрисовка графиков
import matplotlib.pyplot as plt

# %matplotlib inline

# gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l7/tesla.zip', None, quiet=True)

# Распаковка архива в папку writers
# !unzip -qo tesla.zip -d tesla/

# Просмотр содержимого папки
# !ls tesla

  # Объявляем функции для чтения файла. На вход отправляем путь к файлу
def read_text(file_name):

  # Задаем открытие нужного файла в режиме чтения
  read_file = open(file_name, 'r')

  # Читаем текст
  text = read_file.read()

  # Переносы строки переводим в пробелы
  text = text.replace("\n", " ")

  # Возвращаем текст файла
  return text

# Объявляем интересующие нас классы
class_names = ["Негативный отзыв", "Позитивный отзыв"]

# Считаем количество классов
num_classes = len(class_names)

import os
# Создаём список под тексты для обучающей выборки
# Подготовка файлов обучающей выборки (не используется далее)
# texts_list = []
# for j in os.listdir('/content/tesla/'):
#     texts_list.append(read_text('/content/tesla/' + j))
#     print(j, 'добавлен в обучающую выборку')
# texts_len = [len(text) for text in texts_list]
# t_num = 0
# print(f'Размеры текстов по порядку (в символах):')
# for text_len in texts_len:
#     t_num += 1
#     print(f'Текст №{t_num}: {text_len}')
# print(texts_list[0][0:1223])
# train_len_shares = [(i - round(i/5)) for i in texts_len]
# t_num = 0
# for train_len_share in train_len_shares:
#     t_num += 1
#     print(f'Доля 80% от текста №{t_num}: {train_len_share} символов')

from itertools import chain


def load_reviews(path):
    """Read reviews from a text file and return unique non-empty lines."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # remove duplicates while preserving order
    return list(dict.fromkeys(lines))


DATA_DIR = 'tesla'

# Пути к файлам с отзывами
neg_path = os.path.join(DATA_DIR, 'Негативный отзыв.txt')
pos_path = os.path.join(DATA_DIR, 'Позитивный отзыв.txt')

# Загрузка и балансировка базы отзывов
neg_reviews = load_reviews(neg_path)
pos_reviews = load_reviews(pos_path)

min_len = min(len(neg_reviews), len(pos_reviews))
np.random.seed(42)
np.random.shuffle(neg_reviews)
np.random.shuffle(pos_reviews)
neg_reviews = neg_reviews[:min_len]
pos_reviews = pos_reviews[:min_len]

texts = neg_reviews + pos_reviews
labels = [0] * len(neg_reviews) + [1] * len(pos_reviews)

# Разделяем данные на обучающую и проверочную выборки
from sklearn.model_selection import train_test_split

train_texts, val_texts, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Векторизация текстов
vocab_size = 10000
max_len = 50
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)
X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len, padding='post')
X_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=max_len, padding='post')

# Преобразование меток
y_train = utils.to_categorical(y_train, num_classes)
y_val = utils.to_categorical(y_val, num_classes)

# Построение модели
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    SpatialDropout1D(0.2),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение сети
callbacks = []  # EarlyStopping can be added if needed

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=callbacks,
    verbose=2,
)

loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print('Validation accuracy:', accuracy)


