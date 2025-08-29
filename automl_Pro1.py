## Подготовка датасетов

# Импортируем модули
import os # работа с файлами
import gdown # загрузка данных с Gdrive
import pickle # сериализация объектов (сохранение переменных в файлы)

# Скачиваем датасет по ссылке
dataset_url = 'https://storage.yandexcloud.net/aiueducation/Content/DS/L8/dataset.pickle'
gdown.download(dataset_url, quiet=True)

# Загружаем датасет в переменные со списками
with open('dataset.pickle','rb') as f:
    all_texts_good, all_texts_bad = pickle.load(f)

print(f'ХОРОШИЕ ОТЗЫВЫ ({len(all_texts_good)} штук):\n')
for t in all_texts_good[:3]:
    print(t)
    print('*'*100)

print(f'ПЛОХИЕ ОТЗЫВЫ ({len(all_texts_bad)} штук):\n')
for t in all_texts_bad[:3]:
    print(t)
    print('*'*100)

# Выполним задание с помощью циклов по требуемому количеству отзывов
for n in [10, 50, 100]:
    for category in ['good','bad']:
        dataset_category = os.path.join(str(n), category)
        os.makedirs(dataset_category, exist_ok=True)
        file_path = os.path.join(dataset_category, f'{category}.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            if category=='good':
                f.writelines(all_texts_good[:n])
            else:
                f.writelines(all_texts_bad[:n])

# Ваше решение
# Ваше решение
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score
import os

def load_dataset(n):
    """Читает n хороших и n плохих отзывов из заранее созданных файлов."""
    with open(os.path.join(str(n), 'good', 'good.txt'), encoding='utf-8') as f:
        good = f.readlines()
    with open(os.path.join(str(n), 'bad', 'bad.txt'), encoding='utf-8') as f:
        bad = f.readlines()
    texts = good + bad
    labels = [1]*len(good) + [0]*len(bad)
    return texts, labels

def train_and_evaluate(n):
    """Формирует выборки, обучает модель и возвращает метрики."""
    texts, labels = load_dataset(n)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=0.3, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_val   = vectorizer.transform(X_val).toarray()
    X_test  = vectorizer.transform(X_test).toarray()

    # Ensure data is in the correct float32 format for TensorFlow
    X_train = np.asarray(X_train, dtype="float32")
    X_val   = np.asarray(X_val,   dtype="float32")
    X_test  = np.asarray(X_test,  dtype="float32")
    y_train = np.asarray(y_train, dtype="float32")
    y_val   = np.asarray(y_val,   dtype="float32")
    y_test  = np.asarray(y_test,  dtype="float32")

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=0)

    val_pred  = (model.predict(X_val)  > 0.5).astype(int)
    test_pred = (model.predict(X_test) > 0.5).astype(int)

    val_acc  = accuracy_score(y_val,  val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    val_f1   = f1_score(y_val,  val_pred)
    test_f1  = f1_score(y_test, test_pred)

    return {
        'samples': n*2,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'val_f1': val_f1,
        'test_f1': test_f1
    }

results = [train_and_evaluate(n) for n in [10, 50, 100]]
results_table = pd.DataFrame(results)
print(results_table)
