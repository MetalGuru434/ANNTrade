"""# Задание 2

1. Скачайте и распакуйте "акционный" датасет (ссылка: https://storage.yandexcloud.net/terraai/sources/shares.zip)
2. Разделите данные на обучающую и проверочную выборки (1300 и 165 примеров соответственно) для компании Polymetal (POLYb_1d.csv).
3. Подберите модель предсказания цены закрытия
4. Выведите на экран график предсказания модели на отдельных участках проверочной выборки длиной 30 дней каждый

**Подготовка**
"""

# Импортируем необходимые библиотеки
import pandas as pd
import gdown

"""## Загрузка и анализ данных"""

# Загрузка обучающих данных
URL = 'https://storage.yandexcloud.net/terraai/sources/shares.zip'
download_filename = gdown.download(URL, None, quiet = True)

# Распаковка архива
!unzip -q {download_filename} -d '/content/data'

# Удаление архива
!rm -rf {download_filename}

# Вывод датасета
DATASET_CSV_DIR = 'data/POLYb_1d.csv'


df = pd.read_csv(DATASET_CSV_DIR)
print(df.shape)
df

# Переформатирование датасета, исключение ненужных колонок

df = df[['<DATE>', '<OPEN>', '<HIGH>','<LOW>','<CLOSE>']]
df['<DATE>'] = pd.to_datetime(df['<DATE>'], format='%Y%m%d')
df.head()

# Ваше решение
import os
import io
import zipfile
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ensure dataset is available
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)
csv_path = os.path.join(DATA_DIR, 'POLYb_1d.csv')
if not os.path.exists(csv_path):
    zip_url = 'https://storage.yandexcloud.net/terraai/sources/shares.zip'
    r = requests.get(zip_url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(DATA_DIR)

# load and prepare data
df = pd.read_csv(csv_path)
df = df[['<DATE>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']]
df['<DATE>'] = pd.to_datetime(df['<DATE>'], format='%Y%m%d')

# split into train and validation sets
train_df = df.iloc[:1300].reset_index(drop=True)
val_df = df.iloc[1300:1300 + 165].reset_index(drop=True)

X_train = train_df[['<OPEN>', '<HIGH>', '<LOW>']]
y_train = train_df['<CLOSE>']
X_val = val_df[['<OPEN>', '<HIGH>', '<LOW>']]
val_df['y_true'] = val_df['<CLOSE>']

# model training
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
val_df['y_pred'] = model.predict(X_val)

# visualization on 30-day segments
segment_length = 30
num_segments = len(val_df) // segment_length
for i in range(num_segments):
    segment = val_df.iloc[i * segment_length:(i + 1) * segment_length]
    plt.figure(figsize=(10, 5))
    plt.plot(segment['<DATE>'], segment['y_true'], label='Real')
    plt.plot(segment['<DATE>'], segment['y_pred'], label='Predicted')
    plt.title(f'Segment {i + 1}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

