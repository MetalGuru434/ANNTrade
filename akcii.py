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
