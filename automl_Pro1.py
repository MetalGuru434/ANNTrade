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
