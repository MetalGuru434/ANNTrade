# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# Загрузка датасетов из облака google
import gdown

# Функции операционной системы
import os

# Регулярные выражения
import re

# Работа с внешними процессами и архивами
import subprocess
import zipfile

# @title Libraries
# Работа с массивами данных
import numpy as np

# Функции-утилиты для работы с категориальными данными
from tensorflow.keras import utils

# Класс для конструирования последовательной модели нейронной сети
from tensorflow.keras.models import Sequential

# Основные слои
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D

# Рисование схемы модели
from tensorflow.keras.utils import plot_model

# Матрица ошибок классификатора
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Отрисовка графиков
import matplotlib.pyplot as plt

# Вывод объектов в ячейке colab
try:
    from IPython.display import display
except ImportError:  # Если скрипт запускается вне Jupyter
    def display(obj):
        return obj

# %matplotlib inline

# Работа с текстом
subprocess.run(["pip", "install", "-q", "nltk", "pymorphy3"], check=True)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3

# Подсчет частоты элементов
from collections import Counter

# Загрузим датасет из облака, если архив отсутствует локально
WRITERS_ARCHIVE = 'writers.zip'
if not os.path.exists(WRITERS_ARCHIVE):
    try:
        gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l7/writers.zip',
                       WRITERS_ARCHIVE,
                       quiet=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('Не удалось скачать архив writers.zip. '
                           'Поместите архив в корневую папку проекта вручную.') from exc
else:
    print('Обнаружен локальный архив writers.zip, скачивание пропущено.')

# Распакуем архив в папку writers, если она еще не создана
if not os.path.exists('writers'):
    with zipfile.ZipFile(WRITERS_ARCHIVE, 'r') as zip_ref:
        zip_ref.extractall('writers')

# Настройка констант для загрузки данных
FILE_DIR  = 'writers'                     # Папка с текстовыми файлами
SIG_TRAIN = 'обучающая'                   # Признак обучающей выборки в имени файла
SIG_TEST  = 'тестовая'                    # Признак тестовой выборки в имени файла

# Подготовим пустые списки

CLASS_LIST = []  # Список классов
text_train = []  # Список для оучающей выборки
text_test = []   # Список для тестовой выборки

# Получим списка файлов в папке
file_list = os.listdir(FILE_DIR)

for file_name in file_list:
    # Выделяем имя класса и типа выборки из имени файла
    m = re.match('\((.+)\) (\S+)_', file_name)
    # Если выделение получилось, то файл обрабатываем
    if m:

        # Получим имя класса
        class_name = m[1]

        # Получим имя выборки
        subset_name = m[2].lower()

        # Проверим тип выборки
        is_train = SIG_TRAIN in subset_name
        is_test = SIG_TEST in subset_name

        # Если тип выборки обучающая либо тестовая - файл обрабатываем
        if is_train or is_test:

            # Добавляем новый класс, если его еще нет в списке
            if class_name not in CLASS_LIST:
                print(f'Добавление класса "{class_name}"')
                CLASS_LIST.append(class_name)

                # Инициализируем соответствующих классу строки текста
                text_train.append('')
                text_test.append('')

            # Найдем индекс класса для добавления содержимого файла в выборку
            cls = CLASS_LIST.index(class_name)
            print(f'Добавление файла "{file_name}" в класс "{CLASS_LIST[cls]}", {subset_name} выборка.')

            # Откроем файл на чтение
            with open(f'{FILE_DIR}/{file_name}', 'r') as f:

                # Загрузим содержимого файла в строку
                text = f.read()
            # Определим выборку, куда будет добавлено содержимое
            subset = text_train if is_train else text_test

            # Добавим текста к соответствующей выборке класса. Концы строк заменяются на пробел
            subset[cls] += ' ' + text.replace('\n', ' ')

# Определим количество классов
CLASS_COUNT = len(CLASS_LIST)

# Выведем прочитанные классы текстов
print(CLASS_LIST)

# Посчитаем количество текстов в обучающей выборке
print(len(text_train))

# Проверим загрузки: выведем начальные отрывки из каждого класса

for cls in range(CLASS_COUNT):                   # Запустим цикл по числу классов
    print(f'Класс: {CLASS_LIST[cls]}')           # Выведем имя класса
    print(f'  train: {text_train[cls][:200]}')   # Выведем фрагмент обучающей выборки
    print(f'  test : {text_test[cls][:200]}')    # Выведем фрагмент тестовой выборки
    print()

"""## Решение

Преобразование текстовых данных в числовые и векторные представления для обучения нейросети
"""

# ваше решение
# Базовые параметры преобразования, используемые для начального запуска
BASE_VOCAB_SIZE = 20000                   # Объем словаря для токенизатора по умолчанию
BASE_WIN_SIZE   = 1000                    # Длина отрезка текста (окна) в словах по умолчанию
BASE_WIN_HOP    = 100                     # Шаг окна разбиения текста на векторы по умолчанию

"""Токенизация и преобразование в последовательности"""

# Подготовка NLTK и pymorphy3
nltk.download("punkt")
nltk.download("stopwords")

morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words("russian"))

"""Напишем функцию для предобработки текста. Функция будет токенизировать текст, удалять знаки препинания, стоп слова, и выполнять лемматизацию:

"""

# Функция предобработки текста
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Токенизация и приведение к нижнему регистру
    tokens = [token for token in tokens if token.isalpha()]  # Оставляем только буквенные токены
    tokens = [morph.parse(token)[0].normal_form for token in tokens if token not in stop_words]  # Лемматизация и удаление стоп-слов
    return tokens

"""Подготовим обучабщую и тестовую выборки, и создадим частотный словарь:
all_tokens: Собирает все токены (слова) из всех текстов обучающей выборки в один список.
Список создаётся через вложенное перечисление: токены из каждого текста объединяются в общий список.
Counter: Считает частоту появления каждого токена в данных.
Результат — частотный словарь, где ключи — это слова, а значения — их частота.
"""

"""Подготовка предобработанных данных, построение словаря и преобразование текстов
в последовательности индексов будут вынесены в отдельные функции, чтобы их можно было
переиспользовать при различных наборах гиперпараметров."""

# Применение предобработки к обучающим и тестовым данным
tokens_train = [preprocess_text(text) for text in text_train]
tokens_test = [preprocess_text(text) for text in text_test]

# Построение частотного словаря по обучающей выборке
all_tokens = [token for tokens in tokens_train for token in tokens]
token_counts = Counter(all_tokens)


def build_word_index(token_counter, vocab_size):
    """Формирование словаря токенизатора заданного размера."""
    word_index = {"<OOV>": 1}
    word_index.update({word: idx + 2 for idx, (word, _) in enumerate(token_counter.most_common(vocab_size - 2))})
    return word_index


"""Словарь word_index: Отображает каждое слово из частотного словаря в уникальный индекс.
<OOV> (Out-Of-Vocabulary) — специальный токен для слов, которые отсутствуют в словаре, его индекс = 1.
Метод most_common(VOCAB_SIZE) возвращает список из VOCAB_SIZE самых частотных слов.
Слова из token_counts.most_common(VOCAB_SIZE - 2) получают индексы начиная с 2 (так как 1 индекс зарезервирован под редкие слова).
"""


def texts_to_sequences(tokens_list, word_index):
    """Преобразование списка токенов в последовательности индексов."""
    return [[word_index.get(token, 1) for token in tokens] for tokens in tokens_list]


# Базовый словарь для параметров по умолчанию
base_word_index = build_word_index(token_counts, BASE_VOCAB_SIZE)

# Пример предобработки
print("Фрагмент обучающего текста:")
print("Оригинальный текст:              ", text_train[1][:101])
print("После токенизации:               ", tokens_train[1][:20])
print("В виде последовательности индексов:", texts_to_sequences([tokens_train[1]], base_word_index)[0][:20])

"""Создание обучающей и проверочной выборки

Опишите две функции, которые будут использоваться одна в другой. Первая необходима для разбиения последовательности на отрезки:

sequence – последовательность индексов;
win_size – размер окна;

hop – шаг окна.

Вторая уже формирует сами выборки, используя первую функцию.



"""

# Функция разбиения последовательности на отрезки скользящим окном
# На входе - последовательность индексов, размер окна, шаг окна
def split_sequence(sequence, win_size, hop):
    # Последовательность разбивается на части до последнего полного окна
    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, hop)]


# Функция формирования выборок из последовательностей индексов
# формирует выборку отрезков и соответствующих им меток классов в виде one hot encoding
def vectorize_sequence(seq_list, win_size, hop):
    # В списке последовательности следуют в порядке их классов
    # Всего последовательностей в списке ровно столько, сколько классов
    ClassCount = len(seq_list)

    # Списки для исходных векторов и категориальных меток класса
    x, y = [], []

    # Для каждого класса:
    for cls in range(ClassCount):
        # Разбиение последовательности класса cls на отрезки
        vectors = split_sequence(seq_list[cls], win_size, hop)
        # Добавление отрезков в выборку
        x += vectors
        # Для всех отрезков класса cls добавление меток класса в виде OHE
        y += [utils.to_categorical(cls, ClassCount)] * len(vectors)

    # Возврат результатов как numpy-массивов
    return np.array(x, dtype=np.int32), np.array(y, dtype=np.float32)


def prepare_datasets(word_index, tokens_train, tokens_test, win_size, win_hop):
    """Создание обучающего и тестового наборов для заданных параметров окна."""
    train_seq = texts_to_sequences(tokens_train, word_index)
    test_seq = texts_to_sequences(tokens_test, word_index)
    x_train, y_train = vectorize_sequence(train_seq, win_size, win_hop)
    x_test, y_test = vectorize_sequence(test_seq, win_size, win_hop)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

"""Сервисные функции
Напишите три уже стандартные функции:

первая – создание, компиляция, обучение и вывод статистики по модели;

вторая – вывод результатов оценки модели;

третья – функция, объединяющая первую и вторую.
"""

#Функция компиляции и обучения модели нейронной сети
def compile_train_model(model,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        optimizer='adam',
                        epochs=50,
                        batch_size=128,
                        figsize=(20, 5)):

    # Компиляция модели
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Вывод сводки
    model.summary()

    # Вывод схемы модели
    display(plot_model(model, dpi=60, show_shapes=True))

    # Обучение модели с заданными параметрами
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val))

    # Вывод графиков точности и ошибки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('График процесса обучения модели')
    ax1.plot(history.history['accuracy'],
               label='Доля верных ответов на обучающем наборе')
    ax1.plot(history.history['val_accuracy'],
               label='Доля верных ответов на проверочном наборе')
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.set_xlabel('Эпоха обучения')
    ax1.set_ylabel('Доля верных ответов')
    ax1.legend()

    ax2.plot(history.history['loss'],
               label='Ошибка на обучающем наборе')
    ax2.plot(history.history['val_loss'],
               label='Ошибка на проверочном наборе')
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_xlabel('Эпоха обучения')
    ax2.set_ylabel('Ошибка')
    ax2.legend()
    plt.show()


# Функция вывода результатов оценки модели на заданных данных
def eval_model(model, x, y_true,
               class_labels=[],
               cm_round=3,
               title='',
               figsize=(10, 10)):
    # Вычисление предсказания сети
    y_pred = model.predict(x)
    # Построение матрицы ошибок
    cm = confusion_matrix(np.argmax(y_true, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')
    # Округление значений матрицы ошибок
    cm = np.around(cm, cm_round)

    # Отрисовка матрицы ошибок
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Нейросеть {title}: матрица ошибок нормализованная', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    plt.gca().images[-1].colorbar.remove()  # Стирание ненужной цветовой шкалы
    plt.xlabel('Предсказанные классы', fontsize=16)
    plt.ylabel('Верные классы', fontsize=16)
    fig.autofmt_xdate(rotation=45)          # Наклон меток горизонтальной оси при необходимости
    plt.show()

    print('-'*100)
    print(f'Нейросеть: {title}')

    # Для каждого класса:
    for cls in range(len(class_labels)):
        # Определяется индекс класса с максимальным значением предсказания (уверенности)
        cls_pred = np.argmax(cm[cls])
        # Формируется сообщение о верности или неверности предсказания
        msg = 'ВЕРНО :-)' if cls_pred == cls else 'НЕВЕРНО :-('
        # Выводится текстовая информация о предсказанном классе и значении уверенности
        print('Класс: {:<20} {:3.0f}% сеть отнесла к классу {:<20} - {}'.format(class_labels[cls],
                                                                               100. * cm[cls, cls_pred],
                                                                               class_labels[cls_pred],
                                                                               msg))

    # Средняя точность распознавания определяется как среднее диагональных элементов матрицы ошибок
    avg_accuracy = cm.diagonal().mean()
    print('\nСредняя точность распознавания: {:3.0f}%'.format(100. * avg_accuracy))
    return avg_accuracy


# Совместная функция обучения и оценки модели нейронной сети
def compile_train_eval_model(model,
                             x_train,
                             y_train,
                             x_test,
                             y_test,
                             class_labels=CLASS_LIST,
                             title='',
                             optimizer='adam',
                             epochs=50,
                             batch_size=128,
                             graph_size=(20, 5),
                             cm_size=(10, 10)):

    # Компиляция и обучение модели на заданных параметрах
    # В качестве проверочных используются тестовые данные
    compile_train_model(model,
                        x_train, y_train,
                        x_test, y_test,
                        optimizer=optimizer,
                        epochs=epochs,
                        batch_size=batch_size,
                        figsize=graph_size)

    # Вывод результатов оценки работы модели на тестовых данных
    avg_accuracy = eval_model(model, x_test, y_test,
                              class_labels=class_labels,
                              title=title,
                              figsize=cm_size)
    return avg_accuracy

"""
Embedding(50) + BLSTM(8)x2 + GRU(16)x2 + Dense(200)"""

def create_model(input_length, vocab_size):
    """Создание модели с параметрами словаря и длиной входной последовательности."""
    model = Sequential()
    model.add(Input(shape=(input_length,)))
    model.add(Embedding(vocab_size, 50))
    model.add(SpatialDropout1D(0.4))
    model.add(BatchNormalization())
    # Два двунаправленных рекуррентных слоя LSTM
    model.add(Bidirectional(LSTM(8, return_sequences=True)))
    model.add(Bidirectional(LSTM(8, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    # Два рекуррентных слоя GRU
    model.add(GRU(16, return_sequences=True, reset_after=True))
    model.add(GRU(16, reset_after=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    # Дополнительный полносвязный слой
    model.add(Dense(CLASS_COUNT, activation='softmax'))
    return model


def run_experiment(description, vocab_size, win_size, win_hop, cached_word_indices=None):
    """Запуск одного эксперимента обучения с заданными гиперпараметрами."""
    print('\n' + '=' * 120)
    print(f'Эксперимент: {description}')
    if cached_word_indices is not None and vocab_size in cached_word_indices:
        word_index = cached_word_indices[vocab_size]
    else:
        word_index = build_word_index(token_counts, vocab_size)
        if cached_word_indices is not None:
            cached_word_indices[vocab_size] = word_index

    x_train, y_train, x_test, y_test = prepare_datasets(word_index,
                                                        tokens_train,
                                                        tokens_test,
                                                        win_size,
                                                        win_hop)

    model = create_model(win_size, vocab_size)
    avg_accuracy = compile_train_eval_model(model,
                                            x_train, y_train,
                                            x_test, y_test,
                                            optimizer='rmsprop',
                                            epochs=40,
                                            batch_size=512,
                                            class_labels=CLASS_LIST,
                                            title=description)
    return avg_accuracy


# Словарь для кеширования построенных словарей токенизатора
cached_word_indices = {BASE_VOCAB_SIZE: base_word_index}

# Список результатов экспериментов
experiment_results = []

# Базовый эксперимент с исходными параметрами
x_train, y_train, x_test, y_test = prepare_datasets(base_word_index,
                                                    tokens_train,
                                                    tokens_test,
                                                    BASE_WIN_SIZE,
                                                    BASE_WIN_HOP)

base_model = create_model(BASE_WIN_SIZE, BASE_VOCAB_SIZE)
base_accuracy = compile_train_eval_model(base_model,
                                         x_train, y_train,
                                         x_test, y_test,
                                         optimizer='rmsprop',
                                         epochs=40,
                                         batch_size=512,
                                         class_labels=CLASS_LIST,
                                         title='Базовая модель')
experiment_results.append(('VOCAB_SIZE=20000, WIN_SIZE=1000, WIN_HOP=100', base_accuracy))

# Эксперименты с изменением размера словаря
for vocab_size in (5000, 10000, 40000):
    description = f'VOCAB_SIZE={vocab_size}, WIN_SIZE=1000, WIN_HOP=100'
    accuracy = run_experiment(description,
                              vocab_size,
                              BASE_WIN_SIZE,
                              BASE_WIN_HOP,
                              cached_word_indices)
    experiment_results.append((description, accuracy))

# Эксперименты с изменением размеров окна при фиксированном словаре
vocab_size = BASE_VOCAB_SIZE
for win_size, win_hop in ((500, 50), (2000, 200)):
    description = f'VOCAB_SIZE={vocab_size}, WIN_SIZE={win_size}, WIN_HOP={win_hop}'
    accuracy = run_experiment(description,
                              vocab_size,
                              win_size,
                              win_hop,
                              cached_word_indices)
    experiment_results.append((description, accuracy))

print('\nИтоговая средняя точность распознавания для каждой модели:')
for description, accuracy in experiment_results:
    print(f'{description}: {accuracy * 100:.2f}%')
