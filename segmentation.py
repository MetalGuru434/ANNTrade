
## Подготовка

### Импорт библиотек
#

# Импортируем модели keras: Model
from tensorflow.keras.models import Model

# Импортируем стандартные слои keras
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation
from tensorflow.keras.layers import MaxPooling2D, Conv2D, BatchNormalization, UpSampling2D

# Импортируем колбэки для обучения
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Импортируем оптимизатор Adam
from tensorflow.keras.optimizers import Adam

# Импортируем модуль pyplot библиотеки matplotlib для построения графиков
import matplotlib.pyplot as plt

# Импортируем модуль image для работы с изображениями
from tensorflow.keras.preprocessing import image

# Импортируем библиотеку numpy
import numpy as np

# Импортируем методделения выборки
from sklearn.model_selection import train_test_split

# загрузка файлов по HTML ссылке
import gdown

# Для работы с файлами
import os

# Для распаковки архивов
import zipfile

# Для генерации случайных чисел
import random

import time

# импортируем модель Image для работы с изображениями
from PIL import Image

# очистка ОЗУ
import gc

"""### Загрузка датасета

грузим и распаковываем архив картинок
"""

# Глобальные параметры

IMG_WIDTH = 192               # Ширина картинки
IMG_HEIGHT = 256             # Высота картинки
NUM_CLASSES = 7               # Задаем количество классов на изображении
TRAIN_DIRECTORY = 'train'     # Название папки с файлами обучающей выборки
VAL_DIRECTORY = 'val'         # Название папки с файлами проверочной выборки

# Загрузка датасета из облака

DATASET_URL = 'https://storage.yandexcloud.net/aiueducation/Content/base/l14/construction_256x192.zip'
ZIP_PATH = 'construction_256x192.zip'

if not os.path.exists(ZIP_PATH):
    try:
        gdown.download(DATASET_URL, ZIP_PATH, quiet=False)
    except Exception as download_error:
        raise RuntimeError(
            "Не удалось скачать датасет. Загрузите архив вручную и поместите его рядом с скриптом."
        ) from download_error
else:
    print('Архив найден, пропускаем загрузку.')

if not (os.path.isdir(TRAIN_DIRECTORY) and os.path.isdir(VAL_DIRECTORY)):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('.')

"""### Предварительная подготовка данных

"""

"""Соберем список классов:"""

FLOOR = (100, 100, 100)         # Пол (серый)
CEILING = (0, 0, 100)           # Потолок (синий)
WALL = (0, 100, 0)              # Стена (зеленый)
COLUMN = (100, 0, 0)            # Колонна (красный)
APERTURE = (0, 100, 100)        # Проем (темно-бирюзовый)
DOOR = (100, 0, 100)            # Дверь (бордовый)
WINDOW = (100, 100, 0)          # Окно (золотой)
EXTERNAL = (200, 200, 200)      # Внешний мир (светло-серый)
RAILINGS = (0, 200, 0)          # Перила (светло-зеленый)
BATTERY = (200, 0, 0)           # Батареи (светло-красный)
PEOPLE = (0, 200, 200)          # Люди (бирюзовый)
LADDER = (0, 0, 200)            # Лестница (светло-синий)
INVENTORY = (200, 0, 200)       # Инвентарь (розовый)
LAMP = (200, 200, 0)            # Лампа (желтый)
WIRE = (0, 100, 200)            # Провод (голубой)
BEAM = (100, 0, 200)            # Балка (фиолетовый)

CLASSES = (
    ("FLOOR", (FLOOR,)),
    ("CEILING", (CEILING,)),
    ("WALL", (WALL,)),
    ("APERTURE_DOOR_WINDOW", (APERTURE, DOOR, WINDOW)),
    ("COLUMN_RAILINGS_LADDER", (COLUMN, RAILINGS, LADDER)),
    ("INVENTORY", (INVENTORY,)),
    (
        "LAMP_WIRE_BEAM_EXTERNAL_BATTERY_PEOPLE",
        (LAMP, WIRE, BEAM, EXTERNAL, BATTERY, PEOPLE),
    ),
)

COLOR_TO_CLASS_INDEX = {
    color: class_index
    for class_index, (_, colors) in enumerate(CLASSES)
    for color in colors
}

INDEX_TO_COLOR = {
    class_index: np.array(colors[0], dtype=np.uint8)
    for class_index, (_, colors) in enumerate(CLASSES)
}


def map_mask_to_class_indices(mask_image):
    """Преобразует RGB-маску сегментации в карту индексов новых классов."""

    mask_array = np.array(mask_image, dtype=np.uint8)
    class_map = np.zeros(mask_array.shape[:2], dtype=np.uint8)

    for color, class_index in COLOR_TO_CLASS_INDEX.items():
        matches = np.all(mask_array == color, axis=-1)
        class_map[matches] = class_index

    return class_map

"""Загрузим оригинальные изображения:"""

train_images = [] # Создаем пустой список для хранений оригинальных изображений обучающей выборки
val_images = [] # Создаем пустой список для хранений оригинальных изображений проверочной выборки

cur_time = time.time()  # Засекаем текущее время

# Проходим по всем файлам в каталоге по указанному пути
for filename in sorted(os.listdir(TRAIN_DIRECTORY+'/original')):
    # Читаем очередную картинку и добавляем ее в список изображений с указанным target_size
    train_images.append(image.load_img(os.path.join(TRAIN_DIRECTORY+'/original',filename),
                                       target_size=(IMG_WIDTH, IMG_HEIGHT)))

# Отображаем время загрузки картинок обучающей выборки
print ('Обучающая выборка загружена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='')

# Отображаем количество элементов в обучающей выборке
print ('Количество изображений: ', len(train_images))

cur_time = time.time() # Засекаем текущее время

# Проходим по всем файлам в каталоге по указанному пути
for filename in sorted(os.listdir(VAL_DIRECTORY+'/original')):
    # Читаем очередную картинку и добавляем ее в список изображений с указанным target_size
    val_images.append(image.load_img(os.path.join(VAL_DIRECTORY+'/original',filename),
                                     target_size=(IMG_WIDTH, IMG_HEIGHT)))

# Отображаем время загрузки картинок проверочной выборки
print ('Проверочная выборка загружена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='')

# Отображаем количество элементов в проверочной выборке
print ('Количество изображений: ', len(val_images))

"""Загрузим сегментированные изображения:"""

train_segments = [] # Создаем пустой список для хранений оригинальных изображений обучающей выборки
val_segments = [] # Создаем пустой список для хранений оригинальных изображений проверочной выборки

cur_time = time.time() # Засекаем текущее время

for filename in sorted(os.listdir(TRAIN_DIRECTORY+'/segment')): # Проходим по всем файлам в каталоге по указанному пути
    # Читаем очередную картинку и преобразуем ее в карту индексов классов
    mask_image = image.load_img(
        os.path.join(TRAIN_DIRECTORY+'/segment', filename),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode='rgb'
    )
    train_segments.append(map_mask_to_class_indices(mask_image))

# Отображаем время загрузки картинок обучающей выборки
print ('Обучающая выборка загружена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='')

# Отображаем количество элементов в обучающем наборе сегментированных изображений
print ('Количество изображений: ', len(train_segments))

cur_time = time.time() # Засекаем текущее время

for filename in sorted(os.listdir(VAL_DIRECTORY+'/segment')): # Проходим по всем файлам в каталоге по указанному пути
    # Читаем очередную картинку и преобразуем ее в карту индексов классов
    mask_image = image.load_img(
        os.path.join(VAL_DIRECTORY+'/segment', filename),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode='rgb'
    )
    val_segments.append(map_mask_to_class_indices(mask_image))

# Отображаем время загрузки картинок проверочной выборки
print ('Проверочная выборка загружена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='')

# Отображаем количество элементов в проверочном наборе сегментированных изображений
print ('Количество изображений: ', len(val_segments))

"""# Решение"""

# Ваше решение


def simple_unet(input_shape, num_classes, base_filters=32):
    """Строит симметричную архитектуру U-Net для задачи сегментации."""

    def conv_block(x, filters, block_name):
        x = Conv2D(filters, (3, 3), padding="same", activation="relu", name=f"{block_name}_conv1")(x)
        x = BatchNormalization(name=f"{block_name}_bn1")(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu", name=f"{block_name}_conv2")(x)
        x = BatchNormalization(name=f"{block_name}_bn2")(x)
        return x

    inputs = Input(shape=input_shape, name="input_layer")

    # Downsampling path
    c1 = conv_block(inputs, base_filters, "enc1")
    p1 = MaxPooling2D((2, 2), name="enc1_pool")(c1)

    c2 = conv_block(p1, base_filters * 2, "enc2")
    p2 = MaxPooling2D((2, 2), name="enc2_pool")(c2)

    c3 = conv_block(p2, base_filters * 4, "enc3")
    p3 = MaxPooling2D((2, 2), name="enc3_pool")(c3)

    c4 = conv_block(p3, base_filters * 8, "enc4")
    p4 = MaxPooling2D((2, 2), name="enc4_pool")(c4)

    # Bottleneck
    bottleneck = conv_block(p4, base_filters * 16, "bottleneck")

    # Upsampling path
    u1 = Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same", name="dec1_up")(bottleneck)
    u1 = concatenate([u1, c4], axis=-1, name="dec1_concat")
    c5 = conv_block(u1, base_filters * 8, "dec1")

    u2 = Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same", name="dec2_up")(c5)
    u2 = concatenate([u2, c3], axis=-1, name="dec2_concat")
    c6 = conv_block(u2, base_filters * 4, "dec2")

    u3 = Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same", name="dec3_up")(c6)
    u3 = concatenate([u3, c2], axis=-1, name="dec3_concat")
    c7 = conv_block(u3, base_filters * 2, "dec3")

    u4 = Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same", name="dec4_up")(c7)
    u4 = concatenate([u4, c1], axis=-1, name="dec4_concat")
    c8 = conv_block(u4, base_filters, "dec4")

    outputs = Conv2D(num_classes, (1, 1), activation="softmax", name="output_layer")(c8)

    return Model(inputs=inputs, outputs=outputs, name="simple_unet")


def normalize_images(images):
    return np.array([np.asarray(img, dtype=np.float32) / 255.0 for img in images], dtype=np.float32)


def prepare_masks(masks):
    mask_array = np.array(masks, dtype=np.uint8)
    return np.expand_dims(mask_array, axis=-1)


def decode_class_indices(class_map):
    color_map = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for index, color in INDEX_TO_COLOR.items():
        color_map[class_map == index] = color
    return color_map


# Подготовка данных
train_images_np = normalize_images(train_images)
val_images_np = normalize_images(val_images)

train_masks_np = prepare_masks(train_segments)
val_masks_np = prepare_masks(val_segments)

# Построение и обучение модели
input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
model = simple_unet(input_shape, NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

checkpoint_cb = ModelCheckpoint(
    "simple_unet_best.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)
early_stopping_cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

history = model.fit(
    train_images_np,
    train_masks_np,
    validation_data=(val_images_np, val_masks_np),
    epochs=25,
    batch_size=8,
    callbacks=[checkpoint_cb, early_stopping_cb],
    verbose=1,
)

# Визуализация истории обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Динамика функции потерь")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Динамика точности")
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# Демонстрация предсказаний
num_examples = min(3, val_images_np.shape[0])
sample_images = val_images_np[:num_examples]
sample_masks = val_masks_np[:num_examples]

predictions = model.predict(sample_images)
predicted_masks = np.argmax(predictions, axis=-1)

plt.figure(figsize=(12, num_examples * 4))
for idx in range(num_examples):
    original_image = sample_images[idx]
    true_mask = sample_masks[idx, ..., 0]
    pred_mask = predicted_masks[idx]

    plt.subplot(num_examples, 3, idx * 3 + 1)
    plt.imshow(original_image)
    plt.title("Оригинал")
    plt.axis("off")

    plt.subplot(num_examples, 3, idx * 3 + 2)
    plt.imshow(decode_class_indices(true_mask))
    plt.title("Истинная маска")
    plt.axis("off")

    plt.subplot(num_examples, 3, idx * 3 + 3)
    plt.imshow(decode_class_indices(pred_mask))
    plt.title("Предсказание")
    plt.axis("off")

plt.tight_layout()
plt.show()
