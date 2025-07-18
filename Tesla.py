import os
import zipfile
import random
import shutil
import gdown
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Bidirectional,
    Dropout,
    Dense,
)
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import clean_text


DATA_URL = 'https://storage.yandexcloud.net/aiueducation/Content/base/l7/tesla.zip'
DATA_DIR = 'tesla'
ZIP_PATH = 'tesla.zip'

# Labels used in evaluation plots
class_names = ["Негативный", "Позитивный"]


def download_and_extract():
    """Ensure dataset is available locally."""
    if os.path.isdir(DATA_DIR) and \
       os.path.isfile(os.path.join(DATA_DIR, 'Негативный отзыв.txt')) and \
       os.path.isfile(os.path.join(DATA_DIR, 'Позитивный отзыв.txt')):
        return
    if os.path.isfile('Негативный отзыв.txt') and os.path.isfile('Позитивный отзыв.txt'):
        os.makedirs(DATA_DIR, exist_ok=True)
        shutil.copy('Негативный отзыв.txt', os.path.join(DATA_DIR, 'Негативный отзыв.txt'))
        shutil.copy('Позитивный отзыв.txt', os.path.join(DATA_DIR, 'Позитивный отзыв.txt'))
        return
    try:
        print('Downloading dataset...')
        gdown.download(DATA_URL, ZIP_PATH, quiet=False)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(DATA_DIR)
        print('Dataset extracted to', DATA_DIR)
    except Exception as e:
        print('Failed to download dataset:', e)
        if not os.path.isdir(DATA_DIR):
            raise


def read_reviews(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return list(dict.fromkeys(lines))


def prepare_datasets():
    download_and_extract()
    neg_path = os.path.join(DATA_DIR, 'Негативный отзыв.txt')
    pos_path = os.path.join(DATA_DIR, 'Позитивный отзыв.txt')

    neg_reviews = [clean_text(r) for r in read_reviews(neg_path)]
    pos_reviews = [clean_text(r) for r in read_reviews(pos_path)]

    min_len = min(len(neg_reviews), len(pos_reviews))
    random.seed(42)
    random.shuffle(neg_reviews)
    random.shuffle(pos_reviews)
    neg_reviews = neg_reviews[:min_len]
    pos_reviews = pos_reviews[:min_len]

    texts = neg_reviews + pos_reviews
    labels = [0] * len(neg_reviews) + [1] * len(pos_reviews)
    return train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)


def vectorize_text(
    train_texts,
    val_texts,
    vocab_size: int = 20000,
    max_len: int = 100,
):
    """Tokenize and pad text sequences."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_texts), maxlen=max_len, padding="post"
    )
    X_val = pad_sequences(
        tokenizer.texts_to_sequences(val_texts), maxlen=max_len, padding="post"
    )
    return X_train, X_val, tokenizer


def build_model(vocab_size: int = 10000, max_len: int = 100) -> Sequential:
    """Build a text classification model with an LSTM layer."""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# main
X_train_texts, X_val_texts, y_train, y_val = prepare_datasets()
X_train, X_val, _ = vectorize_text(X_train_texts, X_val_texts, max_len=100)
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)

model = build_model(max_len=100)
callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=2,
    callbacks=callbacks,
)

# Plot training and validation accuracy
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluation metrics
y_pred = model.predict(X_val).argmax(axis=1)
y_true = y_val.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.title('Confusion Matrix')
plt.show()

print("Classification report:\n", classification_report(y_true, y_pred))

print('Validation accuracy:', history.history['val_accuracy'][-1])
