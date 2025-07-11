import os
import zipfile
import shutil
import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown

DATA_URL = 'https://storage.yandexcloud.net/aiueducation/Content/base/l7/tesla.zip'
DATA_DIR = 'tesla'
ZIP_PATH = 'tesla.zip'


def ensure_dataset():
    """Make sure tesla reviews are available locally."""
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
    except Exception as e:
        print('Failed to download dataset:', e)
        if not os.path.isdir(DATA_DIR):
            raise


def load_reviews(path):
    """Read reviews from a text file and return unique non-empty lines."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return list(dict.fromkeys(lines))


def main():
    ensure_dataset()
    neg_path = os.path.join(DATA_DIR, 'Негативный отзыв.txt')
    pos_path = os.path.join(DATA_DIR, 'Позитивный отзыв.txt')

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

    from sklearn.model_selection import train_test_split

    train_texts, val_texts, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vocab_size = 10000
    max_len = 50
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len, padding='post')
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=max_len, padding='post')

    num_classes = 2
    y_train = utils.to_categorical(y_train, num_classes)
    y_val = utils.to_categorical(y_val, num_classes)

    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        SpatialDropout1D(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=2,
    )

    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print('Validation accuracy:', accuracy)


if __name__ == '__main__':
    main()
