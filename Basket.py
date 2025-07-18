import os
import zipfile
import shutil
import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.model

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
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


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

