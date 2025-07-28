import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    GlobalAveragePooling1D,
    Concatenate,
)


def parse_float(value: str) -> float:
    """Parse a string to float, ignoring non-digit characters."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    value = value.replace(',', '.')
    cleaned = re.sub(r"[^0-9.]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_int(value: str) -> int:
    """Parse a string to int, ignoring non-digit characters."""
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    cleaned = re.sub(r"\D", "", value)
    return int(cleaned) if cleaned else 0


def load_dataset(path: str) -> pd.DataFrame:
    """Load Moscow apartment dataset and parse numeric columns."""
    df = pd.read_csv(path)
    # Example expected columns. Adjust to your dataset structure.
    for col in ["rooms", "square", "floor"]:
        if col in df.columns:
            if col == "floor":
                df[col] = df[col].apply(parse_int)
            else:
                df[col] = df[col].apply(parse_float)
        else:
            df[col] = 0
    if "price" not in df.columns:
        raise ValueError("Dataset must contain 'price' column")
    df["price"] = df["price"].apply(parse_float)
    df["description"] = df.get("description", "")
    df["description"] = df["description"].fillna("").astype(str)
    return df


def build_text_data(texts: List[str], vocab_size: int = 20000, max_len: int = 100) -> Tuple[np.ndarray, Tokenizer]:
    """Convert texts to padded token sequences."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_len, padding="post")
    return padded, tokenizer


def vectorize_texts(texts: List[str], tokenizer: Tokenizer, max_len: int) -> np.ndarray:
    """Tokenize and pad texts using an existing tokenizer."""
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post")


def prepare_train_data(
    df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tokenizer,
    StandardScaler,
]:
    """Split dataset, vectorize text and normalize numeric features."""
    X_num = df[["rooms", "square", "floor"]].values.astype(np.float32)
    y = df["price"].values.astype(np.float32)
    X_train_num, X_test_num, y_train, y_test, desc_train, desc_test = train_test_split(
        X_num, y, df["description"].tolist(), test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    X_train_txt, tokenizer = build_text_data(desc_train)
    X_test_txt = vectorize_texts(desc_test, tokenizer, max_len=X_train_txt.shape[1])
    return X_train_num, X_train_txt, X_test_num, X_test_txt, y_train, y_test, tokenizer, scaler


def build_model(num_features: int, vocab_size: int, max_len: int = 100) -> Model:
    """Build regression model processing numeric and text inputs."""
    num_input = Input(shape=(num_features,), name="numeric")
    txt_input = Input(shape=(max_len,), name="text")
    x_txt = Embedding(vocab_size, 64, input_length=max_len)(txt_input)
    x_txt = GlobalAveragePooling1D()(x_txt)

    x = Concatenate()([num_input, x_txt])
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(1)(x)

    model: Model = Model([num_input, txt_input], out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    df = load_dataset("moscow.csv")
    X_train_num, X_train_txt, X_test_num, X_test_txt, y_train, y_test, tokenizer, scaler = prepare_train_data(df)
    model: Model = build_model(
        X_train_num.shape[1],
        vocab_size=len(tokenizer.word_index) + 1,
        max_len=X_train_txt.shape[1],
    )
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model.fit(
        [X_train_num, X_train_txt],
        y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        callbacks=[tb_callback],
        verbose=2,
    )
    preds = model.predict([X_test_num, X_test_txt])
    mse = tf.keras.metrics.mean_squared_error(y_test, preds.squeeze()).numpy().mean()
    mae = tf.keras.metrics.mean_absolute_error(y_test, preds.squeeze()).numpy().mean()
    print(f"Test MSE: {mse:.2f}")
    print(f"Test MAE: {mae:.2f}")


if __name__ == "__main__":
    main()
