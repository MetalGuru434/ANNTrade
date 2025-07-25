# ANNTrade

This repository demonstrates small projects using neural networks for working with text and financial data.

## Contents
- `Lukoil.py` – example of stock price prediction on Lukoil data.
- `Basket.py` – basketball score prediction with textual features.
- `Texts_processing.py` – sentiment classification for reviews.
- `preprocess.py` – helper utilities used in tests.
- `cifar10_convnet.py` – convolutional network for CIFAR-10 images.

## Usage
Install requirements (TensorFlow, pandas, scikit-learn, etc.) and run any script with Python 3:

```bash
python3 Lukoil.py
python3 cifar10_convnet.py
```

Datasets are downloaded automatically via `gdown`.

When working with imbalanced datasets, you can oversample the minority
class with `sklearn.utils.resample` or pass `class_weight` to
`model.fit` for better training stability.
