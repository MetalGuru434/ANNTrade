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

## Moscow apartment dataset
The `moscow_regression.py` example expects a CSV file with information
about Moscow real estate. You can obtain such data from open sources
like [data.mos.ru](https://data.mos.ru) or community datasets on Kaggle.
Save the file as `moscow.csv` or pass its path via the `--dataset`
argument (or `MOSCOW_DATA` environment variable).

Typical column names in Russian should be mapped to the names used by
the script:

| Russian column | Used in code |
| -------------- | ------------ |
| `Комнат`       | `rooms`      |
| `Площадь`      | `square`     |
| `Этаж`         | `floor`      |
| `Цена`         | `price`      |

Make sure to rename the columns accordingly before running the model.
