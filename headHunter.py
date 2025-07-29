import os
import re
import gdown
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Utility functions

def to_categorical(index: int, num_classes: int) -> np.ndarray:
    vec = np.zeros(num_classes, dtype=np.float32)
    if 0 <= index < num_classes:
        vec[index] = 1.0
    return vec


def purify(x):
    if isinstance(x, str):
        return x.replace('\n', ' ').replace('\xa0', '').strip().lower()
    return x


def extract_year(x):
    try:
        return int(re.search(r"\d\d.\d\d.(\d{4})", x)[1])
    except (IndexError, TypeError, ValueError):
        return 0


# Classes and thresholds
currency_rate = {
    'usd': 65.0,
    'kzt': 0.17,
    'грн': 2.6,
    'белруб': 30.5,
    'eur': 70.0,
    'kgs': 0.9,
    'сум': 0.007,
    'azn': 37.5,
}

age_class = [0, [18, 23, 28, 33, 38, 43, 48, 53, 58, 63]]
experience_class = [0, [7, 13, 25, 37, 61, 97, 121, 157, 193, 241]]
city_class = [0, {
    'москва': 0,
    'санкт-петербург': 1,
    'новосибирск': 2,
    'екатеринбург': 2,
    'нижний новгород': 2,
    'казань': 2,
    'челябинск': 2,
    'омск': 2,
    'самара': 2,
    'ростов-на-дону': 2,
    'уфа': 2,
    'красноярск': 2,
    'пермь': 2,
    'воронеж': 2,
    'волгоград': 2,
    'прочие города': 3,
}]

employment_class = [0, {
    'стажировка': 0,
    'частичная занятость': 1,
    'проектная работа': 2,
    'полная занятость': 3,
}]

schedule_class = [0, {
    'гибкий график': 0,
    'полный день': 1,
    'сменный график': 2,
    'удаленная работа': 3,
}]

education_class = [0, {
    'высшее образование': 0,
    'higher education': 0,
    'среднее специальное': 1,
    'неоконченное высшее': 2,
    'среднее образование': 3,
}]

# Compute class sizes
for desc in [age_class, experience_class, city_class,
             employment_class, schedule_class, education_class]:
    if isinstance(desc[1], list):
        desc[0] = len(desc[1]) + 1
    else:
        desc[0] = max(desc[1].values()) + 1


# Encoding helpers

def int_to_ohe(arg: int, class_list) -> np.ndarray:
    num_classes = class_list[0]
    for i in range(num_classes - 1):
        if arg < class_list[1][i]:
            cls = i
            break
    else:
        cls = num_classes - 1
    return to_categorical(cls, num_classes)


def str_to_multi(arg: str, class_dict) -> np.ndarray:
    num_classes = class_dict[0]
    result = np.zeros(num_classes, dtype=np.float32)
    if not isinstance(arg, str):
        return result
    for value, cls in class_dict[1].items():
        if value in arg:
            result[cls] = 1.0
    return result


def extract_sex_age_years(arg: str):
    sex = 1.0 if isinstance(arg, str) and 'муж' in arg else 0.0
    try:
        years = 2019 - int(re.search(r"\d{4}", arg)[0])
    except (IndexError, TypeError, ValueError):
        years = 0
    return sex, years


def age_years_to_ohe(arg: int) -> np.ndarray:
    return int_to_ohe(arg, age_class)


def experience_months_to_ohe(arg: int) -> np.ndarray:
    return int_to_ohe(arg, experience_class)


def extract_salary(arg: str) -> float:
    try:
        value = float(re.search(r"\d+", arg)[0])
        for curr, rate in currency_rate.items():
            if curr in arg:
                value *= rate
                break
    except TypeError:
        value = 0.0
    return value / 1000.0


def extract_city_to_ohe(arg: str) -> np.ndarray:
    num_classes = city_class[0]
    split_array = re.split(r"[ ,.:()?!]", arg)
    for word in split_array:
        city_cls = city_class[1].get(word, -1)
        if city_cls >= 0:
            break
    else:
        city_cls = num_classes - 1
    return to_categorical(city_cls, num_classes)


def extract_employment_to_multi(arg: str) -> np.ndarray:
    return str_to_multi(arg, employment_class)


def extract_schedule_to_multi(arg: str) -> np.ndarray:
    return str_to_multi(arg, schedule_class)


def extract_education_to_multi(arg: str) -> np.ndarray:
    result = str_to_multi(arg, education_class)
    if result[2] > 0.0:
        result[0] = 0.0
    return result


def extract_experience_months(arg: str) -> int:
    try:
        years = int(re.search(r"(\d+)\s+(год.?|лет)", arg)[1])
    except (IndexError, TypeError, ValueError):
        years = 0
    try:
        months = int(re.search(r"(\d+)\s+месяц", arg)[1])
    except (IndexError, TypeError, ValueError):
        months = 0
    return years * 12 + months


def extract_row_data(row) -> tuple[np.ndarray, np.ndarray]:
    sex, age = extract_sex_age_years(row[COL_SEX_AGE])
    sex_vec = np.array([sex], dtype=np.float32)
    age_ohe = age_years_to_ohe(age)
    city_ohe = extract_city_to_ohe(row[COL_CITY])
    empl_multi = extract_employment_to_multi(row[COL_EMPL])
    sched_multi = extract_schedule_to_multi(row[COL_SCHED])
    edu_multi = extract_education_to_multi(row[COL_EDU])
    exp_months = extract_experience_months(row[COL_EXP])
    exp_ohe = experience_months_to_ohe(exp_months)
    salary = extract_salary(row[COL_SALARY])
    salary_vec = np.array([salary], dtype=np.float32)

    x_data = np.hstack([
        sex_vec,
        age_ohe,
        city_ohe,
        empl_multi,
        sched_multi,
        edu_multi,
        exp_ohe,
    ]).astype(np.float32)
    return x_data, salary_vec


def construct_train_data(row_list):
    x_data, y_data = [], []
    for row in row_list:
        x, y = extract_row_data(row)
        if y[0] > 0:
            x_data.append(x)
            y_data.append(y)
    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)


# Load dataset
DATA_URL = 'https://storage.yandexcloud.net/aiueducation/Content/base/l10/hh_fixed.csv'
if not os.path.isfile('hh_fixed.csv'):
    gdown.download(DATA_URL, 'hh_fixed.csv', quiet=True)

df = pd.read_csv('hh_fixed.csv', index_col=0)
df = df.applymap(purify)

COL_SEX_AGE = df.columns.get_loc('Пол, возраст')
COL_SALARY = df.columns.get_loc('ЗП')
COL_POS_SEEK = df.columns.get_loc('Ищет работу на должность:')
COL_POS_PREV = df.columns.get_loc('Последеняя/нынешняя должность')
COL_CITY = df.columns.get_loc('Город')
COL_EMPL = df.columns.get_loc('Занятость')
COL_SCHED = df.columns.get_loc('График')
COL_EXP = df.columns.get_loc('Опыт (двойное нажатие для полной версии)')
COL_EDU = df.columns.get_loc('Образование и ВУЗ')
COL_UPDATED = df.columns.get_loc('Обновление резюме')


# Build feature matrix and target vector
x_all, y_all = construct_train_data(df.values)

# Scale features
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

# Convert to tensors
X = torch.tensor(x_all, dtype=torch.float32)
y = torch.tensor(y_all, dtype=torch.float32)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


class SalaryModel(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train(model, loader, criterion, optimizer, epochs: int = 10):
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()


model = SalaryModel(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, criterion, optimizer, epochs=20)


def eval_net(model, x_val, y_val):
    model.eval()
    with torch.no_grad():
        pred = model(x_val)
        mae = mean_absolute_error(y_val.numpy(), pred.numpy())
        print('Средняя абсолютная ошибка:', mae, '\n')
        for i in range(min(10, len(y_val))):
            print(
                'Реальное значение: {:6.2f}  Предсказанное значение: {:6.2f}  Разница: {:6.2f}'.format(
                    y_val[i].item(),
                    pred[i].item(),
                    abs(y_val[i].item() - pred[i].item()),
                )
            )
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_val.numpy(), pred.numpy())
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.plot(plt.xlim(), plt.ylim(), 'r')
        plt.xlabel('Правильные значения')
        plt.ylabel('Предсказания')
        plt.grid()
        plt.title('Сравнение предсказанных и реальных значений')
        plt.show()


eval_net(model, X_test, y_test)
