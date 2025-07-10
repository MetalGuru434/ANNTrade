# Загрузка из google облака
import gdown
gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l10/basketball.csv', None, quiet=True)

# Библиотека для работы с базами
import pandas as pd
df = pd.read_csv('basketball.csv', encoding= 'cp1251', sep=';', header=0, index_col=0) # Загружаем базу
df.head()

#Извлекаем текстовые данные из колонки `info` таблицы, помещаем в переменную `data_text`. 
#Выводим длину списка:

data_text = df['info'].values #

len(data_text) #

#Задаем максимальное кол-во слов в словаре, помещаем в переменную все символы, 
#которые хотим вычистить из текста.
#Токенизируем текстовые данные:

# Импортируем токенайзер
from tensorflow.keras.preprocessing.text import Tokenizer

maxWordsCount = 5000

sim_for_del='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

tokenizer = Tokenizer (num_words=maxWordsCount,
                       filters=sim_for_del,
                       lower=True,
                       split=' ',
                       oov_token='unknown',
                       char_level=False)

tokenizer.fit_on_texts(data_text)

# Переводим в Эмбеддинг пространство
Sequences = tokenizer.texts_to_sequences(data_text)

# Вариант  Bag of Words
xBOW_text = tokenizer.sequences_to_matrix(Sequences)

#Преобразуем данные в numpy, подготовим наборы для обучения
# Библиотека работы с массивами
import numpy as np

xTrain = np.array(df[['Ком. 1','Ком. 2', 'Минута', 'Секунда','ftime']].astype('int'))
yTrain = np.array(df['fcount'].astype('int'))

print(xTrain.shape)
print(yTrain.shape)
print(xBOW_text.shape)


# Функция по проверке ошибки

def check_MAE_predictl_DubbleInput (model,
                                    x_data,
                                    x_data_text,
                                    y_data_not_scaled,
                                    plot=False):

  mae = 0 # Инициализируем начальное значение ошибки
  y_pred = (model.predict([x_data,x_data_text])).squeeze()

  for n in range (0,len(x_data)):
    mae += abs(y_data_not_scaled[n] - y_pred[n]) # Увеличиваем значение ошибки для текущего элемента
  mae /= len(x_data) # Считаем среднее значение
  print('Среднаяя абслолютная ошибка {:.3f} очков это {:.3f}% от общей выборки в {} игры'.format(mae, (mae/y_data_not_scaled.mean(axis=0))*100,len(x_data)))

  if plot:
     plt.scatter(y_data_not_scaled, y_pred)
     plt.xlabel('Правильные значение')
     plt.ylabel('Предсказания')
     plt.axis('equal')
     plt.xlim(plt.xlim())
     plt.ylim(plt.ylim())
     plt.plot([0, 250], [0, 250])
     plt.show()

# ваше решение

# Split the numerical and textual data into training and test parts
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


x_num_train, x_num_test, x_text_train, x_text_test, y_train, y_test = train_test_split(
    xTrain, xBOW_text, yTrain, test_size=0.2, random_state=42
)

# Numerical branch of the model
num_input = Input(shape=(x_num_train.shape[1],), name="num_input")
num_branch = Dense(64, activation="relu")(num_input)
num_branch = Dense(32, activation="relu")(num_branch)

# Text branch of the model
text_input = Input(shape=(x_text_train.shape[1],), name="text_input")
text_branch = Dense(128, activation="relu")(text_input)
text_branch = Dense(64, activation="relu")(text_branch)

# Combine two branches
combined = Concatenate()([num_branch, text_branch])
combined = Dense(64, activation="relu")(combined)
output = Dense(1, activation="linear")(combined)

model = Model(inputs=[num_input, text_input], outputs=output)
model.compile(optimizer=Adam(0.001), loss="mse")

model.fit(
    [x_num_train, x_text_train],
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    verbose=1,
)

check_MAE_predictl_DubbleInput(model, x_num_test, x_text_test, y_test, plot=True)




