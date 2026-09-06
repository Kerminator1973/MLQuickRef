from tensorflow.keras.datasets import imdb

# База данных IMDB из Keras (точнее, dataset обзоров фильмов IMDb) чаще всего используется для задач
# классификации текста. В частности, для бинарной классификации тональности (sentiment analysis).
#
# Определяют, положительный или отрицательный отзыв: метки train_labels и test_labels - это 0
# (отрицательный отзыв) или 1 (положительный).

# num_words - сохраняться будет только 10 000 наиболее часто встречающихся слов в обучающем наборе
MAX_WORDS_COUNT=10000

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=MAX_WORDS_COUNT)

# train_data[0] - это список индексов слов (токенов)
#print(train_data[0])
#print(train_labels[0])

# Осуществляем декодирование отзыва в последовательность слов на английском языке
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Пропускаем первые 3 слова, т.к. в IMDB они зарезервированы для:
# 0 - padding (заполнитель)
# 1 - начало последовательности (start of sequence)
# 2 - "редкое слово" (out-of-vocabulary/unknown)
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print(decoded_review)

# Список целых чисел нельзя передавать в нейронную сеть не обработанными. Подготовка данных
# может осуществлятсья двемя способами:
# - привести списки к одинаковой длине, преобразовать их в тензоры целых чисел. Первый слой - Embedding
# - выполнить прямое кодирование списков в векторы нулей и единиц. Например, преобразование последовательности [8, 5]
# в 10 000-мерный вектор, все элементы которого содержат нули, кроме элементов с индексами 8 и 5.
# Затем их можно передать в первый слой сети типа Dense, способный обрабатывать векторизованные данные
# с вещественными числами.

# Реализация второго подхода:
import numpy as np

def vectorize_sequence(sequences, dimension=MAX_WORDS_COUNT):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

#print(x_train)

# Также необходимо векторизовать метки
y_train = np.array(train_labels).astype('float32')
y_test = np.array(test_labels).astype('float32')

# В этом месте данные векторизованы и их можно передавать в нейронную сеть.
# Однако здесь нужно принять два архитектурных решения:
# - сколько слоев использовать (первый параметр в Dense)
# - сколько скрытых нейронов выбрать для каждого слоя
#
# Также мы можем выбрать функцию активации, например, "relu"
#
# TODO: в задании требуется:
# - вместо двух слоёв использовать 1, или 3
# - использовать большее количество нейронов в сети: 32, 64, и т.д.
# - использовать другую функцию активации вместо "relu", например, "tanh"

from tensorflow import keras
from tensorflow.keras import layers

# Определяем модель:
#
# Два скрытых слоя по 16 нейронов - это классическая "учебная" архитектура из книги Ф.Шолле про Keras.
# Её используют, чтобы показать базовый пайплайн.
# ReLU в скрытых слоях - стандартный выбор: помогает бороться с затуханием градиентов и ускоряет обучение.
#
# Это простой полносвязный (dense) классификатор. Он работает, если на вход подать уже векторизованные данные
#
# Для бинарной классификации нужен 1 нейрон на выходе: он будет выдавать вероятность принадлежности к классу 1.
# Уровень с sigmoid (сигматоидная функция) даёт значение от 0 до 1. Что позволяет оценивать результат, как
# вероятность.
#
# Большее количество скрытых нейронов позволяет модели обучаться на более сложных представлениях,
# но при этом увеличивается вычислительная стоимость модели. Побочка: можно переобучить модель
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Вариант с Embedding-слоем:
#model = keras.Sequential([
#    layers.Embedding(max_words, 16, input_length=MAX_WORDS_COUNT),
#    layers.GlobalAveragePooling1D(),          # усредняет эмбеддинги
#    layers.Dense(16, activation="relu"),
#    layers.Dense(1, activation="sigmoid")
#])

# Настраиваем модель (компилируем):
#
# Сейчас чаще вместо RMSprop берут adam как более универсальный вариант, но для этой задачи RMSprop вполне адекватен.
#
# TODO: в упражнениях нужно заменить 'binary_crossentropy' на функцию потерь 'mse'
model.compile(optimizer='rmsprop',
              loss="binary_crossentropy",
              metrics=['accuracy'])

# Поскольку Keras за нас обучить и проверит нейронную сеть, необходимо подготовить для него проверочный набор
x_val = x_train[:MAX_WORDS_COUNT]
partial_x_train = x_train[MAX_WORDS_COUNT:]
y_val = y_train[:MAX_WORDS_COUNT]
partial_y_train = y_train[MAX_WORDS_COUNT:]

# Теперь нужно провести обучение, указав количество эпох. Эпоха - это итерация обучения
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Результат обучения - метрики
history_dict = history.history
#print(history_dict.keys())

# Визуализируем результат используя MatPlotLib. Отображаем как график потерь,
# так и график точности
import matplotlib.pyplot as plt

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# Потери
axs[0].plot(epochs, loss_values, "bo", label="Обучение")
axs[0].plot(epochs, val_loss_values, "b", label="Проверка")
axs[0].set_title("Потери на этапах обучения и проверки")
axs[0].set_ylabel("Потери")
axs[0].legend()

# Точность
axs[1].plot(epochs, acc_values, "go", label="Обучение")
axs[1].plot(epochs, val_acc_values, "g", label="Проверка")
axs[1].set_title("Точность на этапах обучения и проверки")
axs[1].set_xlabel("Эпохи")
axs[1].set_ylabel("Точность")
axs[1].legend()

plt.tight_layout()
plt.show()

# Для решения практических задач используется метод predict()
#print(model.predict(x_test))
