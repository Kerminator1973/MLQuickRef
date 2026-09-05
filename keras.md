# Создать стенд с Keras

При установке OpenIDE (рекомендуется) необходимо указать plug-ins для Python. При скачивании IDE следует обратить внимание на версию продукта: OpenIDE PRO - платная версия, требующая приобретения лицензии. Бесплатная версия - Community Edition.

Преимущества OpenIDE по сравнению с GigaIDE:

- более свежие сборки кода
- инсталлятор с нормальной цифровой подписью
- лучшая поддержка разных языков программирования
- можно выполнить Uninstall

>JetBrains поддерживает режим санкций, что крайне затрудняет использование PyCharm в России. Настоятельно не рекомендуется к использованию. Red flag!

Запустив IDE следует создать новый проект и выбрать Interpreter type "Project venv". При выборе версии Python следует принять во внимание, что TensorFlow может быть не совместим с актуальными версиями Python. В сентябре 2026, установка TensorFlow осуществлялась успешно в случае выбора Python 3.11.

Команда установки библиотеки:

```shell
pip install tensorflow
```

Необходимо помнить, что виртуальное окружение может быть очень большим по объёму; для проектов с ML - 5 ГБ и больше.

## Начальный проверочный код

```py
from tensorflow.keras.datasets import imdb

# num_words - сохраняться будет только 10 000 наиболее часто встречающихся слов в обучающем наборе
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# train_data[0] - это список индексов слов (токенов)
print(train_data[0])
print(train_labels[0])

# Осуществляем декодирование отзыва в последовательность слов на английском языке
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Пропускаем первые 3 слова, т.к. в IMDB они зарезервированы для:
# 0 - padding (заполнитель)
# 1 - начало последовательности (start of sequence)
# 2 - "редкое слово" (out-of-vocabulary/unknown)
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print(decoded_review)
```
