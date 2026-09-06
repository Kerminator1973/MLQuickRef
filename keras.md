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

Также нужно установить MatPlotLib для визуализации:

```shell
pip install matplotlib
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

## Выводы по изменениям параметров (проверка)

При переходе на 32 нейрона потери существенно выросли, точность несущественно уменьшилась. Обобщение ухудшилось.

При переходе на 64 нейрона потери существенно выросли c локальным всплеском на 14 итерации, точность уменьшилась. Обобщение ухудшилось.

При использовании одного слоя потери уменьшились, точность - более пологая.

При использовании трёх слоёв потери значительно выросли.

При использовании функции активации **tanh** потери увеличились, но оба графика более пологие в правой части. Средние активации tanh ближе к нулю, что иногда стабилизирует обучение - результаты более сглаженные. Функция tanh хорошо работала в старых архитектурах. ReLU обучается гораздо быстрее.

При использовании функции потерь "mse" (Mean Squared Error) потери критически снизились, точность более выравненная в правой части. Считается, что **MSE** хорош для регрессии, тогда как **Binary Crossentropy** лучше работает для — для бинарной классификации.

Общий вывод: для конкретного dataset-а увеличение количества нейронов приводит к переобучению. Количество своёв в два - кажется оптимальным. Функция активации tanh стабилизрует обучение, но не приводит к значимому повышению точности. MSE, как будто-бы, уменьшает потери.

## Вопросы производительности ML на специализированном железе

Замечания по запускаемому коду:

```
I0000 00:00:1788682999.432081   20612 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
```

Это предупреждение означает, что начиная с TensorFlow 2.11 GPU-ускорение на нативном Windows больше не поддерживается. TensorFlow 2.10 был последней версией с GPU на "голом" Windows. Дальше Google направил всех на WSL2 или плагин DirectML.

```shell
# Внутри WSL2 (Ubuntu)
sudo apt update && sudo apt upgrade -y

# Установить Python и pip
sudo apt install python3 python3-pip python3-venv -y

# Создать виртуальное окружение
python3 -m venv tf-env
source tf-env/bin/activate

# Установить TensorFlow с поддержкой GPU
pip install tensorflow[and-cuda]
```

Проверочный код, который показал, что TensorFlow действительно запустился без аппаратной акселлерации:

```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import tensorflow as tf
build_info = tf.sysconfig.get_build_info()
print(build_info)
```
