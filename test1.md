# Тестовое задание

Тестовое задание: взять CSV-файл с данными для настройки spam-фильтра и попробовать обучить с разными значениями параметра max_depth, классификатора **DecisionTreeClassifier**. Построить график точности (accuracy) работы классификатора на тестовых данных, используя matplot.

Значение **max_depth** - максимальная высота (или глубина) дерева решений.

Наше решение:

```py
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import kagglehub
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.tokenize import word_tokenize
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Функция осуществляет пред-обработку тестовых данных, потенциально,
# убирая знаки пунктуации и изменяя словоформы
def preprocess_text(line, punct=False, process='none'):

  tokens = word_tokenize(line)

  if punct == False:
    tokens = [t for t in tokens if re.match(r"\w+", t, flags=re.UNICODE)]

  if process == 'lemma':
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

  elif process == 'stem':
    ps = PorterStemmer()
    tokens = list(map(ps.stem, tokens))

  preprocessed_text = " ".join(map(str, tokens))

  return preprocessed_text

# Функция обучает классификатор типа DecisionTreeClassifier на предоставленных
# данных, принимая в качестве параметра значение max_depth
def f(X_train, X_test, y_train, y_test, i):

    first_tree = DecisionTreeClassifier(max_depth=i, min_samples_split=4, min_samples_leaf=1)
    first_tree.fit(X_train, y_train)

    y_true = np.array(y)
    y_pred = first_tree.predict(X_test)

    return metrics.accuracy_score(y_test, y_pred)

# Загрузка файла. Используется для загрузки csv-файла с локальной машины, если его
# ещё нет в виртуальной машине Google Colab
#from google.colab import files
#uploaded = files.upload()

df = pd.read_csv('spam.csv', encoding='latin1', sep=',')

# Осуществляем предобработку текстовых данных
nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

df['text_clean'] = df['text'].apply(lambda s: preprocess_text(s, False, "lemma"))

if df['label'].dtype == object:
    df['label_num'] = df['label'].map(lambda x: 1 if str(x).lower().startswith('spam') else 0)
else:
    df['label_num'] = df['label']

X = df['text_clean']
y = df['label_num']

# Выполняем векторизацию текстовых данных, используя TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=2)
X_vec = vectorizer.fit_transform(X)

# Разбиваем наши данные на два подмножества: обучения и проверки. Это необходимо для того,
# чтобы вычислить эффективность классификатора
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

# Генерируем значения от 1 до 20 включительно и получает для каждого значения
# max_depth точность работы классификатора с ним
i_vals = list(range(1, 21))
y_vals = [f(X_train, X_test, y_train, y_test, i) for i in i_vals]

# Выводим график зависимости точность распознавания от значения max_depth
plt.figure(figsize=(8, 5))
plt.plot(i_vals, y_vals, marker='o', linestyle='-')
plt.title('Зависимость точность классификатора от параметра max_depth')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.grid(True)
plt.show()
```

Ключевая особенность кода в преобразовании строковых данных в числовой вектор:

```py
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=2)
X_vec = vectorizer.fit_transform(X)
```

Первая строка создаёт объект vectorizer, который умеет преобразовывать входные данные в числовой вектор. Входными данными (токены) может быть отдельное слово (unigarm), или пара последовательных слов (bigram). Именно количество слов в токене задаётся параметром ngram_range. Параметр max_df указывает, что нужно игнорировать слишком часто встречаемые токены, которые появляются в 95% документов. Также нужно игнорировать слова, которые появляются в двух и меньшем количестве документов - за это отвечает параметр min_df.

Результатом выполнения fit_transform() является матрица, каждая из строк которой соответствует строке входного вектора, а каждый столбец - unigram/bigram. Значение каждой ячейки - вес unigram/bigram в строке документа (из файла "span.csv").

Для каждого значения max_depth создаётся отдельный классификатор, который использует доступную ему таблицу для построение дерева решений. Целью задания является вычисление точности классификатора для изменяемых значений:

```py
def f(X_train, X_test, y_train, y_test, i):

    first_tree = DecisionTreeClassifier(max_depth=i, min_samples_split=4, min_samples_leaf=1)
    first_tree.fit(X_train, y_train)

    y_true = np.array(y)
    y_pred = first_tree.predict(X_test)

    return metrics.accuracy_score(y_test, y_pred)
```

Таким образом можно говорить о том, что задачи машинного обучения укладываются в некоторую общую схему:

- Предобработка "сырых" данных (см. preprocess_text)
- Векторизация данных, т.е. преобразование их в набор векторов/матриц числовых значений
- Разделение данных случайным образом на dataset обучения и dataset тестирования
- Обучение классификатора, возможно, с автоматическим поиском оптимального набора параметров
- Тестирование качества обучения
- Подготовка классификатора для промышленного использования (создание package/библиотеки)
