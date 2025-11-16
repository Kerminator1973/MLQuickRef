# Студенческое задание по классификации

В качестве студенческого задания по классификации было предложено разработать spam-фильтр.

Данные для обучения были представлены в файле "spam.cvs", в котором колонка "label" содержала признак спама, или обычного сообщения (spam, ham), а колонка "text" - соответствующие сообщения.

Определить список возможных значений колонки "label" можно следующим кодом:

```py
import pandas as pd
df = pd.read_csv('spam.csv')
df['label'].unique()
```

Параметры выборки:

```py
df.shape
df['label'].value_counts()
```

В рамках задачи, требовалось выполнить предобработку текстовых данных, используя библиотеку NLTK, которая позволяет выделить из слова корень (stem), а также привести слова к их базовым формам (леммам). Например, слова "бегал", "бегаю", "бегать" преобразовываются в одну лемму. 

Ламматизация помогает уменьшить размер словаря (_vocabulary_), что особенно важно для машинного обучения и анализа данных.

Стемминг просто обрезает окончание слов.

Был разработан следующий код, который позволяет настраивать способ обработки текста:

```py
from nltk.tokenize import word_tokenize
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

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
```

Тестовые процедуры:

```py
print(preprocess_text("Totoro, totoro!", True))
print(preprocess_text("Totoro, totoro!", False))
print(preprocess_text("gone goes men man knocked cows cow worker sitting upstairs forgiving laughed scary preparations", False, "lemma"))
print(preprocess_text("gone goes men man knocked cows cow worker sitting upstairs forgiving laughed scary preparations", False, "stem"))
```

Также был разработан код, который использует dataset для обучения нейронной сети классификации текста, как спам, или обычное сообщение. Не смотря на то, что точность модели (по тестовым данным) была уровня 0.98, есть некоторые сомнения в корректности кода:

```py
import pandas as pd
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('spam.csv', encoding='latin1', sep=',')

df['text_clean'] = df['text'].apply(lambda s: preprocess_text(s, False, "lemma"))

if df['label'].dtype == object:
    df['label_num'] = df['label'].map(lambda x: 1 if str(x).lower().startswith('spam') else 0)
else:
    df['label_num'] = df['label']

X = df['text_clean']
y = df['label_num']

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vect = CountVectorizer(ngram_range=(1,1), stop_words=None)
clf = MultinomialNB()

model = make_pipeline(vect, clf)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nДругой вариант\n")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model2 = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), stop_words=None, min_df=2),
    LogisticRegression(max_iter=1000)
)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("Accuracy (LR):", accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
```

Тем не менее, в примере есть развитая преобработка данных. Например, из текста в Dataset удаляется пунктуация, а затем осуществляется его ламмаизация:

```py
df['text_clean'] = df['text'].apply(lambda s: preprocess_text(s, False, "lemma"))
```

Также признак spam/ham заменяется на целочисленное значение 1, или 0 соответственно:

```py
df['label_num'] = df['label'].map(lambda x: 1 if str(x).lower().startswith('spam') else 0)
```

Из dataset выбираются Series "обработанный текст" и "метки" и затем все данные разделяются на данные для обучения и данные для проверки точности классификации:

```py
X = df['text_clean']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

X_train и y_train - данные для обучения, а X_test и y_test - данные для получения метрик классификации.

Один из вариантов обучения выглядит следующим образом:

```py
vect = CountVectorizer(ngram_range=(1,1), stop_words=None)
clf = MultinomialNB()

model = make_pipeline(vect, clf)
model.fit(X_train, y_train)
```

В данном коде создается классификатор на основе метода наивного Байеса с использованием векторизации текста.

Класс **CountVectorizer** из библиотеки **sklearn.feature_extraction.text** используется для преобразования текстовых данных в числовые векторы. `ngram_range=(1,1)` указывает, что мы будем использовать одиночные слова (т. е. униграммы), не рассматривая более сложные комбинации слов (биграммы и триграммы). Строка `stop_words=None` означает, что никакие стоп-слова (например, "и", "в", "на") не будут удалены из текста. Поскольку в задании мы используем предобработку, то удалять стоп-слова нужно было бы в функции preprocess_text().

Вызов функции **MultinomialNB**() позволяет создать классификатор наивного Байеса (NB = naive Bayes), который хорошо подходит для задач классификации с несколькими классами и работает с дискретными признаками, что делает его идеальным для работы с текстовыми данными.

Функция **make_pipeline**() создает последовательность шагов (п pipeline), включая векторизацию текста и классификацию. Это позволяет объединить векторизатор и классификатор в один объект, который будет легко использовать.

Метод **model.fit**() обучает модель на обучающем наборе данных X_train, y_train.

После того как классификатор обучен на данных, можно выполнить классификацию тестовых данных и получить оценку точности полученной модели:

```py
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

Во втором варианте используется другой тип классификаторов - TF-IDF и логистической регрессии:

```py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model2 = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), stop_words=None, min_df=2),
    LogisticRegression(max_iter=1000)
)
```

**TfidfVectorizer** используется для преобразования текстов в векторы с учетом термина "обратной частоты" (TF-IDF).

TF-IDF (TF — term frequency, IDF — inverse document frequency)  — это способ векторизации текста, отражающий важность слова в документе (относительно некоторого набора документов).

## Автоматический подбор параметров классификатора

В SKLearn реализована библиотека GridSearchCV которой можно задать диапазон допустимых параметров классификатора и эта библиотека определит, какие из них подходят для классификации конкретного набора данных наилучшим образом.

>**Гиперпараметры** - это те характеристики модели, которые мы задаем изначально при её инициализации. Библиотеки, подобные GridSearch позволяют автоматизировать поиск оптимальных гиперпараметров модели.

Допустим, что мы используем классификатор из предыдущего примера:

```py
model2 = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), stop_words=None, min_df=2),
    LogisticRegression(max_iter=1000)
)
```

Мы явным образом указали некоторые параметры классификатора: ngram_range, stop_words и min_df. Какие-то параметры классификации мы не указали и они используются по умолчанию. Таким параметром может быть, например - max_df. Также мы не указывали параметры LogisticRegression: C и penalty.

Ниже приведён код, который подбирает оптимальные параметры:

```py
from sklearn.model_selection import GridSearchCV

param_grid = {
    # TfidfVectorizer: prefix is the step name auto-generated by make_pipeline: "tfidfvectorizer"
    "tfidfvectorizer__ngram_range": [(1,1), (1,2)],
    "tfidfvectorizer__min_df": [1, 2, 5],
    "tfidfvectorizer__max_df": [0.85, 0.95, 1.0],
    "tfidfvectorizer__stop_words": [None],
    "tfidfvectorizer__sublinear_tf": [False, True],
    # LogisticRegression:
    "logisticregression__C": [0.01, 0.1, 1, 10],
    "logisticregression__penalty": ["l2"],     # l1 possible with solver='liblinear' or 'saga'
    # If you want to test different solvers/penalties:
    # "logisticregression__solver": ["liblinear"],
}

# Choose scoring metric appropriate for your problem, e.g., 'accuracy', 'f1_macro'
gs = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    refit=True
)

gs.fit(X_train, y_train)

print("Best score (cv):", gs.best_score_)
print("Best params:")
for k, v in gs.best_params_.items():
    print(f"  {k}: {v}")

# Evaluate on held-out test set if available:
# print("Test score:", gs.score(X_test, y_test))

# Inspect top few candidates (optional)
results = gs.cv_results_
```
