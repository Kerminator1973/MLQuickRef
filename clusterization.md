# Кластеризация

**Кластеризация** — это неуправляемый метод машинного обучения, который группирует объекты в наборы (кластеры) так, чтобы объекты внутри одного кластера были более похожи друг на друга, чем на объекты из других кластеров.

Необходимо выбрать метрику расстояния (например, евклидово, манхэттенское) для измерения сходства между объектами.

Необходимо выбрать алгоритм, который последовательно распределяет объекты по кластерам, минимизируя внутрикластерное разбросание и/или максимизируя межкластерное различие.

Наиболее популярные типы алгоритмов кластеризации:

- Центроидные (prototype-based). Пример алгоритма: **k-means** минимизирует сумму квадратов расстояний до центров кластеров. Быстрый и простой, но требует заранее задать k, чувствителен к выбросам и масштабу признаков
- Иерархические. Пример алгоритма: Divisive clustering (дивизивная) - начинает со всех точек и рекурсивно делит.
Плюс: можно не фиксировать число кластеров заранее (выбирают «срез» дендрограммы). Минус: на больших выборках может быть тяжёлой по памяти/времени
- Плотностные. Пример алгоритма - DBSCAN: кластеры как области высокой плотности + выделение шума/выбросов. Хорош для «не-сферических» кластеров, но чувствителен к параметрам eps и min_samples, хуже при сильно разной плотности
- Модельные (вероятностные). Пример алгоритма: Байесовские смеси (Dirichlet Process Mixture) - могут автоматически подбирать число кластеров (в рамках модели), но сложнее и тяжелее
- Графовые / спектральные. Пример алгоритма: Spectral clustering - строит граф похожести и кластеризует в спектральном представлении (через собственные векторы лапласиана). Хорош для сложной структуры, но дорог по вычислениям на больших n.
- Прочие: Mean Shift, Affinity Propagation, BIRCH

## Пример приложения

В приведённом ниже примере из тестовых dataset-ов SKLearn извлекаются сообщения из группы новостей по двум темам:

```py
# Получаем данные из новостных групп
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
import re

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Этап получения данных и их предобработки
def preprocess(text: str) -> str:

    # Разделяем текст на отдельные слова, удаляем символы не являющиеся буквами,
    # удаляем stop-слова. Собираем отдельные слова в строки с пробелом в
    # качестве символа-разделителя
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

# Выбираем два подмножества статей из групп "криптография" и "мотоциклы"
newsgroups_train = fetch_20newsgroups(
    subset='train',
    categories=['rec.motorcycles', 'sci.crypt'],
    remove=('headers', 'footers', 'quotes')
)

data = newsgroups_train.data
preprocessed = [preprocess(t) for t in data]
```

Список доступных групп в новостях можно посмотреть так:

```py
# Загрузить только тестовый набор и получить список групп
test_data = fetch_20newsgroups(subset='test', download_if_missing=True)
group_names = test_data.target_names

# Распечатываем все доступные нам группы
print(group_names)
```

Кластеризация и вывод результата:

```py
# Этап кластеризации и вывода результатов
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Функция осуществляет кластеризацию данных. Мы могли бы задать не два, а большее
# количество кластеров, если бы работали с большим количеством новостных групп
def clustering(data, vectorizer, n_clusters=2):

    # Преобразуем строку слов в числовой вектор
    X = vectorizer.fit_transform(data)

    # Выполняем кластеризацию данных с помощью алгоритма KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Определяем качество кластеризации, используя подсчёт Silhouette Score
    score = silhouette_score(X, labels)

    return X, labels, score, kmeans

# Вспомогательный метод для печати наиболее часто встречаемых слов в кластере
def print_top_words(vectorizer, model, n_words=10):

    feature_names = np.array(vectorizer.get_feature_names_out())
    for i, center in enumerate(model.cluster_centers_):
        top_idx = center.argsort()[::-1][:n_words]
        words = feature_names[top_idx]

        print(f"Кластер {i}: {', '.join(words)}")

print("СРАВНЕНИЕ ЭФФЕКТИВНОСТИ КЛАСТЕРИЗАЦИИ")

# Программа будет выполнять четыре итерации кластеризации, используя
# предобработанные/не обработанные данные, а также CountVectorizer/TfidfVectorizer
batches = [
    ("CountVectorizer без предобработки", data, CountVectorizer()),
    ("CountVectorizer с предобработкой", preprocessed, CountVectorizer()),
    ("TfidfVectorizer без предобработки", data, TfidfVectorizer()),
    ("TfidfVectorizer с предобработкой", preprocessed, TfidfVectorizer())
]

for name, data, vectorizer in batches:
    print(name)

    X, labels, score, model = clustering(data, vectorizer)

    print(f"Silhouette Score: {score:.4f}")

    print("Ключевые слова:")
    print_top_words(vectorizer, model)
```
