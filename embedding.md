# Что такое Embedding

Embedding (эмбеддинг) — это технология машинного обучения, преобразующая сложные объекты, такие как слова, изображения или пользователи, в плотные числовые векторы фиксированной длины в многомерном пространстве. Основная цель — зафиксировать семантические взаимосвязи, чтобы похожие объекты находились близко друг к другу в этом векторном пространстве. Это позволяет моделям ИИ обрабатывать и понимать данные, которые иначе были бы слишком сложными для анализа.

Предположим, что у нас есть некоторый dataset, загруженный из сети интернет:

```py
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space'])
```

Обучаем модель на тренировочных данных:

```py
import sentence_transformers

model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(newsgroups_train.data)
```

```py
l1 = LogisticRegressionCV()
l1.fit(embeddings, newsgroups_train.target)
```
