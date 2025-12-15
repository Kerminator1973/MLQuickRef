# Тестовые задания Евгения (скопировал из его блокнота для проверки)

Подзадание №2: напишите итератор, который будет подавать в цикл переданное ему число элементов. Например, если мы итерируемся по списку [1, 2, 3, 4, 5], а на вход итератору передано число 2, он сначала вернёт (1, 2), затем (3, 4) и на этом завершит итерации.

Решение:

```py
class IntegerIterator:
    def __init__(self, int_list, int_count):
        if not isinstance(int_list, list) or not all(isinstance(i, int) for i in int_list):
            raise ValueError("Input must be a list of integers.")
        self.int_list = int_list
        self.index = 0

        # Сохраняем количество элементов, которые нужно возвращать на каждой итерации
        self.count = int_count

    def __iter__(self):
        return self

    def __next__(self):
        if self.index + self.count - 1 < len(self.int_list):
            result = self.int_list[self.index:self.index+self.count]
            self.index += self.count
            return tuple(result)
        else:
            raise StopIteration

# Пример использования
int_list = [1, 2, 3, 4, 5]
iterator = IntegerIterator(int_list, 2)

for number in iterator:
    print(number)
```

Подзадание №3: напишите класс, на вход которому при создании экземпляра подаётся список строк, в каждой из которых обязательно содержится как минимум 2 слова. Класс должен поддерживать следующие методы:

- get_tf(word, number of doc): принимает на вход слово и номер строки, возвращает значение TF
- get_idf(word, number of doc): принимает на вход те же самые параметры, возвращает значение IDF
- get_tf_idf(word, number of doc, ignore_stopwords=True): возвращает значение TF-IDF. По умолчанию при его вычислении не учитываются стопслова, если при вызове данного метода установить значение ignore_stopwords=False, учитывать стопслова при вычислении TF-IDF. Стопслова сделайте переменной класса.

```py
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfIdfCalculator:
    def __init__(self, list_words):
        self.list_words = list_words
        self.stop_words = []

    # Вспомогательный метод - для уменьшения объёма класса
    def _get_estimation(self, word, number_of_doc, _use_idf):

        # Создание и настройка TF-векторизатора
        # use_idf=False отключает IDF, оставляя только TF
        vectorizer = TfidfVectorizer(use_idf=_use_idf, norm=None)

        # Обучение векторизатора на данных
        tf_matrix = vectorizer.fit_transform(self.list_words)

        # Получение TF (частота слова в документе)
        feature_names = vectorizer.get_feature_names_out()

        if word in feature_names:
            word_index = list(feature_names).index(word)
            tf_values = tf_matrix[:, word_index].toarray().flatten()
            return tf_values[number_of_doc]
            
        return 0

    # Метод возвращает TF (Term Frequency) — Частота термина
    def get_tf(self, word, number_of_doc):
        return self._get_estimation(word, number_of_doc, False)

    # Метод возвращает IDF (Inverse Document Frequency) — Обратная частота документа
    def get_idf(self, word, number_of_doc):

        # binary=True означает, что слово либо есть (1), либо нет (0) в документе
        # Это даст нам IDF, умноженный на 1
        vectorizer = TfidfVectorizer(binary=True, norm=None, use_idf=True)
        tfidf_matrix = vectorizer.fit_transform(self.list_words)

        # Получить чистые IDF-значения
        feature_names = vectorizer.get_feature_names_out()

        if word in feature_names:
            idf_values = vectorizer.idf_
            return idf_values[number_of_doc]

        return 0

    # Метод возвращает TF-IDF = TF × IDF
    def get_tf_idf(self, word, number_of_doc, ignore_stopwords=True):
        return self._get_estimation(word, number_of_doc, True)

# Примеры использования
base_strings = ["apple banana plastic", "banana pineapple cherry banana", "cherry watermelon durian melon plastic", "date melon", "fig watermelon", "grape pear"]

obj = TfIdfCalculator(base_strings)
obj.stop_words = ["plastic", "coal"]

val = obj.get_tf("banana", 1)
print(f"TF слова 'banana': {val}")

# Интерпретация IDF:
# Высокое значение IDF = слово редкое (встречается в малом количестве документов)
# Низкое значение IDF = слово частое (встречается во многих документах)
val = obj.get_idf("cherry", 2)
print(f"IDF слова 'cherry': {val}")

# Слово важно для документа, если оно часто встречается в этом документе (высокий TF),
# но при этом редко встречается в других документах (высокий IDF)
val = obj.get_tf_idf("durian", 2, False)
print(f"IDF слова 'durian': {val}")
```
