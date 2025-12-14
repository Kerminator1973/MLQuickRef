# Тестовые задания Евгения (скопировал из его блокнота для проверки)

Евгений решает свои задания в [Google Colab](https://colab.research.google.com/), что удобно с точки зрения нулевых расходов на настройку окружения. Однако отладка кода в Colab отсутствует. На мой взгляд, имеет смысл поставить Conda и работать в GIGA IDE, или OpenIDE.

Подзадание №1: с веб-страницы https://misis.ru/applicants/admission/magistracy/faculties/lingmg/digling/ вытащите все ссылки, ведущие на страницы на сайте misis.ru. Вывести все эти ссылки на экран в виде полных веб-адресов (https:// ... .ru) в алфавитном порядке.

Решение:

```py
import re       # Регулярные выражения
import requests # Библиотека для выполнения http-запросов

sites = requests.get('https://misis.ru/applicants/admission/magistracy/faculties/lingmg/digling/').text

# Маска для поиска ссылок с помощью регулярных выражений. Маска ищет подстроку которая:
# 1. Начинается с имени аттрибута href HTML-тэга
# 2. Строка внутри href начинается с http://, или c https://
regex_mask = r'href=["\'](http[s]?://[^"\']+)["\']'

# Обрабатываем весь HTML-документ и находим все строки соответствующие маске
links = re.findall(regex_mask, sites)

# Выводим все найденные строки на экран
print(links)

sorted_links = sorted(links)
print(sorted_links)
```

Замечу, что в контексте заданий предлагается использовать функцию split(), а не findall(). Функция re.split() менее гибкая; findall() можно настроить на значительно более гибкие схемы поиска. С другой стороны, split() сохраняет в результате HTML-тэги.

Подзадание №2: Загрузите с веб-страницы https://www.gutenberg.org/files/4300/4300-h/4300-h.htm#chap01 текст 1-й главы романа "Улисс". Подсчитайте частоту всех использованных слов с помощью класса Counter из библиотеки Collections. Напишите функцию, которая выводит на экран и записывает в файл txt все случаи употребления заданного слова в заданной длине контекста. Функция должна принимать на вход 4 аргумента:

- текст документа;
- слово, случаи употребления которого нужно найти;
- длина левого контекста; 
- длина правого контекста.

Также задайте в данной функции параметр cut_length, значение которого по умолчанию False. Если задать значение True, то функция должна выводить на экран и записывать в файл фрагменты правого и левого контекста ТОЛЬКО в рамках одного предложения (допустим, если длина правого контекста равна пяти, а искомое слово идёт в данном предложении последним, выводить все последующие 5 слов из другого предложения не нужно).

Решение:

```py
import re       # Регулярные выражения
import requests # Библиотека для выполнения http-запросов
from collections import Counter

def get_left_context(words, index, left_context_len):
    if index >= left_context_len:
        return ' '.join(words[index - left_context_len:index])
    else:
        return ' '.join(words[:index])

def get_right_context(words, index, right_context_len):
    if index + right_context_len < len(words):
        return ' '.join(words[index + 1:index + right_context_len + 1])
    else:
        return ' '.join(words[index + 1:])

# Функция создаёт файл для вывода результата поиска контекста
def process_text(chapter, search_word, left_context_len, right_context_len, cut_length = False):

    # Сохраняем в файл результат работы функции
    with open('example.txt', 'w', encoding='utf-8') as file:

        if cut_length:

            # Разбиваем текст на предложения и обрабатываем их по отдельности
            sentences = re.split(r'[.!?]', chapter)
            for sentence in sentences:

                if search_word in sentence.lower():
                    specify_context(file, sentence, search_word, left_context_len, right_context_len)

        else:
            # Если не нужно резать контекст по границе предложения, передаём на обработку весь текст
            specify_context(file, chapter, search_word, left_context_len, right_context_len)

# Функция для получения контекста слова в
def specify_context(file, part, search_word, left_context_len, right_context_len):

    # Удаляем знаки пунктуации
    words = part.lower().replace('.', '').replace('!', '').replace('?', '').replace(',', '').split()

    # Фильтрация по первому символу "Левая угловая скобка", чтобы исключить HTML-теги
    filtered_words = [w for w in words if not w.startswith('<')]

    for index, word in enumerate(filtered_words):
        if word == search_word:

            left_context = get_left_context(filtered_words, index, left_context_len)
            right_context = get_right_context(filtered_words, index, right_context_len)

            file.write(f"{left_context} {search_word} {right_context}\n")

# Загружаем файл с сайта Gutenberg
chapter1 = requests.get('https://www.gutenberg.org/files/4300/4300-h/4300-h.htm#chap01').text

# С помощью регулярных выражений находим все слова, исключая знаки пунктуации
flt_words = chapter1.lower().replace('.', '').replace(',', '').split()

# Подсчитываем частоту использования слов
word_counts = Counter(flt_words)

# Выводим на экран количество использованных слов
print(word_counts)

# Выводим контекст для каждого использования слова
process_text(chapter1, "his", 2, 4, True)
```

Подзадание №3: cравните скорость и качество работы re.split и токенизаторов из библиотек **transformers** и **nltk**. Для этого разбейте текст 1-й главы Улисса (предварительно очищенной от тэгов!) на слова с помощью функции re.split и на токены с помощью любого токенизатора. С помощью библиотеки time зафиксируйте время, которое ушло в каждом случае на обработку текста. Подсчитайте количество различных токенов, получившихся в результате обработки текста различными методами. Чем отличаются полученные наборы токенов?

```py
import requests
from transformers import BertTokenizer
import nltk
import time
from html.parser import HTMLParser

# Вспомогательный класс и функция для удаления тэгов из строки с HTML
class TagStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self) -> str:
        return ''.join(self.text_parts)

def strip_tags(html: str) -> str:
    parser = TagStripper()
    parser.feed(html)
    return parser.get_text()

# Инициализация библиотеки ntlk посредством добавления правил пунктуации
nltk.download('punkt')

# Считываем первую главу "Улисс"
chapter1 = requests.get('https://www.gutenberg.org/files/4300/4300-h/4300-h.htm#chap01').text

clean_chapter1 = strip_tags(chapter1)

# Подготавливаем токенайзер к работе

start = time.perf_counter()   # Используем таймер высокой точности
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elapsed = time.perf_counter() - start

print(f"Execution time (BertTokenizer.from_pretrained): {elapsed:.6f} seconds")

# Осуществляем токенизация текста (получим список токенов и их соответствие индексам)
start = time.perf_counter()
tokens = tokenizer.tokenize(clean_chapter1)
elapsed = time.perf_counter() - start

print(f"Execution time (tokenize): {elapsed:.6f} seconds")

print(f"Total tags found (tokenizer): {len(tokens)}")

start = time.perf_counter()
flt_words = clean_chapter1.lower().replace('.', '').replace(',', '').split()
elapsed = time.perf_counter() - start

print(f"Execution time (re): {elapsed:.6f} seconds")

print(f"Total tags found (tokenizer): {len(flt_words)}")

# Для контроля корректности работы можно сравнить набор токенов
#print(tokens)
#print(flt_words)
```

Не смотря на то, что чистка текста от HTML-токенов была грубой, глубокая оптимизация не осуществлялась, очень хорошо видно, что "медленные" регулярные выражения кратно быстрее, чем профессиональный лингвистический токенайзер.

Попробовать улучшить код можно подключив библиотеку BeautifulSoup, которая позволяет более аккуратно обрабатывать HTML-документы.
