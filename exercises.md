# Выполненные упражнения из книги "Pandas в действии" Бориса Пасхавера

Примеры CSV-файлов, необходимых для выполнения тестовых заданий можно скачать в [официальном репозитарии](https://github.com/paskhaver/pandas-in-action/tree/master/chapter_04_the_dataframe_object) книги.

Упражнения из главы 2:

```py
import pandas as pd

# Входные данные
superheroes = [
    "Batman",
    "Superman",
    "Spider-Man",
    "Iron Man",
    "Captain America",
    "Wonder Woman"
]
strength_levels = (100,120,90,95,110,120)

# Exercise 1: создание серии из списка
sh_series = pd.Series(superheroes)
print(sh_series)

# Exercise 2: создание серии из кортежа
sh_levels = pd.Series(strength_levels)
print(sh_levels)

# Exercise 3: создание серии с указанием как данных, так и индекса
heroes = pd.Series(strength_levels, index = superheroes)
print(heroes)

# Exercise 4: получить первые два значения серии
h1 = heroes.head(2)
print(h1)

# Exercise 5: получить четыре последних значения серии
h2 = heroes.tail(4)
print(h2)

# Exercise 6: подсчитать количество значений в серии
print(heroes.nunique())

# Exercise 7: подсчитать среднее значение
print(heroes.mean())

# Exercise 8: найти минимальное и максимальные значения
print(heroes.min())
print(heroes.max())

# Exercise 9: домножить все значения на 2
h3 = heroes * 2
print(h3)

# Exercise 10: преобразовать серию в ассоциативный массив (dictionary)
h4 = dict(heroes)
print(h4)
```

## Упражнение из главы 3

Задача: использовать файл "revolutionary_war.csv", в котором приведены даты исторических сражений во время Войны за независимость США. Требуется определить, сколько сражений было в каждый из дней недели.

Решение задачи приведено для Google Colab. Закоментированный код используется для проверки корректности выполненных операций. Решение:

```py
import pandas as pd

# Загрузка файла в папку проекта в Google Colab
#from google.colab import files
#uploaded = files.upload()

# Загружаем только колонку "Start Date" из CSV и явным образом указываем
# поле, в котором хранится дата сражения
df = pd.read_csv('revolutionary_war.csv', 
                 parse_dates = ["Start Date"],
                 usecols=['Start Date'])

#print(df.head())
#print(df['Start Date'].dtype)

# Преобразуем даты сражения в дни недели
day_names = df['Start Date'].apply(lambda x: x.day_name())

# Выполняем подсчёт количества элементов с каждым из дней недели
sorted_counts = day_names.value_counts()
print(sorted_counts)
```

В книге описывается применение функции `strftime("%A")`, которая возвращает день недели, но я её не использовал явным образом.

## Упражнения из главы 4

Для выполнения заданий требуется файл "nfl.csv", который содержит информацию об игроках национальной футбольной лиги (NFL).

```py
# Exercise 1: импортировать данные из CSV, корректно считать значение поля Birthday
import pandas as pd

data = pd.read_csv(
    "nfl.csv",
    parse_dates = ["Birthday"]
)
print(data)
```

Для решения задачи №3 необходимо выполнить группировку по каждой из команд, а затем пересчитать количество элементов в каждой группе:

```py
# Exercise 3: сколько человек в каждой команде
teams = players.groupby("Team")
print(teams["Name"].count().head())
```

Задача №4: какие игроки являются наиболее высоко-оплачиваемыми:

```py
print(players.sort_values(by=["Salary"], ascending=False).head())
```

Ключевым в задании является указание, что сортировка должна осуществляться **по убыванию** значений поля "Salary".

Задание №5 - отсортировать набор данных сначала по командам в алфавитном порядке, а затем по зарплатам в порядке убывания.

Решение:

```py
print(players.sort_values(by=["Team", "Salary"],ascending=[True, False]))
```

Задание №6: кто самый возрастной игрок в списке команды "Нью-Йорк Джетс" и когда он родился.

Решение:

```py
jets = players[players["Team"] == "New York Jets"]
sorted_by_birthday = jets.sort_values(by=["Birthday"],ascending=True)
print(sorted_by_birthday.iloc[0])
```

Ключевая строка - первая, в которой мы указываем массив логических значений (True/False) в качестве индексатора FrameData players. Т.е. сначала мы готовим Series, в каждом элементе которого вычисляем True, или False (`players["Team"] == "New York Jets"`), а потом уже формируем новые FrameData.

После сортировки данных, извлекаем первый элемент таблицы - это **Ryan Kalil**.

### Проверка

Первое решение - Ok.

Вторую задачу я не делал за очевидностью решения: `nfl.set_index("Name")`

Третье решение - OK, но автор книги сдела проще:

```py
nfl["Team"].value_counts().head()
```

Четвертое и пятое решение - OK.

Шестое решение автор книги сделал по другому:

```py
nfl = nfl.reset_index().set_index(keys = "Team")
nfl.loc["New York Jets"].sort_values("New York Jets").head(1)
```

И ответ действительно - Ryan Kalil.

Т.е. мои решения корректные, но нуждаются в оптимизации.

## Упражнения из главы 5

Первое задание - оптимизация хранения данных в ОЗУ:

```py
import pandas as pd

# Загружаем только колонку "Start Date" из CSV
df = pd.read_csv('netflix.csv')

# Просматриваем информацию о загруженных данным (объём кода).
# До оптимизации: memory usage: 182.5 KB
#df.info()

# Осуществляем оптимизацию данных
df["type"] = df["type"].astype("category")  # 142.6 KB
df["director"] = df["director"].astype("category")  # 132.1 KB
df['date_added'] = pd.to_datetime(df['date_added'], format='%d-%b-%y', errors='coerce')

df.info()

# Выводим первые строки
print(df.head())
```

Мой вариант реализует два действия, которых нет в ответе автора книги - он не оптимизирует "director" как категорию и он выполняет импорт даты и времени используя read_csv(), тогда как я явным образом преобразовываю дату с указанием конкретного формата.

Мои решений со второго по четвертое:

```py
# Второе упражнение
limitless = df[df["title"] == "Limitless"]
print(limitless)

# Третье задание
director = df["director"] == "Robert Rodriguez"
movie = df["type"] == "Movie"
print(df[director & movie].head())

# Четвёртое задание
director = df["director"] == "Robert Altman"
date_added = df["date_added"] == '2019-07-31'
print(df[director | date_added].head())
```

Решение пятого задания:

```py
directors = ["Orson Welles", "Aditya Kripalani", "Sam Raimi"]
star_teams = df["director"].isin(directors)
print(df[star_teams].head())
```

Решение шестого задания:

```py
start_date = df["date_added"] >= '2019-05-01'
end_date = df["date_added"] < '2019-06-01'
print(df[start_date & end_date].head())
```

Решение седьмого и восьмого заданий:

```py
cleaned = df.dropna(subset = ["director"])
print(cleaned)

# Восьмое задание
only_one = df.drop_duplicates(subset = ["date_added"], keep = False)
print(only_one)
```

## Упражнения из главы 6

На первый взгляд, решение задачи типовое - выполнить split() по запятой, а потом - новое поле "Street" по пробелу:

```py
import pandas as pd

customers = pd.read_csv('customers.csv')

# Задача: есть столбец Address, который нужно разбить на четыре новых столбца: Street, City, State и Zip
# Примеры адресов:
#   "6461 Quinn Groves, East Matthew, New Hampshire, 16656"
#   "1360 Tracey Ports Apt. 419, Kyleport, Vermont, 31924"
#
# Zip - последнее значение из списка
# State - предпоследнее
# City - перед штатом
# Street - это самый первый элемент, но без номера дома

customers[["Street", "City", "State", "Zip"]] = customers["Address"].str.split(pat = ",", expand = True)
customers["Street"] = customers["Street"].apply(lambda x: x.split(" ", 1)[1] if len(x.split(" ", 1)) > 1 else None)

print(customers.head())
```

В приведённом выше решении применяется лямбда-функция для обеспечения защиты от данных, не соответствующих предположению о том, что номер дома и улица разделены пробелом. Однако, если предположение верное, то сработает следующий код, который выглядит более понятно и очевидно:

```py
customers["Street"] = customers["Street"].str.split(pat = " ", n = 1, expand=True)[1]
```

Для того, чтобы вывести все колонки DataFrame с изменённой структурой, может потребоваться добавить специализированные настройки:

```py
#print(customers.columns)
#
# Показать все колонки
pd.set_option('display.max_columns', None)

# Дополнительно: показать больше строк (None — все строки)
#pd.set_option('display.max_rows', None)

# Можно также увеличить ширину колонок
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
```

Единственное, что я не сделал - не удалил исходный столбец "Address", что можно было бы сделать так:

```py
customers.drop(labels = "Address", axis = "columns")
```

Альтернативный вариант:

```py
del customers["Address"]
```

## Использование регулярных выражений в Pandas

Можно применять регулярные выражения для обработки значений полей DataFrame и Series:

```py
customer["Street"].str.replace("\d{4,}", "*", regex = True);
```

## Мульти-индексные объекты DataFrame
