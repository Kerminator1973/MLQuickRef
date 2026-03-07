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
