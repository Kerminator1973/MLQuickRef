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
