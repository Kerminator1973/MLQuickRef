# Выполненные упражнения из книги "Pandas в действии" Бориса Пасхавера

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
