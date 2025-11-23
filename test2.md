# Тестовое задание №2

Предоставлен CSV-файл с описанием пользователей игры Dota. В колонке "roles" содержится список ролей пользователя. Требуется построить гистограмму с распределением пользовательских ролей по всем представленным пользователям.

Пример решения:

```py
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка файла. Используется для загрузки csv-файла с локальной машины, если его
# ещё нет в виртуальной машине Google Colab
#from google.colab import files
#uploaded = files.upload()

df = pd.read_csv('dota_hero_stats.csv')

def parse_roles(x):
    # Преобразовываем строку с описание Python-контейнера в контейнер
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
    except Exception:
        return []
    return []

# Создаём Series с количеством ролей каждого пользователя Dota
lengths = df['roles'].apply(lambda x: len(parse_roles(x)))

# Настраиваем стиль отбражения гистограммы
sns.set(style="whitegrid")

# Построение гистограммы частот длин
plt.figure(figsize=(8, 4))
sns.histplot(lengths, bins=range(0, max(lengths)+2), discrete=True)
plt.xlabel("Роли")
plt.ylabel("Количество")
plt.title("Количество ролей пользователей")
plt.xticks(range(0, max(lengths)+1))
plt.tight_layout()
plt.show()
```

Одним из ключевых решений является использование встроенного модуля Python **Abstract Syntax Tree** (ast). Функция literal_eval() позволяет создать переменную Python из строки, в которой она описана. В CSV-файле описание ролей это именно строка, в которой описывается список, например: `['Carry', 'Escape', 'Nuker']`

В качестве входных данных для построения гистограммы мы создаём Series, в каждом из элементов которого хранится количество элементов в списке поля "roles".

Библиотека **Seaborn**, построенная поверх Matplotlib, упрощает визуализацию данных для Python с помощью готовых тем, цветовых палитр и функций, автоматически учитывающих группировку и агрегирование данных. Она позволяет создавать статистически информативные графики: корреляционные матрицы, распределения, регрессионные линии и т.д.
