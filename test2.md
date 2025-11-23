# Тестовое задание №2

```py
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
import pandas as pd

# Загрузка файла. Используется для загрузки csv-файла с локальной машины, если его
# ещё нет в виртуальной машине Google Colab
#from google.colab import files
#uploaded = files.upload()

df = pd.read_csv('dota_hero_stats.csv')

# Пример:
# df = pd.DataFrame({'roles': ["['Carry', 'Escape', 'Nuker']", "['Support']", np.nan, "[]", ['Carry', 'Support']]})

def parse_roles(x):
    # Строка: пытаемся распарсить литерал Python
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
    except Exception:
        return []
    return []

#считаем длину
lengths = df['roles'].apply(lambda x: len(parse_roles(x)))

# Настройка стиля
sns.set(style="whitegrid")

# Построение гистограммы частот длин
plt.figure(figsize=(8, 4))
sns.histplot(lengths, bins=range(0, max(lengths)+2), discrete=True)
plt.xlabel("Количество ролей")
plt.ylabel("Значение")
plt.title("Гистограмма количества ролей пользователей")
plt.xticks(range(0, max(lengths)+1))
plt.tight_layout()
plt.show()
```
