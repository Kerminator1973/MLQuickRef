import pandas as pd

investments = pd.read_csv('investments.csv')

# Выводим количество уникальных значений по каждому столбцу.
# Это позволит нам понять, какие категориальные данных хорошо подходят
# на роль уровней индекса
print(investments.nunique())

# Принятие решения по создаваемым категориям: чем меньше значений в категории
# тем раньше находится её индекс. Например, в конкретной выборке:
#   Name              27763
#   Market              693
#   Status                3
#   State                61
#   Funding Rounds       16
# Добавляем индексы в следующем порядке:
investments = investments.set_index(
    keys = ['Status', 'Funding Rounds', 'State']
).sort_index()

# Exercise №1: Извлеките все строки с состоянием "Closed"
#print(investments.xs(key = "Closed", level = "Status").head())

# Вариант решения из книги. Результаты одинаковые
#print(investments.loc["Closed",].head())

# Exercise №2: Извлеките все строки с состоянием "Acquired" и десятью раундами инвестиций
#print(investments.xs(key = ("Acquired", 10), level = [0, 1]).head())

# Вариант решения из книги. Результаты одинаковые
#print(investments.loc[("Acquired", 10)].head())

# Exercise №3: Извлеките все строки с состоянием "Operating", шестью раундами инвестиций и штатом "NJ"
#print(investments.loc[("Operating", 6, "NJ")].head())

# Exercise №4: Извлеките все строки с состоянием "Closed" и восемью раундами инвестиций.
# Извлеките только столбец "Name"
#print(investments.loc[("Closed", 8, )]["Name"].head())

# Вариант решения из книги. Результаты одинаковые
#print(investments.loc[("Closed", 8),("Name",)].head())

# Exercise №5: Извлеките все строки со штатом "NJ". Вне зависимости от значений уровней Status и Funding Round
#print(investments.xs(key = "NJ", level = "State").head())

# Exercise №6: верните уровни мультииндекса обратно в число столбцов объекта DataFrame
print(investments.head())
investments = investments.reset_index()
print(investments.head())