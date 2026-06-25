import pandas as pd

citi_bikes = pd.read_csv('citibike.csv')
#print(citi_bikes.head())

# Задача 1: преобразуйте строковые значения в столбцах start_time и stop_time в значения типа Timestamp.
citi_bikes["start_time"] = pd.to_datetime(citi_bikes["start_time"])
citi_bikes["stop_time"] = pd.to_datetime(citi_bikes["stop_time"])

# Решение из книги:
#for column in ["start_time", "stop_time"]:
#    citi_bikes[column] = pd.to_datetime(citi_bikes[column])

#print(citi_bikes.info())

# Задача 2: Подсчитайте количество поездок, совершаемых по дням недели (понедельник, вторник и т.д.).
# В какой будний день совершается больше всего велосипедных поездок? Используйте столбец start_time в
# качестве отправной точки

# Важно, что формат полей start_time и stop_time - уже дата/время, а не строка
#citi_bikes["day_of_week"] = citi_bikes["start_time"].dt.dayofweek
#groups = citi_bikes.groupby("day_of_week")
#print(groups.count())

# Задача 3: Подсчитайте количество поездок за каждую неделю в течение месяца. Для этого округлите все
# даты в столбце start_time до предыдущего или текущего понедельника. Предположим, что каждая неделя начинается
# в понедельник и заканчивается в воскресенье. Соответственно, первая неделя июня будет начинаться
# в понедельник 1 июня и заканчиваться в воскресенье 7 июня.

#weekly_counts = citi_bikes.groupby(citi_bikes['start_time'].dt.to_period('W')).size()
#print(weekly_counts.head())

#days_away_from_monday = citi_bikes["start_time"].dt.dayofweek
#dates_rounded_to_monday = citi_bikes["start_time"] - pd.to_timedelta(days_away_from_monday, unit="day")
#print(dates_rounded_to_monday.value_counts().head())

# Задача 4: рассчитайте продолжительность каждой поездки и сохраните результаты в новый столбец duration
citi_bikes["duration"] =citi_bikes["stop_time"] - citi_bikes["start_time"]
print(citi_bikes.head())

# Задача 5: Найдите среднюю продолжительность поездки
print(citi_bikes["duration"].mean())

# Задача 6: Извлеките из набора данных пять самых долгих поездок
print(citi_bikes["duration"].sort_values(ascending = False).head())