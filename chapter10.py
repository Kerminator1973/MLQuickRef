import pandas as pd

week_1_sales = pd.read_csv('week_1_sales.csv')
week_2_sales = pd.read_csv('week_2_sales.csv')

#print(week_1_sales.info())
#print(week_2_sales.info())

# Задача №1: Объедините данные о продажах за две недели в один DataFrame. Назначьте данных из набора week1 ключ "Week 1", а данными из набора week2 - ключ "Week 2".
#sales = pd.concat(objs = [week_1_sales, week_2_sales], keys = ["Week 1", "Week 2"])
#print(sales.head())

# Задача №2: Найдите клиентов, которые посещали ресторан на каждой из рассматриваемых недель.
#print(week_1_sales.head())
#print(week_2_sales.head())

#every_week = week_1_sales.merge(right = week_2_sales, how = "inner", on = "Customer ID"
#                                ).drop_duplicates(subset=["Customer ID"])
#print(every_week.count())   # Всего найдено 46 уникальных покупателей

# Задача №3: Найдите клиентов, который посещали ресторан на каждой рассматриваемой неделе и каждую неделю
# заказывали одно и то же блюдо
#every_week = week_1_sales.merge(right = week_2_sales, how = "inner", on = ["Customer ID", "Food ID"])
#print(every_week)

# Задача №4: Найдите клиентов, посещавших ресторан только на первой неделе и только на второй неделе
#just_one_week = week_1_sales.merge(right = week_2_sales, how = "outer", on = "Customer ID", indicator = True)
#print(just_one_week)

# Задача №5: Каждая строка в наборе данных week1 идентифицирует клиента, заказавшего блюдо. Для каждой
# строки в week1 извлеките информацию о клиенте из набора данных customer

customers = pd.read_csv('customers.csv', index_col='ID')

with_customers_data = week_1_sales.merge(right=customers, how="left", left_on="Customer ID", right_index=True)
print(with_customers_data.head())
