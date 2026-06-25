import pandas as pd

#cars = pd.read_csv('used_cars.csv')
#print(cars.head())

# Задание 1
#cars_by_fuel_type = cars.pivot_table(values="Price", index = "Fuel", aggfunc = "sum")
#print(cars_by_fuel_type.head())

# Задание 2
#cars_count = cars.pivot_table(values="Price", index = "Manufacturer", aggfunc = "count", columns="Transmission", margins=True)
#print(cars_count.head())

# Задание 3
#cars_mean = cars.pivot_table(values="Price", index = ["Year", "Fuel"], columns="Transmission", aggfunc = "mean")
#print(cars_mean.head())

# Задание 4
#cars_mean = cars.pivot_table(values="Price", index = ["Year", "Fuel"], columns="Transmission", aggfunc = "mean").stack()
#print(cars_mean.head())

cars = pd.read_csv('minimum_wage.csv')
print(cars.head())

years_columns=[
    "2010", "2011", "2012", "2013",
    "2014", "2015", "2016", "2017"
]
car_melted=cars.melt(id_vars = "State", var_name="Year", value_name="Wage")
print(car_melted.head())
