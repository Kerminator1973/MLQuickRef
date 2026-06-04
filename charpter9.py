import pandas as pd

cereals = pd.read_csv('cereals.csv')
#print(cereals.head())

groups = cereals.groupby("Manufacturer")

# Задания №1 и 2 - группировка по производителю, количество групп и завтраков
#print(groups.size())

# Задание №3 - список завтраков от Nabisco
#print(groups.get_group("Nabisco"))

# Задание №4 - средние значения по столбцам в каждой из групп
'''
aggregations = {
    "Calories": "mean",
    "Fiber": "mean",
    "Sugars": "mean"
}

print(groups.agg(aggregations))
'''

'''
aggregations = {
    "Sugars": "max",
    "Fiber": "min"
}

print(groups.agg(aggregations))
'''

def get_smallest_row(df):
    return df.nsmallest(1, "Sugars")

new_df = groups.apply(get_smallest_row)
print(new_df)
