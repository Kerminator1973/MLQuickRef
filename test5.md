# Лабораторное задание

Формулировка лабораторного задания:

- Произвести классификацию породы пингвинов (dataset `sns.load_dataset('penguins')`)
- Изучить влияние признаков на конечную точность классификации породы пингвина. Выполнить классификацию породы пингвина при изменении количества признаков от 2 до 6
- Составить [матрицу несоответствия](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

## Решение

Загрузка необходимых библиотек (зависимостей):

```shell
pip install pandas seaborn matplotlib scikit-learn
```

Код, сгенерированный GLM 5.2 по формулировке лабораторного задания:

```py
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ──────────────────────────────────────────────────────────────
# 1. Загрузка и подготовка данных
# ──────────────────────────────────────────────────────────────
penguins = sns.load_dataset('penguins')

# Удаляем строки с пропущенными значениями
penguins = penguins.dropna()

# Целевая переменная — порода пингвина
y = penguins['species']

# Категориальные признаки кодируем числами
X_raw = penguins.drop(columns=['species'])
X = pd.get_dummies(X_raw, columns=['island', 'sex'], drop_first=False)

# Итоговый список всех доступных признаков
all_features = list(X.columns)
print("Признаки, доступные в датасете:")
for i, f in enumerate(all_features, 1):
    print(f"  {i}. {f}")
print(f"\nВсего признаков: {len(all_features)}\n")

# Разделение на обучающую и тестовую выборки (стратификация по классам)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ──────────────────────────────────────────────────────────────
# 2. Классификация при изменении числа признаков от 2 до 6
# ──────────────────────────────────────────────────────────────
n_features_range = range(2, 7)          # 2, 3, 4, 5, 6
results = []                            # список словарей с результатами

for n in n_features_range:
    # Перебираем все сочетания из n признаков
    for combo in itertools.combinations(all_features, n):
        X_tr = X_train_full[list(combo)]
        X_te = X_test_full[list(combo)]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        results.append({
            'n_features': n,
            'features': combo,
            'accuracy': acc
        })

results_df = pd.DataFrame(results)

# Усреднённая точность по числу признаков
mean_acc = results_df.groupby('n_features')['accuracy'].agg(['mean', 'min', 'max'])
print("Зависимость точности от числа признаков (по всем комбинациям):")
print(mean_acc.round(4))
print()

# ──────────────────────────────────────────────────────────────
# 3. Лучшая комбинация для каждого количества признаков
#    и матрица несоответствия для неё
# ──────────────────────────────────────────────────────────────
class_names = sorted(y.unique())        # ['Adelie', 'Chinstrap', 'Gentoo']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

# Для удобства также фиксируем "эталонную" модель на всех признаках
# (чтобы попасть на 6-й подграфик, если нужно)
for idx, n in enumerate(n_features_range):
    subset = results_df[results_df['n_features'] == n]
    best_row = subset.loc[subset['accuracy'].idxmax()]

    best_features = list(best_row['features'])
    best_acc = best_row['accuracy']

    # Переобучаем модель на лучшей комбинации признаков
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_full[best_features], y_train)
    y_pred = clf.predict(X_test_full[best_features])

    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
    axes[idx].set_title(f'{n} признаков\nТочность: {best_acc:.4f}')

# Если осталось пустое место — убираем неиспользуемый подграфик
for j in range(len(n_features_range), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Матрицы несоответствия для лучших комбинаций признаков',
             fontsize=14, y=1.02)
plt.show()

# ──────────────────────────────────────────────────────────────
# 4. Полная таблица результатов (топ-комбинации по точности)
# ──────────────────────────────────────────────────────────────
top_results = results_df.sort_values('accuracy', ascending=False).head(10)
print("\nТоп-10 комбинаций признаков по точности:")
for _, row in top_results.iterrows():
    print(f"  n={row['n_features']}, acc={row['accuracy']:.4f}, "
          f"признаки={list(row['features'])}")

# ──────────────────────────────────────────────────────────────
# 5. График зависимости средней точности от числа признаков
# ──────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
mean_acc['mean'].plot(marker='o', ax=ax2, label='Средняя точность')
ax2.fill_between(mean_acc.index,
                 mean_acc['min'], mean_acc['max'],
                 alpha=0.2, label='Разброс min–max')
ax2.set_xlabel('Количество признаков')
ax2.set_ylabel('Точность')
ax2.set_title('Влияние числа признаков на точность классификации')
ax2.set_xticks(list(mean_acc.index))
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

Приложение выводит признаки, доступные в dataset:

1. bill_length_mm
2. bill_depth_mm
3. flipper_length_mm
4. body_mass_g
5. island_Biscoe
6. island_Dream
7. island_Torgersen
8. sex_Female
9. sex_Male

Признаки из dataset включают 4 количественных (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g) и категориальные island и sex, которые переводятся в числовой вид (через `pd.get_dummies`). Соответственно, признаки 5, 6 и 7 "схлопываются" в "остров", а 8 и 9 в "пол". Что даёт шесть отдельных признаков, по которым и будет осуществляться классификация:

1. bill_length_mm
2. bill_depth_mm
3. flipper_length_mm
4. body_mass_g
5. island
6. sex

Затем приложение отображает "матрицe несоответствия" и график "влияния числа признаков на точность классификации". Выводит в двух отдельных окнах Matplot.

Ожидаемые выводы по лабораторной:

- Два морфометрических признака (bill_length_mm + bill_depth_mm) уже дают высокую точность (~0.92–0.95), так как форма клюва хорошо разделяет породы
- Добавление flipper_length_mm и body_mass_g поднимает точность до ~0.97–0.99
- Признаки island и sex коррелируют с породой территориально и дают умеренное улучшение
- Оптимальный баланс точности и числа признаков — 3–4 признака
- В матрицах несоответствия основная путаница — между Adelie и Chinstrap при малом n; Gentoo отделяется чисто
