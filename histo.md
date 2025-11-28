# Генерация гистограммы и её визуализация

Мой вопрос Cloude Sonet 4.5: "_Я занимаюсь анализом изображений на языке Си. Для принятия решений я использую гистограммы. В инструментальных средствах, которые мне доступны есть упоминание о том, что классификация изображений строиться на основании длины проекции гистограммы на ось x. При формировании гистограммы используются параметры lower и upper, которые, возможно, используются как фильтр шумов. Авторы алгоритма построения гистограммы утверждают, что сумма значений lower и upper должна быть равна единице. Однако я знаю, что гистограмма не размещается по центру изображения. Как работает алгоритм построения гистограммы?_"

Судя по вашему описанию, речь идёт о **гистограмме проекций** (projection histogram), которая часто используется в анализе изображений для сегментации, распознавания текста и классификации объектов.

Горизонтальная проекция вычисляется как сумма интенсивностей пикселей в каждой строке изображения.

"Длина проекции на ось x" — это, вероятно, ширина значимой (ненулевой или превышающей порог) части гистограммы.

## Параметры lower и upper

Эти параметры определяют область интереса в гистограмме и работают как весовые коэффициенты для обрезки краёв:

- lower — отсекает нижнюю (левую) часть распределения
- upper — отсекает верхнюю (правую) часть распределения

Условие lower + upper = 1 означает, что они задают процентные доли от полного диапазона

### Механизм работы с несимметричным расположением

Если гистограмма не центрирована, алгоритм, вероятно, работает так:

- Строится гистограмма проекции
- Находится диапазон значимых значений (например, где гистограмма > порога)
- lower определяет, какую долю от начала этого диапазона отсечь
- upper определяет, какую долю от конца отсечь
- "Длина проекции" = оставшийся диапазон после обрезки

Важно: Проверьте документацию вашего инструмента — возможно, lower/upper определяют не отсечки, а **квантили распределения** (например, отбросить 10% самых низких и 10% самых высоких значений), что также объясняет условие суммы = 1.

Эффект параметров:

- lower = 0.0, upper = 1.0 — полная проекция (чувствительна к шуму)
- lower = 0.1, upper = 0.9 — игнорирует 10% краевых пикселей (устойчиво)
- lower = 0.2, upper = 0.8 — сильная фильтрация (может пропустить детали)

**Квантиль распределения** — это значение, ниже которого находится заданный процент наблюдений в наборе данных. Например, медиана — это 0,5‑квантиль, а 0,25‑квантиль называется первым квартилем (_значение, ниже которого 25% данных_). Квантили позволяют описать форму распределения, сравнивать выборки и строить боксплоты.

Квантили фиксируют позиционные характеристики распределения, поэтому их набор сразу показывает, как «распределены» данные вдоль оси значений.

- 0‑квантиль (минимум) - Наименьшее наблюдение - Границы распределения
- 0.25 (первый квартиль) - 25 % наблюдений ≤ значение - Нижняя «плотность»; если Q1 сильно от минимума, хвост слева короток
- 0.5 (медиана)	- 50 % наблюдений ≤ значение - Центр; смещение медианы от среднего указывает на асимметрию
- 0.75 (третий квартиль) - 75 % наблюдений ≤ значение - Верхняя «плотность»; большой разрыв Q3‑Q1 → широкий разброс
- 0.95, 0.99 … - Дальние правые хвосты - Наличие экстремальных значений, «толстый» хвост

>Наиболее часто используемые виды **квантилей**:
>
>Квартили — делят данные на 4 равные части (0.25, 0.5, 0.75)
>Децили — делят данные на 10 равных частей (0.1, 0.2, ..., 0.9)
>Процентили — делят данные на 100 равных частей (0.01, 0.02, ..., 0.99)
>
>В Pandas метод `quantile()` реализует вычисление квантилей как для одномерных (Series), так и для многомерных (DataFrame) данных.

Как квантили формируют представление о форме:

- Симметрия vs. асимметрия – если расстояния от медианы до Q1 и Q3 одинаковы, распределение симметрично; иначе — скошено в сторону более длинного хвоста
- Куртозис (толщина хвостов) – большие разрывы между 0.95‑квантилем и 0.5‑квантилем (или между 0.05 и 0.5) указывают на "толстый" хвост, то есть более высокую вероятность экстремальных значений
- Разброс – интерквартильный размах (Q3‑Q1) измеряет центральный разброс; сравнение его с полным диапазоном (max‑min) показывает, насколько данные сосредоточены в середине
- Множественные моды – если квантили меняются резко (например, большой скачок между соседними процентилями), это может свидетельствовать о наличии нескольких "скоплений" значений
- Таким образом, последовательность квантилей (или их графическое представление в виде боксплота) даёт компактный, но информативный профиль формы распределения без необходимости строить полную гистограмму

### Квантильная интерпретация (наиболее вероятная)

Если lower и upper работают как квантили, то:

```c
// Накопленная сумма гистограммы (CDF)
int total_pixels = 0;
for (int i = 0; i < size; i++) {
    total_pixels += histogram[i];
}

// Найти границы по квантилям
int cumsum = 0;
int left_boundary = 0;
int right_boundary = size - 1;

// Найти left_boundary (где cumsum достигает lower * total)
for (int i = 0; i < size; i++) {
    cumsum += histogram[i];
    if (cumsum >= total_pixels * lower) {
        left_boundary = i;
        break;
    }
}

// Найти right_boundary (где cumsum достигает (1 - upper) * total)
cumsum = 0;
for (int i = 0; i < size; i++) {
    cumsum += histogram[i];
    if (cumsum >= total_pixels * (1.0 - upper)) {
        right_boundary = i;
        break;
    }
}

projection_length = right_boundary - left_boundary;
```

Пример:

- Гистограмма: [0, 0, 5, 10, 20, 15, 8, 3, 0, 0] (индексы 0-9)
- lower = 0.1, upper = 0.9 (отбросить 10% слева, 10% справа)
- Всего пикселей: 61
- Левая граница: где накопилось ≥ 6.1 пикселей → индекс 2-3
- Правая граница: где накопилось ≥ 54.9 пикселей → индекс 7
- Длина проекции = 7 - 3 = 4 (в единицах индексов, умножить на масштаб)

Для вашего случая (несимметричная гистограмма):

- Если пик смещён влево, lower = 0.1 отрежет небольшую часть слева
- upper = 0.9 отрежет большую часть справа
- Результат: проекция будет измерять основную массу объекта, игнорируя шум


## Вариант 2: Пропорциональная обрезка диапазона

Если алгоритм работает с геометрическими границами:

```cpp
// Найти фактические границы сигнала
int first_nonzero = -1, last_nonzero = -1;

for (int i = 0; i < size; i++) {
    if (histogram[i] > threshold) {
        if (first_nonzero == -1) first_nonzero = i;
        last_nonzero = i;
    }
}

// Полный диапазон
int full_range = last_nonzero - first_nonzero;

// Обрезка пропорционально lower/upper
int cut_left = (int)(full_range * lower);
int cut_right = (int)(full_range * upper);

int effective_start = first_nonzero + cut_left;
int effective_end = last_nonzero - cut_right;

projection_length = effective_end - effective_start;
```

Эффект параметров:
lower = 0.0, upper = 1.0 — полная проекция (чувствительна к шуму)
lower = 0.1, upper = 0.9 — игнорирует 10% краевых пикселей (устойчиво)
lower = 0.2, upper = 0.8 — сильная фильтрация (может пропустить детали)

## Подбор параметров

Рекомендуемые значения по типу изображения

- Чёткие края: 0.05 / 0.95 - Малая обрезка
- Зашумлённые: 0.15 / 0.85 - Средняя фильтрация
- С тенями: 0.25 / 0.75 - Агрессивная

## Визуализация в Pandas и matplot

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Параметры
width, height = 50, 50
total_pixels = width * height

# Генерация данных
np.random.seed(42)
pixels_bright = np.random.randint(170, 186, int(total_pixels * 0.8))
pixels_very_bright = np.random.randint(230, 246, int(total_pixels * 0.2))
all_pixels = np.concatenate([pixels_bright, pixels_very_bright])
np.random.shuffle(all_pixels)

# Создание DataFrame
df = pd.DataFrame({'pixel_intensity': all_pixels})

# Вычисление границ (10% слева и справа)
lower_percentile = df['pixel_intensity'].quantile(0.10)
upper_percentile = df['pixel_intensity'].quantile(0.90)

print(f"Нижняя граница (10-й перцентиль): {lower_percentile:.2f}")
print(f"Верхняя граница (90-й перцентиль): {upper_percentile:.2f}")

# Фильтрация данных
df['filtered'] = (df['pixel_intensity'] >= lower_percentile) & (df['pixel_intensity'] <= upper_percentile)
df_kept = df[df['filtered']]
df_removed = df[~df['filtered']]

print(f"\nВсего пикселей: {len(df)}")
print(f"Сохранено: {len(df_kept)} ({len(df_kept)/len(df)*100:.1f}%)")
print(f"Отсечено: {len(df_removed)} ({len(df_removed)/len(df)*100:.1f}%)")

# Визуализация 1: Гистограмма с выделением отсечённых данных
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
bins = np.arange(165, 251, 2)

# Гистограмма сохранённых данных
plt.hist(df_kept['pixel_intensity'], bins=bins, alpha=0.7, color='steelblue', 
         edgecolor='black', label='Сохранённые данные')

# Гистограмма отсечённых данных
plt.hist(df_removed['pixel_intensity'], bins=bins, alpha=0.7, color='red', 
         edgecolor='black', label='Отсечённые данные (10% слева + 10% справа)')

# Границы отсечения
plt.axvline(lower_percentile, color='darkred', linestyle='--', linewidth=2, 
            label=f'Нижняя граница: {lower_percentile:.1f}')
plt.axvline(upper_percentile, color='darkred', linestyle='--', linewidth=2, 
            label=f'Верхняя граница: {upper_percentile:.1f}')

plt.xlabel('Интенсивность пикселей', fontsize=11)
plt.ylabel('Количество пикселей', fontsize=11)
plt.title('Гистограмма с отсечением 10% слева и справа', fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.xlim(165, 250)

# Визуализация 2: Box plot для наглядности
plt.subplot(1, 2, 2)
data_for_box = [df['pixel_intensity'], df_kept['pixel_intensity']]
bp = plt.boxplot(data_for_box, labels=['Все данные', 'После фильтрации'],
                 patch_artist=True, widths=0.6)

# Раскраска box plot
colors = ['lightcoral', 'lightblue']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('Интенсивность пикселей', fontsize=11)
plt.title('Box plot: до и после фильтрации', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# Визуализация 3: Изображения до и после фильтрации
plt.figure(figsize=(14, 6))

# Исходное изображение
plt.subplot(1, 3, 1)
image_original = all_pixels.reshape(height, width)
plt.imshow(image_original, cmap='gray', vmin=165, vmax=250)
plt.title('Исходное изображение\n(все пиксели)', fontsize=11, fontweight='bold')
plt.colorbar(label='Интенсивность')
plt.axis('off')

# Маска отсечённых пикселей
plt.subplot(1, 3, 2)
mask = df['filtered'].values.reshape(height, width)
image_mask = np.where(mask, image_original, np.nan)

plt.imshow(image_original, cmap='gray', vmin=165, vmax=250, alpha=0.3)
removed_pixels = np.where(~mask, image_original, np.nan)
plt.imshow(removed_pixels, cmap='Reds', vmin=165, vmax=250, alpha=0.8)

plt.title('Отсечённые пиксели\n(выделены красным)', fontsize=11, fontweight='bold')
plt.axis('off')

# Изображение после фильтрации
plt.subplot(1, 3, 3)
image_filtered = np.where(mask, image_original, np.median(df_kept['pixel_intensity']))
plt.imshow(image_filtered, cmap='gray', vmin=165, vmax=250)
plt.title('После фильтрации\n(отсечённые = медиана)', fontsize=11, fontweight='bold')
plt.colorbar(label='Интенсивность')
plt.axis('off')

plt.tight_layout()
plt.show()

# Статистика
print("\n=== Статистика ===")
print("\nДО фильтрации:")
print(df['pixel_intensity'].describe())
print("\nПОСЛЕ фильтрации:")
print(df_kept['pixel_intensity'].describe())
```
