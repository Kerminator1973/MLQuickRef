# Conda - менеджер пакетов и среды (В РАБОТЕ)

Ключевая особенность Conda состоит в том, что Conda является и менеджером пакетов, и менеджером среды. Таким образом, он совмещает функции как **pip**, так и **Venv**. Также важно заметить, что pip останавливает пакеты только из репозитария PyPI, а Conda из репозитариев Anaconda Repository и Anaconda Cloud. Пакеты из PyPI можно установить используя pip в активной среде Conda.

Anaconda — это набор бинарных систем, включающий в себя Scipy, Numpy, Pandas и их зависимости.

- Scipy — это пакет статистического анализа.
- Numpy — это пакет числовых вычислений.
- Pandas — уровень абстракции данных для объединения и преобразования данных.

Anaconda Navigator — это графический интерфейс пользователя на рабочем столе (GUI), включенный в дистрибутив Anaconda, который позволяет запускать приложения и легко управлять пакетами, средами и каналами conda без использования команд командной строки.

Официальный сайт [проекта Anaconda](https://www.anaconda.com/).

Начать работу с Conda можно используя **Anaconda Prompt**.

## Основные команды управления средами

Создание среды: `conda create --name <имя_окружения> python=<версия>`. Пример значения параметра python: `python=3.9`

Активация среды: `conda activate <имя_окружения>`

Деактивация среды: `conda deactivate`

Список всех сред: `conda env list`

Удаление среды: `conda env remove --name <имя_окружения>`. Для удаление среды её сначала нужно деактивировать.

## Управление пакетами

Установка пакетов: `conda install <название_пакета>`. Пример: `conda install numpy pandas`

Установка из конкретного канала: `conda install -c <имя_канала> <название_пакета>`

Установка с помощью pip (в активной среде): `pip install <название_пакета>`

Список пакетов в среде: `conda list`

## Загрузка Anaconda

Для загрузки Anaconda необходимо авторизоваться по электронной почте. Можно использовать авторизации на GitHub, Google, или Microsoft.

После подтверждения электронной почты, сайт предлагает либо скачать полный пакет (Anaconda Distribution), который включает Jupyter, JupyterLab, 8000+ библиотек и Spyder IDE, или Miniconda: Python, Conda и минимальные зависимости. Miniconda занимает около 500 МБ дискового пространства.

Ключевая идея состоит в том, что окружение и проект не связаны друг с другом, т.е. можно создать некоторое окружение (базовые библиотеки и их зависимости) и использовать его для нескольких проектов. Переключение между окружениями позволяет адаптировать окружение под особенности конкретного проекта (кодовой базы).

После установки, в приложениях появляются ссылки на "Anaconda PowerShell Prompt" и "Anaconda Prompt". При использовании этих ссылок, `conda` будет доступна из командной строки.

Также можно добавить путь к скриптам Miniconda в переменную Path: `C:\Users\<YourUser>\miniconda3\Scripts`

Проверить установленную версию conda можно командой:

```shell
conda --version
```

## Практическая задача - постоить гистограмму по графическому файлу

На странице [histo.md](https://github.com/Kerminator1973/MLQuickRef/blob/main/histo.md) есть пример приложения, которое создаёт гистограмму по некоторому графическому изображению. Приложение разрабатывалось в среде Google Colab. Цель данного эксперимента - запустить её локально, используя GigaIDE в качестве среды разработки.

Создание нового окружения:

```shell
conda create --name histo
conda activate histo
conda env list
```

Поскольку в проекте используеются numpy, Pandas и matplotlib, они загружаются в окружение:

```shell
conda install numpy pandas
conda install matplotlib
conda install pillow
conda list
```

Теперь можно перейти в папку, в котором находится файл с исходными текстами "main.py" (а также с файлом данных "image.bmp") и можно запустить проект:

```shell
python main.py
```

Как результат, последовательно будут созданы два изображения, одно с областью файла "image.bmp" по которой считается гистограмма, а второй - результирующая гистограмма.

Результат в Google Colab выглядит кратно красивее (тоже, наверняка, можно сказать и об Jupyter), однако локальная копия позволяет работать с приложением без подключения к интернету, а также вести полноценную отладку кода.

## Создание приложения в GigaIDE с подключением окружения histo

GigaIDE поддерживаем работу с окружениями conda, но наилучший режим для этой IDE - создать окружение с нуля. По этой причине, я выполняю деактивацию и удаление ранее созданного через консоль окружения:

```shell
conda deactivate
conda env remove --name histo
conda env list
```

Далее следует в параметрах проекта указать, что в качестве окружения будет использоваться conda. После создания проекта следует добавить необходимые библиотеки через Terminal/Console среды разработки.

Приложение запускается из IDE, работает отладчик.

Однако если выполнить из "Anaconda Prompt" команду отображения доступных окружений, то новое созданное окружение будет отбражаться в списке, но оно не будет выбрано (активным останется base). Попытка запуска приложения "main.py" приведёт к сообщению об ошибке - отсутствия зависимостей (библиотек), что говорит нам о том, что GigaIDE активирует нужное нам окружение при запуске приложения из своей среды, а не изменяет его глобально.

### Чем хороша GigaIDE для разработки ML-кода на Python

В IDE автоматически работает Linter, который подсказал, что мне не нужно отдельно выполнять импорт Pandas, т.е. следующая строка лишняя:

```py
import pandas as pd
```

И действительно, именно в этом проекте Pandas не был использован. Соответственно, инструкция по установке могла бы быть чуть короче. Т.е. Linter - это значимый бонус.

Для применения фильтрации с использованием перцентелей, был использован GigaChat, который предложил следующий вариант решения:

```py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

# Загрузка BMP в оттенках серого
img = Image.open('image.bmp').convert('L')
arr = np.array(img)  # 2D массив пикселей (0-255)

# Координаты региона
col, row = 112 + 3, 44 + 6
width, height = 5, 7
region = arr[row:row+height, col:col+width]
values = region.ravel()

# Вычисление перцентилей
low_percentile = np.percentile(values, 10)
high_percentile = np.percentile(values, 90)

# Маски для фильтрации
low_mask = values < low_percentile
high_mask = values > high_percentile
filtered_mask = ~ (low_mask | high_mask)  # Внутри диапазона
removed_values = values[low_mask | high_mask]

# ---- Первое окно: гистограмма ----
plt.figure(figsize=(8, 5))
plt.hist(values[filtered_mask], bins=range(0, 257, 5),
         edgecolor='black', alpha=0.7, label='Kept (10–90%)')
plt.hist(removed_values, bins=range(0, 257, 5),
         color='orange', edgecolor='black', alpha=0.8,
         label='Removed (<10% or >90%)')
plt.title('Histogram of 5×7 region at (115,50) with percentile-based filtering')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show(block=False)  # Не блокируем выполнение

# ---- Второе окно: изображение с рамкой ----
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(arr, cmap='gray', origin='upper')
ax.set_title('Grayscale BMP with selected 5×7 area')
rect = patches.Rectangle(
    (col, row),
    width, height,
    linewidth=1,
    edgecolor='red',
    facecolor='none'
)
ax.add_patch(rect)
ax.axis('off')

plt.show()
```

Кроме добавления и визуализации фильтрации, так же с помощью GigaChat была изменена логика отображения окон с гистограммой и изображением банкноты - теперь они могут отображаться на экране одновременно и их можно перемещать, см.: `plt.show(block=False)`

Отображение двух отдельных окон не очень удобно. Кажется, что более рациональным было бы размещать оба элемента (гистограмма и изображение) в одном окне и перемещаться между ними с помощью кнопок "Назад" / "Вперёд". 

К сожалению, стандартные кнопки "Back" и "Forward" в панели инструментов Matplotlib (matplotlib.widgets.BackToolbar) не предназначены для переключения между разными графиками, а используются для навигации по состояниям масштабирования и панорамирования (например, "назад к предыдущему зуму").

GigaChat, Cloude 4.5 Sonet и Gemini 3 Flash предложили вариант с добавлением кнопок в пользовательский интерфейс главного окна, или с обработкой клавиш клавиатуры `Left` и `Right`. Вариант от GigaIDE показался наиболее красивым:

```py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from matplotlib.widgets import Button

# --- Загрузка изображения ---
img = Image.open('image.bmp').convert('L')
arr = np.array(img)

# --- Параметры региона ---
col, row = 112 + 3, 44 + 6
width, height = 5, 7
region = arr[row:row+height, col:col+width]
values = region.ravel()

# --- Фильтрация по перцентилям ---
low_percentile = np.percentile(values, 10)
high_percentile = np.percentile(values, 90)
removed_values = values[(values < low_percentile) | (values > high_percentile)]
kept_values = values[(values >= low_percentile) & (values <= high_percentile)]

# --- Подготовка интерфейса ---
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2)  # Место под кнопки

# --- Кнопки ---
ax_back = plt.axes([0.3, 0.05, 0.1, 0.075])
ax_forward = plt.axes([0.6, 0.05, 0.1, 0.075])
btn_back = Button(ax_back, 'Back')
btn_forward = Button(ax_forward, 'Forward')

# --- Страницы ---
pages = ['histogram', 'image']
current_page = 0

def show_page(page):
    ax.clear()
    if page == 'histogram':
        ax.hist(kept_values, bins=range(0, 257, 5), edgecolor='black', alpha=0.7, label='Kept (10–90%)')
        ax.hist(removed_values, bins=range(0, 257, 5), color='orange', edgecolor='black', alpha=0.8, label='Removed')
        ax.set_title('Histogram: Pixel Intensity Distribution')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(axis='y', alpha=0.75)
    elif page == 'image':
        ax.imshow(arr, cmap='gray', origin='upper')
        ax.set_title('Grayscale Image with 5×7 Region')
        rect = patches.Rectangle((col, row), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.axis('off')
    fig.canvas.draw_idle()

# --- Обработчики кнопок ---
def on_forward(event):
    global current_page
    current_page = (current_page + 1) % len(pages)
    show_page(pages[current_page])

def on_back(event):
    global current_page
    current_page = (current_page - 1) % len(pages)
    show_page(pages[current_page])

btn_forward.on_clicked(on_forward)
btn_back.on_clicked(on_back)

# --- Показать первую страницу ---
show_page(pages[current_page])

# --- Отобразить окно ---
plt.show()
```

Однако во всех вариантах нарушалось маштабирование гистограммы по оси Y и форма становилась абсолютно не читаемой. Все три ИИ не предложили работоспособного варианта решения данной проблемы. Более того, предложенные (несколько) варианты были либо приводили к падению приложения, либо оказались на работоспособными (использованием `set_ylim()`).

При анализе кода узнал, что matplotlib.pyplot для открытия окон и отрисовки графиков и изображений использует GUI-бэкенды, такие как:

- TkAgg (на основе Tkinter)
- Qt5Agg (на основе PyQt5)

Вот на чём базируется моё утверждение:

```py
try:
    import tkinter as tk
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
except:
    # Если Tk недоступен, используем примерные значения
    screen_width, screen_height = 1920, 1080
```

```py
# Перемещение окна
manager1 = plt.get_current_fig_manager()
try:
    # Для Tkinter
    manager1.window.wm_geometry(f"+{pos_left[0]}+{pos_left[1]}")
except:
    pass
```

```
File "C:\Users\kermi\miniconda3\envs\HistoProject\Lib\site-packages\matplotlib\backends\backend_qt.py", line 657, in start_main_loop
  with _allow_interrupt_qt(qapp):
```

В результате многочисленных экспериментов с ИИ откатился до варианта с двумя отдельными окнами. Нужно более серьёзно погружаться в техническую документацию, чтобы решить те проблемы, которые не смог адекватно решить ИИ.
