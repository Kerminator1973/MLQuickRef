# Ollama - локальные LLM

[Ollama](https://ollama.com/) -  это инструмент для простого локального запуска больших языковых моделей (LLM) одной командой. Его часто сравнивают с Docker — но применительно к AI‑моделям: достаточно выполнить одну команду, и модель запускается без необходимости разбираться в зависимостях, конфигурациях и драйверах.

Пример команды:

```shell
ollama run llama2
```

Ollama работает с рядом известных открытых LLM, в том числе:

- LLaMA (включая Llama 3)
- Mistral
- CodeLlama

Ключевое свойство - автоматическая оптимизация:

- определяет доступные ресурсы (CPU/GPU)
- подстраивает загрузку модели под возможности оборудования
- поддерживает квантование и оптимизацию моделей

Небольшие модели можно запускать и на бюджетных процессорах, например, на Core i3 13100, но рекомендуется использовать только 3B‑модели (например, llama3.2:3b, phi3:mini):

```shell
# Запуск Phi‑3 Mini
ollama run phi3:mini

# Запуск Llama 3.2 3B
ollama run llama3.2:3b

# Запуск квантованной Mistral 7B
ollama run mistral:7b-instruct-v0.1-q4_k_m
```

Core i3‑13100 подходит для экспериментов с Ollama и LLM, но не более того.

Настройки для повышения производительности:

- количество параллельных запросов не более 1–2
- квантованные версии моделей (формат GGUF, 4–8 бит)
- все фоновые ресурсоёмкие приложения должны быть закрыты
- для снижения нагрузки следует уменьшить размер контекста. Например: `--num-ctx 2048`

## Hugging Face

[Hugging Face](https://huggingface.co/) - это тоже самое для машинного обучения, что GitHub для кода. На Hugging Face размещаются ИИ модели, решающие различные задачи. В частности, есть много узкоспециализированных ИИ моделей, например, только для генерации кода на ИИ. Узкоспециализированные модели могут эффективно решать сложные задачи, расходуя кратно меньшее количество ресурсов, чем при использовании универсальных моделей.

>Hugging Face после нескольких лет плодотворного сотрудничества купила команду llama.cpp. Предполагается, что ПО останется открытым, но у разработчиков появится больше ресурсов для совершенствования продуктов.

В репозитариях Hugging Face находятся как модели, так и Datasets.

>Hugging Face - это эмодзи "обнимашка". Миссия Hugging Face: "ИИ для исследователей, демократизация ИИ".

Для запуска моделей с Hugging Face необходима Ollama. Можно зарузить Docker-контейнер, но для этого нужне API-ключ (DASHSCOPE_API_KEY), а также поддержку containerd engine в Docker.

Благодаря использованию карточек моделей и поиска, на Hugging Face довольно удобно искать специализированные модели.

## Установка Ollama

Команда установки Ollama в Linux:

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

Проверка версии:

```shell
ollama --version
```

### Установка в Microsoft Windows

Инсталлятор клиента Ollama в Microsoft Windows загружается напрямую с сайта. Его размер составляет 1.2 ГБ. Настроек установки на конкретный носитель нет, т.е. клиент устанавливается безальтернативно на диск c. Доступно всего несколько моделей. Размер модели gemma3:4b - 3.1 ГБ.

Ollama выполняет модель gemma3:4b вполне шустро, приблизительно, со скоростью чуть медленнее вывода на Duck.io в web, на компьютере с процессором Core i3 13100, без дискретной графики. При этом процессор нагревается до 60+ градусов.

Приложения "ollama app.exe" и "ollama.exe" находятся в папке: `C:\Users\[User]\AppData\Local\Programs\Ollama\ollama app.exe`

На портале Hugging Face можно найти модель в формате GGUF. На странице модели нужно выбрать "Use this model" и "Ollama", а затем получить команду для загрузки в консоли Ollama. Например:

```shell
ollama run hf.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M
```

В описании модели есть список аппаратной совместимости. LLM любят VRAM и FP8: "_The Ministral 3 family is designed for edge deployment, capable of running on a wide range of hardware. Ministral 3 8B can even be deployed locally, capable of fitting in 12GB of VRAM in FP8, and less if further quantized._"

Для задач edge deployment минимальная аппаратная рекомендация - использование дискретной видео-карты с 16 ГБ ОЗУ VRAM.

### Повторная попытка

В консоли Hugging Face, в фильтрах выбрал размер модели меньше 3B, GUFF и совместимость с Ollama. Выбрал `ruv/ruvltra-claude-code`.

Команда установки (была запущена из под учётной записи администратора):

```shell
ollama run hf.co/ruv/ruvltra-claude-code:Q4_K_M
```

Размер модели всего 400 МБ, т.е. это крошечная модель - она была специально выбрана для проверки установки моделей из Hugging Face.

Установка прошла успешно.

Эта крошечная модель работает на Core i3 13100 очень быстро. Существенное ограничение - поддержка только английского языка. Для теста попросил проанализировать LLM код моего короткого приложения на Python:

```py
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dota_hero_stats.csv')

def parse_roles(x):
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
    except Exception:
        return []
    return []

lengths = df['roles'].apply(lambda x: len(parse_roles(x)))

sns.set(style="whitegrid")

plt.figure(figsize=(8, 4))
sns.histplot(lengths, bins=range(0, max(lengths)+2), discrete=True)
plt.xlabel("Роли")
plt.ylabel("Количество")
plt.title("Количество ролей пользователей")
plt.xticks(range(0, max(lengths)+1))
plt.tight_layout()
plt.show()
```

Ответы LLM:

_The code above is a Python script that loads and manipulates data from two different CSV files. The first file is named 'dota_hero_stats.csv' which contains Dota's player roles, while the second file is called 'roles.csv'._

Галлюцинация: код считывает только один файл - "dota_hero_stats.csv". Код работает с колонкой "Roles" из CSV-файла, а не со вторым файлом.

_Firstly, it uses the numpy module to load the data from the 'dota_hero_stats.csv' file, which contains Dota's player roles_

Галлюцинация: загружает данные из CSV-файла не numpy, а Pandas.

_After loading the data, it uses the pandas module to manipulate this data as it is and creates a series from it. The result is a column with the number of roles for each user._

Да, это верная оценка.

_Next, it applies a list comprehension to iterate over the 'roles' columns in the 'dota_hero_stats.csv' file and stores them into the 'lengths' variable._

Галлюцинация: ошибка, используется не list comprehension, а лямбда.

_Then, it uses seaborn's seaborn library to generate a white grid histrogram plot to show how many users have each role._

Да, корректно.

В целом, эта крошечная модель сделала много существенных, скорее даже критичных ошибок.
