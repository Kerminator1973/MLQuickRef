# Классификатор графических изображений на основе dataset-а Kaggle

В качестве основы взят dataset [Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) "Cats-vs-Dogs", предоставленный Microsoft. Dataset содержит 12500 изображений котов и собак, архив занимает 800+ МБ.

Для создания классификатора необходимо разделить данные на три группы:

- train - используется для обучения данных содержит по 1000 изображений кошек и собак
- test - используется для определения необходимости в дообучении модели данными. Содержит по 500 изображений
- validation - используется для финальной проверки качества модели. Содержит по 1000 изображений

## Подготовка данных для обработки

Подготовка данных из базового dataset-а осуществляется скриптом автоматизации, разработанном на Python:

```py
import os
import sys
import shutil

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Исходные папки и целевые диапазоны номеров
SOURCE_DIRS = {"Cat": "Cats", "Dog": "Dogs"}
RANGES = {
    "train": (0, 999),
    "test": (1000, 1499),
    "validation": (1500, 2499),
}

DEST_DIR = "cats_and_dogs"


def get_number(filename):
    """Возвращает число из имени файла (например, '123.jpg' -> 123) или None."""
    name, ext = os.path.splitext(filename)
    if ext.lower() != ".jpg":
        return None
    try:
        return int(name)
    except ValueError:
        return None


def main():
    # Создаём папку cats_and_dogs, подпапки test, train, validation
    # и в каждой из них подпапки Cats и Dogs
    for split in RANGES:
        for animal in SOURCE_DIRS.values():
            os.makedirs(os.path.join(DEST_DIR, split, animal), exist_ok=True)

    copied = 0
    for source, animal in SOURCE_DIRS.items():
        for filename in os.listdir(source):
            number = get_number(filename)
            if number is None:
                continue
            for split, (start, end) in RANGES.items():
                if start <= number <= end:
                    src_path = os.path.join(source, filename)
                    dst_path = os.path.join(DEST_DIR, split, animal, filename)
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                    break

    print(f"Скопировано файлов: {copied}")
    for split, (start, end) in RANGES.items():
        for animal in SOURCE_DIRS.values():
            count = len(os.listdir(os.path.join(DEST_DIR, split, animal)))
            print(f"  {split}/{animal}: {count}")


if __name__ == "__main__":
    main()
```

## Подготовка инструментальных средств

Разработка классификатора может осуществляться в среде Google Colab, по причине доступности в среде средств аппаратного ускорения (GPU с 16 ГБ VRAM).

Вместе с тем, ограничения длительности сессии в бесплатном тарифном плане могут потребовать использования видео-ускорителя в локальном ПК и создания соответствующего стенда.

Ключевой момент, который критически влияет на конфигурацию системы: **RTX 5060** — это архитектура Blackwell, compute capability 12.0. Отсюда два жёстких следствия:

- Нужен CUDA 12.8+ и cuDNN 9.8+. Более старые сборки TensorFlow просто не содержат ядер для sm_120 — вы увидите no kernel image is available for execution on the device или тихий откат на CPU
- С TensorFlow 2.11 официальные Windows-сборки не поддерживают GPU. На "голой" Windows с GPU TensorFlow работать не будет (кроме экспериментального tensorflow-directml, который для Blackwell не подходит)

Таким образом, работоспособная рекомендация: WSL2 + Ubuntu, TensorFlow внутри.

### Конфликт VMware и WSL

Раньше этот вариант не походил для моей машины, т.к. у меня установлена VMware для других рабочих задач, а Windows не позволяет использовать одновременно для гипервизора. Возможно, что проблема решена — WSL2 и VMware Workstation могут работать одновременно на одной машине, но только при соблюдении трёх условий:

- Windows 10 2004+ / Windows 11
- использовать актуальные версии VMware Workstation - 17.5/17.6.x и выше
- включённая функция Windows Hypervisor Platform

Конфликт, с которым я столкивался раньше, был актуален для старых связок "VMware < 15.5 + любое состояние Hyper-V".

Цена совместимости — VMware-гость работает медленнее, чем в "чистом" режиме с захватом VT-x, и теряет вложенную виртуализацию.

Изначально VMware Workstation запускал свой собственный гипервизор в ring-0 и требовал эксклюзивного доступа к VT-x/EPT. Hyper-V тоже гипервизор и тоже забирает VT-x себе. Два монополиста на одном процессоре не уживались — отсюда сообщения вроде «VMware Workstation and Device/Credential Guard are not compatible» или просто отказ стартовать VM.

WSL2 устроен иначе, чем WSL1: это лёгкая служебная Hyper-V-машина (utility VM) с реальным ядром Linux. Следовательно, WSL2 требует запущенного гипервизора Windows. Как только вы включали Virtual Machine Platform и hypervisorlaunchtype auto, VMware терял эксклюзивный доступ к VT-x и падал.

Выход VMware нашёл в 2019 году: начиная с Workstation 15.5 VMware умеет работать «поверх» Hyper-V, используя API Windows Hypervisor Platform (WHP) вместо собственного гипервизора. Гостевые ОС исполняются через WHP, конфликт исчезает, и обе системы живут параллельно. В 16.x и 17.x этот режим довели до приемлемой производительности.

Ограничения редакции Home здесь почти не мешают: WSL2 работает на собственном легковесном гипервизоре Windows через компонент Virtual Machine Platform, а не через полноценную роль Hyper-V. Критично и то, что поддержка CUDA/GPU в WSL2 доступна на Home так же, как на Pro — RTX 5060 будет видна из Ubuntu.

Проверка совместимости через PowerShell (от администратора):

```powershell
Get-ComputerInfo | Select-Object WindowsProductName, OsBuildNumber, HyperVisorPresent
```

Windows выдаёт не корректную информацию об операционной системе:

```output
WindowsProductName OsBuildNumber HyperVisorPresent
------------------ ------------- -----------------
Windows 10 Home    26300                      True
```

Следует проверить, что тип запуска гипервизора выставлен корректно:

```shell
bcdedit /enum | Select-String hypervisorlaunchtype
bcdedit /set hypervisorlaunchtype auto     # авто — требуется для WSL2
```

Должен быть выставлен "auto" и таким он и установлен:

```output
hypervisorlaunchtype    Auto
```

Установка WSL:

```shell
wsl --install -d Ubuntu-24.04
```

После перезагрузки:

```shell
wsl --version
```

```output
Версия WSL: 3.0.3.0
Версия ядра: 6.18.54.1-1
Версия WSLg: 1.0.79
Версия MSRDC: 1.2.7391
Версия Direct3D: 1.611.1-81528511
Версия DXCore: 10.0.26100.1-240331-1435.ge-release
Версия Windows: 10.0.26300.9711
```

Однако дистрибутив не был установлен предыдущей командой. Список доступных для загрузки дистрибутивов:

```shell
wsl.exe --list --online
```

Установка:

```shell 
wsl.exe --install Ubuntu-24.04
```

Запуск из командной строки `wsl`. Текущая папки Linux - та, текущая для "Command Prompt" на момент запуска `wsl`.

Внутри WSL2 драйвер и CUDA Toolkit ставить НЕ нужно — WSL получает драйвер от хоста.

Проверить, что драйвер работает можно командой (запускать в Linux):

```bash
nvidia-smi
```

### Формирование данных для Linux и подключение диск к Google Colab

С помощью приведённого выше скрипта данные были разбиты по папкам.

Затем в WSL была выполнена команда:

```bash
tar -czf my_data.tar.gz ./cats_and_dogs/
```

В результате был создан архив с именем "my_data.tar.gz", который можно загрузить на Google Drive.

Находясь в виртуальной машине Google Colab можно подключить личный диск пользователя:

```py
from google.colab import drive
drive.mount('/content/drive')
```

После этого появляется папка "drive/MyDrive" внутри которой находятся личные файлы и архив "my_data.tar.gz".

Следует открыть терминал в Google Colab и перейти в папку с данными:

```bash
cd ./drive/MyDrive
```

После этого следует распаковать данные на виртуальный диск Colab:

```bash
tar -xzf my_data.tar.gz -C /content
```

В папке "/content" верхнего уровня появится папка с данными "cats_and_dogs".

Размонтировать Google Drive можно командой:

```py
from google.colab import drive
drive.flush_and_unmount()
```

Всё. Теперь можно использовать данные в виртуальной машине Google Colab.

## Установка Python и других средства выполнения

Обновление списка зависимостей:

```bash
sudo apt update && sudo apt upgrade -y
```

Установка инструментальных средств, включая Python 3:

```bash
sudo apt install -y build-essential git curl wget unzip \
                    software-properties-common \
                    python3.12 python3.12-venv python3.12-dev \
                    python3-pip pipx
```

TensorFlow 2.19/2.20 поддерживает 3.9–3.13, но 3.12 — самый безопасный выбор.

Установка virtualenv:

```bash
sudo apt install python3-virtualenv
```

Проверка установленной версии:

```bash
virtualenv --version
```

Какие Python-библиотеки установлены в системе:

```bash
apt list --installed | grep python3
```

Добрался до следующей рекомендации: `Windows: клонируйте репозиторий ВНУТРЬ файловой системы WSL — работа с /mnt/c замедляет чтение датасета в 3–5 раз`. И в этом месте появилось желание вернуться к Google Colab - полноценная настройка системы выполнения на локальной машине занимает слишком много времени.

### Удаление ОС в WSL и самой WSL

Список установленных операционных систем в WSL:

```shell
wsl.exe --list
```

Разрегистрация Ubuntu в WSL:

```shell
wsl --unregister Ubuntu-24.04
```

Удалить WSL:

```shell
wsl --uninstall
```
