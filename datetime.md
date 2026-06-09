# Дата и время в Pandas

В Python есть встроенные типы данных для работы с датой и временем: date, time, datetime и timedelta. Эти типы необходимо импортировать:

```py
import datetime as dt
```

Примеры инициализации объектов:

```py
birthday = dt.date(1991, 4, 12)
birthday = dt.date(year = 1991, month = 4, day = 12)
```

Объект date - неизменяемый. После создания изменить его состояние нельзя.

Пример инициализации объекта time:

```py
alarm_clock = dt.time(6, 43, 25)
alarm_clock = dt.time(hour = 6, minute = 43, second = 25)
```

Pandas добавляет дополнительные объекты **Timestamp** и **Timedelta**. Импорт:

```py
import pandas as pd
```
