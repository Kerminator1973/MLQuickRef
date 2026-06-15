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

В действительности, Pandas добавляет целую кучу дополнительных модулей осуществляющих работы у календарями, преобразованием времени, служебными функциями, и т.д. Специфика состоит в том, что для аналитической обработки данных, стандарного datatime не достаточно.

Конструктор Timestamp принимает те же параметры, что и datatime. Например:

```py
pd.Timestamp(1991, 4, 12)
pd.Timestamp(year = 1991, month = 4, day = 12)
```

Однако Timestamp умеет обрабатывать множество других форматов данных, в том числе строковые представления. Например:

```py
pd.Timestamp("2015-03-31")
pd.Timestamp("2015/03/31")
pd.Timestamp("03/31/2015")
pd.Timestamp("2021-03-08 8:35:15")
```

### Хранение нескольких отметок времени в DatetimeIndex

Получить доступ к индексу Series или DataFrame можно через атрибут **index**. Например:

```python
pd.Series([1,2,3]).index
```

**DatatimeIndex** - это индекс, хранящий объекты **Timestamp**. Присоединить индекс с отметками времени к данным можно следующий образом:

```py
timestamps = [
    pd.Timestamp("2020-01-01"),
    pd.Timestamp("2020-02-01"),
    pd.Timestamp("2020-03-01")
]
pd.Series([1,2,3], index = timestamps).index
```

Для сортировки содержимого структуры по индексу DatatimeIndex можно использовать метод sort_index:

```py
s.sort_index()
```

Для Timestamp поддерживаются различные операции сортировки и сравнения.

При считывании данных из CSV-файла, мы можем явным образом указывать, какое поле содержит время:

```py
disney = pd.read_csv("disney.csv", parse_dates = ["Date"])
```

Явное преобразование даты из строчного представления в DatatimeStamp может выглядеть так:

```py
disney["Date"] = pd.to_datetime(disney["Date"])
```

