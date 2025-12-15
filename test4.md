# Тестовые задания Евгения (скопировал из его блокнота для проверки)

Подзадание №2: напишите итератор, который будет подавать в цикл переданное ему число элементов. Например, если мы итерируемся по списку [1, 2, 3, 4, 5], а на вход итератору передано число 2, он сначала вернёт (1, 2), затем (3, 4) и на этом завершит итерации.

Решение:

```py
class IntegerIterator:
    def __init__(self, int_list, int_count):
        if not isinstance(int_list, list) or not all(isinstance(i, int) for i in int_list):
            raise ValueError("Input must be a list of integers.")
        self.int_list = int_list
        self.index = 0

        # Сохраняем количество элементов, которые нужно возвращать на каждой итерации
        self.count = int_count

    def __iter__(self):
        return self

    def __next__(self):
        if self.index + self.count - 1 < len(self.int_list):
            result = self.int_list[self.index:self.index+self.count]
            self.index += self.count
            return tuple(result)
        else:
            raise StopIteration

# Пример использования
int_list = [1, 2, 3, 4, 5]
iterator = IntegerIterator(int_list, 2)

for number in iterator:
    print(number)
```
