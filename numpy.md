# numpy и MatPlot.lib

Библиотека numpy обеспечивает работу с многомерными массивами и матрицами, а также включает в себя большое количество математических функций для выполнения операций над этими данными.

Построить синусоиду можно следующим кодом:

```py
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10*np.pi, 1000)
y = np.sin(x)
plt.plot(x, y)
```

В строке `x = np.linspace(0, 10*np.pi, 1000)` мы определяем линейное пространство - область в заданных границах по оси x.

В строке `y = np.sin(x)` мы задаём функцию вычисления значения y по заданному x во всём диапазоне.

Функция `plt.plot(x, y)` строит график функции в заданном диапазоне.

В следующем примере мы генерируем массив 500x2 случайных значений в диапазоне от 0 до 1. Мы сдвигаем случайные координаты на половину по каждой из осей, чтобы точки равномерно попали в разные четверти системы координат:

```py
import numpy as np
import matplotlib.pyplot as plt

L = np.random.random((500, 2))
X = L[:,0] - 0.5
Y = L[:,1] - 0.5
COLORS = np.sign(X * Y)
plt.scatter(X, Y, c=COLORS)
```

Цвета точек показывают их расположение в разных квадрантах относительно осей. Цвет точки вычисляется как знак произведения X и Y.

Следующий пример строит "пончик" - точки размещаются вокруг центра координат в заданном диапазоне:

```py
import numpy as np
import matplotlib.pyplot as plt

L = np.random.random((500, 2))
D = L[:,0] + 1
A = L[:,1] * np.pi * 2
X = D * np.cos(A)
Y = D * np.sin(A)
plt.scatter(X, Y)
```

## Загрузка и обработка изображений

Для работы с изображениям может быть использована библиотека PIL (Pillow). В приведённом ниже примере загружается изображение из файла с именем "lena.png", посредством усреднения цвета изображение преобразовывается в gray-scale (для удобства дальнейшей обработки). 

Функция convolve2d() из библиотеки SciPy применяется для выполнения двумерной свертки двух массивов. В нашем случае мы используем convolve2d() для формирования изображения удобного для поиска границ на оригинальном изображении (edge detection):

```py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import convolve2d

im = Image.open('lena.png')
gray = np.mean(im, axis=2)

plt.subplot(1,2,1)
plt.imshow(gray, cmap='gray')

#gx = hx * gray
hx = np.array([
  [1, 0,  -1],
  [2, 0, -2],
  [1, 0,  -1]])

gradx = convolve2d(gray, hx, boundary='symm', mode='same')

##plt.subplot(1,4,2)
##plt.imshow(gradx, cmap='gray');

#gy = hy * gray
hy = np.array([
  [1, 2,  1],
  [0, 0, 0],
  [-1, -2, -1]])
grady = convolve2d(gray, hy, boundary='symm', mode='same')

##plt.subplot(1,4,3)
##plt.imshow(grady, cmap='gray');

g = np.sqrt(gradx ** 2 + grady ** 2)
plt.subplot(1,2,2)
plt.imshow(g, cmap='gray');
```
