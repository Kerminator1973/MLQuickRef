# Инструкция по настройке CUDA в Linux

Инструкция заимствована из статьи [Тестируем выделенный L40S и vGPU на 16 ГБ по производительности (llama.cpp, ComfyUI)](https://habr.com/ru/companies/first/articles/1042142/) by FirstVDS.

1. Обновить репозитории и все текущие пакеты в системе:

```shell
apt update
apt upgrade
```

2. Установать компилятор gcc, необходимый для сборки CUDA:

```shell
apt install gcc-12 g++-12
```

3. Подключить официальные репозитории от Nvidia:

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
```

4. Устанавить СUDA из официального репозитария Nvidia. Драйверы NVDIA добавятся автоматически как зависимости:

```shell
apt install cuda
```


5. Прописать переменные окружения. Действие нужно повторить для каждого пользователя, который будет работать с CUDA:

```shell
echo 'export PATH="/sbin:/bin:/usr/sbin:/usr/bin:${PATH}:/usr/local/cuda/bin"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

6. После установки желательно перезагрузить сервер.

7. Далее проверяем, что ОС видит видеокарту и утилиты доступны:

```shell
nvidia-smi
lsmod | grep nvi 
nvcc -V
```
