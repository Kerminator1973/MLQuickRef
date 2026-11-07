# Контроль работы агентов (UNDER CONSTRUCTION)

Для просмотра выполняемых запросов к модели, можно использовать plug-in [opencode-llm-capture](https://github.com/aixmoyu/opencode-llm-capture)

Рекомендуется следовать [online-инструкции](https://github.com/aixmoyu/opencode-llm-capture/blob/main/INSTALL.md)

В случае Windows, необходимо скачать plugin в папку `%USERPROFILE%\.config\opencode\opencode-llm-capture`.

Процедура инсталляции размещает файл "llm-capture.ts" в папке "C:\Users\[пользователь]\.config\opencode\plugins\llm-capture.ts" и создаёт символическую ссылку на этот файл в папке "C:\Users\[пользователь]\.config\opencode\opencode-llm-capture\llm-capture.ts".

Проверить, если ли символическая ссылка в папке можно командой PowerShell:

```shell
Get-Item "$env:USERPROFILE\.config\opencode\plugins\llm-capture.ts"
```

Удалить символическую ссылку можно через Command Prompt:

```shell
del "%USERPROFILE%\.config\opencode\plugins\llm-capture.ts"
```

Получена ошибка "несовместимый плагин":

```
Server plugin error
Plugin: ~/.config/opencode/plugins/llm-capture.ts
Status: failed
Runtime: server
Source: ~/.config/opencode/plugins/llm-capture.ts
Error: Plugin must export a default definition with an id and an effect or setup function.
Reference: err_33cd541b
```

Модель предложила исправить ошибку и нашла решение, но, посколько выполнялась модификация глобального "package.json" я не стал эту модификацию применять. Потребовалось бы вычистить глобальную папку с зависимостями Node.js и переустановить OpenCode.

## Что передаёт агент на сервер

- Роль ассистента-разработчика
- Правила модификации кода: не удалять несвязанный код, сохранять стиль форматирования, отступы и комментарии
- Инструкции по безопасности и работе с файловой системой
- Схема доступных инструментов (Tools / Function Calling)
- Пользовательский запрос (User Message)
- Метаданные окружения и репозитория
- Структура проекта (дерево каталогов верхнего уровня или список ключевых файлов)
- Рабочая директория и используемый язык программирования / стек

Содержимое файла:

- Если файл подгружается сразу (Pre-fetching): плагин/утилита определяет по тексту путь к файлу XXX, считывает его с диска и включает в контекст целиком (или с нумерацией строк), оборачивая в специальные маркеры (например, <file path="XXX">...код файла...</file>)
- Если используется агентный подход (ReAct): сначала модель получает только ваш запрос и описание инструментов. Модель принимает решение вызвать инструмент read_file("XXX"), среда выполнения считывает файл и отправляет его содержимое модели вторым шагом как результат вызова функции (tool_result)
