# Использование ИИ агентов

Практический опыт генерации кода с использованием LLM (Android-приложение для обмена данными с микроконтроллером через USB/CDC, а также задача подготовки explosion view для схем приборов), позволяет утверждать, что chat-bot-ы пока не способны самостоятельно решать более-менее сложные задачи, связанные с генерацией кода.

При этом "грубая ИИ сила", используя множество попыток, способна генерировать работоспособные решения. 

Можно выделить несколько основных схем, которые позволяют бороться с галлюцинирующим нейрослопом:

- разделение задачи на простые подзадачи
- поиск консенсуса нескольких LLM
- использование специализированное ПО, упрощающего LLM задачу по генерации кода (MCP-сервера)

## Организация pipeline с устранением ошибок

Об этом чаще всего пишет Андрей Карпати (со-основатель OpenAI), который ввёл в обиход термин _vibe coding_.

Идея состоит в том, что один агент генерирует требования в соответствии с некоторой схемой. Например, агент может сгенерировать качественные технические требования к REST API, используя промпт оператора. При этом, генерация может осуществляться в виде описания OpenAPI (ранее Swagger).

Далее это описание передаётся другим агентам, например:

- генерация Backend-а по описанию REST API и описанию технологического стека
- генерация Frontend-а по описанию REST API и описанию технологического стека
- генерация тестов по описанию REST API

Ключевым моментом является тот факт, что агент, генерирующий тесты ничего не знает о результате генерации кода Backend-а, т.е. формально, Unit-тестирование максимально честное.

>Здесь я имею ввиду не Unit-тесты, а тестирование ответов Endpoint-ов по спецификациям OpenAPI

Далее, ещё один агент (manager) применяет тесты к Backend и те тесты, которые были выполнены не успешно, передаются на вход генератору Backend-а. С некоторой вероятностью, процесс будет сходящимся и через несколько итераций Backend будет проходить тестирование.

Смысл мульти-агентной системы состоит в том, что несколько агентов в совокупности представляют pipeline по разработке кода, который включает этапы устранения ошибок и при этом сильно похоже на то, как такую же работу выполняет команда разработчиков (аналитики, программисты, тестировщики и devops).

## Поиск консенсуса нескольких LLM

Современные LLM умеют давать простые ответы на сложные вопросы.

Например, в проекте [AndroidUsbCDC](https://github.com/Kerminator1973/AndroidUsbCDC) я выполнил промпт, целью которого было [улучшение качества моего кода](https://github.com/Kerminator1973/AndroidUsbCDC/blob/main/ai-prompt.md). Я использовал локальную копию Gemma 4, которая сформировала целый ряд предложений по улучшению кода. Часть изменений я перенёс в моё решение и поставил ту же самую задачу, но уже Claude Sonnet 4.6. По одному из файлов получил вот такой вывод:

```
CdcPortData.kt — unchanged, already idiomatic.
```

Т.е. Claude Sonnet 4.6 подтвердил качество изменений предложенных LLM Gemma 4 для моего кода.

Таким образом, можно рассматривать схему, в которой решение о том, что задача была выполнена качественно принимает не человек, а несколько LLM, используя правило консенсуса.

## Использование MCP-серверов

**MCP-сервер** - это сервис (коннектор), который по протоколу Model Context Protocol (MCP) предоставляет ИИ‑моделям доступ к внешним ресурсам, инструментам и данным в стандартизированном виде.

Существуют разные типы MCP-серверов:

- Tools — исполняемые функции. Позволяют модели не просто читать данные, а выполнять действия: делать API‑запросы, создавать задачи, обновлять записи и т. д.
- Resources — источники данных и контекста: содержимое файла, запись из базы, ответ API, карточка клиента, внутренний документ
- Prompts — шаблоны взаимодействия. Структурируют запросы и подсказки, чтобы обмен данными между клиентом и сервером был предсказуемым

Один из способов улучшить качество генерации кода - интегрировать генерацию кода LLM и MCP-сервера. Предположим, что создаётся некоторый набор форм для 1С. Calude Sonnet что-то знает о 1С, но не много и самостоятельно не сможет качественно сгенерировать такие формы. Но он может использовать специализированный MCP-сервер, код которого сгенерирует нужные формы по шаблонам, передаст их LLM, а LLM уже поправит их в соответствии с промптом оператора.

Подход позволяет достичь двух критически важных целей: кратно уменьшить расход токенов, т.к. задача для LLM может быть сильно упрощена, а качество результа может быть существено выше, чем у самостоятельно работающей LLM.

>Эту схему я подсмотрел у разработчиков 1C, в нескольких видео на RuTube, которые именно для разработки форм 1С используют связку из специализированных MCP-серверов и Claude Sonnet 4.6

Скорее всего, именно подобную схему имеет ввиду Дженсен Хуанг (руководитель Nvidia), когда говорит об изменении роли программиста, которая предполагает не только оркестрацию ИИ разработкой, но и сменой парадигмы - "не ИИ помогает человеку писать код, а человек помогает ИИ генерировать качественный код".

## Другие варианты использования агентов

Рекомендуется к ознакомлению RuTube-ролик: [Как AI помогает запускать стартапы вайбкодинг, OpenClaw и ограничения реального бизнеса](https://rutube.ru/video/4807e5897434926ba476c2a7e4a76eee/) опубликованное на канале **Организованное программирование**.

Пример из практической жизни. OpenClaw используется как агент сопровождения пользователей. Выполняет следующие действия:

- получает почту и анализирует каждое письмо
- определяется суть претензии пользователя: 
    - требование возврата денег, поскольку результат не удалось использовать
    - качество работы оказалось низким, нужны ещё бесплатные попытки
- OpenClaw подключается к базе данных пользователей и проверяет, что конкретный пользователь платил за услуги. Далее, в зависимости от требования:
    - осуществляется возврат денежных средств
    - начисляются дополниетльные попытки использования сервиса

Следует заметить, что OpenClaw не выполняет возврат средств самостоятельно, а пересылает человеку описание выполняемых действий и ссылку на действие "Refund". Если человек согласен, то он нажимает кнопку и попадает в интернет-банкинг, в котором есть реквизиты платежа, но необходимо авторизовать операцию вручную.

Механизм достаточно хорошо масштабируется, не нужно создавать отдельный отдел и нанимать в него дополнительных людей.

Особенность OpenClaw: очень прожорливый в части потребления контекста. Контекст накапливается и всё время передаётся в ИИ, что приводит к большой стоимости запросов.

## Режимы работы агентов

- Plan - только анализ, безопасно
- Build - с возможностью внесения изменений. Это довольно опасный инструмент, который требует внимательности

## OpenCode

OpenCode устанавливается автоматически при первом запуске, но в случае Windows, консоль должна быть открыта с привелегиями локального администратора:

```shell
ollama launch opencode
```

Для анализа структуры проекта следует перейти в соответствующую папку и запустить OpenCode. Рекомендуется запускать команду из папки, в которой находится описание проекта, например - папка с файлом *.sln.

Далее необходимо выполнить команду `/init` для формирования структуры проекта. Эта команда сформирует файл "AGENTS.md". Следует заметить, что генерация файла "AGENTS.md" осуществляется только в режиме "build". В режиме "plan" у OpenCode нет права на формирование файла.

Пример сгенерированного файла - "AGENTS-SPCD.md". Этот файл описывает структуру проекта и действия, необходимые для его сборки и запуска.

Задача программиста - актуализировать содержимое файла "AGENTS.md", добавляя важные особенности проекта для работы ИИ агента. Пример:

```
# AGENTS.md — инструкции для AI‑агента

## Назначение
Помогать с анализом, рефакторингом и добавлением функционала в C#‑проекте. Агент должен соблюдать архитектурные правила и соглашения команды.

## Стек
- C# 13
- .NET 9
- ASP.NET Core
- xUnit для тестов
- Entity Framework Core

## Структура проекта
- `src/App` — точка входа и конфигурация хоста.
- `src/Api` — контроллеры и DTO.
- `src/Core` — бизнес‑логика и доменные сервисы.
- `src/Infrastructure` — доступ к данным, внешние клиенты.
- `tests/` — модульные и интеграционные тесты.

## Архитектурные правила
- Контроллеры должны быть тонкими: только валидация и вызов сервисов.
- Вся бизнес‑логика — в сервисах в `src/Core`.
- Не создавать статические классы для логики; избегать синглтонов без необходимости.
- Все внешние вызовы (HTTP, БД) — через слой `Infrastructure`.

## Соглашения по коду
- Классы и методы: PascalCase.
- Локальные переменные и параметры: camelCase.
- Имена сервисов заканчиваются на `Service` (например, `OrderService`).
- Не использовать `var` в публичных сигнатурах методов.

## Команды
- Сборка: `dotnet build`
- Запуск: `dotnet run --project src/App/App.csproj`
- Тесты: `dotnet test`

## Ограничения
- Не менять файлы в `migrations/` и автогенерируемые классы.
- Не удалять или изменять файлы в `.github/` и `docker/`.

## Примеры задач
- При добавлении нового эндпоинта: создать DTO в `src/Api/Dtos`, контроллер в `src/Api`, сервис в `src/Core`, тест в `tests/`.
- При рефакторинге: сначала предложить план, не применять изменения автоматически без подтверждения.
```

Стартовый промпт для анализа кода проекта (вставка текста через соответствующий инструмент "терминала Windows"):

```
You are an experienced C#/.NET architect and senior developer. I’m working on a large C# codebase and need high‑level architectural improvements.

First, analyze the project structure and key components (services, controllers, data access, tests, etc.) to understand the current architecture. Pay special attention to separation of concerns, dependency management, and error handling.

Then, propose 3–5 concrete architectural improvements that would increase maintainability, scalability, and testability. For each improvement:

- Briefly describe the problem it addresses.
- Explain the proposed change in 2–3 sentences.
- List 1–2 specific files or folders where this change should be applied.
- Mention any potential risks or effort level (low/medium/high).

Do not rewrite code yet—focus on architecture and strategy. Return the result as a numbered list with clear headings for each improvement.
```

По проекту SPCD был подготовлен следующий анализ:

```
Based on the existing multi-layered structure (ServicePartners.sln orchestrating Shared, SDK, DorsFileStorage, ServicePartners.Server, and ServicePartners.Client), here are several architectural improvements focused on increasing maintainability, scalability, and testability within this C#/.NET codebase.

1. Implement Clean/CQRS Architecture for Business Logic

- Problem: The current pattern likely mixes business logic execution with API handling (Controllers/endpoints), leading to fat controllers and poor separation of concerns.
- Proposed Change: Introduce the Command Query Responsibility Segregation (CQRS) pattern, possibly using a library like MediatR. Controllers should only map HTTP requests into specific Commands or Queries, which are then executed by dedicated handlers without knowing about the HTTP context. This isolates I/O handling from business rules and data fetching queries.
- Affected Files/Folders: ServicePartners.Server (Controllers), Shared (Interfaces for commands/queries), new dedicated folder: Application/Handlers.
- Risks/Effort: Medium. Requires updating every Controller to use the "Command Mediator pattern" and introducing a dependency on an additional NuGet package.

2. Standardize Cross-Cutting Concerns with MediatR or an Aspect-Oriented Approach (AOP)

- Problem: Logging, auditing, caching, and validation are likely implemented ad-hoc throughout various services and controllers, making it difficult to maintain consistency across the codebase.
- Proposed Change: Centralize these concerns using a mediator pattern listener or by implementing an Aspect-Oriented Programming framework (e.g., leveraging interceptors in ASP.NET Core middleware). This ensures that all interactions flow through standardized interception points for non-business logic tasks.
- Affected Files/Folders: The ASP.NET Core Startup/Program setup files within ServicePartners.Server, and potentially extending the base classes for Controllers.
- Risks/Effort: Low to Medium. If using built-in framework features (like middleware) or a standard pattern, it's relatively low effort; if it requires complex interception logic, it increases risk.

3. Introduce an Explicit Feature Module Structure (Vertical Slicing)

- Problem: The codebase appears organized by technical layer (Server, Client, Shared), which sometimes leads to unintended cross-layer couplings or difficulty localizing changes for a specific feature (e.g., "User Management").
- Proposed Change: Adopt a Vertical Slice Architecture grouping all components required for a single business domain or feature together. A "Feature Module" directory would contain its own dedicated models, handlers (CQRS), services, and DTOs, minimizing global changes when adding a new feature like "Invoicing."
- Affected Files/Folders: The root of ServicePartners.Server (new structure: Features/UserManagement, Features/ProductCatalog). This will significantly reorganize folder structures but improves cohesion.
- Risks/Effort: Medium to High. This is a major refactoring effort, requiring careful dependency mapping and breaking down existing modules into self-contained units.

4. Formalize Error Handling with Custom Exceptions and Fallbacks

- Problem: Using standard try/catch blocks throughout the business logic is brittle and often catches too much or not enough context. Errors may propagate inconsistently from the data layer (DorsFileStorage) up to the API response layer.
- Proposed Change: Define a standardized set of custom exception types (e.g., ResourceNotFoundException, BusinessRuleViolationException). Implement a global middleware handler in the server application that intercepts these specific exceptions and maps them uniformly to appropriate HTTP status codes (404, 400, etc.).
- Affected Files/Folders: ServicePartners.Server startup/middleware configuration files, and an extension within the Shared project for defining custom exception types.
- Risks/Effort: Low. Highly beneficial and addresses common enterprise pitfalls without major structural overhaul.
```

На что обращает внимание ИИ (Gemma 4):

- уровень бизнес-логики перемешан с обработкой запросов к API. Контроллер получается слишком "жирным" и, как следствие, разделение ответственности (_separation of concerns_) реализуется очень слабо
- логирование, кэширование (где Gemma его нашёл!?), аудит и валидация "размазаны" по различным сервисам и контроллерам, что усложняет управление ими
- кодовая база организована по техническим уровня (клиент/сервер/разделяемые данные), что приводит к межуровневой сцепленности и усложнению в локализации изменений для таких сущностей, как "UserManagement" и "ProductCatalog"
- обработка исключений не систематизирована, реализована довольно слабо

Как отправная точка - результат можно оценить как хороший.
