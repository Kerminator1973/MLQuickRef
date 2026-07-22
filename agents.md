# Использование ИИ агентов

Практический опыт генерации кода с использованием LLM (Android-приложение для обмена данными с микроконтроллером через USB/CDC, а также задача подготовки explosion view для схем приборов), позволяет утверждать, что chat-bot-ы пока не способны самостоятельно решать более-менее сложные задачи, связанные с генерацией кода.

При этом "грубая ИИ сила", используя множество попыток, способна генерировать работоспособные решения. 

Можно выделить несколько основных схем, которые позволяют бороться с галлюцинирующим нейрослопом:

- разделение задачи на простые подзадачи
- поиск консенсуса нескольких LLM
- использование специализированного ПО, упрощающего модели (LLM) задачу по генерации кода (через вспомогательные MCP-сервера)

>В 2026 году, программирование выглядит как сместь генерации кода скриптами (например, генерации миграций из описания модели базы данных), генерация кода ИИ и, в меньшей степени, ручная разработка кода. При этом наибольшая эффективность при создании когда достигается, когда два вида автоматической генерации кода удачно синхронизованы.

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

Смысл мульти-агентной системы состоит в том, что несколько агентов в совокупности представляют pipeline по разработке кода, который включает этапы устранения ошибок и это похоже на то, как такую же работу выполняет команда разработчиков (аналитики, программисты, тестировщики и devops).

## Агентный цикл

Довольно часто рассматривают агентный цикл, который должен обеспечить разработку работоспособного приложения. На входе указывается цель, а далее повторяется следующий цикл:

- Чтение контекста
- Планирование достижения результата
- Запуск инструментов для генерации кода
- Получение результатов
- Проверка результатов: если проверка пройдена, то генерация завершается. Если проверка не пройдена, уходим на следующий цикл

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

## Режимы работы агентов (OpenCode)

- **Plan** - только анализ, безопасно
- **Build** - с возможностью внесения изменений. Это довольно опасный инструмент, который требует внимательности

## OpenCode

OpenCode устанавливается автоматически при первом запуске, но в случае Windows, консоль должна быть открыта с привелегиями локального администратора:

```shell
ollama launch opencode
```

Для анализа структуры проекта следует перейти в соответствующую папку и запустить OpenCode. Рекомендуется запускать команду из папки, в которой находится описание проекта, например - папка с файлом *.sln.

Далее необходимо выполнить команду `/init` для формирования структуры проекта. Эта команда сформирует файл "AGENTS.md". Следует заметить, что генерация файла "AGENTS.md" осуществляется только в режиме "build". В режиме "plan" у OpenCode нет права на формирование файла.

>Замечу, что использовать OpenCode совместно с традиционными IDE крайне неудобно. По всем видимости, предполагается, что разработчик не будет писать код вручную, а делегирует эту функцию LLM на 100%.
>
>Этой гипотезе есть подтверждение практическим опытом - LLM очень не любят, когда программист изменяет в сгенерированном LLM коде что-либо. Чаще всего LLM просто игнорирует эти изменения, но иногда такие изменения вызывают галлюцинации.
>
>Допускаю, что **vide coding** придуман не по причине его высочайшей эффективности, а из-за того, что гибридный режим модификации кода (LLM + человек) работает значительно хуже, чем генерация кода без активного участия человека.

### AGENTS.md

Файл "AGENTS.md" часто называют **памятью проекта**. Этот файл описывает структуру проекта, его назначения, действия, необходимые для его сборки и запуска, а также особенности проекта о которых должна знать LLM для более эффективного выполнения действий с помощью агента.

Пример сгенерированного файла - "AGENTS-SPCD.md". 

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

### Как лучше использовать эти промпты в OpenCode

- Запускайте после /init, чтобы агент уже проиндексировал файлы
- Если файлов очень много, можно дополнительно указать: "Focus especially on the `src/Core` and `src/Api` folders; ignore generated files and test projects unless they illustrate a problem"
- Чтобы избежать слишком абстрактных советов, фраза "provide concrete file paths" (как в примерах) сильно повышает полезность ответа
- Для C#‑проектов полезно напомнить агенту про DI, testing, and EF Core patterns — это снижает риск советов, несовместимых с .NET

### Запрос для поиска Razor-компонентов нуждающихся в улучшении

```
Act as a senior Blazor developer with deep expertise in client-side Blazor (Blazor WebAssembly) architecture and performance optimization.

I’m working on a Blazor WebAssembly application. Below is a brief description of the app’s domain and some key components I’ve implemented (you can treat this as context; if no specific code is provided, reason about a typical mid‑sized Blazor Wasm app):

[Insert your app description or paste relevant code snippets / component list here.]

Your task: analyze the provided context (or assume a typical Blazor Wasm project structure if nothing is specified) and generate a prioritized list of 5–10 client‑side Blazor components that are most likely to benefit from significant improvements. For each component, provide:

- Component name (e.g., `DashboardWidget`, `OrderList`) or a clear description if the name isn’t known.
- Primary issue or limitation (e.g., excessive re-renders, large payload size, poor accessibility, tight coupling to services, inefficient state management, lack of virtualization, missing error boundaries, overuse of JS interop, etc.).
- One concrete improvement idea (1–2 sentences) with a specific technique or pattern (e.g., “Use `Virtualize` for long lists,” “Move heavy logic to a background service,” “Replace direct service calls with a shared state container,” “Add `RenderMode.InteractiveWebAssembly` with proper fallbacks,” “Implement lazy loading for this component,” etc.).
- Expected impact (e.g., “reduce initial load time by ~30%,” “cut render cycles in half for large datasets,” “improve accessibility score,” “decrease bundle size,” etc.).

Prioritize the list by potential impact on performance, maintainability, and user experience (highest impact first). If you need any additional details about the project (tech stack version, typical data volumes, target browsers, etc.) to make better recommendations, ask 2–3 focused questions at the end.

Do not list generic “best practices” without tying them to specific component types. Focus on tangible, high‑leverage improvements.
```

### Skills

**Skill** - это часто повторяемая операция ИИ.

В OpenCode **skill** — это отдельный файл с инструкциями, который описывает агенту, как решать конкретную категорию задач. В OpenCode рекомендуется сохранять skill в папке `.opencode/skills/<skill-name>/SKILL.md`. Строгое соответствие имени ("SKILL.md") критично для успешности поиска. Также структура документа должна быть корректной - OpenCode может проигнорировать файл, если посчитает его содержимое не валидным. Описание для LLM осуществляется на Markdown.

Допустим, мы разрабатываем навык **blazor-component-review** для ревью Blazor‑компонентов.

Описание skill-а может выглядеть следующим образом:

```markdown
---
name: blazor-component-review
description: Review Blazor components for best practices, performance, and maintainability.
version: 1.0.0
author: Maksim Rozhkov
tags:
  - blazor
  - csharp
  - code-review
triggers:
  - type: keyword
    patterns:
      - "review Blazor component"
      - "check Blazor component for issues"
      - "analyze Blazor component quality"
allowed-tools:
  - read_file
  - list_files
constraints:
  - Only analyze .razor and .cs files related to components.
  - Do not modify code; only propose changes.
---

# Blazor Component Review Skill

## Purpose

This skill is used to review Blazor components (`.razor` and associated `.cs`) for:
- adherence to Blazor best practices
- performance concerns (unnecessary renders, large cascading parameters)
- maintainability (large components, tight coupling, magic strings)
- testability and separation of concerns

## Workflow

1. Identify all relevant `.razor` files in the requested scope.
2. For each component, check:
   - Component size: if > ~300–400 lines, suggest splitting.
   - Parameter usage: prefer `[Parameter]` with clear names; avoid excessive cascading.
   - State management: look for patterns that cause unnecessary re-renders.
   - Code-behind: ensure proper separation if used.
   - Dependency injection: correct usage in `.razor` vs `.cs`.
3. Produce a concise report per file with:
   - 3–5 most important observations
   - concrete suggestions and example snippets (C#/Razor)
   - estimated impact (high/medium/low)

## Checklist

- [ ] Component size within reasonable limits
- [ ] Parameters are explicit and documented
- [ ] No unnecessary `StateHasChanged()` calls
- [ ] Proper use of `EventCallback` vs direct events
- [ ] DI registered correctly and injected cleanly

## Examples

**User request:** “Review the `OrderSummary.razor` component.”  
**Agent behavior:** Load this skill, analyze `OrderSummary.razor`, produce a bullet list with observations and code snippets.

**User request:** “Check all Blazor components in `src/Client/Components`.”  
**Agent behavior:** Use `list_files` to find `.razor` files, then apply the same review process to each.
```

Чтобы проверить, что Skill подключился, в консоли OpenCode следует перейти в панель команд OpenCode (crtl+p), найти раздел Skills и посмотреть, попал ли наш skill в список доступных. К сожалению, OpenCode не всегда подхватывает необходимые данные.

>Scanner skill-ов загружает зависимости в папку ".opencode" проекта ~55 МБ. При практической работе необходимо следить, чтобы эти зависимости не попали в репозитарий проекта.

OpenCode автоматически ищет и подключает папки со скиллами. Затем достаточно будет явно указать на необходимость использования конкретного skill-а в запросе, например: `Use the blazor-component-review skill to analyze \ServicePartners.Client\Components\SanitizedInput.razor`

- Делайте skills узкими. Один навык — одна задача: "review компонентов", "поиск N+1 запросов в EF Core", "генерация тестов для сервисов". Так проще поддерживать и тестировать
- Используйте теги и триггеры. Это повышает точность автовыбора навыка
- Храните skills в репозитории. Папка .opencode/skills должна быть под Git вместе с проектом: это часть "операционной мудрости" вашей команды
- Тестируйте на реальных файлах. Проверьте, что агент действительно следует чеклисту и не выходит за рамки constraints
