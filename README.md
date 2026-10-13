# ML Platform

Небольшой pet-проект с собственными классификаторами и локальным трекингом экспериментов на SQLite.

## Установка

Для работы с классификаторами и tracking API достаточно основных зависимостей:

```bash
uv sync --no-dev
```

Чтобы запустить baseline на SMS Spam Collection, установите optional extra `baseline`:

```bash
uv sync --extra baseline
```

## Датасет и baseline

Датасет не хранится в репозитории: каталог `ml_platform/data/raw/` исключён из Git. Скачайте SMS Spam Collection с [официальной страницы UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), распакуйте файл `SMSSpamCollection` и поместите его сюда:

```text
ml_platform/data/raw/SMSSpamCollection
```

Из корня репозитория запустите обучение и сравнение четырёх значений `alpha` через CLI:

```bash
uv run --extra baseline ml-platform baseline run
```

Путь к датасету можно переопределить через `--data-path`, а путь к JSON-отчёту — через `--report-path`.

Для каждого кандидата создаётся run с validation-метриками. Отдельный финальный run содержит test-метрики выбранной модели и majority baseline. Test-метрики не используются при выборе `alpha`. Сводный JSON-отчёт по-прежнему записывается в `ml_platform/reports/baseline.json`.

## CLI

Получить справку и запустить сохранённое демо классификатора:

```bash
uv run ml-platform --help
uv run ml-platform demo
```

Список runs и подробности конкретного запуска:

```bash
uv run ml-platform runs list --experiment sms-baseline
uv run ml-platform runs list --experiment sms-baseline --format json
uv run ml-platform runs show RUN_ID
uv run ml-platform runs show RUN_ID --format json
```

Текстовый вывод предназначен для чтения человеком; JSON-формат — для скриптов. Команда `baseline run` требует optional extra `baseline`; просмотр runs и запуск `demo` доступны без него.

## Dashboard

Установите UI-extra и запустите локальный dashboard:

```bash
uv sync --extra ui
uv run --extra ui ml-platform ui
```

По умолчанию FastAPI доступен на `http://127.0.0.1:8000` (OpenAPI: `/docs`), а Streamlit — на `http://127.0.0.1:8501`. Порты можно изменить:

```bash
uv run --extra ui ml-platform ui --api-port 8100 --ui-port 8600
```

Команда запускает оба локальных сервиса, открывает dashboard в браузере и останавливает их по Ctrl+C. Для headless-среды используйте `--no-browser`. UI работает только с localhost и предоставляет read-only просмотр: выбор эксперимента, runs, параметров и сравнение метрик. Он использует тот же файл SQLite, который задаётся через `ML_PLATFORM_TRACKING_PATH`.

## Tracking API

По умолчанию SQLite-файл создаётся как `ml_platform.db` в текущей рабочей директории. Путь можно переопределить переменной окружения `ML_PLATFORM_TRACKING_PATH`.

```python
from ml_platform.tracking import (
    get_run_metrics,
    get_run_params,
    list_experiments,
    list_runs,
    log_metric,
    log_param,
    start_run,
)

with start_run("example", "first-run") as run:
    run_id = run.run_id
    log_param("alpha", 0.5)
    log_metric("validation_spam_f1", 0.91)

for run in list_runs("example"):
    print(run.run_name, run.status.value, get_run_params(run.run_id))
    print(get_run_metrics(run.run_id))
```

`list_experiments()` возвращает эксперименты по имени; `list_runs(experiment_name)` возвращает runs выбранного эксперимента от новых к старым. Параметры читаются через `get_run_params(run_id)`, а история скалярных метрик с шагами и timestamp — через `get_run_metrics(run_id)`.

При использовании SMS Spam Collection укажите источник данных UCI в материалах проекта; датасет предоставляется его авторами на условиях, описанных в архиве.
