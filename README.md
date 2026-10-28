# Улучшение baseline-модели

Учебный проект посвящён прогнозированию стоимости недвижимости и продолжает работу предыдущего спринта по подготовке данных и обучению CatBoostRegressor. Основная метрика — RMSE. В проекте исследовано влияние генерации и отбора
признаков, а также настройки гиперпараметров. Результаты экспериментов и версии моделей сохранены в MLflow.

## Особенности выполнения проекта

В связи с проблемами доступности учебной облачной инфраструктуры Яндекс Практикума проект был выполнен в локальном окружении. Вместо учебной виртуальной машины использовались Docker Desktop, локально развёрнутые PostgreSQL и
MLflow Tracking Server. Артефакты экспериментов сохранялись в локальном файловом хранилище. Это позволило выполнить все этапы, провести эксперименты и зарегистрировать модели в MLflow Model Registry. Yandex Object Storage (S3)
фактически не использовалось.

## Используемые технологии

Python 3.10, pandas, scikit-learn, CatBoost, autofeat, mlxtend, Optuna, MLflow 2.7.1, PostgreSQL, Docker Desktop и Jupyter Notebook. Для визуализаций использованы Matplotlib и Seaborn.

## Этапы работы

1. **Stage 1 — Baseline.** Исходная CatBoost-модель восстановлена на основе результатов предыдущего спринта, проверена по историческим данным и метрикам и зарегистрирована как V1. Метаданные происхождения сохранены в
   `model_improvement/baseline/`.
2. **Stage 2 — EDA.** Исследованы распределения цены и признаков, корреляции, зависимости, выбросы и возможные факторы стоимости. Выводы сохранены в notebook, отдельном [Markdown-отчёте](model_improvement/eda_conclusions.md) и
   MLflow.
3. **Stage 3 — Feature Generation.** Проверены предметные признаки, `PolynomialFeatures`, `KBinsDiscretizer` и `autofeat`. Преобразования объединены через `ColumnTransformer` и `sklearn.Pipeline`. Для продолжения выбрана
   конфигурация F2 с 24 признаками; модель зарегистрирована как V2.
4. **Stage 4 — Feature Selection.** Выполнены Sequential Forward Selection (SFS) и Sequential Backward Selection (SBS) из `mlxtend`. Выбрана конфигурация с восемью признаками, зарегистрирована V3.
5. **Stage 5 — Hyperparameter Tuning.** Сравнены `RandomizedSearchCV` и Optuna с `TPESampler` и `MLflowCallback`. По групповой кросс-валидации выбрана конфигурация RandomizedSearchCV. Итоговый Pipeline зарегистрирован как V4.

## Результаты

Оценка на historical test:

| Версия | Этап | RMSE, руб. | MAE, руб. | R² |
|---|---|---:|---:|---:|
| V1 | Baseline | 10 547 270,06 | 3 744 622,81 | 0,894899 |
| V2 | Feature Generation | 10 594 103,15 | 3 749 084,73 | 0,893964 |
| V3 | Feature Selection | 11 811 402,85 | 3 795 846,93 | 0,868196 |
| V4 | Hyperparameter Tuning | 10 995 267,83 | 3 533 863,89 | 0,885782 |

По основной метрике RMSE лучший результат сохранился у V1. Настройка гиперпараметров снизила RMSE относительно V3, а V4 получила минимальную MAE среди четырёх моделей. Улучшить исходный baseline по RMSE не удалось: усложнение
обработки признаков не всегда повышает качество прогнозирования.

## Структура проекта

```text
mle-mlflow/
├── mlflow_server/                   # запуск MLflow и логирование Stage 1–2
├── model_improvement/
│   ├── baseline/                    # метаданные исходной модели
│   ├── model_improvement.ipynb       # единый notebook Stage 2–5
│   ├── eda_conclusions.md           # выводы EDA
│   ├── stage3/                      # генерация признаков
│   ├── stage4/                      # отбор признаков
│   └── stage5/                      # настройка гиперпараметров
├── .env_template
├── requirements.txt
└── README.md
```

## Локальный запуск

Команды ниже рассчитаны на настроенное Windows-окружение с PowerShell, Docker Desktop, Python 3.10 и виртуальным окружением `.venv` в корне репозитория. Версии зависимостей указаны в `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Docker Compose находится **вне репозитория**, в `..\mle-local-infra` (в текущем окружении — `C:\Users\egpoz\mle-local-infra`). Для запуска нужны этот каталог с конфигурацией и подготовленные БД и пользователь `mlflow`. Настройки
подключения задаются в локальном `.env` по образцу `.env_template`; секреты в Git не включены.

Если PostgreSQL и MLflow ещё не запущены, из корня репозитория:

```powershell
docker compose -f ..\mle-local-infra\docker-compose.yml up -d postgres
.\mlflow_server\run_server_local.ps1
```

PostgreSQL доступен на `127.0.0.1:5433`. Команда Compose запускает сервер БД, но не восстанавливает историю экспериментов. Launcher MLflow работает в текущем терминале и явно выбирает локальное хранение.

В другом терминале можно проверить сервер и открыть notebook:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
.\.venv\Scripts\python.exe -m jupyterlab model_improvement/model_improvement.ipynb
```

Интерфейс MLflow: [http://127.0.0.1:5000](http://127.0.0.1:5000).

Для повторного выполнения нужны historical `train.csv` и `test.csv` предыдущего спринта, расположенные в `..\mle-baseline-recovery\data`. Train содержит 112 734 строки, test — 28 184. CSV не входят в Git; их происхождение и
контрольные суммы описаны в метаданных baseline. Путь к train для EDA задаётся в начале notebook.

Каталоги `mle-local-infra`, `mle-baseline-recovery` и `mle-mlflow-runtime` не создаются одним клонированием этого репозитория. Для повторного выполнения части ячеек нужны сохранённые runtime JSON или их восстановление из MLflow. В
частности, Stage 5 использует `reports/results.json` и `design/winner_before_C.json`, восстановленные как `results.json` и `winner_before_C.json` в `..\mle-mlflow-runtime\stage5` либо в каталоге `STAGE5_RUNTIME_DIR`.

Готовые таблицы, графики и выводы доступны в сохранённых outputs [основного notebook](model_improvement/model_improvement.ipynb) без повторного обучения. Полные эксперименты Stage 3–5 запускаются отдельными модулями этапов с флагом `--run`;
обычное чтение сохранённых результатов их не повторяет.

## MLflow

- Experiment: `real_estate_model_improvement`.
- Локальный Experiment ID: `1`. Он относится к этому экземпляру PostgreSQL/MLflow и может отличаться в другом окружении.
- Registered Model: `real_estate_price_model`.
- Зарегистрированные версии: `1`, `2`, `3`, `4`, статус `READY`.
- Финальная модель Stage 5: `models:/real_estate_price_model/4`.
- Итоговый Stage 5 Run ID: `52b6764852554b4eb42d49cc5106430a`.

Основные runs: `reconstructed_baseline_v1`, `stage2_eda`, `stage3_feature_generation`, `stage4_feature_selection` и `stage5_hyperparameter_tuning`. В MLflow сохранены параметры, метрики, код, модели, окружения, signatures и отчёты
соответствующих этапов.

Артефакты находятся в `..\mle-mlflow-runtime\artifacts`, история Tracking и Registry — в локальной PostgreSQL. Модели, артефакты и база данных не хранятся в GitHub. Для доступа к текущим `file://` URI нужен тот же локальный
artifact storage.

В конфигурации предусмотрен S3-launcher `mlflow_server/run_server.sh`, но эксперименты проекта выполнены с файловым хранилищем. Сохранённые notebook и отчёты в MLflow являются снимками этапов; сообщения о статусе логирования или
регистрации внутри ранних снимков относятся к моменту их сохранения.

## Ограничения результатов

В данных встречаются повторяющиеся комбинации характеристик объектов, поэтому для внутренних оценок применялась групповая валидация. Historical test имеет известные пересечения с train и уже использовался в истории разработки.
Полученные метрики не являются строгой независимой оценкой качества на совершенно новых объектах. Кроме того, CV Stage 5 относится к набору признаков, ранее выбранному на Development, а не ко всей процедуре отбора и настройки.
