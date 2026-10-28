# Улучшение модели стоимости недвижимости

Учебный проект: исследование данных и улучшение baseline CatBoostRegressor.
MLflow **2.7.1** использует PostgreSQL для Tracking и Model Registry.
Эксперимент: `real_estate_model_improvement`, модель: `real_estate_price_model`.
Version **1** — восстановленный baseline. Stage 2 выполнен; Stage 3–5 впереди.
Stage 2 успешно залогирован в локальном MLflow; run имеет статус **FINISHED**. Baseline Registry Version 1 не изменена.

## Структура и этапы

- **Stage 1:** восстановление baseline и проверка исторических метрик.
- **Stage 2:** [единый notebook](model_improvement/model_improvement.ipynb) и [выводы EDA](model_improvement/eda_conclusions.md).
- **Stage 3:** генерация признаков — предстоит.
- **Stage 4:** отбор признаков — предстоит.
- **Stage 5:** подбор гиперпараметров — предстоит.

## Stage 2: исследовательский анализ

Использован только historical train: **112 734 строки, 15 столбцов**.
Данные читаются из `../mle-baseline-recovery/data/train.csv`; абсолютный путь задаётся
в начале notebook. CSV не копируются в Git. Выполняйте ячейки по порядку в существующем окружении Python 3.10.
Основные таблицы и графики сохранены в notebook.

Проверены пропуски, дубликаты, распределение цены, числовые и категориальные признаки,
география, Pearson/Spearman и гипотезы об относительных площадях, этажах и взаимодействиях.

- Пропусков нет; полных повторов сверх первого вхождения — **5 858**.
- Средняя цена **18,52 млн ₽**, медиана **11,85 млн ₽**, асимметрия **10,17**.
  На объекты выше P99 приходится **73,81%** суммы квадратов отклонений target от среднего.
- Pearson общей и жилой площадей — **0,961**; найдено **4** нарушения суммы площадей.
- Цены различаются по этажам, периодам, типам зданий и местоположению.
  Эти наблюдения не доказывают причинность или прирост качества новых признаков.
- Приоритеты Stage 3: доли площадей, положение этажа, пространственные взаимодействия;
  отдельно проверить логарифмирование target. CatBoost уже умеет учитывать нелинейности.

**Ограничение historical split:** отдельный предыдущий аудит выявил точное совпадение с train
у **2 930 строк test (10,396%)**, по 14 признакам — у **6 299 (22,350%)**.
Историческая оценка может быть оптимистичной. Test в EDA не загружается и не используется для выбора
преобразований; split сохраняется для сопоставимости с baseline. Дубликаты и выбросы не удаляются.

### Логирование Stage 2

Имя run: `stage2_eda`, существующий experiment: `real_estate_model_improvement`.
Локальный MLflow Server восстановлен. Experiment ID: **1**.
Run ID: `4c473e93c8814486a68005667ca7dc31`, статус: **FINISHED**.
[Открыть run](http://127.0.0.1:5000/#/experiments/1/runs/4c473e93c8814486a68005667ca7dc31).

Проверены три artifacts: `stage2/model_improvement.ipynb`, `stage2/eda_conclusions.md`,
`stage2/log_stage2.py`; их SHA256 совпадают с локальными файлами. Baseline Version 1 не изменена.
Notebook и отчёт сохранены как снимок исследования до восстановления сервера: их прежние
сообщения о недоступности MLflow относятся к предыдущей проверке; текущий статус указан здесь.
Снимок сохранён без изменений для соответствия MLflow artifacts.

Для воспроизводимого повторного запуска логирования того же снимка:

```powershell
.\.venv\Scripts\python.exe -B mlflow_server/log_stage2.py
# При необходимости явно указать исторический train:
.\.venv\Scripts\python.exe -B mlflow_server/log_stage2.py --train-path C:\Users\egpoz\mle-baseline-recovery\data\train.csv
```

Скрипт проверяет health, соответствие PostgreSQL/Registry и расположение artifacts,
затем логирует notebook, Markdown-отчёт, исходник логирования, показатели EDA,
параметры исследования и контрольные суммы train. Тот же снимок файлов повторно не создаёт run.
При недоступности сервера останавливается; file-based tracking и новые эксперименты не создаёт.
Модель и baseline Version 1 не меняются.

## Локальная разработка — Stage 1 проверен

Локальные health, PostgreSQL backend, filesystem artifacts, Registry Version 1,
загрузка модели и три исторические метрики проверены **2026-10-26**.
Это сохранённый результат Stage 1, а не утверждение о текущей доступности сервера.

Docker Compose в `../mle-local-infra` сохраняет PostgreSQL и Airflow.
MinIO исключён после нехватки диска; для локальных artifacts используется файловая система.
Существующие Docker volumes и таблицы `flats`/`buildings` сохраняются.

- PostgreSQL: отдельная БД/пользователь `mlflow`, хост `127.0.0.1`, порт `5433`.
- Tracking URI: `http://127.0.0.1:5000`, только loopback. SQLite не используется.
- Artifact root: file URI каталога `../mle-mlflow-runtime/artifacts`, вычисляется из пути checkout.
- Логи, временные файлы и результаты проверок: `../mle-mlflow-runtime`, вне Git.
- Используется существующее окружение Python 3.10 `.venv`.

Сохраните рабочий `.env`. Для новой установки скопируйте `.env_template` и заполните
пароль БД и путь восстановления локально. Не перезаписывайте рабочие секреты.
`MLFLOW_STORAGE_MODE=local` выбирает файловое хранение; локальный launcher явно выбирает local независимо от S3-переменных.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\mlflow_server\run_server_local.ps1
# Совместимый alias: .\mlflow_server\run_server.ps1
# В другом терминале:
Invoke-RestMethod http://127.0.0.1:5000/health
.\.venv\Scripts\python.exe -B mlflow_server/register_baseline.py
# Проверка целостности исходных данных только на чтение:
.\.venv\Scripts\python.exe -B mlflow_server/provision_postgres.py --verify-only
```

Для перезапуска используются существующие миграции: provisioning и обновление схемы не нужны.
Launchers работают на переднем плане. Ранее фоновый запуск сохранил PID в
`../mle-mlflow-runtime/mlflow_server.pid`, логи — в `logs/` этого runtime.
Останавливайте только соответствующие MLflow-процессы; Docker volumes не удаляйте.

При `--no-serve-artifacts` клиенту нужен доступ к тем же локальным путям на Windows.
Расположение artifacts эксперимента хранится в PostgreSQL: смена server default не переносит старые artifacts.
Старый `Default` может ссылаться на прежний S3 root; используйте именованный эксперимент проекта.

## Yandex / S3 — подготовлено, production-проверка не выполнена

Реальные production credentials ещё не проверялись. Launcher `mlflow_server/run_server.sh`
явно выбирает S3 и читает переменные окружения или локальный untracked `.env`.
Для PostgreSQL задаются `MLFLOW_DB_HOST`, `MLFLOW_DB_PORT`, `MLFLOW_DB_NAME=mlflow`,
`MLFLOW_DB_USER=mlflow`, `MLFLOW_DB_PASSWORD`.

```sh
export MLFLOW_STORAGE_MODE=s3
export MLFLOW_ARTIFACT_ROOT=s3://YOUR_BUCKET/YOUR_PREFIX
export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_DEFAULT_REGION=ru-central1
# AWS_ACCESS_KEY_ID и AWS_SECRET_ACCESS_KEY передаются через секретное окружение.
# Также задайте MLFLOW_DB_*, BASELINE_RECOVERY_DIR и нужный MLFLOW_TRACKING_URI.
sh mlflow_server/run_server.sh
.venv/bin/python -B mlflow_server/register_baseline.py
```

Заранее создайте bucket и предоставьте сервисному аккаунту чтение/запись объектов.
Доступ к artifacts прямой, без MLflow artifact proxy. Сервер слушает loopback:
для удалённого доступа нужен аутентифицированный TLS reverse proxy или SSH tunnel.
Переменная `PYTHON` позволяет выбрать совместимый интерпретатор.

Для финального запуска используйте **отдельный production-инстанс PostgreSQL** с выделенной
БД/пользователем `mlflow` и зарегистрируйте Version 1 там. Повторное использование локальной БД
не переносит artifacts в S3. Скрипт регистрации отклоняет несовпадение artifact location
и по умолчанию повторно использует существующую Version 1.
Локальный Stage 2 logger не предназначен для S3; production-проверка остаётся отдельной задачей.

## Восстановленный baseline и проверка Stage 1

Регистрация читает `../mle-baseline-recovery` без изменения файлов и переобучения.
До загрузки доверенного pickle проверяются размеры и MD5. Логируются параметры модели/разбиения,
метрики, provenance, контрольные суммы, исходники, окружения, модель CatBoost,
signature, пример из пяти строк и зависимости модели. CSV и pickle в Git не добавляются.

| Метрика | Точное значение |
|---|---:|
| RMSE | 10547270.062205384 |
| MAE | 3744622.8052026215 |
| R2 | 0.8948994777286887 |

| Файл | Исторический MD5 |
|---|---|
| initial_data.csv | 84045cd0f0e5a6f7a36eb4e2320fa41f |
| train.csv | 2d4bb3ba2091eb269bc56a93f05f9119 |
| test.csv | 1d5874c7f4950c457e1ff2678d44a121 |
| metrics.json | c76d1f590d0cafcb94f4078b26faca6f |
| fitted_model.pkl | ee7aa528bbdcdc207747e7ddc1194182 |

MD5 восстановленного pickle: `bd7cc21e56d9b904832d7c3ed2878e82`.
CSV и метрики совпадают побайтово; pickle отличается. Исторических индивидуальных предсказаний
нет, восстановленная модель воспроизводит агрегированные метрики.
SHA исходного проекта: `e2decc886f6a9a5d633ff6e710ba2a74cf2db6a2`.

Скрипт Stage 1 проверяет связи experiment/run, метрики, параметры, теги, artifacts и signature,
загружает `models:/real_estate_price_model/1`, пересчитывает метрики на 28 184 historical test-строках.
Предсказания должны совпадать с восстановленной моделью; допуски:
`rel_tol=1e-12`, `abs_tol=1e-10`.
Результат: `../mle-mlflow-runtime/stage1_verification.json`.
Это проверка Stage 1, а не использование test в EDA.

Повторный обычный запуск переиспользует Version 1 без нового run/version.
Локальный эксклюзивный lock предотвращает одновременную регистрацию на этой машине;
распределённого lock нет. `--force-new-version` — явный opt-in, для Stage 1 не нужен.

## Файлы и зависимости

`model_improvement/baseline/` содержит небольшие метаданные воспроизводимости.
Корневые `run_mlflow_server*.sh` и `test_mlflow_*.ipynb` — старые упражнения;
используйте launchers из `mlflow_server/`. MinIO-сборки и загрузка его исходников не нужны.

`.gitignore` исключает `.env`, `.venv`, runtime artifacts, CSV, модели, SQLite и checkpoints.
`requirements.txt` фиксирует проверенные прямые зависимости Python 3.10 и выбранные
совместимые зависимости MLflow, но не является полным транзитивным lock-файлом.
Исследовательские библиотеки сохранены для дальнейших этапов.
`psycopg2-binary` нужен backend `postgresql+psycopg2`;
setuptools закреплён из-за использования MLflow 2.7.1 модуля `pkg_resources`.
Собственный загрузчик `.env` не требует python-dotenv; wheel не является runtime-зависимостью.
