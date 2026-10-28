"""Логирование Stage 2 в существующий локальный MLflow 2.7.1 без обучения."""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from common import ROOT, backend_uri, configure, required

URI = "http://127.0.0.1:5000"
EXPERIMENT = "real_estate_model_improvement"
TRAIN_MD5 = "2d4bb3ba2091eb269bc56a93f05f9119"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path)
    args = parser.parse_args()
    # Проверяем сервер до конфигурации и до обращения к данным/БД.
    with urllib.request.urlopen(URI + "/health", timeout=5) as response:
        if response.status != 200 or response.read(64).strip() != b"OK":
            raise RuntimeError("Неожиданный health-ответ")
    configure()
    if required("MLFLOW_TRACKING_URI").rstrip("/") != URI:
        raise ValueError("Ожидается существующий локальный Tracking URI")

    import mlflow
    import numpy as np
    import pandas as pd
    from mlflow.tracking import MlflowClient
    from sqlalchemy import create_engine, text

    if mlflow.__version__ != "2.7.1":
        raise RuntimeError("Требуется существующая версия MLflow 2.7.1")
    client = MlflowClient(tracking_uri=URI, registry_uri=URI)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None or experiment.lifecycle_stage != "active":
        raise RuntimeError("Существующий experiment не найден или неактивен")
    expected_location = (ROOT.parent / "mle-mlflow-runtime" / "artifacts" / EXPERIMENT).resolve().as_uri()
    if experiment.artifact_location.rstrip("/") != expected_location:
        raise RuntimeError("Artifact location не соответствует существующей локальной конфигурации")
    baseline = client.get_model_version("real_estate_price_model", "1")
    # Только SELECT. Не выполняем provisioning, миграции, регистрацию или изменение модели.
    engine = create_engine(backend_uri())
    try:
        with engine.connect() as connection:
            row = connection.execute(text(
                "SELECT experiment_id FROM experiments WHERE name=:name AND lifecycle_stage='active'"
            ), {"name": EXPERIMENT}).first()
            version = connection.execute(text(
                "SELECT run_id FROM model_versions WHERE name=:name AND version=1"
            ), {"name": "real_estate_price_model"}).first()
            if row is None or str(row[0]) != str(experiment.experiment_id):
                raise RuntimeError("Tracking experiment не совпадает с существующей PostgreSQL")
            if version is None or str(version[0]) != str(baseline.run_id):
                raise RuntimeError("Baseline Tracking/Registry не совпадает с PostgreSQL")
    finally:
        engine.dispose()

    train_path = (args.train_path or Path(required("BASELINE_RECOVERY_DIR")) / "data/train.csv").expanduser().resolve()
    notebook = ROOT / "model_improvement/model_improvement.ipynb"
    report = ROOT / "model_improvement/eda_conclusions.md"
    payload = train_path.read_bytes()
    digest = hashlib.md5(payload).hexdigest()
    if digest != TRAIN_MD5:
        raise RuntimeError("Контрольная сумма historical train не совпадает")
    import io
    train = pd.read_csv(io.BytesIO(payload))
    if train.shape != (112734, 15):
        raise RuntimeError("Неожиданная размерность train")
    columns = [c for c in train.columns if c != "price"]
    price = train.price
    q = price.quantile([.95, .99])
    squared = (price - price.mean()) ** 2
    metrics = {
        "train_rows": len(train), "train_columns": len(train.columns),
        "missing_values": int(train.isna().sum().sum()),
        "full_duplicates": int(train.duplicated().sum()),
        "feature_duplicates": int(train.duplicated(columns).sum()),
        "unique_feature_sets": len(train[columns].drop_duplicates()),
        "different_price_groups": int((train.groupby(columns, dropna=False).price.nunique() > 1).sum()),
        "price_min_rub": float(price.min()), "price_max_rub": float(price.max()),
        "price_mean_rub": float(price.mean()), "price_median_rub": float(price.median()),
        "price_std_rub": float(price.std()), "price_skew": float(price.skew()),
        "price_mean_median_ratio": float(price.mean() / price.median()),
        "price_p95_rub": float(q.loc[.95]), "price_p99_rub": float(q.loc[.99]),
        "above_p95_pct": float((price > q.loc[.95]).mean() * 100),
        "above_p99_pct": float((price > q.loc[.99]).mean() * 100),
        "above_p99_squared_deviation_pct": float(squared[price > q.loc[.99]].sum() / squared.sum() * 100),
        "area_sum_violations": int((train.living_area + train.kitchen_area > train.total_area).sum()),
        "area_living_pearson": float(train.total_area.corr(train.living_area)),
    }
    params = {
        "train_path": str(train_path), "train_md5": digest, "train_bytes": len(payload),
        "train_sha256": hashlib.sha256(payload).hexdigest(), "test_loaded": False,
        "geo_sample_n": min(15000, len(train)), "random_state": 42,
        "geo_cell_degrees": .02, "group_min_n": 100, "std_ddof": 1,
        "percentile_rule": "strictly_above", "target": "price",
        "data_changed": False, "training_performed": False,
        "pandas_version": pd.__version__, "numpy_version": np.__version__,
        "mlflow_version": mlflow.__version__,
    }
    files = [notebook, report, Path(__file__).resolve()]
    signature = hashlib.sha256(
        (digest + "".join(hashlib.sha256(f.read_bytes()).hexdigest() for f in files)).encode()
    ).hexdigest()
    prior = client.search_runs([experiment.experiment_id],
                              filter_string="tags.stage = 'stage2_eda'",
                              order_by=["attributes.start_time DESC"], max_results=1000)
    for run in prior:
        if run.data.tags.get("input_signature") == signature:
            if run.info.status != "FINISHED":
                raise RuntimeError("Такой снимок уже имеет незавершённый run; требуется проверка")
            listed = {a.path for a in client.list_artifacts(run.info.run_id, "stage2")}
            if not {"stage2/" + f.name for f in files}.issubset(listed):
                raise RuntimeError("В существующем run отсутствуют artifacts; требуется проверка")
            print(json.dumps({"reused": True, "experiment_id": experiment.experiment_id,
                              "run_id": run.info.run_id, "run_name": "stage2_eda"}, ensure_ascii=False))
            return
    run = client.create_run(experiment.experiment_id, tags={
        "mlflow.runName": "stage2_eda", "stage": "stage2_eda",
        "input_signature": signature, "baseline_version": "1",
        "historical_test_used": "false", "training_performed": "false",
    })
    run_id = run.info.run_id
    try:
        for key, value in params.items():
            client.log_param(run_id, key, value)
        for key, value in metrics.items():
            client.log_metric(run_id, key, float(value))
        for artifact in files:
            client.log_artifact(run_id, str(artifact), artifact_path="stage2")
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    print(json.dumps({"experiment_id": experiment.experiment_id, "run_id": run_id,
                      "run_name": "stage2_eda", "artifacts": ["stage2/" + f.name for f in files]},
                     ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Не печатаем исключения драйвера БД: они могут содержать секреты.
        print("Логирование Stage 2 остановлено (" + type(error).__name__ +
              "). Проверьте доступность существующего сервера и локальную конфигурацию. Run может не быть создан; инфраструктура не запускается.",
              file=sys.stderr)
        sys.exit(2)
