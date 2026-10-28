"""Последовательные эксперименты Stage 3. CSV и Registry не изменяются."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_improvement"))
sys.path.insert(0, str(ROOT / "mlflow_server"))
from common import configure, redact
configure()
RUNTIME = ROOT.parent / "mle-mlflow-runtime" / "stage3"
RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ.pop("NUMBA_DISABLE_JIT", None)
os.environ["NUMBA_CACHE_DIR"] = str(RUNTIME / "numba")
os.environ["MPLCONFIGDIR"] = str(ROOT.parent / "mle-mlflow-runtime/matplotlib")
import json
import time
import hashlib
import subprocess
import gc
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
import numpy as np
import pandas as pd
import psutil
import joblib
from threadpoolctl import threadpool_limits
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from stage3.features import RAW_FEATURES, BASE_PARAMS, make_pipeline, ManualFeatures
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

URI = "http://127.0.0.1:5000"
TRAIN = ROOT.parent / "mle-baseline-recovery/data/train.csv"
TEST = TRAIN.with_name("test.csv")
RUN_NAMES = dict(baseline="stage3_baseline_grouped", manual="stage3_manual_features",
                 sklearn="stage3_sklearn_features", autofeat="stage3_autofeat")
RESULTS = RUNTIME / "results.json"


def event(message, **kwargs):
    print(json.dumps(dict(event=message, **kwargs), ensure_ascii=False), flush=True)


def save(result):
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def resources(label):
    gc.collect()
    free = psutil.disk_usage(str(ROOT)).free / 2**30
    available = psutil.virtual_memory().available / 2**30
    event("resources", phase=label, disk_gib=round(free, 3), ram_gib=round(available, 3))
    if free < 1:
        raise RuntimeError("Свободно менее 1 ГиБ на C:; операция не запущена")
    if available < .55:
        raise MemoryError("Доступно менее 0,55 ГиБ RAM; операция не запущена")


def metrics(y, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError("Неконечные предсказания")
    return dict(rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                mae=float(mean_absolute_error(y, prediction)), r2=float(r2_score(y, prediction)))


def registry_snapshot(client):
    version = client.get_model_version("real_estate_price_model", "1")
    versions = [(v.version, v.run_id, v.source) for v in client.search_model_versions("name='real_estate_price_model'")]
    run1 = client.get_run(version.run_id)
    run2 = client.get_run("4c473e93c8814486a68005667ca7dc31")
    return dict(version=dict(version), versions=versions,
                run1=run1.to_dictionary(), run2=run2.to_dictionary())


def run_fit(name, X, y, train_idx, valid_idx):
    resources(name)
    model = make_pipeline(name)
    before = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
    fit_time = time.perf_counter() - before
    before = time.perf_counter()
    prediction = model.predict(X.iloc[valid_idx])
    prediction_time = time.perf_counter() - before
    transformed = model.named_steps["preprocess"].transform(X.iloc[valid_idx[:32]])
    assert isinstance(transformed, pd.DataFrame)
    assert transformed["base__building_type_int" if name == "sklearn" else "building_type_int"].dtype.kind in "iu"
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()
    row = dict(config=name, input_features=14, output_features=transformed.shape[1],
               fit_seconds=fit_time, predict_seconds=prediction_time, **metrics(y.iloc[valid_idx], prediction))
    if name == "autofeat":
        block = model.named_steps["preprocess"].named_steps["autofeat"]
        row.update(autofeat_fit_rows=block.fit_rows_, autofeat_formulas=block.formulas_)
    event("fit_done", **row)
    del model, transformed, prediction
    gc.collect()
    return row


def checks(X):
    sample = X.iloc[:12].copy()
    original = sample.copy(deep=True)
    derived = ManualFeatures().fit_transform(sample)
    pd.testing.assert_frame_equal(sample, original)
    assert derived.shape[1] == 19
    assert np.allclose(derived.relative_floor, sample.floor / sample.floors_total)
    for field in ["total_area", "floors_total"]:
        bad = sample.copy(); bad.loc[bad.index[0], field] = 0
        try:
            ManualFeatures().transform(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Нулевой знаменатель не отклонён")
    from sklearn.base import clone
    for name in ["baseline", "manual", "sklearn", "autofeat"]:
        assert clone(make_pipeline(name)) is not None
    ct = make_pipeline("sklearn").named_steps["preprocess"]
    # Только проверка технического интерфейса на train; кандидаты ниже создаются заново.
    transformed = ct.fit_transform(sample)
    assert transformed.shape[1] == 24 and transformed["base__building_type_int"].dtype.kind in "iu"
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()
    event("transformer_checks_passed")


def log_candidate(client, experiment_id, row, result):
    name = row["config"]
    run = client.create_run(experiment_id, tags={"mlflow.runName": RUN_NAMES[name],
        "stage": "stage3", "config": name, "training_performed": "true",
        "historical_test_used_for_selection": "false"})
    run_id = run.info.run_id
    try:
        for key, value in {**BASE_PARAMS, "config": name, "train_md5": result["train_md5"],
                           "split": "GroupShuffleSplit", "split_seed": 42,
                           "inner_train_rows": 90215, "inner_validation_rows": 22519,
                           "group_definition": "exact_14_original_features_without_price",
                           "input_features": 14, "output_features": row["output_features"]}.items():
            client.log_param(run_id, key, value)
        for key in ["rmse", "mae", "r2", "fit_seconds", "predict_seconds"]:
            client.log_metric(run_id, "validation_" + key, row[key])
        for fold in result["cv"]:
            if fold["config"] == name:
                for key in ["rmse", "mae", "r2", "fit_seconds"]:
                    client.log_metric(run_id, "cv_" + key, fold[key], step=fold["fold"])
        path = RUNTIME / (name + "_validation.json")
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        client.log_artifact(run_id, str(path), artifact_path="validation")
        client.set_terminated(run_id, "FINISHED")
    except Exception:
        client.set_terminated(run_id, "FAILED"); raise
    return run_id


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Выполнить эксперименты и финальное логирование")
    args = parser.parse_args()
    if not args.run:
        print("Для обучения и MLflow logging явно укажите --run."); return
    if RESULTS.exists():
        raise RuntimeError("results.json уже существует: проверьте завершённый запуск, не создавайте дубликаты")
    resources("start")
    mlflow.set_tracking_uri(URI); mlflow.set_registry_uri(URI)
    client = MlflowClient(tracking_uri=URI, registry_uri=URI)
    experiment = client.get_experiment_by_name("real_estate_model_improvement")
    if experiment is None or experiment.lifecycle_stage != "active":
        raise RuntimeError("Существующий experiment недоступен")
    expected = (ROOT.parent / "mle-mlflow-runtime/artifacts/real_estate_model_improvement").resolve().as_uri()
    if experiment.artifact_location.rstrip("/") != expected:
        raise RuntimeError("Несовпадение artifact location")
    before_registry = registry_snapshot(client)
    payload = TRAIN.read_bytes()
    train_md5 = hashlib.md5(payload).hexdigest()
    assert train_md5 == "2d4bb3ba2091eb269bc56a93f05f9119"
    frame = pd.read_csv(TRAIN)
    assert frame.shape == (112734, 15)
    X = frame[RAW_FEATURES].copy(); y = frame.price.copy()
    original_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    groups, keys = pd.factorize(pd.MultiIndex.from_frame(X), sort=False)
    train_idx, valid_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
    assert len(train_idx) == 90215 and len(valid_idx) == 22519
    assert not np.intersect1d(groups[train_idx], groups[valid_idx]).size
    split_digest = hashlib.sha256(train_idx.tobytes() + valid_idx.tobytes()).hexdigest()
    result = dict(train_md5=train_md5, train_sha256=hashlib.sha256(payload).hexdigest(),
                  split_sha256=split_digest, groups=len(keys), inner_train_rows=len(train_idx),
                  inner_validation_rows=len(valid_idx), group_overlap=0,
                  selection_rule="baseline_and_best_nonbaseline_3fold_cv; require_0.5pct_rmse_improvement; prefer_simpler",
                  holdout=[], cv=[], runs={}, limitations=[],
                  environment={n: __import__("importlib.metadata", fromlist=["version"]).version(n)
                               for n in ["numpy","pandas","scikit-learn","catboost","mlflow","autofeat","numba"]})
    cutoff = float(y.iloc[train_idx].quantile(.99))
    result["split_diagnostics"] = {}
    for label, idx in [("inner_train",train_idx),("validation",valid_idx)]:
        v = y.iloc[idx]
        result["split_diagnostics"][label] = dict(rows=len(v),mean=float(v.mean()),median=float(v.median()),
              p99=float(v.quantile(.99)),max=float(v.max()),above_training_p99=int((v > cutoff).sum()),
              above_training_p99_pct=float((v > cutoff).mean()*100))
    result["split_diagnostics"]["training_p99_cutoff"] = cutoff
    checks(X)
    del payload
    for name in ["baseline", "manual", "sklearn"]:
        result["holdout"].append(run_fit(name, X, y, train_idx, valid_idx)); save(result)
    # Обычный JIT включён. Изолированный импорт с кешем только в runtime.
    resources("autofeat_import")
    probe = subprocess.run([sys.executable, "-B", "-c",
        "import os; assert not os.environ.get('NUMBA_DISABLE_JIT'); import autofeat; print('AUTOFEAT_IMPORT_OK')"],
        env=os.environ.copy(), capture_output=True, text=True)
    result["autofeat_import"] = dict(returncode=probe.returncode,stdout=probe.stdout.strip(),stderr=probe.stderr[-3000:])
    if probe.returncode:
        result["limitations"].append("autofeat import: " + probe.stderr[-1500:])
    elif psutil.virtual_memory().available / 2**30 < .9:
        result["limitations"].append("autofeat не запущен: доступно менее 0,9 ГиБ RAM")
    else:
        try:
            result["holdout"].append(run_fit("autofeat", X, y, train_idx, valid_idx))
        except Exception as error:
            result["limitations"].append("autofeat не завершён: " + type(error).__name__ + ": " + str(error))
            event("autofeat_failed", error=type(error).__name__, detail=str(error))
    save(result)
    candidate = min([r for r in result["holdout"] if r["config"] != "baseline"],key=lambda r:r["rmse"])["config"]
    for name in ["baseline", candidate]:
        for i,(a,b) in enumerate(GroupKFold(n_splits=3).split(X,y,groups),1):
            assert not np.intersect1d(groups[a],groups[b]).size
            row = run_fit(name,X,y,a,b); row["fold"]=i
            result["cv"].append(row); save(result)
    cv_table = pd.DataFrame(result["cv"]).groupby("config").rmse.mean()
    winner = candidate if cv_table[candidate] < cv_table["baseline"] * .995 else "baseline"
    result["winner"] = winner
    result["winner_holdout"] = next(r for r in result["holdout"] if r["config"] == winner)
    result["cv_summary"] = pd.DataFrame(result["cv"]).groupby("config")[["rmse","mae","r2","fit_seconds"]].agg(["mean","std"]).to_json()
    # Выбор фиксируется на диске ДО первого чтения historical test.
    decision = dict(winner=winner, rule=result["selection_rule"],train_md5=train_md5,
                    split_sha256=split_digest,holdout=result["holdout"],cv=result["cv"])
    lock = RUNTIME / "selection_before_test.json"
    with lock.open("x",encoding="utf-8") as stream: json.dump(decision,stream,ensure_ascii=False,indent=2)
    decision_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    result["selection_sha256"] = decision_sha
    event("selection_locked",winner=winner,sha256=decision_sha)
    resources("final_train")
    final = make_pipeline(winner)
    start=time.perf_counter()
    with threadpool_limits(limits=1):final.fit(X,y)
    result["final_fit_seconds"]=time.perf_counter()-start
    check_input = X.sample(n=128,random_state=123).copy()
    check_prediction = final.predict(check_input)
    assert np.isfinite(check_prediction).all()
    # Единственная итоговая оценка: результат не влияет на конфигурацию.
    test_payload=TEST.read_bytes()
    result["test_md5"]=hashlib.md5(test_payload).hexdigest()
    assert result["test_md5"]=="1d5874c7f4950c457e1ff2678d44a121"
    test=pd.read_csv(TEST)
    start=time.perf_counter(); pred=final.predict(test[RAW_FEATURES])
    result["test_predict_seconds"]=time.perf_counter()-start
    result["test_metrics"]=metrics(test.price,pred)
    result["test_rows"]=len(test)
    assert hashlib.sha256(lock.read_bytes()).hexdigest()==decision_sha
    assert original_hash==hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    assert hashlib.md5(TRAIN.read_bytes()).hexdigest()==train_md5
    assert hashlib.md5(TEST.read_bytes()).hexdigest()==result["test_md5"]
    save(result)
    for row in result["holdout"]:
        result["runs"][RUN_NAMES[row["config"]]]=log_candidate(client,experiment.experiment_id,row,result)
        save(result)
    resources("final_logging")
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name="stage3_feature_generation",
        tags={"stage":"stage3","winner":winner,"registry_registration":"false",
              "test_used_only_after_selection":"true","selection_sha256":decision_sha}) as active:
        final_id=active.info.run_id
        result["runs"]["stage3_feature_generation"]=final_id
        mlflow.log_params({**BASE_PARAMS,"winner":winner,"input_features":14,
            "train_md5":train_md5,"test_md5":result["test_md5"],"selection_sha256":decision_sha,
            "split_sha256":split_digest,"groups":len(keys),"cv_folds":3,
            "inner_train_rows":len(train_idx),"inner_validation_rows":len(valid_idx),
            "group_definition":"exact_14_original_features_without_price",
            "test_not_used_for_selection":True,"serialization_format":"cloudpickle",
            **{n+"_version":v for n,v in result["environment"].items()}})
        mlflow.log_metrics({"test_"+k:v for k,v in result["test_metrics"].items()})
        mlflow.log_metrics({"validation_"+k:result["winner_holdout"][k] for k in ["rmse","mae","r2"]})
        mlflow.log_metric("final_fit_seconds",result["final_fit_seconds"])
        mlflow.sklearn.log_model(final,artifact_path="model",
            code_paths=[str(ROOT/"model_improvement/stage3")],
            signature=infer_signature(check_input,check_prediction),input_example=check_input.head(5),
            pip_requirements=[f"{key}=={value}" for key,value in result["environment"].items()]
                             + ["scipy==1.12.0","joblib==1.3.2","cloudpickle==2.2.1"],
            serialization_format="cloudpickle")
        mlflow.log_artifact(str(lock),artifact_path="validation")
        sample=RUNTIME/"verification_sample.pkl"
        joblib.dump((check_input,check_prediction),sample)
        verify_code = """import sys,json,joblib,numpy as np,mlflow,mlflow.sklearn
mlflow.set_tracking_uri('http://127.0.0.1:5000')
sample,expected=joblib.load(sys.argv[2])
model=mlflow.sklearn.load_model('runs:/'+sys.argv[1]+'/model')
actual=model.predict(sample)
assert np.isfinite(actual).all()
np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-6)
print(json.dumps({'rows':len(actual),'max_abs_difference':float(np.max(np.abs(actual-expected))),'isolated_process':True}))
"""
        verification=subprocess.run([sys.executable,"-I","-B","-c",verify_code,final_id,str(sample)],
            cwd=RUNTIME,env=os.environ.copy(),capture_output=True,text=True)
        result["reload_verification"]=dict(returncode=verification.returncode,
             stdout=verification.stdout.strip(),stderr=verification.stderr[-1500:])
        if verification.returncode:
            raise RuntimeError("Проверка модели в новом процессе не пройдена: "+verification.stderr[-1000:])
        save(result)
        mlflow.log_artifact(str(RESULTS),artifact_path="reports")
    # Не изменяем Registry и предыдущие runs.
    assert registry_snapshot(client)==before_registry
    result["registry_and_stage1_stage2_unchanged"]=True
    result["completed"]=True
    save(result)
    client.log_artifact(final_id,str(RESULTS),artifact_path="reports")
    event("completed",winner=winner,test_metrics=result["test_metrics"],runs=result["runs"],
          reload=result["reload_verification"]["stdout"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        event("error",type=type(error).__name__,message=redact(str(error)))
        raise
