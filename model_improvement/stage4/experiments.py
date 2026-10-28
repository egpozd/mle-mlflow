"""Stage 4: screening, SFS/SBS и фиксированная проверка; без Registry/коммитов."""
import os
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_improvement"))
sys.path.insert(0, str(ROOT / "mlflow_server"))
from common import configure, backend_uri, redact
configure()
RUNTIME = ROOT.parent / "mle-mlflow-runtime/stage4"
RUNTIME.mkdir(parents=True, exist_ok=True)
import time
import json
import hashlib
import gc
import subprocess
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
import numpy as np
import pandas as pd
import psutil
import joblib
import importlib.metadata as metadata
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, get_scorer
from threadpoolctl import threadpool_limits
from catboost import CatBoostRegressor
from mlxtend.feature_selection import SequentialFeatureSelector
from stage3.features import RAW_FEATURES, BASE_PARAMS
from stage4.features import (CATEGORY, CatBoostSubsetAdapter, FixedFeatureSelector,
                             encode_search_frame, make_pipeline, reset_fit_audit, FIT_AUDIT)
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sqlalchemy import create_engine, text

TRAIN = ROOT.parent / "mle-baseline-recovery/data/train.csv"
TEST = TRAIN.with_name("test.csv")
RESULTS = RUNTIME / "results.json"
URI = "http://127.0.0.1:5000"
TOLERANCE = .005


def event(kind, **values):
    print(json.dumps(dict(event=kind, **values), ensure_ascii=False), flush=True)


def save(result):
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def resource_check(phase, min_ram=.55):
    gc.collect()
    available = psutil.virtual_memory().available / 2**30
    free = psutil.disk_usage(str(ROOT)).free / 2**30
    event("resources", phase=phase, ram_gib=round(available, 3), disk_gib=round(free, 3))
    if free < 1 or available < min_ram:
        raise MemoryError("Операция остановлена: требуется >=1 ГиБ диска и >=%.2f ГиБ RAM" % min_ram)
    return dict(ram_gib=available, disk_gib=free)


def score(y, prediction):
    assert np.isfinite(prediction).all()
    return dict(rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                mae=float(mean_absolute_error(y, prediction)), r2=float(r2_score(y, prediction)))


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def make_splits(X, y, groups, pool_size):
    dev, C = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
    pool, _ = next(GroupShuffleSplit(n_splits=1, train_size=min(.95, pool_size/len(dev)),
                                    random_state=31415).split(X.iloc[dev], y.iloc[dev], groups[dev]))
    local_A, local_B = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=2718).split(
        X.iloc[dev[pool]], y.iloc[dev[pool]], groups[dev[pool]]))
    A, B = dev[pool[local_A]], dev[pool[local_B]]
    assert len(dev) == 90215 and len(C) == 22519
    for first, second in [(dev,C),(A,B),(A,C),(B,C)]:
        assert np.intersect1d(groups[first], groups[second]).size == 0
    return dev, A, B, C


def preprocess_selection(X, y, A, B):
    preprocessor = make_pipeline().named_steps["preprocess"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        XA = preprocessor.fit_transform(X.iloc[A], y.iloc[A])
    XB = preprocessor.transform(X.iloc[B])
    assert list(XA.columns) == list(XB.columns) and XA.shape[1] == 24
    assert XA[CATEGORY].dtype.kind in "iu"
    assert np.isfinite(XA.to_numpy(dtype=float)).all() and np.isfinite(XB.to_numpy(dtype=float)).all()
    return preprocessor, XA, XB, [str(w.message) for w in caught]


def screening(XA, yA):
    start = time.perf_counter()
    estimator = CatBoostRegressor(**BASE_PARAMS, cat_features=[CATEGORY])
    with threadpool_limits(limits=1):
        estimator.fit(XA, yA)
    importance = estimator.get_feature_importance(type="PredictionValuesChange")
    ranked = sorted(range(len(importance)), key=lambda i: (-float(importance[i]), i))
    indices = ranked[:16]
    forced = XA.columns.get_loc(CATEGORY) not in indices
    if forced:
        indices[-1] = XA.columns.get_loc(CATEGORY)
    # Сохраняем исходный порядок F2, а не порядок importance.
    columns = tuple(c for i,c in enumerate(XA.columns) if i in indices)
    assert len(columns) == 16 and CATEGORY in columns
    rows = [dict(feature=c, importance=float(importance[i]), selected=c in columns,
                 original_position=i) for i,c in enumerate(XA.columns)]
    return dict(importance=rows, columns=list(columns), forced_category=forced,
                fit_seconds=time.perf_counter()-start, method="PredictionValuesChange")


def adapter_checks(XA, XB, yA, yB, columns):
    numeric = [c for c in columns if c != CATEGORY]
    subset = [numeric[0], CATEGORY, numeric[1]]
    subset = list(dict.fromkeys(subset))
    a = XA.iloc[:1024][subset]
    b = XB.iloc[:256][subset]
    adapter = CatBoostSubsetAdapter(iterations=100)
    clone(adapter).set_params(iterations=100)
    adapter.fit(encode_search_frame(a).to_numpy(), np.asarray(yA)[:len(a)])
    direct = CatBoostRegressor(**{**BASE_PARAMS,"iterations":100},cat_features=[CATEGORY])
    direct.fit(a,np.asarray(yA)[:len(a)])
    np.testing.assert_allclose(adapter.predict(encode_search_frame(b).to_numpy()), direct.predict(b),
                               rtol=1e-12,atol=1e-6)
    # Категория переносится на любую позицию; признак обязателен для каждого подмножества.
    rotated = subset[::-1]
    clone(adapter).fit(encode_search_frame(a[rotated]).to_numpy(),np.asarray(yA)[:len(a)])
    negative = get_scorer("neg_root_mean_squared_error")(
        adapter,encode_search_frame(b).to_numpy(),np.asarray(yB)[:len(b)])
    expected = -score(np.asarray(yB)[:len(b)],adapter.predict(encode_search_frame(b).to_numpy()))["rmse"]
    assert np.isclose(negative,expected) and negative <= 0
    combined = pd.concat([encode_search_frame(a),encode_search_frame(b)],ignore_index=True)
    yy = np.r_[np.asarray(yA)[:len(a)],np.asarray(yB)[:len(b)]]
    cv = [(np.arange(len(a)),np.arange(len(a),len(combined)))]
    smoke = SequentialFeatureSelector(CatBoostSubsetAdapter(iterations=100),
        forward=True,floating=False,k_features=(2,3),fixed_features=(subset.index(CATEGORY),),
        cv=cv,scoring="neg_root_mean_squared_error",n_jobs=1,verbose=0)
    smoke.fit(combined,yy)
    assert all(subset.index(CATEGORY) in v["feature_idx"] for v in smoke.subsets_.values())
    selector = FixedFeatureSelector(tuple(columns)).fit(XA)
    original = XB.iloc[:5].copy(deep=True)
    transformed = selector.transform(original)
    pd.testing.assert_frame_equal(original,XB.iloc[:5])
    assert transformed[CATEGORY].dtype.kind in "iu"
    clone(make_pipeline(tuple(columns)))
    assert "model__depth" in make_pipeline(tuple(columns)).get_params()
    event("adapter_checks_passed",direct_prediction_max_difference=0,scoring_sign="negative_RMSE")


def benchmark(XA, yA, columns):
    rows = []
    for iterations in [100,100,150]:
        resource_check("benchmark_"+str(iterations))
        model = CatBoostSubsetAdapter(iterations=iterations)
        start=time.perf_counter()
        model.fit(encode_search_frame(XA[list(columns)]).to_numpy(),yA)
        rows.append(dict(iterations=iterations,fit_seconds=time.perf_counter()-start,rows=len(XA)))
        event("benchmark",**rows[-1])
    # Берём минимальное разрешённое число итераций для ограниченного последовательного поиска.
    estimate = float(np.mean([v["fit_seconds"] for v in rows if v["iterations"]==100])) * 204
    return dict(runs=rows,search_iterations=100,estimated_search_fits=204,
                estimated_search_seconds=estimate,max_planned_search_seconds=1800)


def fit_progress(phase, count, fit_seconds):
    # Только метаданные, без копирования матриц/данных.
    with (RUNTIME/"search_fit_log.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(dict(method=phase,count=count,fit_seconds=fit_seconds))+"\n")
    if count == 1 or count % 5 == 0:
        event("search_progress",method=phase,fits=count,last_fit_seconds=round(fit_seconds,3))
        resource_check(phase+"_progress",min_ram=.35)


def run_selection(method, XA, XB, yA, yB, columns, iterations):
    combined = pd.concat([encode_search_frame(XA[list(columns)]),
                          encode_search_frame(XB[list(columns)])],ignore_index=True)
    target = np.r_[np.asarray(yA),np.asarray(yB)]
    split = [(np.arange(len(XA)),np.arange(len(XA),len(combined)))]
    estimator = CatBoostSubsetAdapter(iterations=iterations)
    selector = SequentialFeatureSelector(
        estimator,k_features=(8,12),forward=(method=="sfs"),floating=False,
        fixed_features=(list(columns).index(CATEGORY),),scoring="neg_root_mean_squared_error",
        cv=split,n_jobs=1,verbose=0,clone_estimator=True)
    reset_fit_audit(method,fit_progress)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        selector.fit(combined,target)
    elapsed=time.perf_counter()-start
    trajectory=[]
    for k,entry in sorted(selector.subsets_.items()):
        names=[columns[i] for i in entry["feature_idx"]]
        assert CATEGORY in names and np.isfinite(entry["avg_score"]) and entry["avg_score"] <= 0
        trajectory.append(dict(method=method,k=int(k),features=names,
            rmse=float(-entry["avg_score"]),negative_score=float(entry["avg_score"]),
            cv_scores=[float(s) for s in entry["cv_scores"]]))
    count=FIT_AUDIT["count"]
    reset_fit_audit("idle")
    return dict(method=method,fit_count=count,elapsed_seconds=elapsed,
                trajectory=trajectory,best_mlxtend_k=int(len(selector.k_feature_idx_)),
                best_mlxtend_score=float(-selector.k_score_))


def validation_fit(label, columns, X, y, a, b):
    resource_check(label)
    pipeline=make_pipeline(None if columns is None else tuple(columns))
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        pipeline.fit(X.iloc[a],y.iloc[a])
    fit_seconds=time.perf_counter()-start
    start=time.perf_counter();pred=pipeline.predict(X.iloc[b])
    predict_seconds=time.perf_counter()-start
    transformed=pipeline.named_steps["preprocess"].transform(X.iloc[b[:32]])
    assert transformed.shape[1]==24
    final_input=pipeline.named_steps["selection"].transform(transformed) if columns else transformed
    assert final_input[CATEGORY].dtype.kind in "iu" and np.isfinite(final_input.to_numpy(dtype=float)).all()
    row=dict(config=label,features=final_input.shape[1],fit_seconds=fit_seconds,
             predict_seconds=predict_seconds,warnings=[str(w.message) for w in caught],
             **score(y.iloc[b],pred))
    event("validation_fit",**{k:v for k,v in row.items() if k!="warnings"})
    return row


def protected_snapshot(client):
    versions={v.version:dict(client.get_model_version("real_estate_price_model",v.version))
              for v in client.search_model_versions("name='real_estate_price_model'")}
    assert set(versions)=={"1","2"},"Ожидались только Registry Versions 1 и 2"
    protected_ids=[versions["1"]["run_id"],"4c473e93c8814486a68005667ca7dc31",
        "9ecc43b535844bd9a6b3d4197d73eb57","8e941f8cb1a74c1daab595d24da906d6",
        "fd6641b87e484d2e80307f7041665191","b4f74d76c869431fb9a94b323b2961f8",
        "3fd3fc80f8254c8b93437c4e18b930f8"]
    return dict(versions=versions,runs={rid:client.get_run(rid).to_dictionary() for rid in protected_ids})


def log_search(client, exp_id, name, result, payload):
    run=client.create_run(exp_id,tags={"mlflow.runName":name,"stage":"stage4",
                                      "historical_test_used_for_selection":"false"})
    rid=run.info.run_id
    try:
        params=dict(selection_A_rows=result["splits"]["A"]["rows"],selection_B_rows=result["splits"]["B"]["rows"],
            train_md5=result["train_md5"],category=CATEGORY,search_iterations=100,search_depth=8,
            search_learning_rate=.05,search_seed=42,thread_count=2,n_jobs=1,cv="single_explicit_A_to_B",
            scoring="neg_root_mean_squared_error",tolerance=TOLERANCE)
        for k,v in params.items():client.log_param(rid,k,v)
        if name=="stage4_feature_screening":
            client.log_param(rid,"screening_importance","PredictionValuesChange")
            client.log_param(rid,"screening_iterations",300)
            client.log_param(rid,"screened_features",json.dumps(result["screening"]["columns"]))
            client.log_metric(rid,"screening_fit_seconds",result["screening"]["fit_seconds"])
        else:
            client.log_param(rid,"forward",payload["method"]=="sfs")
            client.log_param(rid,"floating",False)
            client.log_param(rid,"k_features","(8,12)")
            client.log_param(rid,"fixed_features",CATEGORY)
            client.log_metric(rid,"fit_count",payload["fit_count"])
            client.log_metric(rid,"elapsed_seconds",payload["elapsed_seconds"])
            for row in payload["trajectory"]:
                client.log_metric(rid,"selection_B_rmse",row["rmse"],step=row["k"])
        path=RUNTIME/(name+".json")
        path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
        client.log_artifact(rid,str(path),artifact_path="selection")
        client.set_terminated(rid,"FINISHED")
    except Exception:
        client.set_terminated(rid,"FAILED");raise
    return rid


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if not args.run:
        print("Для запуска обучения и отбора явно укажите --run.");return
    if RESULTS.exists():
        raise RuntimeError("Stage4 results.json уже существует; не перезаписываем результаты")
    start_all=time.perf_counter()
    resources=resource_check("start")
    mlflow.set_tracking_uri(URI);mlflow.set_registry_uri(URI)
    client=MlflowClient(tracking_uri=URI,registry_uri=URI)
    experiment=client.get_experiment_by_name("real_estate_model_improvement")
    assert experiment and experiment.experiment_id=="1"
    engine=create_engine(backend_uri())
    try:
        with engine.connect() as connection:
            row=connection.execute(text("SELECT experiment_id FROM experiments WHERE name=:name"),
                                   {"name":experiment.name}).first()
            assert row and str(row[0])==experiment.experiment_id
    finally:engine.dispose()
    protected=protected_snapshot(client)
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/"README.md",*list((ROOT/"model_improvement/stage3").glob("*.py"))]}
    payload=TRAIN.read_bytes()
    assert hashlib.md5(payload).hexdigest()=="2d4bb3ba2091eb269bc56a93f05f9119"
    train=pd.read_csv(TRAIN);X=train[RAW_FEATURES].copy();y=train.price.copy()
    original_digest=hashlib.sha256(pd.util.hash_pandas_object(train,index=True).values.tobytes()).hexdigest()
    groups,keys=pd.factorize(pd.MultiIndex.from_frame(X),sort=False)
    assert len(keys)==99839
    pool_size=50000 if resources["ram_gib"]>=1.5 else 25000
    dev,A,B,C=make_splits(X,y,groups,pool_size)
    result=dict(stage="stage4",completed=False,train_md5=hashlib.md5(payload).hexdigest(),
        train_sha256=hashlib.sha256(payload).hexdigest(),group_definition="exact_14_original_features_without_price",
        group_count=len(keys),outer_seed=42,pool_seed=31415,selection_seed=2718,
        tolerance=TOLERANCE,selection_rule="SFS_SBS_minimum_k_within_0.5pct_of_best_B_then_lower_RMSE",
        resource_start=resources,search_iterations=100,screening={},search={},validation=[],cv=[],runs={},
        environment={n:metadata.version(n) for n in ["numpy","pandas","scikit-learn","catboost","mlflow",
                     "mlxtend","joblib","scipy","cloudpickle","psutil","threadpoolctl"]})
    pre,XA,XB,pre_warnings=preprocess_selection(X,y,A,B)
    screen=screening(XA,y.iloc[A]);columns=tuple(screen["columns"])
    adapter_checks(XA,XB,y.iloc[A],y.iloc[B],columns)
    bench=benchmark(XA,y.iloc[A],columns)
    if bench["estimated_search_seconds"]>1800 and pool_size>25000:
        event("reduce_selection_pool",reason="estimated_search_over_30_minutes",
              estimated_seconds=bench["estimated_search_seconds"])
        del pre,XA,XB
        dev,A,B,C=make_splits(X,y,groups,25000)
        pre,XA,XB,pre_warnings=preprocess_selection(X,y,A,B)
        screen=screening(XA,y.iloc[A]);columns=tuple(screen["columns"])
        bench["initial_pool_reduced"]=True
        bench["initial_estimate_seconds"]=bench["estimated_search_seconds"]
        bench["reduced_benchmark"]=benchmark(XA,y.iloc[A],columns)
        if bench["reduced_benchmark"]["estimated_search_seconds"]>1800:
            raise RuntimeError("Даже уменьшенный A/B превышает плановый бюджет поиска 30 минут")
    result["splits"]={}
    cutoff=float(y.iloc[A].quantile(.99))
    for name,idx in [("Development",dev),("A",A),("B",B),("C",C)]:
        if name == "C":
            result["splits"][name]=dict(rows=len(idx),groups=len(np.unique(groups[idx])),indices_sha256=digest(idx))
            continue
        values=y.iloc[idx]
        result["splits"][name]=dict(rows=len(idx),groups=len(np.unique(groups[idx])),
            indices_sha256=digest(idx),mean=float(values.mean()),median=float(values.median()),
            p99=float(values.quantile(.99)),max=float(values.max()),
            above_A_p99_pct=float((values>cutoff).mean()*100))
    result["splits"]["A_p99_threshold"]=cutoff
    result["group_overlap"]={"A_B":0,"A_C":0,"B_C":0,"Development_C":0}
    result["screening"]=screen;result["benchmark"]=bench;result["preprocessing_warnings"]=pre_warnings
    result["feature_order_24"]=list(XA.columns)
    save(result);event("selection_design_fixed",splits=result["splits"],benchmark=bench)
    for label,cols in [("F2_24",tuple(XA.columns)),("screened_16",columns)]:
        estimator=CatBoostSubsetAdapter(iterations=100)
        before=time.perf_counter()
        estimator.fit(encode_search_frame(XA[list(cols)]).to_numpy(),y.iloc[A])
        pred=estimator.predict(encode_search_frame(XB[list(cols)]).to_numpy())
        result.setdefault("selection_references",[]).append(dict(config=label,k=len(cols),
            fit_seconds=time.perf_counter()-before,**score(y.iloc[B],pred)))
    result["runs"]["stage4_feature_screening"]=log_search(client,experiment.experiment_id,
                                                       "stage4_feature_screening",result,screen)
    save(result)
    for method in ["sfs","sbs"]:
        resource_check(method+"_start")
        result["search"][method]=run_selection(method,XA,XB,y.iloc[A],y.iloc[B],columns,100)
        result["runs"]["stage4_"+method]=log_search(client,experiment.experiment_id,
                                                   "stage4_"+method,result,result["search"][method])
        save(result);event("search_completed",method=method,
            fit_count=result["search"][method]["fit_count"],
            seconds=result["search"][method]["elapsed_seconds"])
    candidates=[row for entry in result["search"].values() for row in entry["trajectory"] if 8<=row["k"]<=12]
    best_rmse=min(row["rmse"] for row in candidates)
    eligible=[row for row in candidates if row["rmse"]<=best_rmse*(1+TOLERANCE)]
    chosen=min(eligible,key=lambda row:(row["k"],row["rmse"],row["method"]))
    selected=tuple(chosen["features"])
    result["selected"]=chosen
    result["selection_best_rmse"]=best_rmse
    lock=RUNTIME/"selection_before_C.json"
    decision=dict(selected=chosen,tolerance=TOLERANCE,best_selection_rmse=best_rmse,
        feature_order_24=list(XA.columns),screened_16=list(columns),base_params=BASE_PARAMS,
        train_md5=result["train_md5"],splits=result["splits"],
        selection_rule=result["selection_rule"])
    with lock.open("x",encoding="utf-8") as stream:json.dump(decision,stream,ensure_ascii=False,indent=2)
    result["selection_sha256"]=hashlib.sha256(lock.read_bytes()).hexdigest()
    save(result);event("selection_locked_before_C",selected=chosen)
    c_values=y.iloc[C]
    result["validation_C_target_diagnostics"]=dict(mean=float(c_values.mean()),median=float(c_values.median()),p99=float(c_values.quantile(.99)),max=float(c_values.max()),above_A_p99_pct=float((c_values>cutoff).mean()*100))
    del pre,XA,XB
    for label,cols in [("F2_reference",None),("Stage4_fixed",selected)]:
        result["validation"].append(validation_fit(label,cols,X,y,dev,C));save(result)
    ref=next(row for row in result["validation"] if row["config"]=="F2_reference")
    cand=next(row for row in result["validation"] if row["config"]=="Stage4_fixed")
    result["validation_rmse_change_pct"]=(cand["rmse"]/ref["rmse"]-1)*100
    for label,cols in [("F2_reference",None),("Stage4_fixed",selected)]:
        for fold,(a,b) in enumerate(GroupKFold(n_splits=3).split(X,y,groups),1):
            assert not np.intersect1d(groups[a],groups[b]).size
            row=validation_fit(label,cols,X,y,a,b);row["fold"]=fold
            result["cv"].append(row);save(result)
    # Никакого повторного выбора по C/CV; сверяем неизменность фиксированного решения.
    assert hashlib.sha256(lock.read_bytes()).hexdigest()==result["selection_sha256"]
    final=make_pipeline(selected);resource_check("final_full_train")
    before=time.perf_counter()
    with warnings.catch_warnings(record=True),threadpool_limits(limits=1):
        final.fit(X,y)
    result["final_fit_seconds"]=time.perf_counter()-before
    transformed=final.named_steps["preprocess"].transform(X)
    assert list(transformed.columns)==result["feature_order_24"]
    chosen_frame=final.named_steps["selection"].transform(transformed)
    assert list(chosen_frame)==list(selected) and chosen_frame[CATEGORY].dtype.kind in "iu"
    assert np.isfinite(chosen_frame.to_numpy(dtype=float)).all()
    check_input=X.sample(n=128,random_state=123).copy()
    expected=final.predict(check_input);assert np.isfinite(expected).all()
    # Единственное обращение к historical test после фиксации подмножества.
    test_md5=hashlib.md5(TEST.read_bytes()).hexdigest()
    assert test_md5=="1d5874c7f4950c457e1ff2678d44a121"
    test=pd.read_csv(TEST)
    before=time.perf_counter();pred=final.predict(test[RAW_FEATURES])
    result["test_predict_seconds"]=time.perf_counter()-before
    result["test_metrics"]=score(test.price,pred);result["test_md5"]=test_md5
    result["historical_comparison"]={"Version1":{"rmse":10547270.062205384,"mae":3744622.8052026215,"r2":.8948994777286887},
        "Version2":{"rmse":10594103.150423475,"mae":3749084.7274927297,"r2":.8939640489409297}}
    assert original_digest==hashlib.sha256(pd.util.hash_pandas_object(train,index=True).values.tobytes()).hexdigest()
    assert hashlib.md5(TRAIN.read_bytes()).hexdigest()==result["train_md5"]
    assert hashlib.md5(TEST.read_bytes()).hexdigest()==test_md5
    save(result)
    mlflow.set_experiment(experiment_id=experiment.experiment_id)
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name="stage4_feature_selection",
        tags={"stage":"stage4","selection_method":chosen["method"],"test_used_for_selection":"false",
              "selection_sha256":result["selection_sha256"],"registry_registration":"false"}) as active:
        rid=active.info.run_id;result["runs"]["stage4_feature_selection"]=rid
        mlflow.log_params({**BASE_PARAMS,"selected_k":chosen["k"],"selection_method":chosen["method"],
            "selected_features":json.dumps(list(selected)),"train_md5":result["train_md5"],
            "test_md5":test_md5,"selection_sha256":result["selection_sha256"],
            "search_iterations":100,"selection_tolerance":TOLERANCE,"category":CATEGORY,
            "validation":"group_aware_A_B_C","outer_seed":42,"pool_seed":31415,"selection_seed":2718})
        mlflow.log_metrics({"validation_"+k:cand[k] for k in ["rmse","mae","r2","fit_seconds","predict_seconds"]})
        mlflow.log_metric("validation_rmse_change_pct",result["validation_rmse_change_pct"])
        mlflow.log_metrics({"test_"+k:v for k,v in result["test_metrics"].items()})
        mlflow.log_metric("final_fit_seconds",result["final_fit_seconds"])
        for label in ["F2_reference","Stage4_fixed"]:
            rows=pd.DataFrame([v for v in result["cv"] if v["config"]==label])
            prefix="" if label=="Stage4_fixed" else "reference_"
            mlflow.log_metrics({prefix+"cv_rmse_mean":float(rows.rmse.mean()),
                prefix+"cv_rmse_std":float(rows.rmse.std(ddof=1)),
                prefix+"cv_mae_mean":float(rows.mae.mean()),prefix+"cv_r2_mean":float(rows.r2.mean())})
        requirements=[name+"=="+version for name,version in result["environment"].items()]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="Hint: Inferred schema contains integer column.*")
            mlflow.sklearn.log_model(final,artifact_path="model",
                code_paths=[str(ROOT/"model_improvement/stage3"),str(ROOT/"model_improvement/stage4")],
                signature=infer_signature(check_input,expected),input_example=check_input.head(5),
                pip_requirements=requirements,serialization_format="cloudpickle")
        mlflow.log_artifact(str(lock),artifact_path="selection")
        sample=RUNTIME/"verification_sample.pkl";joblib.dump((check_input,expected),sample)
        code="""import sys,json,joblib,numpy as np,mlflow,mlflow.sklearn,mlflow.pyfunc
mlflow.set_tracking_uri('http://127.0.0.1:5000')
X,expected=joblib.load(sys.argv[2])
uri='runs:/'+sys.argv[1]+'/model'
model=mlflow.sklearn.load_model(uri);pyfunc=mlflow.pyfunc.load_model(uri)
a=model.predict(X);b=pyfunc.predict(X)
np.testing.assert_allclose(a,expected,rtol=1e-12,atol=1e-6)
np.testing.assert_allclose(b,expected,rtol=1e-12,atol=1e-6)
assert np.isfinite(a).all() and len(model.named_steps['selection'].columns)==int(sys.argv[3])
assert model.named_steps['selection'].__class__.__name__=='FixedFeatureSelector'
print(json.dumps({'sklearn':True,'pyfunc':True,'rows':len(X),'input_features':X.shape[1],'selected_features':len(model.named_steps['selection'].columns),'max_abs_difference':float(np.max(np.abs(a-expected))),'pyfunc_max_abs_difference':float(np.max(np.abs(b-expected))),'isolated_process':True}))
"""
        process=subprocess.run([sys.executable,"-I","-B","-c",code,rid,str(sample),str(len(selected))],
            cwd=RUNTIME,env=os.environ.copy(),capture_output=True,text=True)
        result["reload_verification"]=dict(returncode=process.returncode,stdout=process.stdout.strip(),stderr=process.stderr[-1500:])
        if process.returncode:
            raise RuntimeError("Изолированная загрузка Stage4 не пройдена: "+process.stderr[-1000:])
        save(result);mlflow.log_artifact(str(RESULTS),artifact_path="reports")
    assert protected_snapshot(client)==protected
    assert source_hashes=={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in source_hashes}
    result["protected_versions_and_runs_unchanged"]=True
    result["completed"]=True;result["total_seconds"]=time.perf_counter()-start_all
    result["resource_end"]=resource_check("completed")
    save(result);client.log_artifact(rid,str(RESULTS),artifact_path="reports")
    event("completed",selected=chosen,test=result["test_metrics"],runs=result["runs"],total_seconds=result["total_seconds"])


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        event("error",type=type(error).__name__,message=redact(str(error)))
        raise
