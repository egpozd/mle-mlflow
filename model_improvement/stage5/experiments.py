"""Stage 5: равный budget RandomizedSearchCV/Optuna; Registry не изменяется."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"model_improvement"));sys.path.insert(0,str(ROOT/"mlflow_server"))
from common import configure as configure_environment,backend_uri,redact
configure_environment()
RUNTIME=ROOT.parent/"mle-mlflow-runtime/stage5"
RUNTIME.mkdir(parents=True,exist_ok=True)
import json
import time
import gc
import hashlib
import subprocess
import warnings
warnings.filterwarnings("ignore",message="pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore",message="Bins whose width are too small.*")
import numpy as np
import pandas as pd
import joblib
import importlib.metadata as metadata
from scipy.stats import randint,loguniform
from sklearn.base import clone
from sklearn.model_selection import GroupKFold,RandomizedSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.pipeline import Pipeline
from stage3.features import RAW_FEATURES
from stage4.features import make_pipeline as stage4_pipeline,CATEGORY
from stage4.experiments import make_splits
from stage5.monitoring import (AuditedPipeline,CONTEXT,ALL_RECORDS,configure as audit_config,
                               checked_fit,index_hash,resources)
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import optuna
import optuna.importance
from optuna.integration import MLflowCallback
from optuna.integration.mlflow import RUN_ID_ATTRIBUTE_KEY
warnings.filterwarnings("ignore",category=optuna.exceptions.ExperimentalWarning)
from sqlalchemy import create_engine,text

URI="http://127.0.0.1:5000"
TRAIN=ROOT.parent/"mle-baseline-recovery/data/train.csv"
TEST=TRAIN.with_name("test.csv")
RESULTS=RUNTIME/"results.json"
SELECTED=("base__total_area","base__build_year","base__building_type_int","base__latitude",
          "base__longitude","base__flats_count","interactions__total_area rooms","bins__ceiling_height")
SPACE={"iterations":[200,300,400,500,600],"depth":{"low":5,"high_inclusive":9},
       "learning_rate":{"low":.025,"high":.12,"log":True},
       "l2_leaf_reg":{"low":1.0,"high":12.0,"log":True}}
TUNED_KEYS=("iterations","depth","learning_rate","l2_leaf_reg")
CONSTANTS={"loss_function":"RMSE","random_seed":42,"thread_count":2,"verbose":False,
           "allow_writing_files":False,"random_strength":1}
TRACE=RUNTIME/"fit_trace.jsonl"


def event(kind,**values):
    print(json.dumps(dict(event=kind,**values),ensure_ascii=False),flush=True)


def plain(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [plain(v) for v in value]
    return value


def save(result):
    RESULTS.write_text(json.dumps(plain(result),ensure_ascii=False,indent=2),encoding="utf-8")


def pipeline(parameters=None,audited=True):
    base=stage4_pipeline(SELECTED)
    settings={**CONSTANTS,"l2_leaf_reg":3.0}
    settings.update(parameters or {})
    assert set(parameters or {}).issubset(TUNED_KEYS)
    base.set_params(**{"model__"+k:v for k,v in settings.items()})
    assert tuple(base.named_steps["selection"].columns)==SELECTED
    return AuditedPipeline(base.steps) if audited else base


def distributions():
    return {"model__iterations":SPACE["iterations"],"model__depth":randint(5,10),
            "model__learning_rate":loguniform(.025,.12),"model__l2_leaf_reg":loguniform(1,12)}


def validate_parameters(parameters):
    assert set(parameters)==set(TUNED_KEYS)
    assert parameters["iterations"] in SPACE["iterations"]
    assert 5<=parameters["depth"]<=9
    assert .025<=parameters["learning_rate"]<=.12
    assert 1<=parameters["l2_leaf_reg"]<=12


def metrics(y,pred):
    assert np.isfinite(pred).all()
    return dict(rmse=float(np.sqrt(mean_squared_error(y,pred))),
                mae=float(mean_absolute_error(y,pred)),r2=float(r2_score(y,pred)))


def evaluate_cv(parameters,X,y,folds,train_hashes,valid_hashes,phase,reset=True):
    if reset:audit_config(phase,TRACE,train_hashes,valid_hashes)
    before_count=CONTEXT["count"];started=time.perf_counter();rows=[]
    for fold,(a,b) in enumerate(folds,1):
        model=clone(pipeline(parameters))
        model.fit(X.iloc[a],y.iloc[a])
        pred=model.predict(X.iloc[b])
        record=CONTEXT["records"][-1]
        rows.append(dict(fold=fold,train_rows=len(a),validation_rows=len(b),
                         fit_seconds=record["fit_seconds"],**metrics(y.iloc[b],pred)))
        del model,pred;gc.collect()
    assert CONTEXT["count"]-before_count==3
    return dict(parameters=plain(parameters or {"iterations":300,"depth":8,"learning_rate":.05,"l2_leaf_reg":3.0}),
        folds=rows,mean_rmse=float(np.mean([v["rmse"] for v in rows])),
        std_rmse=float(np.std([v["rmse"] for v in rows],ddof=0)),
        mean_mae=float(np.mean([v["mae"] for v in rows])),
        mean_r2=float(np.mean([v["r2"] for v in rows])),
        mean_fit_seconds=float(np.mean([v["fit_seconds"] for v in rows])),
        wall_seconds=time.perf_counter()-started)


def protected_snapshot(client,experiment_id,ids=None):
    versions={v.version:dict(client.get_model_version("real_estate_price_model",v.version))
              for v in client.search_model_versions("name='real_estate_price_model'")}
    assert set(versions)=={"1","2","3"},"Ожидались только Versions 1–3"
    if ids is None:ids=[r.info.run_id for r in client.search_runs([experiment_id],max_results=5000)]
    return dict(versions=versions,runs={rid:client.get_run(rid).to_dictionary() for rid in ids})


def log_cv(values,prefix=""):
    mlflow.log_metrics({prefix+"cv_rmse_mean":values["mean_rmse"],prefix+"cv_rmse_std":values["std_rmse"],
        prefix+"cv_mae_mean":values["mean_mae"],prefix+"cv_r2_mean":values["mean_r2"]})
    for row in values["folds"]:
        for key in ["rmse","mae","r2","fit_seconds"]:
            mlflow.log_metric(prefix+"fold_"+key,row[key],step=row["fold"])


def cv_metadata(folds,dev,groups):
    rows=[]
    for i,(a,b) in enumerate(folds,1):
        assert np.intersect1d(groups[dev[a]],groups[dev[b]]).size==0
        rows.append(dict(fold=i,train_rows=len(a),validation_rows=len(b),
                         train_indices_sha256=index_hash(dev[a]),validation_indices_sha256=index_hash(dev[b])))
    return rows


def create_benchmark(X,y,folds):
    audit_config("benchmark",TRACE)
    settings={"iterations":400,"depth":7,"learning_rate":float(np.sqrt(.025*.12)),
              "l2_leaf_reg":float(np.sqrt(12))}
    before=resources();a,b=folds[0];model=pipeline(settings)
    model.fit(X.iloc[a],y.iloc[a]);pred=model.predict(X.iloc[b[:128]])
    assert np.isfinite(pred).all()
    record=CONTEXT["records"][0]
    bench=dict(resources_before=before,parameters=settings,record=record,
               prediction_finite=True,no_benchmark_quality_metric_used=True)
    seconds=record["fit_seconds"]
    estimate18=seconds*108*1.5;estimate12=seconds*72*1.5
    budget=18 if estimate18<=3600 else 12 if estimate12<=3600 else 0
    design=dict(n_configurations_each=budget,n_splits=3,tpe_startup_trials=6,
        projected_18_seconds=estimate18,projected_12_seconds=estimate12,safety_multiplier=1.5,
        planned_search_budget_seconds=3600,criterion="18_if_projected_<=60min_else_12_if_safe; no score involved",
        decision_before_search=True,random_seed=42,std_ddof=0)
    (RUNTIME/"benchmark.json").write_text(json.dumps(bench,indent=2),encoding="utf-8")
    with (RUNTIME/"budget_before_search.json").open("x",encoding="utf-8") as stream:json.dump(design,stream,indent=2)
    return bench,design


def choose_winner(randomized,optuna_best):
    candidates=[dict(method="randomized",**randomized),dict(method="optuna",**optuna_best)]
    low=min(v["mean_rmse"] for v in candidates)
    gap=(max(v["mean_rmse"] for v in candidates)-low)/low
    order={"randomized":0,"optuna":1}
    if gap<=.002:
        winner=min(candidates,key=lambda v:(v["parameters"]["iterations"],v["parameters"]["depth"],v["mean_rmse"],order[v["method"]]))
        reason="Разница <=0.2%; меньше iterations, затем depth, затем mean RMSE; полный tie: randomized"
    else:
        winner=min(candidates,key=lambda v:(v["mean_rmse"],order[v["method"]]))
        reason="Разница >0.2%; минимальная mean CV RMSE"
    return winner,dict(relative_gap=gap,denominator=low,rule=reason)


def validation_fit(parameters,X,y,dev,C,label):
    audit_config(label,TRACE)
    model=pipeline(parameters)
    model.fit(X.iloc[dev],y.iloc[dev])
    before=time.perf_counter();pred=model.predict(X.iloc[C])
    return dict(config=label,parameters=plain(parameters or {"iterations":300,"depth":8,"learning_rate":.05,"l2_leaf_reg":3.0}),
                fit_seconds=CONTEXT["records"][-1]["fit_seconds"],
                predict_seconds=time.perf_counter()-before,**metrics(y.iloc[C],pred))


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if not args.run:print("Для запуска оптимизации явно укажите --run.");return
    if RESULTS.exists():raise RuntimeError("Stage5 results.json уже существует; случайный повторный запуск запрещён")
    overall=time.perf_counter()
    client=MlflowClient(tracking_uri=URI,registry_uri=URI)
    mlflow.set_tracking_uri(URI);mlflow.set_registry_uri(URI)
    experiment=client.get_experiment_by_name("real_estate_model_improvement")
    assert experiment and experiment.experiment_id=="1"
    mlflow.set_experiment(experiment_id=experiment.experiment_id)
    engine=create_engine(backend_uri())
    try:
        with engine.connect() as connection:
            row=connection.execute(text("SELECT experiment_id FROM experiments WHERE name=:name"),
                                   {"name":experiment.name}).first()
            assert row and str(row[0])==experiment.experiment_id
    finally:engine.dispose()
    protected=protected_snapshot(client,experiment.experiment_id)
    protected_ids=list(protected["runs"])
    previous3=client.get_run(protected["versions"]["3"]["run_id"])
    assert tuple(json.loads(previous3.data.params["selected_features"]))==SELECTED
    assert previous3.info.status=="FINISHED"
    prior_result=ROOT.parent/"mle-mlflow-runtime/stage4/results.json"
    assert prior_result.is_file()
    prior=json.loads(prior_result.read_text(encoding="utf-8"))
    raw=TRAIN.read_bytes();train_md5=hashlib.md5(raw).hexdigest()
    assert train_md5=="2d4bb3ba2091eb269bc56a93f05f9119"
    train=pd.read_csv(TRAIN);assert train.shape==(112734,15)
    X=train[RAW_FEATURES].copy();y=train.price.copy()
    original_hash=hashlib.sha256(pd.util.hash_pandas_object(train,index=True).values.tobytes()).hexdigest()
    groups,unique=pd.factorize(pd.MultiIndex.from_frame(X),sort=False);assert len(unique)==99839
    dev,_,_,C=make_splits(X,y,groups,50000)
    assert index_hash(dev)==prior["splits"]["Development"]["indices_sha256"]
    assert index_hash(C)==prior["splits"]["C"]["indices_sha256"]
    assert np.intersect1d(groups[dev],groups[C]).size==0
    Xdev=X.iloc[dev].copy();ydev=y.iloc[dev].copy()
    folds=list(GroupKFold(n_splits=3).split(Xdev,ydev,groups[dev]))
    fold_meta=cv_metadata(folds,dev,groups)
    train_hashes=[v["train_indices_sha256"] for v in fold_meta]
    valid_hashes=[v["validation_indices_sha256"] for v in fold_meta]
    folds_sha=hashlib.sha256(json.dumps(fold_meta,sort_keys=True).encode()).hexdigest()
    template=pipeline();copy=clone(template);copy.set_params(model__depth=7)
    assert tuple(copy.named_steps["selection"].columns)==SELECTED
    assert copy.get_params()["model__cat_features"]==[CATEGORY]
    assert set(distributions())=={"model__"+name for name in TUNED_KEYS}
    assert TEST.is_file()
    if (RUNTIME/"benchmark.json").exists():
        bench=json.loads((RUNTIME/"benchmark.json").read_text(encoding="utf-8"))
        design=json.loads((RUNTIME/"budget_before_search.json").read_text(encoding="utf-8"))
        assert bench["record"]["train_indices_sha256"]==train_hashes[0]
        assert bench["parameters"]=={"iterations":400,"depth":7,"learning_rate":float(np.sqrt(.025*.12)),"l2_leaf_reg":float(np.sqrt(12))}
    else:bench,design=create_benchmark(Xdev,ydev,folds)
    budget=design["n_configurations_each"];assert budget in [12,18],"Безопасный бюджет поиска не установлен"
    source_guard={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [ROOT/"README.md",ROOT/"requirements.txt",ROOT/"model_improvement/model_improvement.ipynb",
                  *list((ROOT/"model_improvement/stage3").glob("*.py")),*list((ROOT/"model_improvement/stage4").glob("*.py"))]}
    result=dict(stage="stage5",completed=False,train_md5=train_md5,train_sha256=hashlib.sha256(raw).hexdigest(),
        input_features=RAW_FEATURES,selected_features=list(SELECTED),groups=len(unique),space=SPACE,constants=CONSTANTS,
        benchmark=bench,budget=design,folds=fold_meta,folds_sha256=folds_sha,
        splits={"Development":dict(rows=len(dev),indices_sha256=index_hash(dev)),"C":dict(rows=len(C),indices_sha256=index_hash(C))},
        runs={},validation=[],environment={n:metadata.version(n) for n in ["numpy","pandas","scikit-learn","catboost","mlflow",
        "optuna","scipy","joblib","cloudpickle","psutil","threadpoolctl"]},python=sys.version,
        winner_rule="min_mean_CV_RMSE; if gap/min<=0.002: iterations,depth,mean_rmse; full_tie=randomized",
        source_hashes=source_guard)
    save(result);event("design_fixed",budget=budget,folds=fold_meta)
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name="stage5_baseline_grouped",
                         tags={"stage":"stage5","method":"reference"}) as active:
        result["runs"]["stage5_baseline_grouped"]=active.info.run_id
        reference=evaluate_cv(None,Xdev,ydev,folds,train_hashes,valid_hashes,"reference")
        result["reference"]=reference;assert CONTEXT["count"]==3
        log_cv(reference)
        mlflow.log_params(dict(feature_count=8,folds_sha256=folds_sha,std_ddof=0,iterations=300,depth=8,
                              learning_rate=.05,l2_leaf_reg=3,**CONSTANTS))
        save(result);mlflow.log_artifact(str(RUNTIME/"benchmark.json"),artifact_path="design")
        mlflow.log_artifact(str(RUNTIME/"budget_before_search.json"),artifact_path="design")
    save(result)
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name="stage5_randomized_search",
                         tags={"stage":"stage5","method":"RandomizedSearchCV"}) as active:
        result["runs"]["stage5_randomized_search"]=active.info.run_id
        audit_config("randomized",TRACE,train_hashes,valid_hashes)
        search=RandomizedSearchCV(pipeline(),param_distributions=distributions(),n_iter=budget,
            scoring={"rmse":"neg_root_mean_squared_error","mae":"neg_mean_absolute_error","r2":"r2"},
            cv=folds,n_jobs=1,pre_dispatch=1,refit=False,error_score="raise",random_state=42,
            return_train_score=False,verbose=0)
        before=time.perf_counter();search.fit(Xdev,ydev);elapsed=time.perf_counter()-before
        assert CONTEXT["count"]==budget*3
        data=search.cv_results_;rows=[]
        assert len(data["params"])==budget
        for i,parameters in enumerate(data["params"]):
            params={k.removeprefix("model__"):plain(v) for k,v in parameters.items()};validate_parameters(params)
            fold_rows=[]
            for j in range(3):
                fold_rows.append(dict(fold=j+1,rmse=float(-data[f"split{j}_test_rmse"][i]),
                    mae=float(-data[f"split{j}_test_mae"][i]),r2=float(data[f"split{j}_test_r2"][i]),
                    fit_seconds=CONTEXT["records"][i*3+j]["fit_seconds"]))
            mean=float(-data["mean_test_rmse"][i]);std=float(data["std_test_rmse"][i])
            assert np.isclose(mean,np.mean([v["rmse"] for v in fold_rows]),rtol=1e-12)
            assert np.isclose(std,np.std([v["rmse"] for v in fold_rows],ddof=0),rtol=1e-12)
            rows.append(dict(number=i,parameters=params,folds=fold_rows,mean_rmse=mean,std_rmse=std,
                mean_mae=float(-data["mean_test_mae"][i]),mean_r2=float(data["mean_test_r2"][i]),
                mean_fit_seconds=float(data["mean_fit_time"][i]),rank=int(data["rank_test_rmse"][i])))
        assert all(np.isfinite(v[k]) for v in rows for k in ["mean_rmse","std_rmse","mean_mae","mean_r2"])
        best=min(rows,key=lambda v:(v["mean_rmse"],v["parameters"]["iterations"],v["parameters"]["depth"],v["number"]))
        result["randomized"]=dict(configurations=rows,best=best,wall_seconds=elapsed,fit_count=CONTEXT["count"])
        pd.DataFrame(search.cv_results_).to_csv(RUNTIME/"randomized_cv_results.csv",index=False)
        (RUNTIME/"randomized_results.json").write_text(json.dumps(result["randomized"],indent=2),encoding="utf-8")
        log_cv(best,"best_");mlflow.log_metric("search_seconds",elapsed);mlflow.log_metric("cv_fit_count",budget*3)
        mlflow.log_params({"n_iter":budget,"random_state":42,"n_jobs":1,"pre_dispatch":1,"refit":False,
            "error_score":"raise","folds_sha256":folds_sha,"space":json.dumps(SPACE),
            **{"best_"+k:v for k,v in best["parameters"].items()}})
        mlflow.log_artifact(str(RUNTIME/"randomized_cv_results.csv"),artifact_path="search")
        mlflow.log_artifact(str(RUNTIME/"randomized_results.json"),artifact_path="search")
        save(result);del search;gc.collect()
    save(result);event("randomized_complete",best=best,wall_seconds=elapsed)
    mlflow.set_experiment(experiment_id=experiment.experiment_id)
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name="stage5_optuna",
                         tags={"stage":"stage5","method":"TPESampler"}) as active:
        parent=active.info.run_id;result["runs"]["stage5_optuna"]=parent
        mlflow.log_params(dict(n_trials=budget,n_startup_trials=6,seed=42,n_jobs=1,folds_sha256=folds_sha,
                              create_experiment=False,pruning=False,early_stopping=False,space=json.dumps(SPACE)))
        callback=MLflowCallback(tracking_uri=URI,metric_name="cv_rmse_mean",create_experiment=False,
            mlflow_kwargs={"experiment_id":experiment.experiment_id,"nested":True},
            tag_trial_user_attrs=False,tag_study_user_attrs=False)
        audit_config("optuna",TRACE,train_hashes,valid_hashes)
        result["optuna"]=dict(trials=[],fit_count=0)
        @callback.track_in_mlflow()
        def objective(trial):
            params=dict(iterations=trial.suggest_categorical("iterations",SPACE["iterations"]),
                depth=trial.suggest_int("depth",5,9),learning_rate=trial.suggest_float("learning_rate",.025,.12,log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg",1,12,log=True))
            validate_parameters(params)
            row=evaluate_cv(params,Xdev,ydev,folds,train_hashes,valid_hashes,"optuna",reset=False)
            child=mlflow.active_run().info.run_id
            trial.set_user_attr("evaluation",row);trial.set_user_attr("run_id",child)
            mlflow.set_tags({"stage":"stage5_trial","method":"optuna","trial_number":str(trial.number)})
            mlflow.log_params(dict(feature_count=8,folds_sha256=folds_sha,cv_n_splits=3))
            for v in row["folds"]:
                for metric in ["rmse","mae","r2","fit_seconds"]:
                    mlflow.log_metric("fold_"+metric,v[metric],step=v["fold"])
            mlflow.log_metrics(dict(cv_rmse_std=row["std_rmse"],cv_mae_mean=row["mean_mae"],cv_r2_mean=row["mean_r2"]))
            return row["mean_rmse"]
        def persist_trial(study,trial):
            assert trial.state==optuna.trial.TrialState.COMPLETE
            row=dict(number=trial.number,state=trial.state.name,run_id=trial.system_attrs[RUN_ID_ATTRIBUTE_KEY],
                     duration_seconds=trial.duration.total_seconds(),**trial.user_attrs["evaluation"])
            child=client.get_run(row["run_id"])
            assert child.info.experiment_id==experiment.experiment_id and child.info.status=="FINISHED"
            assert child.data.tags["mlflow.parentRunId"]==parent
            assert np.isclose(child.data.metrics["cv_rmse_mean"],row["mean_rmse"],rtol=1e-12)
            result["optuna"]["trials"].append(row);result["optuna"]["fit_count"]=CONTEXT["count"]
            save(result)
            (RUNTIME/"optuna_trials.json").write_text(json.dumps(result["optuna"]["trials"],indent=2),encoding="utf-8")
            event("optuna_trial_complete",number=trial.number,run_id=row["run_id"],mean_rmse=row["mean_rmse"])
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study=optuna.create_study(study_name="stage5_tpe_seed42",direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=6),pruner=optuna.pruners.NopPruner())
        before=time.perf_counter()
        study.optimize(objective,n_trials=budget,n_jobs=1,callbacks=[callback,persist_trial])
        elapsed=time.perf_counter()-before
        assert len(study.trials)==budget and CONTEXT["count"]==budget*3
        assert all(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        best=next(v for v in result["optuna"]["trials"] if v["number"]==study.best_trial.number)
        assert np.isclose(best["mean_rmse"],study.best_value,rtol=1e-12)
        result["optuna"].update(best=best,wall_seconds=elapsed)
        try:
            result["optuna"]["parameter_importances"]=optuna.importance.get_param_importances(
                study,evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42))
        except Exception as error:
            result["optuna"]["importance_unavailable"]=type(error).__name__+": "+str(error)
        children=list(client.search_runs([experiment.experiment_id],
            filter_string="tags.mlflow.parentRunId = '"+parent+"'",max_results=1000))
        assert len(children)==budget
        assert {t["run_id"] for t in result["optuna"]["trials"]}=={v.info.run_id for v in children}
        log_cv(best,"best_");mlflow.log_metrics(dict(search_seconds=elapsed,cv_fit_count=budget*3))
        mlflow.log_params({"best_"+k:v for k,v in best["parameters"].items()})
        (RUNTIME/"optuna_results.json").write_text(json.dumps(result["optuna"],indent=2),encoding="utf-8")
        study.trials_dataframe().to_csv(RUNTIME/"optuna_trials.csv",index=False)
        mlflow.log_artifact(str(RUNTIME/"optuna_results.json"),artifact_path="search")
        mlflow.log_artifact(str(RUNTIME/"optuna_trials.csv"),artifact_path="search")
        save(result)
    event("optuna_complete",best=best,wall_seconds=elapsed)
    winner,reason=choose_winner(result["randomized"]["best"],result["optuna"]["best"])
    result["winner"]=winner;result["winner_reason"]=reason
    decision=dict(winner=winner,reason=reason,budget=design,space=SPACE,constants=CONSTANTS,
        train_md5=train_md5,train_sha256=result["train_sha256"],folds=fold_meta,folds_sha256=folds_sha,
        selected_features=list(SELECTED),rule=result["winner_rule"])
    lock=RUNTIME/"winner_before_C.json"
    with lock.open("x",encoding="utf-8") as stream:json.dump(decision,stream,ensure_ascii=False,indent=2)
    decision_sha=hashlib.sha256(lock.read_bytes()).hexdigest();result["winner_sha256"]=decision_sha
    save(result);event("winner_locked_before_C",method=winner["method"],parameters=winner["parameters"],reason=reason)
    # Никакого использования C до зафиксированного выбора победителя.
    result["validation"].append(validation_fit(winner["parameters"],X,y,dev,C,"Stage5_tuned"));save(result)
    result["validation"].append(validation_fit(None,X,y,dev,C,"Stage4_reference"));save(result)
    assert hashlib.sha256(lock.read_bytes()).hexdigest()==decision_sha
    mlflow.set_experiment(experiment_id=experiment.experiment_id)
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name="stage5_hyperparameter_tuning",
                         tags={"stage":"stage5","winner":winner["method"],"winner_sha256":decision_sha,
                               "registry_registration":"false","test_used_for_selection":"false"}) as active:
        final_id=active.info.run_id;result["runs"]["stage5_hyperparameter_tuning"]=final_id
        model=pipeline(winner["parameters"],audited=False)
        audit_config("final_full_train",TRACE);checked_fit(model.fit,X,y)
        result["final_fit_seconds"]=CONTEXT["records"][-1]["fit_seconds"]
        probe=X.sample(n=128,random_state=123).copy();expected=model.predict(probe)
        assert np.isfinite(expected).all() and tuple(model.named_steps["selection"].columns)==SELECTED
        transformed=model.named_steps["selection"].transform(model.named_steps["preprocess"].transform(probe))
        assert transformed.shape[1]==8 and transformed[CATEGORY].dtype.kind in "iu"
        log_cv(winner,"selected_")
        mlflow.log_params({**CONSTANTS,**winner["parameters"],"winner_method":winner["method"],
            "selected_features":json.dumps(SELECTED),"input_features":14,"n_trials_each":budget,
            "folds_sha256":folds_sha,"winner_sha256":decision_sha,"train_md5":train_md5,"train_sha256":result["train_sha256"],
            "parent_randomized":result["runs"]["stage5_randomized_search"],"parent_optuna":result["runs"]["stage5_optuna"],
            "std_ddof":0,**{name+"_version":version for name,version in result["environment"].items()}})
        mlflow.log_metric("final_fit_seconds",result["final_fit_seconds"])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="Hint: Inferred schema contains integer column.*")
            mlflow.sklearn.log_model(model,artifact_path="model",
                code_paths=[str(ROOT/"model_improvement/stage3"),str(ROOT/"model_improvement/stage4")],
                signature=infer_signature(probe,expected),input_example=probe.head(5),
                pip_requirements=[name+"=="+v for name,v in result["environment"].items()],
                serialization_format="cloudpickle")
        sample=RUNTIME/"verification_sample.pkl";joblib.dump((probe,expected),sample)
        verify="""import sys,json,joblib,numpy as np,mlflow,mlflow.sklearn,mlflow.pyfunc
mlflow.set_tracking_uri('http://127.0.0.1:5000')
X,expected=joblib.load(sys.argv[2])
uri='runs:/'+sys.argv[1]+'/model'
pipeline=mlflow.sklearn.load_model(uri);pyfunc=mlflow.pyfunc.load_model(uri)
a=pipeline.predict(X);b=pyfunc.predict(X)
np.testing.assert_allclose(a,expected,rtol=1e-12,atol=1e-6)
np.testing.assert_allclose(b,expected,rtol=1e-12,atol=1e-6)
assert X.shape==(128,14) and np.isfinite(a).all() and np.isfinite(b).all()
assert pipeline.__class__.__name__=='Pipeline'
assert len(pipeline.named_steps['selection'].columns)==8
print(json.dumps({'sklearn':True,'pyfunc':True,'rows':128,'input_features':14,'max_abs_difference':float(np.max(np.abs(a-expected))),'pyfunc_max_abs_difference':float(np.max(np.abs(b-expected))),'ordinary_pipeline':True,'isolated_process':True}))
"""
        process=subprocess.run([sys.executable,"-I","-B","-c",verify,final_id,str(sample)],cwd=RUNTIME,
                               env=os.environ.copy(),capture_output=True,text=True)
        result["reload_verification"]=dict(returncode=process.returncode,stdout=process.stdout.strip(),stderr=process.stderr[-1500:])
        save(result)
        if process.returncode:raise RuntimeError("Техническая проверка до test не пройдена: "+process.stderr[-1000:])
        # Технические проверки пройдены. Предохранитель не позволяет незаметно повторить test.
        with (RUNTIME/"historical_test_started.json").open("x",encoding="utf-8") as stream:
            json.dump(dict(run_id=final_id,winner_sha256=decision_sha),stream)
        test_md5=hashlib.md5(TEST.read_bytes()).hexdigest()
        assert test_md5=="1d5874c7f4950c457e1ff2678d44a121"
        test=pd.read_csv(TEST);assert len(test)==28184
        before=time.perf_counter();prediction=model.predict(test[RAW_FEATURES])
        result["test_predict_seconds"]=time.perf_counter()-before
        result["test_metrics"]=metrics(test.price,prediction);result["test_md5"]=test_md5
        mlflow.log_metrics({"test_"+k:v for k,v in result["test_metrics"].items()});mlflow.log_param("test_md5",test_md5)
        history={}
        for number in ["1","2","3"]:
            oldrun=client.get_run(protected["versions"][number]["run_id"])
            if number=="1":history["Version1"]={k:oldrun.data.metrics[n] for k,n in [("rmse","RMSE"),("mae","MAE"),("r2","R2")]}
            else:history["Version"+number]={k:oldrun.data.metrics["test_"+k] for k in ["rmse","mae","r2"]}
        result["historical_comparison"]=history
        for row in result["validation"]:
            mlflow.log_metrics({row["config"]+"_C_"+k:row[k] for k in ["rmse","mae","r2","fit_seconds","predict_seconds"]})
        assert hashlib.sha256(lock.read_bytes()).hexdigest()==decision_sha
        assert original_hash==hashlib.sha256(pd.util.hash_pandas_object(train,index=True).values.tobytes()).hexdigest()
        assert hashlib.md5(TRAIN.read_bytes()).hexdigest()==train_md5
        assert hashlib.md5(TEST.read_bytes()).hexdigest()==test_md5
        save(result)
        mlflow.log_artifact(str(lock),artifact_path="design")
        mlflow.log_artifact(str(RUNTIME/"budget_before_search.json"),artifact_path="design")
        mlflow.log_artifact(str(RESULTS),artifact_path="reports")
    assert protected_snapshot(client,experiment.experiment_id,protected_ids)==protected
    assert source_guard=={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in source_guard}
    trace=[json.loads(line) for line in TRACE.read_text(encoding="utf-8").splitlines()]
    counts={phase:sum(v["phase"]==phase for v in trace)
            for phase in ["benchmark","reference","randomized","optuna","Stage5_tuned","Stage4_reference","final_full_train"]}
    assert counts=={"benchmark":1,"reference":3,"randomized":budget*3,"optuna":budget*3,
                    "Stage5_tuned":1,"Stage4_reference":1,"final_full_train":1},counts
    result["fit_counts"]=counts;result["protected_previous_stages_unchanged"]=True
    result["total_seconds_excluding_benchmark"]=time.perf_counter()-overall
    result["resource_end"]=resources();result["completed"]=True
    save(result);client.log_artifact(final_id,str(RESULTS),artifact_path="reports")
    event("completed",winner=winner["method"],parameters=winner["parameters"],test=result["test_metrics"],
          runs=result["runs"],fit_counts=counts)


if __name__=="__main__":
    try:main()
    except Exception as error:
        event("error",type=type(error).__name__,message=redact(str(error)))
        raise
