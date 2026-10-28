"""Лёгкий аудит fit/predict без изменения компонентов Stage 3–4."""
import hashlib
import json
import threading
import time
from pathlib import Path
import numpy as np
import psutil
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

CONTEXT = dict(phase="idle", count=0, path=None, records=[], expected_train=None, expected_valid=None)
ALL_RECORDS = []


def index_hash(index):
    return hashlib.sha256(np.asarray(index, dtype=np.int64).tobytes()).hexdigest()


def configure(phase, path, expected_train=None, expected_valid=None):
    CONTEXT.update(phase=phase, count=0, path=str(path), records=[],
                   expected_train=expected_train, expected_valid=expected_valid)


def resources():
    return dict(available_ram_gib=psutil.virtual_memory().available / 2**30,
                free_disk_gib=psutil.disk_usage("C:/").free / 2**30,
                rss_mib=psutil.Process().memory_info().rss / 2**20)


def checked_fit(call, X, y, **params):
    before=resources()
    if before["available_ram_gib"] < .65 or before["free_disk_gib"] < 1:
        raise MemoryError("Перед fit требуется >=0,65 ГиБ RAM и >=1 ГиБ диска")
    if X.shape[1] != 14 or "price" in X.columns:
        raise ValueError("Ожидаются исходные 14 признаков без target")
    expected=CONTEXT["expected_train"]
    digest=index_hash(X.index)
    sequence=CONTEXT["count"]
    if expected is not None:
        assert digest == expected[sequence % len(expected)], "Неверные train-индексы fold"
    stop=threading.Event()
    peak=[before["rss_mib"]]
    minimum=[before["available_ram_gib"]]
    started=time.perf_counter()

    def monitor():
        previous=started
        while not stop.wait(.25):
            values=resources()
            peak[0]=max(peak[0],values["rss_mib"])
            minimum[0]=min(minimum[0],values["available_ram_gib"])
            if time.perf_counter()-previous >= 25:
                print(json.dumps(dict(event="fit_heartbeat",phase=CONTEXT["phase"],
                    fit=sequence+1,elapsed_seconds=round(time.perf_counter()-started,1),
                    available_ram_gib=round(values["available_ram_gib"],3))),flush=True)
                previous=time.perf_counter()
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        with threadpool_limits(limits=1):
            result=call(X,y,**params)
    finally:
        stop.set();thread.join()
    elapsed=time.perf_counter()-started
    estimator=call.__self__
    hyper=estimator.named_steps["model"].get_params()
    tuned={name:(value.item() if isinstance(value,np.generic) else value)
           for name,value in hyper.items() if name in ["iterations","depth","learning_rate","l2_leaf_reg"]}
    record=dict(phase=CONTEXT["phase"],fit=sequence+1,train_rows=len(X),
                train_indices_sha256=digest,fit_seconds=elapsed,
                peak_rss_mib=peak[0],min_available_ram_gib=minimum[0],parameters=tuned)
    CONTEXT["count"]+=1
    CONTEXT["records"].append(record);ALL_RECORDS.append(record)
    if CONTEXT["path"]:
        with Path(CONTEXT["path"]).open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record)+"\n")
    print(json.dumps(dict(event="fit_complete",**record)),flush=True)
    if minimum[0] < .3:
        raise MemoryError("Зафиксирован критический запас RAM; следующий fit не запускается")
    return result


class AuditedPipeline(Pipeline):
    """Только для CV/benchmark; финальный artifact использует обычный Pipeline."""
    def fit(self, X, y=None, **params):
        return checked_fit(super().fit,X,y,**params)

    def predict(self, X, **params):
        expected=CONTEXT["expected_valid"]
        if expected is not None:
            assert CONTEXT["count"] > 0
            assert index_hash(X.index) == expected[(CONTEXT["count"]-1) % len(expected)], "Неверные validation-индексы fold"
        result=super().predict(X,**params)
        if not np.isfinite(result).all():
            raise ValueError("Неконечные предсказания")
        return result
