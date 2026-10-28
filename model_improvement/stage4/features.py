"""Stage 4: транспорт категории для mlxtend и фиксированный селектор."""
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from stage3.features import BASE_PARAMS, make_pipeline as make_stage3_pipeline

CATEGORY = "base__building_type_int"
CATEGORY_PREFIX = "__stage4_building_type__="
FIT_AUDIT = {"phase": "idle", "count": 0, "callback": None}


def reset_fit_audit(phase, callback=None):
    FIT_AUDIT.update(phase=phase, count=0, callback=callback)


def encode_search_frame(X):
    """Маркер переносит только исходный код категории, не target."""
    if CATEGORY not in X or X[CATEGORY].dtype.kind not in "iu":
        raise ValueError("Требуется целочисленный категориальный столбец F2")
    result = X.copy()
    result[CATEGORY] = result[CATEGORY].map(lambda v: CATEGORY_PREFIX + str(int(v)))
    return result


class CatBoostSubsetAdapter(RegressorMixin, BaseEstimator):
    """Определяем позицию фиксированной категории по явному транспортному маркеру."""
    def __init__(self, iterations=100, depth=8, learning_rate=.05, random_seed=42,
                 thread_count=2):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.thread_count = thread_count

    def _frame(self, X):
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2:
            raise ValueError("Ожидается двумерный массив mlxtend")
        category_positions = [
            j for j in range(arr.shape[1])
            if all(isinstance(v, str) and v.startswith(CATEGORY_PREFIX) for v in arr[:, j])]
        if len(category_positions) != 1:
            raise ValueError("Подмножество должно содержать ровно одну маркированную категорию")
        pos = category_positions[0]
        frame = pd.DataFrame(index=np.arange(len(arr)))
        for j in range(arr.shape[1]):
            if j == pos:
                frame[j] = np.array([int(v[len(CATEGORY_PREFIX):]) for v in arr[:, j]], dtype=np.int64)
            else:
                frame[j] = np.asarray(arr[:, j], dtype=float)
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("NaN/Inf в подмножестве")
        return frame, pos

    def fit(self, X, y):
        frame, pos = self._frame(X)
        self.n_features_in_ = frame.shape[1]
        self.category_position_ = pos
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            random_seed=self.random_seed, thread_count=self.thread_count,
            loss_function="RMSE", verbose=False, allow_writing_files=False,
            cat_features=[pos])
        start = time.perf_counter()
        self.model_.fit(frame, np.asarray(y))
        FIT_AUDIT["count"] += 1
        callback = FIT_AUDIT["callback"]
        if callback is not None:
            callback(FIT_AUDIT["phase"], FIT_AUDIT["count"], time.perf_counter() - start)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        frame, pos = self._frame(X)
        if frame.shape[1] != self.n_features_in_ or pos != self.category_position_:
            raise ValueError("Схема подмножества отличается от fit")
        prediction = self.model_.predict(frame)
        if not np.isfinite(prediction).all():
            raise ValueError("Неконечный прогноз")
        return prediction


class FixedFeatureSelector(TransformerMixin, BaseEstimator):
    """Только фиксированный выбор по именам: никакого повторного поиска в predict."""
    def __init__(self, columns):
        self.columns = columns

    def _check(self, X):
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
            raise TypeError("Ожидается DataFrame с уникальными именами")
        if len(set(self.columns)) != len(self.columns) or CATEGORY not in self.columns:
            raise ValueError("Список должен быть уникальным и сохранять категорию")
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError("Отсутствуют выбранные признаки: " + str(sorted(missing)))
        if X[CATEGORY].dtype.kind not in "iu":
            raise TypeError("Категория должна оставаться целочисленной")

    def fit(self, X, y=None):
        self._check(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        self._check(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Входная схема F2 изменилась")
        result = X.loc[:, list(self.columns)].copy()
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("NaN/Inf после отбора")
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns, dtype=object)


def make_pipeline(columns=None):
    """Переиспользуем F2 Stage 3 без изменения его реализации."""
    base = make_stage3_pipeline("sklearn")
    if columns is None:
        return base
    return Pipeline([
        ("preprocess", base.named_steps["preprocess"]),
        ("selection", FixedFeatureSelector(columns)),
        ("model", CatBoostRegressor(**BASE_PARAMS, cat_features=[CATEGORY])),
    ])
