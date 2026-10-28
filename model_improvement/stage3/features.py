"""Преобразования Stage 3: только исходные признаки, sklearn-совместимый интерфейс."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, KBinsDiscretizer
from catboost import CatBoostRegressor

RAW_FEATURES = ["floor", "kitchen_area", "living_area", "rooms", "is_apartment",
                "total_area", "build_year", "building_type_int", "latitude",
                "longitude", "ceiling_height", "flats_count", "floors_total", "has_elevator"]
BASE_PARAMS = dict(iterations=300, depth=8, learning_rate=.05, loss_function="RMSE",
                   random_seed=42, verbose=False, allow_writing_files=False, thread_count=2)
POLY_COLUMNS = ["total_area", "rooms", "floors_total", "build_year"]
BIN_COLUMNS = ["total_area", "living_area", "kitchen_area", "ceiling_height"]
AF_COLUMNS = ["total_area", "rooms", "ceiling_height", "floors_total"]


def validate_frame(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Ожидается pandas DataFrame с исходными 14 признаками")
    if set(X.columns) != set(RAW_FEATURES) or len(X.columns) != len(RAW_FEATURES):
        raise ValueError("Входная схема должна содержать ровно 14 исходных признаков без price")
    result = X.loc[:, RAW_FEATURES].copy()
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Обнаружены NaN/Inf")
    category = result["building_type_int"]
    if not (category == np.floor(category)).all():
        raise ValueError("Код типа здания должен быть целым")
    result["building_type_int"] = category.astype("int64")
    return result


class SchemaGuard(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        validate_frame(X)
        self.feature_names_in_ = np.array(RAW_FEATURES, dtype=object)
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X):
        return validate_frame(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_FEATURES, dtype=object)


class ManualFeatures(TransformerMixin, BaseEstimator):
    extra_columns = ["living_area_ratio", "kitchen_area_ratio", "relative_floor",
                     "is_first_floor", "is_last_floor"]

    def fit(self, X, y=None):
        self.transform(X)
        self.feature_names_in_ = np.array(RAW_FEATURES, dtype=object)
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X):
        result = validate_frame(X)
        if not result.total_area.gt(0).all() or not result.floors_total.gt(0).all():
            raise ValueError("Знаменатели площадей и этажности должны быть положительными")
        result["living_area_ratio"] = result.living_area / result.total_area
        result["kitchen_area_ratio"] = result.kitchen_area / result.total_area
        result["relative_floor"] = result.floor / result.floors_total
        result["is_first_floor"] = result.floor.eq(1).astype("int64")
        result["is_last_floor"] = result.floor.eq(result.floors_total).astype("int64")
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("Неконечный результат генерации признаков")
        return result

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_FEATURES + self.extra_columns, dtype=object)


class InteractionTerms(TransformerMixin, BaseEstimator):
    """Сохраняем пары PolynomialFeatures, не дублируя исходные столбцы."""
    def fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if " " in c]
        return self

    def transform(self, X):
        return X.loc[:, self.columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_, dtype=object)


class AutoFeatFeatures(TransformerMixin, BaseEstimator):
    """Supervised-генератор обучается на фиксированной подвыборке fit-части."""
    def __init__(self, sample_size=8000, seed=42):
        self.sample_size = sample_size
        self.seed = seed

    def fit(self, X, y):
        from autofeat import AutoFeatRegressor
        frame = validate_frame(X)
        if not frame[AF_COLUMNS].gt(0).all().all():
            raise ValueError("Для ограниченного autofeat ожидаются положительные признаки")
        if y is None:
            raise ValueError("autofeat требует target только обучающей части")
        indices = np.random.RandomState(self.seed).choice(
            len(frame), size=min(self.sample_size, len(frame)), replace=False)
        self.fit_rows_ = len(indices)
        self.fit_positions_sha256_ = __import__("hashlib").sha256(indices.tobytes()).hexdigest()
        self.generator_ = AutoFeatRegressor(
            feateng_cols=AF_COLUMNS, feateng_steps=1, featsel_runs=2, n_jobs=1,
            max_gb=.10, transformations=("1/", "log", "sqrt", "^2"), verbose=0)
        # autofeat использует глобальный RNG: локально фиксируем и восстанавливаем его.
        state = np.random.get_state()
        try:
            np.random.seed(self.seed)
            self.generator_.fit_transform(
                frame.iloc[indices][AF_COLUMNS].astype(float).reset_index(drop=True),
                np.asarray(y)[indices])
        finally:
            np.random.set_state(state)
        self.formulas_ = list(self.generator_.new_feat_cols_)
        if len(self.formulas_) > 64:
            raise MemoryError("autofeat вернул более 64 производных признаков")
        self.output_columns_ = ["autofeat_%03d" % i for i in range(len(self.formulas_))]
        return self

    def transform(self, X):
        result = validate_frame(X)
        chunks = []
        for start in range(0, len(result), 10000):
            block = self.generator_.transform(
                result.iloc[start:start + 10000][AF_COLUMNS].astype(float).reset_index(drop=True))
            chunks.append(block[self.formulas_].to_numpy(dtype=float))
        values = np.vstack(chunks) if chunks else np.empty((0, len(self.formulas_)))
        if not np.isfinite(values).all():
            raise ValueError("autofeat дал NaN/Inf")
        for i, name in enumerate(self.output_columns_):
            result[name] = values[:, i]
        return result

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_FEATURES + self.output_columns_, dtype=object)


def make_pipeline(name):
    schema = SchemaGuard()
    if name == "baseline":
        preprocessing = schema
        category = "building_type_int"
    elif name == "manual":
        preprocessing = Pipeline([("schema", schema), ("manual", ManualFeatures())])
        category = "building_type_int"
    elif name == "sklearn":
        # Scaling применяется только к полиномиальной ветке; абсолютные признаки сохранены.
        interactions = Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("pairs", InteractionTerms()),
        ])
        columns = ColumnTransformer([
            ("base", "passthrough", RAW_FEATURES),
            ("interactions", interactions, POLY_COLUMNS),
            ("bins", KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile",
                                     subsample=None, random_state=42), BIN_COLUMNS),
        ], remainder="drop", sparse_threshold=0, n_jobs=1, verbose_feature_names_out=True)
        columns.set_output(transform="pandas")
        preprocessing = Pipeline([("schema", schema), ("columns", columns)])
        category = "base__building_type_int"
    elif name == "autofeat":
        preprocessing = Pipeline([("schema", schema), ("autofeat", AutoFeatFeatures())])
        category = "building_type_int"
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", preprocessing),
                     ("model", CatBoostRegressor(**BASE_PARAMS, cat_features=[category]))])
