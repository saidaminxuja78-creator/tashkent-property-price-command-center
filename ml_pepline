from __future__ import annotations

import re
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import joblib

RANDOM_STATE = 42
TARGET = "price_uzs"

DROP_RAW_COLUMNS = [
    "id", "url", "created_time", "last_refresh_time", "valid_to_time", "pushup_time",
    "category_id", "category_name", "latitude", "longitude", "seller_id", "seller_name",
    "status"
]

PREFERRED_FEATURES = [
    "region", "district", "type_of_market", "number_of_rooms", "total_area", "total_living_area",
    "floor", "total_floors", "house_type", "building_age", "wc", "furnished", "repairs",
    "comission", "layout", "ceiling_height", "kitchen_area", "photos_count",
    "area_per_room", "floor_ratio", "is_top_floor", "is_first_floor", "amenities_count", "nearby_count",
    "title_length", "description_length"
]

CATEGORICAL_FEATURES = [
    "region", "district", "type_of_market", "house_type", "wc", "furnished", "repairs", "comission", "layout"
]
NUMERIC_FEATURES = [f for f in PREFERRED_FEATURES if f not in CATEGORICAL_FEATURES]


def _extract_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace("\xa0", " ")
    s = s.replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except ValueError:
        return np.nan


def parse_price_to_uzs(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    digits = re.sub(r"\D", "", str(value))
    return float(digits) if digits else np.nan


def parse_year_midpoint(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    years = re.findall(r"(?:19|20)\d{2}", str(value))
    if not years:
        return np.nan
    nums = [int(y) for y in years[:2]]
    if len(nums) == 1:
        return float(nums[0])
    return float(sum(nums) / len(nums))


def parse_ceiling_height(value: Any) -> float:
    x = _extract_number(value)
    if pd.isna(x):
        return np.nan
    # OLX listings often store 280/300/320 as centimetres. Convert to metres.
    if x > 20:
        x = x / 100.0
    return x


def count_items(value: Any) -> int:
    if pd.isna(value):
        return 0
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return len(parts)


def load_raw_data(path: str | Path = "olx_massive_real_estate.xlsx") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in [".xlsx", ".xls", ".pkl"]:
        # The uploaded dataset may have an incorrect extension but is an Excel workbook.
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean_real_estate_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Numeric parsing
    df["price_uzs"] = df.get("price", pd.Series(index=df.index)).apply(parse_price_to_uzs)
    for col in ["total_area", "total_living_area", "kitchen_area"]:
        if col in df.columns:
            df[col] = df[col].apply(_extract_number)
    if "ceiling_height" in df.columns:
        df["ceiling_height"] = df["ceiling_height"].apply(parse_ceiling_height)
    if "year_of_construction_sale" in df.columns:
        built_year = df["year_of_construction_sale"].apply(parse_year_midpoint)
        df["building_age"] = 2026 - built_year
    else:
        df["building_age"] = np.nan

    # Text-derived features
    df["title_length"] = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str).str.len()
    df["description_length"] = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str).str.len()
    df["amenities_count"] = df.get("more", pd.Series(index=df.index)).apply(count_items)
    df["nearby_count"] = df.get("near_is", pd.Series(index=df.index)).apply(count_items)

    # Numeric sanity handling
    for col in ["number_of_rooms", "floor", "total_floors", "photos_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["number_of_rooms"] = df["number_of_rooms"].clip(1, 12)
    df["floor"] = df["floor"].clip(1, 60)
    df["total_floors"] = df["total_floors"].clip(1, 80)
    df["is_top_floor"] = (df["floor"] >= df["total_floors"]).astype(int)
    df["is_first_floor"] = (df["floor"] == 1).astype(int)
    df["floor_ratio"] = df["floor"] / df["total_floors"].replace(0, np.nan)
    df["area_per_room"] = df["total_area"] / df["number_of_rooms"].replace(0, np.nan)

    # Select rows with usable target and property size.
    df = df[df["price_uzs"].notna() & df["total_area"].notna()].copy()
    df = df[(df["price_uzs"] >= 100_000_000) & (df["price_uzs"] <= 10_000_000_000)]
    df = df[(df["total_area"] >= 15) & (df["total_area"] <= 500)]
    df = df[(df["number_of_rooms"] >= 1) & (df["number_of_rooms"] <= 12)]

    # IQR filtering on price-per-square-metre is more stable than filtering absolute price only.
    df["price_per_sqm"] = df["price_uzs"] / df["total_area"]
    q_low, q_high = df["price_per_sqm"].quantile([0.01, 0.99])
    df = df[(df["price_per_sqm"] >= q_low) & (df["price_per_sqm"] <= q_high)].copy()

    # Ensure feature columns exist.
    for col in PREFERRED_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[PREFERRED_FEATURES + [TARGET, "price_per_sqm"]].reset_index(drop=True)


def get_dense_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", get_dense_onehot_encoder()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")


def wrap_regressor(regressor):
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", make_preprocessor()), ("model", regressor)]),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def get_candidate_models(mode: str = "quick") -> Dict[str, Any]:
    n_estimators = 120 if mode == "quick" else 240
    return {
        "Dummy Baseline": wrap_regressor(DummyRegressor(strategy="median")),
        "Linear Regression": wrap_regressor(LinearRegression()),
        "Ridge Regression": wrap_regressor(Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        "Random Forest": wrap_regressor(RandomForestRegressor(
            n_estimators=n_estimators, max_depth=18 if mode == "quick" else None,
            min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=1
        )),
        "Extra Trees": wrap_regressor(ExtraTreesRegressor(
            n_estimators=n_estimators, max_depth=22 if mode == "quick" else None,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1
        )),
        "Gradient Boosting": wrap_regressor(GradientBoostingRegressor(
            n_estimators=120 if mode == "quick" else 220,
            learning_rate=0.06, max_depth=4, random_state=RANDOM_STATE
        )),
    }


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    within_10 = np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1) <= 0.10) * 100
    within_20 = np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1) <= 0.20) * 100
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, "MedianAE": medae,
        "MAPE": mape, "Within10Pct": within_10, "Within20Pct": within_20
    }


def evaluate_models(df: pd.DataFrame, mode: str = "quick", max_rows: Optional[int] = None) -> Dict[str, Any]:
    data = df.copy()
    if max_rows and len(data) > max_rows:
        data = data.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    X = data[PREFERRED_FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

    models = get_candidate_models(mode)
    cv_splits = 3 if mode == "quick" else 5
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    leaderboard_rows = []
    fold_rows = []
    predictions = []
    fitted = {}

    for name, model in models.items():
        cv_scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            return_train_score=False, n_jobs=1
        )
        for i, (mae_neg, r2_val) in enumerate(zip(cv_scores["test_mae"], cv_scores["test_r2"]), start=1):
            fold_rows.append({"Model": name, "Fold": i, "MAE": -mae_neg, "R2": r2_val})
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        met = regression_metrics(y_test, pred)
        met.update({
            "Model": name,
            "CV_MAE_mean": -cv_scores["test_mae"].mean(),
            "CV_MAE_std": cv_scores["test_mae"].std(),
            "CV_R2_mean": cv_scores["test_r2"].mean(),
            "CV_R2_std": cv_scores["test_r2"].std(),
        })
        leaderboard_rows.append(met)
        fitted[name] = model
        if name != "Dummy Baseline":
            for actual, p in zip(y_test, pred):
                predictions.append({"Model": name, "Actual": actual, "Predicted": p, "AbsoluteError": abs(actual-p), "APE": abs(actual-p)/actual*100})

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(["MAE", "R2"], ascending=[True, False]).reset_index(drop=True)
    best_model_name = leaderboard.iloc[0]["Model"]
    best_model = fitted[best_model_name]

    return {
        "mode": mode,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": PREFERRED_FEATURES,
        "leaderboard": leaderboard,
        "fold_metrics": pd.DataFrame(fold_rows),
        "predictions": pd.DataFrame(predictions),
        "best_model_name": best_model_name,
        "best_model": best_model,
        "fitted_models": fitted,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "dataset_rows": len(data),
    }


def market_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": len(df),
        "median_price": float(df[TARGET].median()),
        "mean_price": float(df[TARGET].mean()),
        "median_ppsqm": float(df["price_per_sqm"].median()),
        "districts": int(df["district"].nunique()),
        "features": len(PREFERRED_FEATURES),
    }


def predict_single(model, payload: Dict[str, Any]) -> float:
    row = {col: payload.get(col, np.nan) for col in PREFERRED_FEATURES}
    X = pd.DataFrame([row])
    return float(model.predict(X)[0])


def save_model(model, path: str | Path):
    joblib.dump(model, path)


def model_feature_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE, scoring="neg_mean_absolute_error", n_jobs=1)
    return pd.DataFrame({
        "Feature": X.columns,
        "Importance_MAE_Increase": result.importances_mean,
        "Std": result.importances_std,
    }).sort_values("Importance_MAE_Increase", ascending=False)


def bootstrap_metric_ci(y_true, y_pred, n_boot: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for metric_name, fn in {
        "MAE": lambda a, p: mean_absolute_error(a, p),
        "RMSE": lambda a, p: math.sqrt(mean_squared_error(a, p)),
        "R2": lambda a, p: r2_score(a, p),
        "MAPE": lambda a, p: np.mean(np.abs((a-p)/np.maximum(a,1))) * 100,
    }.items():
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_true), len(y_true))
            vals.append(fn(y_true[idx], y_pred[idx]))
        rows.append({
            "Metric": metric_name,
            "Estimate": fn(y_true, y_pred),
            "CI95_Lower": np.percentile(vals, 2.5),
            "CI95_Upper": np.percentile(vals, 97.5),
        })
    return pd.DataFrame(rows)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    return pd.util.hash_pandas_object(df[PREFERRED_FEATURES + [TARGET]], index=False).sum().astype(str)


def environment_metadata() -> Dict[str, str]:
    import sklearn
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "random_state": str(RANDOM_STATE),
    }
