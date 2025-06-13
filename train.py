# train.py

import os
import numpy as np
import pandas as pd
import cloudpickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

set_config(transform_output="pandas")


# Base transform

def base_transform(df):
    df = df.copy()
    hoteles_df = pd.read_csv("hotels.csv")

    df = df[df["reservation_status"] != "Booked"]

    df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors="coerce")
    df["lead_time"] = (df["arrival_date"] - df["booking_date"]).dt.days

    df["arrival_month"] = df["arrival_date"].dt.month
    df["booking_month"] = df["booking_date"].dt.month

    hoteles_df = hoteles_df.rename(columns={"country": "country_hotel"})
    df = df.merge(hoteles_df, on="hotel_id", how="left", suffixes=("", "_hotel"))

    df["room_type"] = df["room_type"].apply(lambda x: x if x in {"A", "D", "E"} else "Other")
    df["distribution_channel"] = df["distribution_channel"].apply(
        lambda x: x if x in {"TA/TO", "Direct", "Corporate"} else "Other"
    )
    segmentos_validos = {"Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate"}
    df["market_segment"] = df["market_segment"].apply(
        lambda x: x if x in segmentos_validos else ("Missing" if pd.isna(x) else "Other")
    )
    df["board"] = df["board"].apply(lambda x: x if x in {"BB", "SC", "HB"} else "Other")

    for col in ["parking", "restaurant", "pool_and_spa"]:
        df[col] = df[col].astype(int)

    df["required_car_parking_spaces"] = df["required_car_parking_spaces"].clip(upper=3).fillna(0)
    df = df[df["stay_nights"] != 0]
    df["stay_nights"] = df["stay_nights"].apply(lambda x: 21 if x > 21 else x)
    df["lead_time"] = df["lead_time"].clip(upper=210)

    df = df.fillna(pd.NA)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.reset_index(drop=True)


# Target

def get_target(df):
    df = df.copy()
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors="coerce")

    cancelada = df["reservation_status"] == "Canceled"
    dias_entre = (df["arrival_date"] - df["reservation_status_date"]).dt.days
    mask_validas = df["arrival_date"].notna() & df["reservation_status_date"].notna()

    target = (
            cancelada &
            (dias_entre <= 30) &
            (dias_entre >= 0) &
            mask_validas
    ).astype(int)

    return target


# Feature Engineering

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rate_median_ = None
        self.review_median_ = None

    def fit(self, X, y=None):
        self.rate_median_ = X["rate"].median()
        self.review_median_ = X["avg_review"].median()
        return self

    def transform(self, X):
        df = X.copy()
        df["rate_per_night"] = df["rate"] / df["stay_nights"]
        df["guests_per_room"] = df["total_guests"] / df["total_rooms"]
        df["short_booking"] = (df["lead_time"] <= 3).astype(int)
        df["high_price"] = (df["rate"] > self.rate_median_).astype(int)
        df["high_review"] = (df["avg_review"] >= self.review_median_).astype(int)
        df["luxury_booking"] = ((df["high_price"] == 1) & (df["high_review"] == 1)).astype(int)
        df["lead_time_group"] = pd.cut(
            df["lead_time"], bins=[0, 7, 30, 90, 210],
            labels=["muy_corto", "corto", "medio", "largo"]
        ).astype(str)
        df["many_special_requests"] = (df["special_requests"] > 2).astype(int)
        return df


# Preprocessing pipeline


drop_columns = [
    "hotel_id", "arrival_date", "booking_date", "reservation_status_date",
    "reservation_status", "country"
]

def drop_unwanted_columns(df):
    return df.drop(columns=drop_columns)

col_remover = FunctionTransformer(drop_unwanted_columns, validate=False)

class TopCountryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=5):
        self.top_n = top_n
        self.top_countries_ = None

    def fit(self, X, y=None):
        self.top_countries_ = X["country"].value_counts().head(self.top_n).index
        return self

    def transform(self, X):
        X = X.copy()
        X["country_grouped"] = X["country"].apply(
            lambda x: x if x in self.top_countries_ else "Other"
        )
        return X

num_zero_cols = ["required_car_parking_spaces", "special_requests"]

num_transformer_zero = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", StandardScaler())
])

num_transformer_no_zero = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first"))
])

transformer = ColumnTransformer([
    ("num_zero", num_transformer_zero, num_zero_cols),
    ("num_no_zero", num_transformer_no_zero,
     selector(dtype_exclude="object", pattern="^(?!.*required_car_parking_spaces|.*special_requests).*")),
    ("cat", cat_transformer, selector(dtype_include="object"))
], verbose_feature_names_out=True)

preprocess_pipeline = Pipeline([
    ("group_country", TopCountryGrouper()),
    ("col_remover", col_remover),
    ("cast_types", FunctionTransformer(lambda df: df.infer_objects(), validate=False)),
    ("transformer", transformer),
    ("var_threshold", VarianceThreshold(threshold=0.01))
])


# Modelo completo

def get_pipeline():
    return Pipeline([
        ("add_features", FeatureEngineer()),
        ("preprocess", preprocess_pipeline),
        ("clf", LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.1668100537200059,
            penalty="l2",
            solver="newton-cg",
            tol=0.001,
            random_state=42
        ))
    ])




def get_X_y():
    df = pd.read_csv(os.environ["TRAIN_DATA_PATH"])
    X = base_transform(df)
    y = get_target(X)
    if len(X) != len(y):
        raise ValueError(f" X and y have different lengths: {len(X)} vs {len(y)}")
    return X, y


if __name__ == "__main__":
    X, y = get_X_y()
    pipe = get_pipeline()
    pipe.fit(X, y)

    # Calcular el mejor threshold
    y_probs = pipe.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_probs)

    valid = (precision + recall) > 0
    f1_scores = 2 * (precision[valid] * recall[valid]) / (precision[valid] + recall[valid])
    thresholds_valid = thresholds[valid[:-1]]
    best_threshold = thresholds_valid[np.argmax(f1_scores)]

    # Guardar el modelo + threshold
    with open(os.environ["MODEL_PATH"], "wb") as f:
        cloudpickle.dump({"pipe": pipe, "threshold": best_threshold}, f)

    print(f"✅ Modelo y threshold guardados correctamente.")

