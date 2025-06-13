# inference.py

import os
import numpy as np
import pandas as pd
import cloudpickle
from sklearn import set_config

set_config(transform_output="pandas")


def base_transform(df):
    df = df.copy()
    hoteles_df = pd.read_csv("hotels.csv")

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

    # Protección contra división por 0 (para evitar infinitos)
    df["stay_nights"] = df["stay_nights"].replace(0, pd.NA)
    df["total_rooms"] = df["total_rooms"].replace(0, pd.NA)

    # Elimina infinitos y valores grandes peligrosos
    df = df.replace([np.inf, -np.inf], pd.NA)
    df = df.fillna(np.nan)

    df = df.loc[:, ~df.columns.duplicated()]
    return df.reset_index(drop=True)


def get_X():
    df = pd.read_csv(os.environ["INFERENCE_DATA_PATH"])
    X = base_transform(df)
    return X, df


if __name__ == "__main__":
    X, original_df = get_X()

    with open(os.environ["MODEL_PATH"], "rb") as f:
        model_dict = cloudpickle.load(f)

    pipe = model_dict["pipe"]
    threshold = model_dict["threshold"]

    y_probs = pipe.predict_proba(X)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    results = original_df.iloc[X.index].copy().reset_index(drop=True)
    results["prediction"] = y_pred

    output_path = os.environ.get("OUTPUT_PATH", "inference_output.csv")
    results.to_csv(output_path, index=False)

    print(f" Predicciones guardadas en {output_path}")
