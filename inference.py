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


def get_X():
    df = pd.read_csv(os.environ["INFERENCE_DATA_PATH"])
    X = base_transform(df)
    return X, df  # X limpio, df original para juntar luego


if __name__ == "__main__":
    X, original_df = get_X()

    with open(os.environ["MODEL_PATH"], "rb") as f:
        model_dict = cloudpickle.load(f)

    pipe = model_dict["pipe"]
    threshold = model_dict["threshold"]

    y_probs = pipe.predict_proba(X)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    results = original_df.loc[X.index].copy()
    results["prediction"] = y_pred

    output_path = os.environ.get("OUTPUT_PATH", "inference_output.csv")
    results.to_csv(output_path, index=False)

    print(f"✅ Predicciones guardadas en {output_path}")
