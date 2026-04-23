from functools import lru_cache
import os
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import (
    aggregate_country_year,
    normalize_columns,
    predict_from_datum,
    read_emissions_input,
    train_sarima,
    wide_to_long,
)


DATA_PATHS = {
    "co2": "data.csv",
    "n2o": "data_n2o.csv",
    "ch4": "data_ch4.csv",
}


class PredictRequest(BaseModel):
    datum: Dict[str, Any]
    country: str | None = None
    gas_type: str = "co2"


@lru_cache(maxsize=4)
def load_data(gas_type: str = "co2") -> pd.DataFrame:
    data_path = DATA_PATHS.get(gas_type, DATA_PATHS["co2"])
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset for gas_type='{gas_type}' not found at '{data_path}'. "
            f"Add the dataset file or use gas_type='co2'."
        )
    # Reuse the same robust ingestion used by Streamlit app.py
    # to handle CSVs with leading blank rows/noisy headers.
    with open(data_path, "rb") as file_obj:
        df = read_emissions_input(uploaded_file=file_obj)
    df = normalize_columns(df)
    df = df.dropna(axis=1, how="all")
    return df


model_cache: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="CO2 SARIMA Backend API", version="1.0.0")

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
origins_env = os.getenv("CORS_ORIGINS", "")
allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()] or default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_or_train_country_model(country: str, gas_type: str = "co2") -> Dict[str, Any]:
    cache_key = f"{gas_type}::{country}"
    if cache_key in model_cache:
        return model_cache[cache_key]

    df = load_data(gas_type=gas_type)
    long_df = wide_to_long(df)
    yearly_df = aggregate_country_year(long_df, country)
    train, test, model_fit, forecast = train_sarima(yearly_df)

    model_cache[cache_key] = {
        "train": train,
        "test": test,
        "model_fit": model_fit,
        "forecast": forecast,
        "yearly_df": yearly_df,
    }
    return model_cache[cache_key]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        gas_type = (payload.gas_type or "co2").strip().lower()
        if gas_type not in DATA_PATHS:
            raise HTTPException(status_code=400, detail="gas_type must be one of: co2, n2o, ch4")

        country = payload.country or payload.datum.get("Country")
        if not country:
            raise HTTPException(status_code=400, detail="country is required (in payload.country or datum.Country).")

        model_bundle = get_or_train_country_model(str(country), gas_type=gas_type)
        pred_value, pred_year, _ = predict_from_datum(
            payload.datum,
            model_bundle["model_fit"],
            model_bundle["train"],
        )
        response = {
            "country": country,
            "year": int(pred_year),
            "selected_year": int(pred_year),
            "gas_type": gas_type,
        }
        if gas_type == "n2o":
            response["predicted_n2o"] = float(pred_value)
        elif gas_type == "ch4":
            response["predicted_ch4"] = float(pred_value)
        else:
            response["predicted_co2"] = float(pred_value)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
