"""
SCENARIO FORECASTING API
=========================
This is the API layer for your scenario forecasting module.

In plain English:
  Your Streamlit page is great for researchers to explore interactively.
  But the React frontend (your teammate's dashboard) needs to fetch data
  programmatically — it can't "click" your Streamlit page.

  So this API acts as the bridge:
    React frontend sends a request → this API runs the scenario → returns JSON

  It follows the EXACT same pattern as your teammate's FastAPI backend,
  so it plugs straight into the existing system.

HOW IT FITS IN THE ARCHITECTURE:
  React Frontend (:5173)
      ↓ POST /scenarios/forecast
  Your Scenario API (:8502)          ← THIS FILE
      ↓ runs SARIMA + scenarios
  Returns JSON with all scenario trajectories

  Your teammate's API (:8000) handles single SARIMA predictions.
  Your API (:8502) handles multi-scenario comparisons.
  They work side by side.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# APP SETUP
#
# FastAPI() creates the API application.
# The title and description show up in the
# auto-generated docs at /docs
# ─────────────────────────────────────────────
app = FastAPI(
    title="Climate Scenario Forecasting API",
    description="""
    Scenario-based emissions forecasting built on top of SARIMA baseline predictions.
    
    This API is part of the Climate Emissions Forecasting System.
    It accepts a country + gas + horizon and returns multiple scenario trajectories.
    
    Part of the microservices architecture:
    - React Frontend: :5173
    - SARIMA Inference API (teammate): :8000  
    - Scenario Forecasting API (this): :8502
    - Streamlit Research UI: :8501
    """,
    version="1.0.0"
)


# ─────────────────────────────────────────────
# CORS MIDDLEWARE
#
# CORS = Cross-Origin Resource Sharing.
# Without this, the React frontend (running on
# port 5173) cannot talk to this API (port 8502)
# because browsers block cross-origin requests
# by default for security.
#
# allow_origins=["*"] means any origin can call
# this API — fine for a hackathon/development.
# In production you'd lock this to specific domains.
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# SCENARIO DEFINITIONS
# (Same as in scenario_forecasting.py — kept
# in sync so both the UI and API behave identically)
# ─────────────────────────────────────────────
SCENARIOS = {
    "business_as_usual": {
        "label": "Business As Usual (BAU)",
        "description": "Current emission trends continue unchanged. No new policies are enacted.",
        "annual_reduction_pct": 0.0,
        "shock_year": None,
        "shock_magnitude": 1.0,
        "trend_acceleration": 1.0,
        "color": "#f85149",
    },
    "paris_agreement": {
        "label": "Paris Agreement (1.5°C Path)",
        "description": "Aggressive decarbonisation requiring ~7% annual emission reductions.",
        "annual_reduction_pct": 7.0,
        "shock_year": None,
        "shock_magnitude": 1.0,
        "trend_acceleration": 1.0,
        "color": "#3fb950",
    },
    "moderate_policy": {
        "label": "Moderate Policy (NDC Targets)",
        "description": "Countries meet their Nationally Determined Contributions from COP summits.",
        "annual_reduction_pct": 3.0,
        "shock_year": None,
        "shock_magnitude": 1.0,
        "trend_acceleration": 0.97,
        "color": "#e3b341",
    },
    "economic_shock": {
        "label": "Economic Shock / Crisis",
        "description": "Sudden economic disruption causes a sharp short-term drop, followed by rebound.",
        "annual_reduction_pct": -2.0,
        "shock_year": 2026,
        "shock_magnitude": 0.75,
        "trend_acceleration": 1.02,
        "color": "#a371f7",
    },
    "green_technology": {
        "label": "Green Technology Breakthrough",
        "description": "Rapid adoption of renewables and carbon capture accelerates beyond projections.",
        "annual_reduction_pct": 5.0,
        "shock_year": 2027,
        "shock_magnitude": 0.85,
        "trend_acceleration": 0.93,
        "color": "#58a6ff",
    },
}

# GWP factors for CO2 equivalent conversion
GWP = {"CO2": 1, "CH4": 28, "N2O": 273}

# In-memory model cache — same strategy as your teammate's API
# Key: "country_gas", Value: trained SARIMA result
model_cache = {}


# ─────────────────────────────────────────────
# REQUEST & RESPONSE MODELS
#
# Pydantic models define exactly what shape the
# JSON request and response must be.
# FastAPI automatically validates incoming requests
# against these models and returns clear errors
# if something is wrong.
#
# This matches the style of your teammate's API:
#   POST /predict → { "country": "Kenya", "gas": "CO2", "forecast_horizon": 10 }
# ─────────────────────────────────────────────
class ScenarioForecastRequest(BaseModel):
    country: str = Field(..., example="Canada", description="Country name as it appears in the dataset")
    gas: str = Field(default="CO2", example="CO2", description="Greenhouse gas: CO2, CH4, or N2O")
    forecast_horizon: int = Field(default=15, ge=1, le=30, example=15, description="Years to forecast (1-30)")
    scenarios: list[str] = Field(
        default=["business_as_usual", "paris_agreement"],
        example=["business_as_usual", "paris_agreement", "moderate_policy"],
        description="List of scenario keys to include in the response"
    )
    custom_scenario: Optional[dict] = Field(
        default=None,
        example={
            "annual_reduction_pct": 4.0,
            "shock_year": 2028,
            "shock_magnitude": 0.9,
            "trend_acceleration": 0.98
        },
        description="Optional custom scenario parameters"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "country": "Canada",
                "gas": "CO2",
                "forecast_horizon": 15,
                "scenarios": ["business_as_usual", "paris_agreement", "economic_shock"]
            }
        }


class YearValue(BaseModel):
    year: int
    value: float


class ScenarioResult(BaseModel):
    scenario_key: str
    label: str
    description: str
    color: str
    forecast: list[YearValue]
    final_value: float
    cumulative_total: float
    pct_vs_baseline: float
    risk_level: str


class ScenarioForecastResponse(BaseModel):
    country: str
    gas: str
    forecast_horizon: int
    baseline_forecast: list[YearValue]
    historical_data: list[YearValue]
    scenarios: list[ScenarioResult]
    metrics: dict
    model_info: dict


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# (Same logic as scenario_forecasting.py,
#  refactored to work without Streamlit)
# ─────────────────────────────────────────────
def load_data(gas: str = "CO2") -> pd.DataFrame:
    """Load the emissions CSV for the given gas."""
    file_map = {"CO2": "data.csv", "CH4": "data_ch4.csv", "N2O": "data_n2o.csv"}
    filepath = file_map.get(gas, "data.csv")
    try:
        df = pd.read_csv(filepath, skiprows=9)
        df = df.dropna(axis=1, how='all').drop_duplicates().dropna(how='all')
        df = df.dropna(subset=['Name', 'Country_code_A3'])
        year_cols = [c for c in df.columns if c.startswith("Y_")]
        df = df.dropna(subset=year_cols, thresh=int(0.7 * len(year_cols)))
        df[year_cols] = df[year_cols].astype(float)
        return df.reset_index(drop=True)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Data file '{filepath}' not found on server.")


def get_country_series(df: pd.DataFrame, country: str) -> pd.Series:
    """Extract annual emissions time series for one country."""
    country_df = df[df['Name'] == country]
    if country_df.empty:
        raise HTTPException(status_code=404, detail=f"Country '{country}' not found in dataset.")
    year_cols = [c for c in df.columns if c.startswith("Y_")]
    annual = country_df[year_cols].sum(axis=0)
    years = [int(c.replace("Y_", "")) for c in year_cols]
    series = pd.Series(annual.values, index=years)
    return series[series > 0]


def train_sarima(series: pd.Series) -> tuple:
    """Train SARIMA model and return (result, fitted_model)."""
    cache_key = f"{series.name}"
    if cache_key in model_cache:
        return model_cache[cache_key]

    dt_index = pd.date_range(start=str(series.index[0]), periods=len(series), freq="YE")
    ts = pd.Series(series.values, index=dt_index)

    model = SARIMAX(ts, order=(3, 1, 3), seasonal_order=(1, 0, 1, 5),
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    model_cache[cache_key] = (result, ts)
    return result, ts


def apply_scenario(baseline_values: list, forecast_years: list, params: dict) -> list:
    """Apply scenario parameters to baseline forecast."""
    result = []
    reduction_rate = params.get("annual_reduction_pct", 0.0) / 100.0
    shock_year = params.get("shock_year")
    shock_magnitude = params.get("shock_magnitude", 1.0)
    acceleration = params.get("trend_acceleration", 1.0)

    for i, (year, base_val) in enumerate(zip(forecast_years, baseline_values)):
        effective_rate = reduction_rate * (acceleration ** i)
        scenario_val = base_val * ((1 - effective_rate) ** i)

        if shock_year and year >= shock_year:
            years_since_shock = year - shock_year
            if years_since_shock == 0:
                scenario_val *= shock_magnitude
            else:
                recovery = min(years_since_shock * 0.05, 1.0)
                scenario_val = scenario_val * (shock_magnitude + (1 - shock_magnitude) * recovery)

        result.append(max(scenario_val, 0))
    return result


def compute_risk_level(pct_vs_baseline: float) -> str:
    """Return plain-English risk label."""
    if pct_vs_baseline > 20:
        return "Critical"
    elif pct_vs_baseline > 5:
        return "High"
    elif pct_vs_baseline > -10:
        return "Moderate"
    elif pct_vs_baseline > -30:
        return "Low"
    else:
        return "Transformative"


# ══════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════

# ─────────────────────────────────────────────
# HEALTH CHECK
# Your teammate has GET /health — we match that
# pattern so the Docker health check works the
# same way across all services.
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint for Docker and load balancers."""
    return {
        "status": "healthy",
        "service": "scenario-forecasting-api",
        "version": "1.0.0",
        "port": 8502,
        "cached_models": len(model_cache)
    }


# ─────────────────────────────────────────────
# GET AVAILABLE SCENARIOS
# The React frontend calls this first to know
# what scenarios exist, so it can build the UI
# dynamically without hardcoding scenario names.
# ─────────────────────────────────────────────
@app.get("/scenarios")
def get_scenarios():
    """Return all available scenario definitions."""
    return {
        "scenarios": [
            {
                "key": key,
                "label": data["label"],
                "description": data["description"],
                "color": data["color"],
                "parameters": {
                    "annual_reduction_pct": data["annual_reduction_pct"],
                    "shock_year": data.get("shock_year"),
                    "shock_magnitude": data["shock_magnitude"],
                }
            }
            for key, data in SCENARIOS.items()
        ],
        "total": len(SCENARIOS)
    }


# ─────────────────────────────────────────────
# GET AVAILABLE COUNTRIES
# Lets the React frontend populate its country
# dropdown directly from the data, not hardcoded.
# ─────────────────────────────────────────────
@app.get("/countries")
def get_countries(gas: str = "CO2"):
    """Return list of available countries in the dataset."""
    df = load_data(gas)
    countries = sorted(df['Name'].dropna().unique().tolist())
    return {"countries": countries, "total": len(countries)}


# ─────────────────────────────────────────────
# MAIN ENDPOINT — SCENARIO FORECAST
#
# This is the core endpoint. React sends a POST
# request with country/gas/scenarios, and gets
# back all the forecast trajectories as JSON.
#
# Example request:
#   POST /scenarios/forecast
#   {
#     "country": "Canada",
#     "gas": "CO2",
#     "forecast_horizon": 15,
#     "scenarios": ["business_as_usual", "paris_agreement"]
#   }
# ─────────────────────────────────────────────
@app.post("/scenarios/forecast", response_model=ScenarioForecastResponse)
def scenario_forecast(request: ScenarioForecastRequest):
    """
    Generate scenario-based emissions forecasts for a country.
    
    Returns the SARIMA baseline plus all requested scenario trajectories,
    ready for the React frontend to render as charts.
    """
    # 1. Load data and get country time series
    df = load_data(request.gas)
    series = get_country_series(df, request.country)
    series.name = f"{request.country}_{request.gas}"

    if len(series) < 10:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data for {request.country}. Need at least 10 years of data."
        )

    # 2. Train SARIMA (or load from cache)
    try:
        sarima_result, ts = train_sarima(series)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SARIMA training failed: {str(e)}")

    # 3. Generate baseline forecast
    forecast_result = sarima_result.get_forecast(steps=request.forecast_horizon)
    baseline_values = forecast_result.predicted_mean.values.tolist()
    conf_int = forecast_result.conf_int(alpha=0.2).values

    last_year = series.index[-1]
    forecast_years = list(range(last_year + 1, last_year + request.forecast_horizon + 1))

    # 4. Historical data (for the chart)
    historical = [YearValue(year=int(y), value=round(float(v), 2))
                  for y, v in zip(series.index, series.values)]

    # 5. Baseline forecast points
    baseline_forecast = [
        YearValue(year=y, value=round(max(v, 0), 2))
        for y, v in zip(forecast_years, baseline_values)
    ]

    # 6. Apply each requested scenario
    scenario_results = []
    baseline_total = sum(baseline_values)

    # Add custom scenario to the pool if provided
    scenarios_to_run = list(request.scenarios)
    if request.custom_scenario:
        SCENARIOS["custom"] = {
            "label": "Custom Scenario",
            "description": "User-defined scenario parameters",
            "color": "#ff9500",
            **request.custom_scenario
        }
        scenarios_to_run.append("custom")

    for scenario_key in scenarios_to_run:
        if scenario_key not in SCENARIOS:
            continue  # skip unknown keys silently

        params = SCENARIOS[scenario_key]
        scenario_vals = apply_scenario(baseline_values, forecast_years, params)

        final_val = scenario_vals[-1]
        cumulative = sum(scenario_vals)
        pct_vs_baseline = ((final_val - baseline_values[-1]) / baseline_values[-1] * 100) \
            if baseline_values[-1] > 0 else 0

        scenario_results.append(ScenarioResult(
            scenario_key=scenario_key,
            label=params["label"],
            description=params["description"],
            color=params["color"],
            forecast=[
                YearValue(year=y, value=round(max(v, 0), 2))
                for y, v in zip(forecast_years, scenario_vals)
            ],
            final_value=round(final_val, 2),
            cumulative_total=round(cumulative, 2),
            pct_vs_baseline=round(pct_vs_baseline, 2),
            risk_level=compute_risk_level(pct_vs_baseline)
        ))

    # 7. Summary metrics
    gwp = GWP.get(request.gas, 1)
    best_scenario = min(scenario_results, key=lambda s: s.cumulative_total) if scenario_results else None

    return ScenarioForecastResponse(
        country=request.country,
        gas=request.gas,
        forecast_horizon=request.forecast_horizon,
        baseline_forecast=baseline_forecast,
        historical_data=historical,
        scenarios=scenario_results,
        metrics={
            "baseline_total_kt": round(baseline_total, 2),
            "baseline_total_co2e_mt": round(baseline_total * gwp / 1000, 2),
            "best_scenario": best_scenario.scenario_key if best_scenario else None,
            "best_scenario_saving_kt": round(baseline_total - best_scenario.cumulative_total, 2) if best_scenario else 0,
            "confidence_interval_lower": [round(max(v, 0), 2) for v in conf_int[:, 0].tolist()],
            "confidence_interval_upper": [round(max(v, 0), 2) for v in conf_int[:, 1].tolist()],
        },
        model_info={
            "model": "SARIMA(3,1,3)(1,0,1)[5]",
            "data_source": "EDGAR",
            "historical_range": f"{series.index[0]}-{series.index[-1]}",
            "training_points": len(series),
            "cached": f"{request.country}_{request.gas}" in model_cache,
        }
    )


# ─────────────────────────────────────────────
# CLEAR MODEL CACHE
# Useful if new data is uploaded and you want
# the model to retrain from scratch.
# ─────────────────────────────────────────────
@app.delete("/cache")
def clear_cache():
    """Clear the in-memory model cache to force retraining."""
    count = len(model_cache)
    model_cache.clear()
    return {"message": f"Cleared {count} cached models"}


# ─────────────────────────────────────────────
# RUN THE SERVER
# When you run this file directly with Python,
# uvicorn starts the server on port 8502.
# In Docker, this is called automatically.
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scenario_api:app", host="0.0.0.0", port=8502, reload=True)
