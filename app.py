import io
import json
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


MODEL_PATH = "sarima_model.pkl"


def _find_header_row(raw_df: pd.DataFrame) -> int:
    """Find the first row that likely contains year columns."""
    for idx, row in raw_df.iterrows():
        values = [str(v).strip() for v in row.values if pd.notna(v)]
        year_like = [v for v in values if v.startswith("Y_")]
        if len(year_like) >= 5:
            return idx
    return 0


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map incoming columns to expected names where possible."""
    column_map = {
        "Country": ["Country", "Name", "country", "country_name"],
        "ISO_Code": ["ISO_Code", "Country_code_A3", "ISO3", "iso_code"],
        "Sector": ["Sector", "ipcc_code_2006_for_standard_report_name", "sector"],
        "fossil_bio": ["fossil_bio", "fossil_bio ", "fuel_type"],
    }

    rename_map: Dict[str, str] = {}
    for target, candidates in column_map.items():
        for c in candidates:
            if c in df.columns:
                rename_map[c] = target
                break
    return df.rename(columns=rename_map)


def read_emissions_input(
    uploaded_file: Optional[io.BytesIO] = None, df_input: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Data ingestion module:
    Accept either uploaded CSV or API-style DataFrame input.
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if not any(str(c).startswith("Y_") for c in df.columns):
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, header=None)
                header_row = _find_header_row(raw_df)
                header = raw_df.iloc[header_row].tolist()
                data_df = raw_df.iloc[header_row + 1 :].copy()
                data_df.columns = header
                df = data_df
        except Exception:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, header=None)
            header_row = _find_header_row(raw_df)
            header = raw_df.iloc[header_row].tolist()
            data_df = raw_df.iloc[header_row + 1 :].copy()
            data_df.columns = header
            df = data_df
    elif df_input is not None:
        df = df_input.copy()
    else:
        raise ValueError("Provide either uploaded_file or df_input.")

    df = normalize_columns(df)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    return df


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide format (Y_1970...Y_2018) to long format."""
    year_cols = [c for c in df.columns if str(c).startswith("Y_")]
    if not year_cols:
        raise ValueError("No year columns found (expected columns like Y_1970).")

    id_vars = [c for c in ["Country", "ISO_Code", "Sector", "fossil_bio"] if c in df.columns]
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year_col",
        value_name="co2_emission",
    )
    long_df["Year"] = long_df["year_col"].str.extract(r"(\d{4})").astype(float).astype("Int64")
    long_df["co2_emission"] = _safe_numeric(long_df["co2_emission"])
    long_df = long_df.dropna(subset=["Year"]).copy()
    long_df["Year"] = long_df["Year"].astype(int)
    return long_df


def aggregate_country_year(long_df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Aggregate sector-level emissions into yearly totals for a country."""
    if "Country" not in long_df.columns:
        raise ValueError("Country column is required after ingestion.")
    country_df = long_df[long_df["Country"] == country].copy()
    if country_df.empty:
        raise ValueError(f"No rows found for country '{country}'.")

    yearly = (
        country_df.groupby("Year", as_index=False)["co2_emission"]
        .sum()
        .sort_values("Year")
        .reset_index(drop=True)
    )
    return yearly


def train_sarima(yearly_df: pd.DataFrame):
    """Train strict SARIMA model with fixed parameters and 80/20 split."""
    if len(yearly_df) < 12:
        raise ValueError("Not enough annual points to train/test (need at least 12).")

    series = yearly_df.set_index("Year")["co2_emission"].astype(float)
    split_idx = max(int(len(series) * 0.8), 1)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    train.attrs["latest_country_year"] = int(series.index.max())

    # STRICT MODEL CONSTRAINTS (do not modify)
    model = SARIMAX(
        train,
        order=(3, 1, 3),
        seasonal_order=(1, 0, 1, 5),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)

    forecast = pd.Series(dtype=float)
    if len(test) > 0:
        forecast = model_fit.forecast(steps=len(test))
        forecast.index = test.index

    return train, test, model_fit, forecast


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Optional evaluation metrics."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.where(np.abs(y_true.values) < 1e-9, np.nan, y_true.values)
    mape = np.nanmean(np.abs((y_true.values - y_pred.values) / denom)) * 100.0
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE (%)": float(mape)}


def parse_datum(datum):
    """Accept dict / JSON string / single-row DataFrame as API-style datum."""
    if isinstance(datum, pd.DataFrame):
        if len(datum) != 1:
            raise ValueError("DataFrame datum must contain exactly one row.")
        return datum.copy()
    if isinstance(datum, dict):
        return pd.DataFrame([datum])
    if isinstance(datum, str):
        parsed = json.loads(datum)
        if isinstance(parsed, list):
            if len(parsed) != 1:
                raise ValueError("JSON list datum must contain exactly one object.")
            return pd.DataFrame(parsed)
        if isinstance(parsed, dict):
            return pd.DataFrame([parsed])
    raise ValueError("datum must be dict, JSON string, or single-row DataFrame.")


def predict_from_datum(datum, model_fit, train):
    """
    Predict from one raw wide-format API row using existing trained model only.
    Must not retrain model.
    """
    datum_df = normalize_columns(parse_datum(datum))
    long_df = wide_to_long(datum_df)
    yearly = (
        long_df.groupby("Year", as_index=False)["co2_emission"]
        .sum()
        .sort_values("Year")
        .reset_index(drop=True)
    )
    if yearly.empty:
        raise ValueError("Could not derive yearly totals from datum.")

    target_year = int(train.attrs.get("latest_country_year", yearly["Year"].max()))
    last_train_year = int(train.index.max())

    if target_year <= last_train_year:
        # Keep prediction on the requested year (no forced next-year step).
        prediction_result = model_fit.get_prediction(start=target_year, end=target_year)
        prediction_value = float(prediction_result.predicted_mean.iloc[0])
    else:
        steps = target_year - last_train_year
        forecast_values = model_fit.forecast(steps=steps)
        prediction_value = float(forecast_values.iloc[-1])

    return prediction_value, target_year, yearly


def plot_results(
    yearly_df: pd.DataFrame,
    train: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    api_prediction: Optional[Tuple[int, float]] = None,
):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        yearly_df["Year"],
        yearly_df["co2_emission"],
        color="steelblue",
        linewidth=1.8,
        label="Historical CO2",
    )
    ax.plot(train.index, train.values, color="green", linewidth=2.0, label="Train")

    if len(test) > 0:
        ax.plot(test.index, test.values, color="orange", linewidth=2.0, label="Test")
    if len(forecast) > 0:
        ax.plot(
            forecast.index,
            forecast.values,
            color="purple",
            linestyle="--",
            linewidth=2.0,
            label="SARIMA Forecast",
        )

    ax.axvspan(2010, yearly_df["Year"].max() + 2, color="gray", alpha=0.15, label="Post-2010 Stable Regime")

    if api_prediction is not None:
        pred_year, pred_value = api_prediction
        ax.scatter([pred_year], [pred_value], color="red", s=80, zorder=5, label="API Prediction Point")

    ax.set_title("CO2 Emissions Forecasting (SARIMA)")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2 Emissions")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig


def save_model(model_fit, path: str = MODEL_PATH):
    model_fit.save(path)


@st.cache_resource(show_spinner=False)
def load_persisted_model(path: str = MODEL_PATH):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults

        return SARIMAXResults.load(path)
    except Exception:
        return None


def plot_train_test_forecast(yearly_df: pd.DataFrame, train: pd.Series, test: pd.Series, forecast: pd.Series):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(yearly_df["Year"], yearly_df["co2_emission"], color="steelblue", linewidth=1.8, label="Historical")
    ax.plot(train.index, train.values, color="green", linewidth=2.0, label="Train")
    if len(test) > 0:
        ax.plot(test.index, test.values, color="orange", linewidth=2.0, label="Test")
    if len(forecast) > 0:
        ax.plot(forecast.index, forecast.values, color="purple", linestyle="--", linewidth=2.0, label="Forecast")
    ax.axvspan(2010, yearly_df["Year"].max() + 1, color="gray", alpha=0.12, label="Post-2010 regime")
    ax.set_title("Aggregated CO2 Emissions and SARIMA Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2 Emissions")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residual_diagnostics(model_fit):
    resid = pd.Series(model_fit.resid).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(resid, bins=18, color="teal", alpha=0.8)
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    qqplot(resid, line="s", ax=axes[1], markerfacecolor="purple", markeredgecolor="purple", alpha=0.6)
    axes[1].set_title("Residual Q-Q Plot")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="CO2 SARIMA Scientific Console", layout="wide")
    st.title("CO2 Emissions Forecasting - SARIMA Analysis")
    st.caption("Minimal interface focused on model behavior, diagnostics, and reproducible forecasting.")

    uploaded_file = st.file_uploader("Upload emissions CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to begin analysis.")
        return

    try:
        df_raw = read_emissions_input(uploaded_file=uploaded_file)
    except Exception as exc:
        st.error(f"Failed to parse input file: {exc}")
        return

    if "Country" not in df_raw.columns:
        st.error("Missing Country column after normalization. Expected `Country` or `Name` in source.")
        return

    countries = sorted(df_raw["Country"].dropna().astype(str).unique().tolist())
    if not countries:
        st.error("No country values found.")
        return

    with st.expander("Dataset Preview and Structure", expanded=False):
        st.dataframe(df_raw.head(20), use_container_width=True)
        year_cols = [c for c in df_raw.columns if str(c).startswith("Y_")]
        st.write(f"Detected year columns: {len(year_cols)}")
        st.write(f"Detected rows: {len(df_raw)}")

    selected_country = st.selectbox("Select Country", countries)

    try:
        long_df = wide_to_long(df_raw)
        yearly_df = aggregate_country_year(long_df, selected_country)
        train, test, model_fit, forecast = train_sarima(yearly_df)
    except Exception as exc:
        st.error(f"Preprocessing/training failed: {exc}")
        return

    metrics = calculate_metrics(test, forecast) if len(test) > 0 else {"MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train Samples", f"{len(train)}")
    col2.metric("Test Samples", f"{len(test)}")
    col3.metric("AIC", f"{model_fit.aic:.2f}")
    col4.metric("BIC", f"{model_fit.bic:.2f}")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.3f}" if not np.isnan(metrics["MAE"]) else "N/A")
    m2.metric("RMSE", f"{metrics['RMSE']:.3f}" if not np.isnan(metrics["RMSE"]) else "N/A")
    m3.metric("MAPE (%)", f"{metrics['MAPE (%)']:.2f}" if not np.isnan(metrics["MAPE (%)"]) else "N/A")

    st.pyplot(plot_train_test_forecast(yearly_df, train, test, forecast), use_container_width=True)

    st.subheader("Residual Diagnostics")
    st.pyplot(plot_residual_diagnostics(model_fit), use_container_width=True)

    resid = pd.Series(model_fit.resid).dropna()
    lb_df = acorr_ljungbox(resid, lags=[5, 10], return_df=True)
    st.write("Ljung-Box test for residual autocorrelation:")
    st.dataframe(lb_df, use_container_width=True)

    with st.expander("Model Summary (statsmodels)", expanded=False):
        st.text(model_fit.summary().as_text())


if __name__ == "__main__":
    main()
