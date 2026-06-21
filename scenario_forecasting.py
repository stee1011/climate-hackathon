"""
SCENARIO-BASED FORECASTING MODULE
===================================
Your part of the Climate Emissions Forecasting System.

What this does in plain English:
  The SARIMA model (your teammate's work) gives ONE forecast — what happens
  if current trends continue unchanged. That's the baseline.

  Your job is to add "what if?" scenarios on top of that baseline:
    - What if the country aggressively cuts emissions? (Paris Agreement)
    - What if nothing changes? (Business as Usual)
    - What if there's a sudden policy shock or economic collapse?

  Each scenario is a different version of the future. Decision makers and
  researchers use these to compare outcomes and make policy choices.

HOW IT CONNECTS TO YOUR TEAMMATE'S WORK:
  - app.py has train_sarima() and predict_from_datum() — we import those
  - We take the baseline forecast and then mathematically "bend" it
    according to the scenario the user selects
  - The result gets shown on the same chart so you can compare all scenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# PAGE CONFIG
# Streamlit lets you set the browser tab title,
# icon, and layout before anything else renders.
# "wide" layout means it uses the full screen width.
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Scenario Forecaster",
    page_icon="🌍",
    layout="wide"
)


# ─────────────────────────────────────────────
# STYLING
# We inject CSS directly into the Streamlit page.
# This overrides the default look to feel more
# like a serious climate analytics tool.
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background: deep charcoal, like the dark dashboard in your arch doc */
    .stApp { background-color: #0f1117; color: #e8ecf0; }
    
    /* Sidebar gets a slightly lighter shade */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* Headers */
    h1, h2, h3 { color: #58a6ff; font-family: 'Segoe UI', sans-serif; }
    
    /* Metric cards (the big number boxes Streamlit shows) */
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
    }
    
    /* Scenario colour badges shown in the legend */
    .scenario-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 4px;
    }

    /* Info box */
    .info-box {
        background-color: #1c2128;
        border-left: 3px solid #58a6ff;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
        font-size: 0.9em;
    }
    
    /* Warning box */
    .warn-box {
        background-color: #1c1610;
        border-left: 3px solid #d29922;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SCENARIO DEFINITIONS
#
# This is the core intellectual work of your part.
# Each scenario has a name, description, and a set
# of parameters that describe HOW emissions change.
#
# Think of it like this: the SARIMA gives you a
# straight-ish line into the future. Each scenario
# says "now tilt that line up or down by this much,
# and at this speed, and maybe with this shock."
#
# Parameters explained:
#   annual_reduction_pct  → how many % per year do
#                           emissions go DOWN (negative = up)
#   shock_year            → year a sudden event happens
#   shock_magnitude       → how big is that one-time shock
#                           (0.8 = 20% sudden drop, 1.2 = 20% surge)
#   trend_acceleration    → does the trend get faster over time?
#                           (1.0 = constant, 0.95 = slowing reduction)
# ─────────────────────────────────────────────
SCENARIOS = {
    "📈 Business As Usual (BAU)": {
        "description": "Current emission trends continue unchanged. No new policies are enacted. Historical growth rate persists into the future.",
        "annual_reduction_pct": 0.0,
        "shock_year": None,
        "shock_magnitude": 1.0,
        "trend_acceleration": 1.0,
        "color": "#f85149",   # red — bad outcome
        "color_fill": "rgba(248,81,73,0.1)",
    },
    "🌿 Paris Agreement (1.5°C Path)": {
        "description": "Aggressive decarbonisation consistent with limiting warming to 1.5°C. Requires ~7% annual emission reductions, sustained over decades.",
        "annual_reduction_pct": 7.0,
        "shock_year": None,
        "shock_magnitude": 1.0,
        "trend_acceleration": 1.0,
        "color": "#3fb950",   # green — best case
        "color_fill": "rgba(63,185,80,0.1)",
    },
    "🏛️ Moderate Policy (NDC Targets)": {
        "description": "Countries meet their Nationally Determined Contributions — the pledges made at COP summits. Emissions reduce, but not fast enough for 1.5°C.",
        "annual_reduction_pct": 3.0,
        "shock_year": None,
        "shock_magnitude": 1.0,
        "trend_acceleration": 0.97,
        "color": "#e3b341",   # yellow — middle path
        "color_fill": "rgba(227,179,65,0.1)",
    },
    "💥 Economic Shock / Crisis": {
        "description": "A sudden economic disruption (like COVID-19 in 2020 or the 2008 financial crisis) causes a sharp short-term drop, followed by rebound growth.",
        "annual_reduction_pct": -2.0,   # slight rebound growth after shock
        "shock_year": 2026,
        "shock_magnitude": 0.75,        # 25% sudden drop in shock year
        "trend_acceleration": 1.02,     # then emissions creep back up
        "color": "#a371f7",   # purple — unexpected event
        "color_fill": "rgba(163,113,247,0.1)",
    },
    "⚡ Green Technology Breakthrough": {
        "description": "Rapid adoption of renewables and carbon capture technology accelerates beyond current projections — an optimistic but plausible path.",
        "annual_reduction_pct": 5.0,
        "shock_year": 2027,
        "shock_magnitude": 0.85,   # tech deployment causes a step-down
        "trend_acceleration": 0.93,
        "color": "#58a6ff",   # blue — tech-led future
        "color_fill": "rgba(88,166,255,0.1)",
    },
    "🔧 Custom Scenario": {
        "description": "Design your own scenario. Use the sliders below to set your assumptions.",
        "annual_reduction_pct": 0.0,
        "shock_year": None,
        "shock_magnitude": 1.0,
        "trend_acceleration": 1.0,
        "color": "#ff9500",
        "color_fill": "rgba(255,149,0,0.1)",
    }
}


# ─────────────────────────────────────────────
# CORE DATA LOADING
#
# This replicates what your teammate does in app.py.
# We load the emissions CSV, clean it the same way,
# and build a time series for the selected country.
#
# @st.cache_data means: run this function ONCE and
# remember the result. Don't re-run it every time
# the user clicks something. Makes the app fast.
# ─────────────────────────────────────────────
@st.cache_data
def load_emissions_data(filepath=r"C:\Users\USER\Downloads\data (1).csv"):
    """
    Load and clean the CO2 emissions CSV.
    Returns a tidy dataframe ready for time-series work.
    """
    try:
        df = pd.read_csv(filepath, skiprows=9)
    except FileNotFoundError:
        return None

    # Drop empty columns and duplicate rows (same as your teammate's notebook)
    df = df.dropna(axis=1, how='all')
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    df = df.dropna(subset=['Name', 'Country_code_A3'])

    # Keep only the year columns (Y_1970 to Y_2018)
    year_cols = [c for c in df.columns if c.startswith("Y_")]
    df = df.dropna(subset=year_cols, thresh=int(0.7 * len(year_cols)))
    df[year_cols] = df[year_cols].astype(float)
    df = df.reset_index(drop=True)

    return df


@st.cache_data
def get_country_series(df, country_name):
    """
    From the full dataset, extract a single time series
    for one country — total CO2 emissions summed across all
    its sectors, per year.

    This is what gets fed into SARIMA.
    """
    country_df = df[df['Name'] == country_name].copy()
    if country_df.empty:
        return None

    year_cols = [c for c in df.columns if c.startswith("Y_")]
    # Sum across all sectors for this country
    annual = country_df[year_cols].sum(axis=0)
    # Turn "Y_1970" into integer 1970
    years = [int(c.replace("Y_", "")) for c in year_cols]
    series = pd.Series(annual.values, index=years, name=country_name)
    series = series[series > 0]   # drop zero/negative (no data)
    return series


# ─────────────────────────────────────────────
# SARIMA MODEL TRAINING
#
# This trains the forecasting model on historical data.
# SARIMA stands for:
#   S = Seasonal
#   AR = AutoRegressive (uses past values)
#   I = Integrated (differencing to remove trend)
#   MA = Moving Average (uses past errors)
#
# The parameters (3,1,3)(1,0,1)[5] were chosen by
# your teammate. We use the same ones here so our
# baseline matches their API output exactly.
# ─────────────────────────────────────────────
@st.cache_data
def train_sarima_and_forecast(series_values, series_index, horizon):
    """
    Train SARIMA on historical data, then forecast `horizon` years ahead.
    Returns (historical_series, forecast_values, forecast_years, confidence_intervals)
    """
    # Use DatetimeIndex with annual frequency — SARIMA needs this for clean forecasting
    dt_index = pd.date_range(start=str(series_index[0]), periods=len(series_index), freq="YE")
    series = pd.Series(series_values, index=dt_index)

    try:
        model = SARIMAX(
            series,
            order=(3, 1, 3),
            seasonal_order=(1, 0, 1, 5),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False)

        # forecast() gives point estimates + confidence intervals
        forecast_result = result.get_forecast(steps=horizon)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.2)  # 80% confidence band

        last_year = series_index[-1]
        forecast_years = list(range(last_year + 1, last_year + horizon + 1))

        return series, forecast_mean.values, forecast_years, conf_int.values

    except Exception as e:
        return series, None, None, None


# ─────────────────────────────────────────────
# SCENARIO APPLICATION ENGINE
#
# This is the mathematical heart of your module.
#
# Given a SARIMA baseline forecast, we apply the
# scenario's assumptions to generate an alternative
# future trajectory.
#
# The logic:
#   For each year in the forecast:
#     1. Start from the SARIMA baseline value
#     2. Apply compounding annual reduction/growth
#        (like compound interest, but for emissions)
#     3. If there's a shock year, apply it at that point
#     4. Return the modified trajectory
# ─────────────────────────────────────────────
def apply_scenario(baseline_values, forecast_years, scenario_params):
    """
    Take baseline forecast and bend it according to scenario assumptions.

    Returns a list of emission values under this scenario.
    """
    result = []
    reduction_rate = scenario_params["annual_reduction_pct"] / 100.0
    shock_year = scenario_params["shock_year"]
    shock_magnitude = scenario_params["shock_magnitude"]
    acceleration = scenario_params["trend_acceleration"]

    for i, (year, base_val) in enumerate(zip(forecast_years, baseline_values)):
        # Compound the annual reduction year by year
        # e.g. 7% reduction: year 1 = base * 0.93, year 2 = base * 0.93^2
        effective_rate = reduction_rate * (acceleration ** i)
        scenario_val = base_val * ((1 - effective_rate) ** i)

        # Apply shock if this is the shock year
        if shock_year and year >= shock_year:
            years_since_shock = year - shock_year
            if years_since_shock == 0:
                scenario_val *= shock_magnitude
            else:
                # After the shock, slowly recover toward baseline
                recovery = min(years_since_shock * 0.05, 1.0)  # 5% recovery/year
                scenario_val = scenario_val * (shock_magnitude + (1 - shock_magnitude) * recovery)

        # Emissions can't go negative
        result.append(max(scenario_val, 0))

    return result


# ─────────────────────────────────────────────
# CO₂ EQUIVALENT CONVERSION
#
# Different gases have different warming impacts.
# To compare them fairly, scientists convert everything
# to "CO₂ equivalent" (CO₂e) using GWP factors.
#
# GWP = Global Warming Potential over 100 years
#   CO₂ = 1 (the reference)
#   CH₄ = 28 (methane is 28x more warming than CO₂)
#   N₂O = 273 (nitrous oxide is 273x)
# ─────────────────────────────────────────────
GWP = {"CO₂": 1, "CH₄": 28, "N₂O": 273}

def to_co2e(value, gas):
    return value * GWP.get(gas, 1)


# ─────────────────────────────────────────────
# RISK ASSESSMENT ENGINE
#
# This translates forecast numbers into plain-English
# risk ratings that a policy maker can understand
# without knowing what SARIMA means.
# ─────────────────────────────────────────────
def compute_risk_rating(scenario_values, baseline_values):
    """
    Compare a scenario to the baseline and return a risk level.
    """
    final_scenario = scenario_values[-1]
    final_baseline = baseline_values[-1]

    if final_baseline == 0:
        return "Unknown", "grey"

    pct_change = ((final_scenario - final_baseline) / final_baseline) * 100

    if pct_change > 20:
        return "🔴 Critical — Emissions rising well above baseline", "#f85149"
    elif pct_change > 5:
        return "🟠 High — Emissions tracking above baseline", "#d29922"
    elif pct_change > -10:
        return "🟡 Moderate — Close to baseline trajectory", "#e3b341"
    elif pct_change > -30:
        return "🟢 Low — Meaningful reduction achieved", "#3fb950"
    else:
        return "🟦 Transformative — Deep decarbonisation achieved", "#58a6ff"


# ══════════════════════════════════════════════
#  UI STARTS HERE
#  Everything above was setup. Now we build what
#  the user sees on screen.
# ══════════════════════════════════════════════

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("<div style='font-size:4em; margin-top:10px'>🌍</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("## Climate Emissions — Scenario Forecasting")
    st.markdown(
        "<p style='color:#8b949e; margin-top:-10px'>Explore how different policy, economic, and technology scenarios "
        "change future emissions trajectories</p>",
        unsafe_allow_html=True
    )

st.divider()


# ─────────────────────────────────────────────
# SIDEBAR — USER CONTROLS
#
# The sidebar is where users set their parameters.
# st.sidebar.* puts widgets in the left panel.
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Forecast Settings")
    st.markdown("<div class='info-box'>Select a country, gas, and forecast horizon, then choose which scenarios to compare.</div>", unsafe_allow_html=True)

    # Load data to get the country list
    df = load_emissions_data(r"C:\Users\USER\Downloads\data (1).csv")

    if df is not None:
        countries = sorted(df['Name'].dropna().unique().tolist())
    else:
        # Fallback demo countries if data.csv isn't present
        countries = ["Canada", "Kenya", "Germany", "India", "United States"]

    selected_country = st.selectbox("🌐 Country", countries, index=countries.index("Canada") if "Canada" in countries else 0)

    selected_gas = st.selectbox(
        "💨 Greenhouse Gas",
        ["CO₂", "CH₄", "N₂O"],
        help="CO₂ is from burning fossil fuels. CH₄ from agriculture/waste. N₂O from soils and fertilizers."
    )

    forecast_horizon = st.slider(
        "📅 Forecast Horizon (years)",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
        help="How many years into the future to project"
    )

    st.markdown("---")
    st.markdown("### 🎯 Select Scenarios to Compare")
    st.markdown("<div class='info-box'>Pick 2 or more to compare outcomes side by side.</div>", unsafe_allow_html=True)

    selected_scenarios = []
    for scenario_name, scenario_data in SCENARIOS.items():
        if scenario_name == "🔧 Custom Scenario":
            continue  # custom handled separately below
        checked = st.checkbox(
            scenario_name,
            value=(scenario_name in ["📈 Business As Usual (BAU)", "🌿 Paris Agreement (1.5°C Path)"]),
        )
        if checked:
            selected_scenarios.append(scenario_name)

    # Custom scenario toggle
    st.markdown("---")
    use_custom = st.checkbox("🔧 Add Custom Scenario", value=False)
    if use_custom:
        st.markdown("**Custom Scenario Parameters**")
        custom_reduction = st.slider("Annual Reduction %", -5.0, 15.0, 0.0, 0.5,
            help="Positive = emissions fall each year. Negative = emissions rise.")
        custom_shock_year = st.selectbox("Shock Year (optional)", [None] + list(range(2025, 2041)),
            help="Year of a sudden one-off event")
        custom_shock_size = st.slider("Shock Magnitude", 0.5, 1.5, 1.0, 0.05,
            help="1.0 = no shock. 0.7 = 30% sudden drop. 1.3 = 30% sudden spike.")
        SCENARIOS["🔧 Custom Scenario"]["annual_reduction_pct"] = custom_reduction
        SCENARIOS["🔧 Custom Scenario"]["shock_year"] = custom_shock_year
        SCENARIOS["🔧 Custom Scenario"]["shock_magnitude"] = custom_shock_size
        selected_scenarios.append("🔧 Custom Scenario")

    st.markdown("---")
    st.markdown("### 📖 How to Read This")
    st.markdown("""
    - **Solid lines** = scenario forecasts  
    - **Dashed line** = SARIMA baseline  
    - **Shaded band** = model uncertainty  
    - **Vertical dashed** = today (data ends 2018)
    """)


# ─────────────────────────────────────────────
# MAIN CONTENT — LOAD DATA & TRAIN MODEL
# ─────────────────────────────────────────────
if df is None:
    st.error("⚠️ **data.csv not found.** Place your emissions CSV in the same folder as this file.")
    st.markdown("<div class='info-box'>Expected filename: <code>data.csv</code> — the same CO₂ dataset your teammate uses in app.py</div>", unsafe_allow_html=True)
    st.stop()

with st.spinner(f"Loading data for {selected_country}..."):
    series = get_country_series(df, selected_country)

if series is None or len(series) < 10:
    st.warning(f"Not enough data for **{selected_country}**. Try a different country.")
    st.stop()

# Map gas selection to the right CSV file (for future CH4/N2O extension)
gas_file_map = {"CO₂": "data.csv", "CH₄": "data_ch4.csv", "N₂O": "data_n2o.csv"}
if selected_gas != "CO₂":
    st.markdown(f"<div class='warn-box'>⚠️ Showing CO₂ data — {gas_file_map[selected_gas]} not loaded in this session. Connect it through the data layer to enable {selected_gas} forecasts.</div>", unsafe_allow_html=True)

# Train model
with st.spinner("Training SARIMA model on historical data..."):
    hist_series, baseline_forecast, forecast_years, conf_intervals = train_sarima_and_forecast(
        series.values.tolist(),
        series.index.tolist(),
        forecast_horizon
    )

if baseline_forecast is None:
    st.error("SARIMA training failed for this country/gas combination. Try a different selection.")
    st.stop()


# ─────────────────────────────────────────────
# APPLY SCENARIOS
# ─────────────────────────────────────────────
scenario_results = {}
for scenario_name in selected_scenarios:
    params = SCENARIOS[scenario_name]
    scenario_vals = apply_scenario(baseline_forecast, forecast_years, params)
    scenario_results[scenario_name] = scenario_vals


# ─────────────────────────────────────────────
# BUILD THE MAIN CHART
#
# Plotly gives us interactive charts — hover,
# zoom, download. Much better than matplotlib
# for a user-facing dashboard.
# ─────────────────────────────────────────────
fig = go.Figure()

# 1. Historical data (what actually happened 1970-2018)
fig.add_trace(go.Scatter(
    x=list(hist_series.index),
    y=list(hist_series.values),
    name="Historical (observed)",
    line=dict(color="#8b949e", width=2),
    mode="lines",
))

# 2. Confidence band (model uncertainty around baseline)
if conf_intervals is not None:
    fig.add_trace(go.Scatter(
        x=forecast_years + forecast_years[::-1],
        y=list(conf_intervals[:, 1]) + list(conf_intervals[::-1, 0]),
        fill="toself",
        fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Model uncertainty (80%)",
        showlegend=True,
    ))

# 3. Baseline SARIMA forecast (dashed)
fig.add_trace(go.Scatter(
    x=forecast_years,
    y=list(baseline_forecast),
    name="SARIMA Baseline",
    line=dict(color="#58a6ff", width=2, dash="dash"),
    mode="lines",
))

# 4. Each selected scenario
for scenario_name in selected_scenarios:
    params = SCENARIOS[scenario_name]
    vals = scenario_results[scenario_name]
    fig.add_trace(go.Scatter(
        x=forecast_years,
        y=vals,
        name=scenario_name,
        line=dict(color=params["color"], width=2.5),
        mode="lines",
    ))

# 5. Vertical line at 2018 (end of historical data / start of forecast)
fig.add_vline(
    x=2018,
    line=dict(color="#30363d", width=1.5, dash="dot"),
    annotation_text="Data ends 2018 →",
    annotation_position="top left",
    annotation_font_color="#8b949e",
)

fig.update_layout(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font=dict(color="#e8ecf0", family="Segoe UI"),
    title=dict(
        text=f"{selected_country} — {selected_gas} Emissions: Historical + Scenario Forecast",
        font=dict(size=16, color="#e8ecf0"),
    ),
    xaxis=dict(
        title="Year",
        gridcolor="#21262d",
        color="#8b949e",
        tickcolor="#30363d",
    ),
    yaxis=dict(
        title=f"Emissions (kt CO₂ equiv.)",
        gridcolor="#21262d",
        color="#8b949e",
    ),
    legend=dict(
        bgcolor="#161b22",
        bordercolor="#30363d",
        borderwidth=1,
        font=dict(size=11),
    ),
    hovermode="x unified",
    height=500,
    margin=dict(l=60, r=20, t=60, b=50),
)

st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SCENARIO COMPARISON TABLE + RISK CARDS
#
# Below the chart we show:
#   - Key numbers for each scenario
#   - A risk rating in plain English
#   - The scenario description so judges/users
#     understand what they're looking at
# ─────────────────────────────────────────────
st.markdown("### 📊 Scenario Comparison")

if not selected_scenarios:
    st.info("Select at least one scenario from the sidebar to see comparisons.")
else:
    cols = st.columns(len(selected_scenarios))

    for i, scenario_name in enumerate(selected_scenarios):
        params = SCENARIOS[scenario_name]
        vals = scenario_results[scenario_name]
        risk_label, risk_color = compute_risk_rating(vals, list(baseline_forecast))

        # Percentage change vs baseline at end of forecast horizon
        final_baseline = baseline_forecast[-1]
        final_scenario = vals[-1]
        pct_vs_baseline = ((final_scenario - final_baseline) / final_baseline * 100) if final_baseline > 0 else 0

        # Total cumulative emissions over forecast period
        cumulative = sum(vals)
        baseline_cumulative = sum(baseline_forecast)
        cumulative_saving = baseline_cumulative - cumulative

        with cols[i]:
            st.markdown(
                f"<div style='border:1px solid {params['color']}; border-radius:8px; padding:16px; "
                f"background:#161b22; margin-bottom:8px;'>"
                f"<div style='color:{params['color']}; font-weight:700; font-size:0.95em; margin-bottom:8px;'>"
                f"{scenario_name}</div>"
                f"<div style='font-size:0.8em; color:#8b949e; margin-bottom:12px;'>{params['description']}</div>"
                f"<div style='font-size:1.4em; font-weight:700; color:{params['color']};'>"
                f"{final_scenario:,.0f} kt</div>"
                f"<div style='font-size:0.8em; color:#8b949e;'>Emissions in {forecast_years[-1]}</div>"
                f"<div style='margin-top:8px; font-size:0.85em; color:{'#3fb950' if pct_vs_baseline < 0 else '#f85149'};'>"
                f"{'▼' if pct_vs_baseline < 0 else '▲'} {abs(pct_vs_baseline):.1f}% vs baseline</div>"
                f"<div style='margin-top:8px; font-size:0.8em; color:#8b949e;'>"
                f"Cumulative saving: <strong style='color:#e8ecf0;'>"
                f"{cumulative_saving:,.0f} kt</strong></div>"
                f"<div style='margin-top:12px; padding:6px; background:#0f1117; border-radius:4px; "
                f"font-size:0.78em; color:{risk_color};'>{risk_label}</div>"
                f"</div>",
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────
# DETAILED DATA TABLE
# Users can expand this to see the actual numbers
# ─────────────────────────────────────────────
with st.expander("📋 View Raw Forecast Data"):
    table_data = {"Year": forecast_years, "SARIMA Baseline": [f"{v:,.0f}" for v in baseline_forecast]}
    for scenario_name in selected_scenarios:
        short_name = scenario_name.split("(")[0].strip()
        table_data[short_name] = [f"{v:,.0f}" for v in scenario_results[scenario_name]]

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# CO₂ EQUIVALENT IMPACT SECTION
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🌡️ Climate Impact Summary")

impact_col1, impact_col2, impact_col3 = st.columns(3)

gwp_factor = GWP[selected_gas]
baseline_total_co2e = sum(baseline_forecast) * gwp_factor

with impact_col1:
    st.metric(
        label=f"Baseline Total CO₂e ({forecast_years[0]}–{forecast_years[-1]})",
        value=f"{baseline_total_co2e/1000:,.0f} Mt",
        help="Total CO₂-equivalent emissions if nothing changes"
    )

if selected_scenarios:
    best_scenario = min(selected_scenarios, key=lambda s: sum(scenario_results[s]))
    worst_scenario = max(selected_scenarios, key=lambda s: sum(scenario_results[s]))

    best_saving = (baseline_total_co2e - sum(scenario_results[best_scenario]) * gwp_factor) / 1000
    worst_delta = (sum(scenario_results[worst_scenario]) * gwp_factor - baseline_total_co2e) / 1000

    with impact_col2:
        st.metric(
            label=f"Best Case Saving ({best_scenario.split('(')[0].strip()})",
            value=f"{abs(best_saving):,.0f} Mt CO₂e",
            delta=f"{'Reduced' if best_saving > 0 else 'Increased'} vs baseline",
            delta_color="normal"
        )

    with impact_col3:
        st.metric(
            label="Forecast Horizon",
            value=f"{forecast_horizon} years",
            delta=f"{forecast_years[0]} → {forecast_years[-1]}"
        )


# ─────────────────────────────────────────────
# FOOTER — connects back to the rest of the system
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#30363d; font-size:0.8em; text-align:center;'>"
    "Scenario Forecasting Module · Climate Emissions Forecasting System · "
    "SARIMA(3,1,3)(1,0,1)[5] · Data: EDGAR 1970–2018 · "
    "Connected to FastAPI :8000 · React Frontend :5173</p>",
    unsafe_allow_html=True
)
