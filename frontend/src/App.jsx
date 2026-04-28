import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const DEFAULT_ENDPOINT = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/predict`
  : "http://localhost:8000/predict";

const defaultDatum = {
  Country: "Canada",
  ISO_Code: "CAN",
  Sector: "Main Activity Electricity and Heat Production",
  fossil_bio: "fossil",
  Y_1970: 45100.0,
  Y_1971: 49500.0,
  Y_1972: 50300.0,
  Y_1973: 52000.0,
  Y_1974: 50600.0,
  Y_1975: 54100.0,
  Y_1976: 58800.0,
  Y_1977: 65700.0,
  Y_1978: 64400.0,
  Y_1979: 68900.0,
  Y_1980: 74700.0,
  Y_1981: 72900.0,
  Y_1982: 79300.0,
  Y_1983: 85300.0,
  Y_1984: 91900.0,
  Y_1985: 87800.0,
  Y_1986: 80300.0,
  Y_1987: 90000.0,
  Y_1988: 99200.0,
  Y_1989: 109000.0,
  Y_1990: 97200.0,
  Y_1991: 98600.0,
  Y_1992: 105000.0,
  Y_1993: 95700.0,
  Y_1994: 98300.0,
  Y_1995: 102000.0,
  Y_1996: 101000.0,
  Y_1997: 112000.0,
  Y_1998: 125000.0,
  Y_1999: 124000.0,
  Y_2000: 135000.0,
  Y_2001: 138000.0,
  Y_2002: 132000.0,
  Y_2003: 138000.0,
  Y_2004: 130000.0,
  Y_2005: 126000.0,
  Y_2006: 124000.0,
  Y_2007: 133000.0,
  Y_2008: 122000.0,
  Y_2009: 109000.0,
  Y_2010: 111000.0,
  Y_2011: 108000.0,
  Y_2012: 103000.0,
  Y_2013: 102000.0,
  Y_2014: 102000.0,
  Y_2015: 104000.0,
  Y_2016: 101000.0,
  Y_2017: 94800.0,
  Y_2018: 87700.0
};

function toSeriesFromDatum(datumObject) {
  return Object.entries(datumObject)
    .filter(([key]) => key.startsWith("Y_"))
    .map(([key, value]) => ({
      year: Number(key.replace("Y_", "")),
      value: Number(value)
    }))
    .sort((a, b) => a.year - b.year);
}

export default function App() {
  const [endpoint, setEndpoint] = useState(DEFAULT_ENDPOINT);
  const [gasType, setGasType] = useState("co2");
  const [country, setCountry] = useState(defaultDatum.Country);
  const [sector, setSector] = useState(defaultDatum.Sector);
  const [datumText, setDatumText] = useState(JSON.stringify(defaultDatum, null, 2));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [stats, setStats] = useState({ co2: 0, n2o: 0, ch4: 0 });
  const [displayPredicted, setDisplayPredicted] = useState(0);
  const [datumLibrary, setDatumLibrary] = useState([defaultDatum]);

  const countryOptions = useMemo(() => {
    const countries = [...new Set(datumLibrary.map((d) => d.Country))];
    return countries.sort((a, b) => a.localeCompare(b));
  }, [datumLibrary]);

  const sectorOptions = useMemo(() => {
    const sectors = datumLibrary.filter((d) => d.Country === country).map((d) => d.Sector);
    return [...new Set(sectors)];
  }, [country, datumLibrary]);

  useEffect(() => {
    let active = true;
    async function loadCatalog() {
      try {
        const catalogPath = `/data_catalog_${gasType}.json`;
        const response = await fetch(catalogPath);
        if (!response.ok) return;
        const data = await response.json();
        if (!active || !Array.isArray(data) || data.length === 0) return;
        setDatumLibrary(data);
        setCountry(data[0].Country);
        setSector(data[0].Sector);
      } catch {
        // Keep fallback in-memory datum
      }
    }
    loadCatalog();
    return () => {
      active = false;
    };
  }, [gasType]);

  useEffect(() => {
    if (!sectorOptions.includes(sector) && sectorOptions.length > 0) {
      setSector(sectorOptions[0]);
    }
  }, [sectorOptions, sector]);

  useEffect(() => {
    const selected = datumLibrary.find((d) => d.Country === country && d.Sector === sector);
    if (selected) {
      setDatumText(JSON.stringify(selected, null, 2));
    }
  }, [country, sector, datumLibrary]);

  useEffect(() => {
    const duration = 1600;
    const frames = 60;
    let i = 0;
    const timer = setInterval(() => {
      i += 1;
      const p = i / frames;
      setStats({
        co2: +(418 * p).toFixed(1),
        n2o: +(336 * p).toFixed(1),
        ch4: +(1920 * p).toFixed(0)
      });
      if (i >= frames) clearInterval(timer);
    }, duration / frames);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!prediction) {
      setDisplayPredicted(0);
      return;
    }
    const target = Number(prediction.predicted_value || 0);
    const duration = 1200;
    const frames = 50;
    let i = 0;
    const timer = setInterval(() => {
      i += 1;
      const p = i / frames;
      setDisplayPredicted(target * p);
      if (i >= frames) clearInterval(timer);
    }, duration / frames);
    return () => clearInterval(timer);
  }, [prediction]);

  const chartData = useMemo(() => {
    try {
      const parsed = JSON.parse(datumText);
      return toSeriesFromDatum(parsed);
    } catch {
      return [];
    }
  }, [datumText]);

  const narrative = useMemo(() => {
    if (!prediction) return null;
    const gasLabel = gasType.toUpperCase();
    const value = Number(prediction.predicted_value || 0).toLocaleString();
    return `For ${prediction.country}, the SARIMA backend projects ${value} ${gasLabel} units in ${prediction.selected_year}.`;
  }, [prediction, gasType]);

  const gasColors = useMemo(
    () => ({
      co2: "#19c37d",
      n2o: "#3AB4F2",
      ch4: "#FFB347"
    }),
    []
  );

  const globalCurveData = [
    { year: 1750, ppm: 280 },
    { year: 1800, ppm: 282 },
    { year: 1850, ppm: 285 },
    { year: 1900, ppm: 295 },
    { year: 1950, ppm: 310 },
    { year: 1960, ppm: 317 },
    { year: 1970, ppm: 325 },
    { year: 1980, ppm: 338 },
    { year: 1990, ppm: 355 },
    { year: 2000, ppm: 370 },
    { year: 2010, ppm: 390 },
    { year: 2015, ppm: 401 },
    { year: 2020, ppm: 414 },
    { year: 2024, ppm: 418 }
  ];

  const policyWindowData = [
    { year: 2022, noPolicy: 100, withPolicy: 100 },
    { year: 2023, noPolicy: 108, withPolicy: 102 },
    { year: 2024, noPolicy: 118, withPolicy: 101 },
    { year: 2025, noPolicy: 130, withPolicy: 99 },
    { year: 2026, noPolicy: 145, withPolicy: 96 },
    { year: 2027, noPolicy: 160, withPolicy: 93 },
    { year: 2028, noPolicy: 178, withPolicy: 90 }
  ];

  const chapters = [
    {
      icon: "🌿",
      era: "Pre-Industrial (before 1750)",
      text: "Atmospheric CO2 held near 280 ppm for centuries. Natural carbon cycles stayed in equilibrium and fossil tracking did not yet exist."
    },
    {
      icon: "🏭",
      era: "The Coal Age (1750-1950)",
      text: "Coal-powered industry broke equilibrium. Electricity and heat production emerged as a dominant emissions source."
    },
    {
      icon: "🚗",
      era: "The Acceleration (1950-2000)",
      text: "Post-war growth expanded transport, petrochemicals, and cement. CO2 crossed 315 ppm, forcing tighter emissions accounting."
    },
    {
      icon: "🌡️",
      era: "The Crisis Recognition (2000-2015)",
      text: "Kyoto and Paris sharpened targets, but total emissions stayed high globally. Forecasting became core to climate governance."
    },
    {
      icon: "🔮",
      era: "The Forecast Imperative (2015-present)",
      text: "Forecasting by sector became operational for policy windows, carbon budgets, and intervention design."
    }
  ];

  function riskClass(value) {
    if (value < 1000) return "badge badge-low";
    if (value <= 50000) return "badge badge-mod";
    return "badge badge-high";
  }

  async function onSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const datum = JSON.parse(datumText);
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ country, datum, gas_type: gasType })
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "Prediction request failed.");
      }

      const result = await response.json();
      const predictedValue =
        result.predicted_co2 ?? result.predicted_n2o ?? result.predicted_ch4 ?? result.predicted_value ?? 0;
      setPrediction({
        country: result.country,
        selected_year: result.year ?? result.selected_year,
        predicted_value: predictedValue
      });
    } catch (err) {
      setError(err.message || "Invalid request.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      {/* 0. Full-Page Dark Theme */}
      <header className="hero">
        <div className="hero-content">
          <p className="tag">Climate Intelligence Console</p>
          <h1>
            The Atmosphere
            <br />
            is <span className="burning">Burning.</span>
          </h1>
          <p>
            This console supports CO2, N2O, and CH4 forecasting from identical wide-format datasets. Policy teams can submit sector-level emission records and receive
            machine-learned forecasts, because you cannot fix what you cannot see.
          </p>
          <div className="counter-grid">
            <div className="counter-card glow">
              <div className="counter-label">Atmospheric CO2 concentration</div>
              <div className="counter-value">{stats.co2} ppm</div>
            </div>
            <div className="counter-card glow">
              <div className="counter-label">Atmospheric N2O concentration</div>
              <div className="counter-value">{stats.n2o} ppb</div>
            </div>
            <div className="counter-card glow">
              <div className="counter-label">Atmospheric CH4 concentration</div>
              <div className="counter-value">{stats.ch4} ppb</div>
            </div>
          </div>
        </div>
        <div className="hero-right">
          <div className="molecule-wrap">
            <div className="orbital" />
            <div className="bond bond-left" />
            <div className="bond bond-left second" />
            <div className="bond bond-right" />
            <div className="bond bond-right second" />
            <div className="atom oxygen left">O</div>
            <div className="atom carbon center">C</div>
            <div className="atom oxygen right">O</div>
          </div>
          <div className="co2-table">
            <pre>{`┌─────────────────────┐
│  CO₂                │
│  Carbon Dioxide     │
│  Mol. Weight: 44.01 │
│  GWP: 1 (baseline)  │
│  Lifetime: 300–1000 │
│           years     │
└─────────────────────┘`}</pre>
          </div>
        </div>
      </header>

      {/* 2. Narrative Scroll */}
      <section className="card">
        <h2>The Story of Greenhouse Gases</h2>
        <div className="timeline-row">
          {chapters.map((c) => (
            <article key={c.era} className="timeline-card">
              <h3>
                <span>{c.icon}</span> {c.era}
              </h3>
              <p>{c.text}</p>
            </article>
          ))}
        </div>
        <div className="chart-wrap large">
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={globalCurveData}>
              <defs>
                <linearGradient id="globalFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3AB4F2" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="#3AB4F2" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" />
              <XAxis dataKey="year" stroke="#A8BCD2" />
              <YAxis stroke="#A8BCD2" />
              <Tooltip />
              <Area type="monotone" dataKey="ppm" stroke="#3AB4F2" fill="url(#globalFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* 3. Science Panel */}
      <section className="science-grid">
        <article className="card">
          <h3>The Carbon Feedback Loop</h3>
          <div className="loop-ring">Fossil Burn → CO2 Release → Atmosphere Warming → Permafrost Melt → More CO2</div>
          <p className="muted">Warming can unlock additional carbon sources. Forecasting creates intervention time before positive feedback accelerates.</p>
        </article>
        <article className="card">
          <h3>What the Model Sees</h3>
          <ul className="glow-list">
            <li>Trend — long-run direction of national emissions</li>
            <li>Seasonality — 5-year policy and economic cycles</li>
            <li>Autoregression — how last year predicts next year</li>
            <li>Differencing — removes non-stationarity for stable forecasts</li>
          </ul>
          <code>SARIMA(3,1,3)(1,0,1)[5]</code>
        </article>
        <article className="card">
          <h3>The Intervention Value</h3>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={policyWindowData}>
                <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="year" stroke="#A8BCD2" />
                <YAxis stroke="#A8BCD2" />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="noPolicy" stroke="#FF4C4C" strokeWidth={2.5} name="Forecast without policy" />
                <Line type="monotone" dataKey="withPolicy" stroke="#00FFB2" strokeWidth={2.5} name="Forecast with intervention" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="muted">Each year of forecast lead time = one more year of policy options.</p>
        </article>
        <article className="card">
          <h3>Why Forecast CO2, N2O, and CH4 Together</h3>
          <ul className="glow-list">
            <li>CO2 tracks long-lived energy and industry emissions pressure.</li>
            <li>N2O captures agricultural and fertilizer-driven forcing often missed by CO2-only views.</li>
            <li>CH4 highlights short-lived but intense warming pressure from energy, waste, and agriculture.</li>
          </ul>
          <p className="muted">
            Datasets share the same schema, so the same SARIMA pipeline can estimate both gases consistently.
          </p>
        </article>
      </section>

      {/* 4. Inference Console */}
      <main className="grid">
        <section className="card form-card">
          <p className="kicker">INFERENCE REQUEST</p>
          <h2>Inference Request Panel</h2>
          <p className="muted">Submit one raw wide-format datum directly to your backend service.</p>

          <form onSubmit={onSubmit}>
            <label>Backend endpoint</label>
            <input value={endpoint} onChange={(e) => setEndpoint(e.target.value)} placeholder="http://localhost:8000/predict" />

            <label>Country</label>
            <select value={country} onChange={(e) => setCountry(e.target.value)}>
              {countryOptions.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>

            <label>Sector</label>
            <select value={sector} onChange={(e) => setSector(e.target.value)}>
              {sectorOptions.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>

            <label>Emission Gas</label>
            <select value={gasType} onChange={(e) => setGasType(e.target.value)}>
              <option value="co2">CO2</option>
              <option value="n2o">N2O</option>
              <option value="ch4">CH4</option>
            </select>

            <label>Datum JSON Object · Wide-format datum Y_YYYY keys</label>
            <textarea
              rows={14}
              value={datumText}
              onChange={(e) => setDatumText(e.target.value)}
            />

            <button type="submit" disabled={loading}>
              {loading ? "Predicting..." : "Submit To Streamlit Backend"}
            </button>
          </form>

          <p className="legend-note">ℹ Model predicts national totals. Datum provides the target year. Sector values are not used in inference — only year is extracted.</p>
          {error && <p className="error">{error}</p>}
        </section>

        <section className="card">
          <h2>Emission Story</h2>
          <p className="muted">Historical trajectory from the JSON payload.</p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="co2Fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={gasColors[gasType]} stopOpacity={0.5} />
                    <stop offset="100%" stopColor={gasColors[gasType]} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="year" stroke="#9db0bf" />
                <YAxis stroke="#9db0bf" />
                <Tooltip />
                <Area type="monotone" dataKey="value" stroke={gasColors[gasType]} fill="url(#co2Fill)" />
                {prediction && (
                  <Line
                    dataKey={() => null}
                    stroke="#FF4C4C"
                    dot={false}
                    activeDot={false}
                    isAnimationActive={false}
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card result-card">
          <h2>Prediction Output</h2>
          {prediction ? (
            <>
              <div className="result-grid">
                <div>
                  <p className="muted">Country</p>
                  <strong>{prediction.country}</strong>
                </div>
                <div>
                  <p className="muted">Selected Year</p>
                  <strong>{prediction.selected_year}</strong>
                </div>
                <div>
                  <p className="muted">Predicted {gasType.toUpperCase()}</p>
                  <strong>{Number(displayPredicted).toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong>
                </div>
              </div>
              <div className={riskClass(Number(prediction.predicted_value))}>
                {Number(prediction.predicted_value) < 1000
                  ? "LOW TRAJECTORY"
                  : Number(prediction.predicted_value) <= 50000
                  ? "MODERATE TRAJECTORY"
                  : "HIGH TRAJECTORY"}
              </div>
              <p className="story">{narrative}</p>
            </>
          ) : (
            <p className="muted">No prediction yet. Submit the request to generate a forecast story.</p>
          )}
        </section>
      </main>

      {/* 5. Footer */}
      <footer className="card footer">
        <div>
          <p className="quote">
            "We are the first generation to feel the impact of climate change and the last generation that can do something about it."
          </p>
          <p className="muted">— Barack Obama, 2014 UN Summit</p>
        </div>
        <div className="facts">
          <p>🌊 Sea levels rising 3.3mm/year</p>
          <p>🔥 2023 was the hottest year on record</p>
          <p>🧊 Arctic ice declining 13% per decade</p>
        </div>
        <p className="muted">SARIMA Forecast Engine · Emissions data processed locally · No external API calls</p>
      </footer>
    </div>
  );
}
