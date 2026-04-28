# Climate Emissions Forecasting Dashboard

A comprehensive web application for forecasting greenhouse gas emissions (CO₂, N₂O, CH₄) using SARIMA time series models. The system provides both a scientific analysis interface via Streamlit and a professional dashboard via React frontend with a FastAPI backend.

## 🌍 Overview

This project enables climate scientists, policymakers, and researchers to:
- Analyze historical greenhouse gas emission data
- Train and validate SARIMA models for emission forecasting
- Generate predictions for future emission levels
- Visualize emission trends and model diagnostics
- Access predictions through a web API

The application supports three major greenhouse gases with identical data schemas, allowing consistent analysis across CO₂, N₂O, and CH₄ emissions.

## 🏗️ Architecture

The project consists of three main components:

1. **Streamlit Analysis App** (`app.py`) - Scientific interface for model training and diagnostics
2. **FastAPI Backend** (`streamlit_backend_api.py`) - REST API for serving predictions
3. **React Dashboard** (`frontend/`) - Professional web interface for policy teams

## 📊 Data Format

The system expects emission data in wide-format CSV with the following structure:

```csv
Country,ISO_Code,Sector,fossil_bio,Y_1970,Y_1971,...,Y_2018
Canada,CAN,Main Activity Electricity and Heat Production,fossil,45100,49500,...,87700
```

### Required Columns:
- `Country`: Country name
- `ISO_Code`: ISO 3166-1 alpha-3 country code
- `Sector`: Emission sector (e.g., "Main Activity Electricity and Heat Production")
- `fossil_bio`: Fuel type ("fossil" or "bio")
- `Y_YYYY`: Annual emission values (1970-2018)

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd climate-emissions-forecasting
   ```

2. **Create a Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install fastapi uvicorn streamlit pandas numpy scikit-learn statsmodels matplotlib
   ```

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

## 🏃‍♂️ Running the Application

### Option 1: Docker Compose (Recommended - All 3 Services)

**Prerequisites:** Docker and Docker Compose installed. For WSL2 users, ensure Docker Desktop is running with WSL2 backend.

**For Windows with WSL2:**
```bash
wsl bash -lc 'cd /mnt/c/path/to/climate && docker compose up --build'
```

**For macOS/Linux:**
```bash
docker compose up --build
```

This starts all three services simultaneously:
- **React Frontend**: http://localhost:5173
- **FastAPI Backend API**: http://localhost:8000 (docs at `/docs`)
- **Streamlit Analysis Console**: http://localhost:8501

To stop all services:
```bash
docker compose down
```

To rebuild images after code changes:
```bash
docker compose up --build
```

### Option 2: Full Stack (Traditional Local Development)

1. **Start the FastAPI backend:**
   ```bash
   # From project root
   uvicorn streamlit_backend_api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start the React frontend:**
   ```bash
   # In a new terminal, from frontend directory
   cd frontend
   npm run dev
   ```

3. **Access the application:**
   - Frontend Dashboard: http://localhost:5173
   - API Documentation: http://localhost:8000/docs

### Option 3: Streamlit Analysis Interface Only

```bash
# From project root
streamlit run app.py
```

Access at: http://localhost:8501

### Option 4: API Only

```bash
uvicorn streamlit_backend_api:app --host 0.0.0.0 --port 8000 --reload
```

## 📈 Usage

### Frontend Dashboard

1. Select a country and sector from the dropdown menus
2. Choose the emission gas type (CO₂, N₂O, or CH₄)
3. The datum JSON will auto-populate with historical data
4. Click "Submit To Streamlit Backend" to generate a forecast
5. View the prediction results and emission trajectory chart

### API Usage

**Endpoint:** `POST http://localhost:8000/predict`

**Request Body:**
```json
{
  "country": "Canada",
  "gas_type": "co2",
  "datum": {
    "Country": "Canada",
    "ISO_Code": "CAN",
    "Sector": "Main Activity Electricity and Heat Production",
    "fossil_bio": "fossil",
    "Y_1970": 45100,
    "Y_1971": 49500,
    "Y_2018": 87700
  }
}
```

**Response:**
```json
{
  "country": "Canada",
  "year": 2019,
  "selected_year": 2019,
  "gas_type": "co2",
  "predicted_co2": 85000.5
}
```

### Streamlit Interface

1. Upload your emission CSV file
2. Select a country to analyze
3. View model diagnostics, residual analysis, and forecast plots
4. Examine model performance metrics (MAE, RMSE, MAPE)

## 🔬 Model Details

### SARIMA Configuration

The system uses a fixed SARIMA(3,1,3)(1,0,1)[5] model with:
- **AR order (p)**: 3 - Autoregressive terms
- **Differencing (d)**: 1 - First-order differencing for stationarity
- **MA order (q)**: 3 - Moving average terms
- **Seasonal AR (P)**: 1 - Seasonal autoregressive term
- **Seasonal differencing (D)**: 0 - No seasonal differencing
- **Seasonal MA (Q)**: 1 - Seasonal moving average term
- **Seasonal period (s)**: 5 - 5-year policy/economic cycles

### Training Process

1. **Data Aggregation**: Sector-level data aggregated to national yearly totals
2. **Train/Test Split**: 80/20 split for model validation
3. **Model Fitting**: Strict parameter constraints for reproducibility
4. **Forecasting**: Generate predictions for specified years

### Model Caching

Trained models are cached in memory for improved API performance. Models are trained once per country/gas combination and reused for subsequent predictions.

## 📁 Project Structure

```
climate-emissions-forecasting/
├── app.py                          # Streamlit analysis interface
├── streamlit_backend_api.py        # FastAPI prediction service
├── climate_sarima_model.ipynb      # Jupyter notebook (analysis)
├── requirements.txt                # Full Python dependencies
├── requirements-api.txt            # FastAPI service dependencies
├── requirements-streamlit.txt      # Streamlit service dependencies
├── data.csv                        # CO₂ emission dataset
├── data_n2o.csv                    # N₂O emission dataset
├── data_ch4.csv                    # CH₄ emission dataset
├── sarima_model.pkl               # Cached model file
├── docker-compose.yml             # Multi-container orchestration
├── Dockerfile.api                 # FastAPI service image
├── Dockerfile.streamlit           # Streamlit service image
├── .dockerignore                  # Docker build context exclusions
├── frontend/
│   ├── Dockerfile                # React/Vite service image
│   ├── .dockerignore             # Frontend build exclusions
│   ├── package.json              # Node.js dependencies
│   ├── vite.config.js            # Vite configuration
│   ├── index.html                # Main HTML file
│   ├── src/
│   │   ├── App.jsx               # Main React component
│   │   ├── main.jsx              # React entry point
│   │   └── styles.css            # Application styles
│   └── public/
│       ├── data_catalog.json     # Sample CO₂ data
│       ├── data_catalog_n2o.json # Sample N₂O data
│       └── data_catalog_ch4.json # Sample CH₄ data
└── README.md                     # This file
```

## 🐳 Docker Deployment

The application is fully containerized with three separate microservices orchestrated via Docker Compose.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Docker Compose Network (climate-net)          │
├──────────────────┬──────────────────┬──────────────────┤
│   frontend       │      api         │    streamlit     │
│ (Node:20-slim)  │ (Python:3.12)    │ (Python:3.12)    │
│  Port 5173      │  Port 8000       │  Port 8501       │
└──────────────────┴──────────────────┴──────────────────┘
         │                │                   │
         └────────────────┼───────────────────┘
                          │
                   Data & Models
                   Shared Volume
```

### Services

#### 1. **Frontend Service** (`frontend/Dockerfile`)
- **Base Image**: `node:20-alpine`
- **Framework**: React 18.3 + Vite 5.4
- **Port**: 5173
- **Environment Variables**:
  - `VITE_API_URL`: Backend API URL (default: `http://api:8000`)
- **Build Command**: `npm run dev` (development) or `npm run build` (production)

#### 2. **API Service** (`Dockerfile.api`)
- **Base Image**: `python:3.12-slim`
- **Framework**: FastAPI 0.124.2 + Uvicorn
- **Port**: 8000
- **Environment Variables**:
  - `CORS_ORIGINS`: Comma-separated list of allowed origins
- **Dependencies**: Core scientific stack (pandas, numpy, scikit-learn, statsmodels)
- **Data**: All emission CSV files must be in project root

#### 3. **Streamlit Service** (`Dockerfile.streamlit`)
- **Base Image**: `python:3.12-slim`
- **Framework**: Streamlit 1.56
- **Port**: 8501
- **Dependencies**: Same as API service
- **Data**: All emission CSV files must be in project root

### Building Locally

Build individual services:
```bash
# API service
docker build -f Dockerfile.api -t climate-api:latest .

# Streamlit service
docker build -f Dockerfile.streamlit -t climate-streamlit:latest .

# Frontend service
docker build -f frontend/Dockerfile -t climate-frontend:latest frontend/
```

### Environment Configuration

**Docker Compose defaults** (auto-configured):
- Frontend: `VITE_API_URL=http://api:8000`
- API: `CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`
- Streamlit: Uses default configuration

**Custom environment** (create `.env` file):
```env
VITE_API_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:5173,http://frontend:5173,https://yourdomain.com
STREAMLIT_SERVER_PORT=8501
```

Then load with:
```bash
docker compose --env-file .env up
```

### Network Communication

Services communicate via the `climate-net` bridge network:
- Frontend → API: `http://api:8000/predict`
- Frontend ↔ Host: `http://localhost:5173`
- API ↔ Host: `http://localhost:8000`
- Streamlit ↔ Host: `http://localhost:8501`

### System Requirements

- **Docker**: 20.10+
- **Docker Compose**: 1.29+
- **Memory**: Minimum 2GB (4GB+ recommended)
- **Disk**: 2GB for images + data

### Troubleshooting Docker

**Build fails on requirements install:**
- Ensure all apt packages are available in `python:3.12-slim` base image
- Remove unavailable packages: `libatlas-base-dev` is deprecated in Debian Trixie
- Check `Dockerfile.*` for correct package names

**Frontend cannot reach API:**
- Verify `VITE_API_URL` environment variable
- Check network connectivity: `docker exec climate-frontend ping api`
- Ensure API service is running: `docker compose logs api`

**Port conflicts:**
- Modify port mappings in `docker-compose.yml` (left side of `8000:8000`)
- Update `VITE_API_URL` if you change backend port

**WSL2 Performance:**
- Store project inside WSL filesystem (`/home/user/...`) not `/mnt/c/...`
- Improves build and runtime performance significantly

## 🛠️ Development

### Adding New Gas Types

1. Add dataset file (e.g., `data_xe.csv`)
2. Update `DATA_PATHS` in `streamlit_backend_api.py`
3. Add data catalog to `frontend/public/`
4. Update gas type selection in frontend

### Model Customization

Modify the SARIMA parameters in the `train_sarima()` function in `app.py`. Note: The current strict parameters ensure reproducible results across different environments.

### Data Validation

The system includes robust data ingestion that handles:
- Missing headers
- Inconsistent column names
- Non-numeric values
- Variable row structures

## 🔍 Model Diagnostics

The Streamlit interface provides comprehensive diagnostics:

- **Residual Distribution**: Histogram of model residuals
- **Q-Q Plot**: Normality assessment of residuals
- **Ljung-Box Test**: Autocorrelation testing
- **Train/Test Metrics**: MAE, RMSE, MAPE
- **Model Summary**: AIC, BIC, parameter estimates

## 🌐 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Generate emission forecast

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/), [Streamlit](https://streamlit.io/), and [React](https://reactjs.org/)
- Uses [statsmodels](https://www.statsmodels.org/) for SARIMA modeling
- Data visualization powered by [Recharts](https://recharts.org/)

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the Streamlit interface for model diagnostics

---

*"We are the first generation to feel the impact of climate change and the last generation that can do something about it."*  
— Barack Obama, 2014 UN Summit</content>
<parameter name="filePath">c:\Users\codew\Desktop\climate\README.md