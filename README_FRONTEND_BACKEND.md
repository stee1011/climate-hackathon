# React Dashboard + Streamlit SARIMA Backend

This setup keeps your SARIMA logic in `app.py` and adds:

- `streamlit_backend_api.py` (HTTP API bridge for frontend requests)
- `frontend/` (professional React story dashboard UI)

## 1) Run backend API

From project root:

```bash
pip install fastapi uvicorn streamlit pandas numpy matplotlib scikit-learn statsmodels
uvicorn streamlit_backend_api:app --host 0.0.0.0 --port 8000 --reload
```

Backend endpoint:

- `POST http://localhost:8000/predict`
- body:

```json
{
  "country": "Canada",
  "datum": {
    "Country": "Canada",
    "ISO_Code": "CAN",
    "Sector": "Main Activity Electricity and Heat Production",
    "fossil_bio": "fossil",
    "Y_1970": 45100.0,
    "Y_2018": 87700.0
  }
}
```

## 2) Run React dashboard

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Notes

- The frontend submits raw datum JSON directly to backend.
- Backend reuses your strict SARIMA functions from `app.py`.
- Prediction year remains hardcoded to latest available year of selected country model context.
