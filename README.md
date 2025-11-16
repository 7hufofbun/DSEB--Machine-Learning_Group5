````markdown
# Smart Weather — Full Stack Pack 

## Project overview

Smart Weather is a lightweight end-to-end project for forecasting temperature in **Ho Chi Minh City** and explaining *why* a given day is hotter or cooler than usual.

- On the **ML side**, we train LightGBM time-series models on historical weather data (daily + hourly), export them to **ONNX**, and serve them via a **FastAPI** backend. The backend also provides handy endpoints for “current conditions”, multi-day forecasts, and historical statistics.
- On the **product side**, the **React + Vite** frontend turns those forecasts into an explainer-style UI: summary cards (“How today compares”), key drivers (“What’s driving today’s temperature”), and charts that show how each factor is nudging the temperature up or down.
- The entire pack is already wired together, so you just need to **unzip, run the backend + frontend, and you’ll have a Smart Weather assistant running locally**.

_(Quick summary: a project that forecasts and explains Ho Chi Minh City temperatures, with both ML backend + UI frontend included – just unzip and run.)_

---

## 1) Folder structure

smart_weather_full_project_fullrun/
├─ backend/
│  ├─ main.py
│  ├─ requirements.txt
│  ├─ data/
│  │  ├─ weather_hcm_daily.csv
│  │  └─ weather_hcm_hourly.csv
│  ├─ models_onnx/
│  │  └─ temp_t+{1..5}.onnx           
│  └─ smart_weather_ml/               (ML code for training & exporting ONNX)
│     ├─ __init__.py
│     ├─ io.py, utils.py
│     ├─ preprocessing.py, features.py
│     ├─ metrics.py, evaluation.py
│     ├─ export.py
│     └─ model/
│        ├─ pipeline.py, tuning.py, lgbm.py
├─ frontend/
│  └─ Smartweatherassistantuidesign-main/
│     ├─ package.json
│     ├─ .env.local                   (already set: VITE_API_BASE=http://localhost:8000)
│     └─ src/
│        ├─ lib/api.ts                (API helper)
│        └─ (UI components – already wired to the API)
├─ scripts/
│  ├─ setup_backend.ps1               (create venv, install deps, run server)
│  └─ fix_frontend_imports.ps1        (fix @version imports & install UI deps)
└─ README_RUN.md                      (short run guide – included in the pack)
````

---

## 2) Environment requirements

* **Windows + PowerShell**
* **Python 3.10+** (recommended)
* **Node.js LTS** (18 or 20), with npm

---

## 3) Run Backend (FastAPI)

```powershell
cd D:\ML--06_11_25\DSEB--Machine-Learning_Group5\smart_weather_fullrun\backend
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check: open your browser at `http://127.0.0.1:8000/health` → if you see `{ "ok": true, ... }`, it’s working.

### Customizing with your own data

* Replace the files in `backend/data/` **or** set environment variables before running:

  ```powershell
  $env:WEATHER_DAILY_CSV="C:\path\to\your\weather_daily.csv"
  $env:WEATHER_HOURLY_CSV="C:\path\to\your\weather_hourly.csv"
  uvicorn main:app --reload --port 8000
  ```

### Using ONNX models (if available)

* Put `temp_t+1.onnx … temp_t+5.onnx` into `backend/models_onnx/`.
* If you **don’t** have them, the API will fall back to a simple statistical baseline.

### Main endpoints

* `GET /health` – server status.
* `GET /now` – current conditions (prefers hourly; falls back to daily).
* `GET /history?start=YYYY-MM-DD&end=YYYY-MM-DD&group_by=daily|monthly|yearly`
* `GET /history/stats` – quick stats.
* `POST /forecast_detailed` – 5-day forecast (uses ONNX if available).
* `GET /explain?date=YYYY-MM-DD` – forecast explanation (with sample data).

---

## 4) Run Frontend (React + Vite)

```powershell
cd D:\ML--06_11_25\DSEB--Machine-Learning_Group5\smart_weather_fullrun\frontend\Smartweatherassistantuidesign-main
# First-time setup: fix UI imports like @radix-ui/...@version and install deps
powershell -ExecutionPolicy Bypass -File ..\..\scripts\fix_frontend_imports.ps1

# Install dependencies & run
npm install
npm run dev
```

Open the Vite URL shown in the terminal (usually `http://localhost:5173`).

> **Notes**
>
> * `.env.local` is already set: `VITE_API_BASE=http://localhost:8000`.
> * If npm is slow: run `npm config set registry https://registry.npmmirror.com/` then `npm install` again.

### UI tabs (already connected to the API)

* **Current Conditions** → calls `GET /now`
* **The Week Ahead** → calls `POST /forecast_detailed`
* **Historical Explorer** → calls `GET /history`

---

## 5) Common issues & fixes

* **Vite error: “Failed to resolve import '@radix-ui/.[..@x.y.z](mailto:..@x.y.z)'”**
  → Run `fix_frontend_imports.ps1` in the UI folder (as shown above).

* **`vite` not recognized / npm network errors**
  → Install Node.js LTS; if the network is weak, use a mirror:
  `npm config set registry https://registry.npmmirror.com/`

* **Backend: “Error loading ASGI app” / import errors**
  → Make sure you are in the `backend` folder, the venv is activated, and `uvicorn main:app` matches the correct module name.

* **Backend can’t find CSV files**
  → Put the files in `backend/data/` or use the `WEATHER_..._CSV` environment variables.

* **GitHub push blocked due to files >100MB**
  → Use **Git LFS**: `git lfs install` then `git lfs track "*.onnx" "*.csv"` before `git add`.

---

## 6) (Optional) Train & Export ONNX

In `backend` (with venv activated):

```powershell
python -m smart_weather_ml.train
```

* The pipeline will train LightGBM models for the horizons and **export ONNX** files into `backend/models_onnx/`.
* Restart the backend so the API can use the new models.

```
```
