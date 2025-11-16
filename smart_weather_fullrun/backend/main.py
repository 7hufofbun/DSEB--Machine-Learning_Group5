
from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from smart_weather_ml.model.preprocessing import Preprocessor, basic_cleaning
from smart_weather_ml.model.features import feature_engineer

try:
    import onnxruntime as ort
except Exception:
    ort = None

app = FastAPI(title="Smart Weather Assistant API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
DAILY_CSV = os.environ.get("WEATHER_DAILY_CSV", os.path.join(BASE_DIR, "data", "weather_hcm_daily.csv"))
HOURLY_CSV = os.environ.get("WEATHER_HOURLY_CSV", os.path.join(BASE_DIR, "data", "weather_hcm_hourly.csv"))

_feature_cols_candidates = [
    Path(BASE_DIR) / "smart_weather_ml" / "model" / "feature_cols.json",
    Path(BASE_DIR) / "models_onnx" / "feature_cols.json",
]
FEATURE_COLS: Optional[List[str]] = None


def _load_feature_cols() -> Optional[List[str]]:
    """Load ONNX feature column ordering once the metadata file exists."""
    global FEATURE_COLS
    if FEATURE_COLS:
        return FEATURE_COLS
    for candidate in _feature_cols_candidates:
        try:
            with candidate.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    FEATURE_COLS = data
                    return FEATURE_COLS
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return None


if _load_feature_cols() is None:
    print("[forecast] feature_cols.json not found; ONNX inputs may misalign.")

if not os.path.exists(DAILY_CSV) and os.path.exists("/mnt/data/weather_hcm_daily.csv"):
    DAILY_CSV = "/mnt/data/weather_hcm_daily.csv"
if not os.path.exists(HOURLY_CSV) and os.path.exists("/mnt/data/weather_hcm_hourly.csv"):
    HOURLY_CSV = "/mnt/data/weather_hcm_hourly.csv"

_daily_df_cache: Optional[pd.DataFrame] = None
_hourly_df_cache: Optional[pd.DataFrame] = None

def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ["datetime", "sunrise", "sunset"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
            except TypeError:
                # Older pandas versions lack format="mixed"; normalise ISO "T" separator manually.
                if df[c].dtype == object:
                    df[c] = df[c].astype(str).str.replace("T", " ", regex=False)
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def get_daily_df() -> pd.DataFrame:
    global _daily_df_cache
    if _daily_df_cache is None:
        _daily_df_cache = _read_csv_safe(DAILY_CSV)
        if not _daily_df_cache.empty and "datetime" in _daily_df_cache.columns:
            _daily_df_cache = _daily_df_cache.sort_values("datetime").reset_index(drop=True)
    return _daily_df_cache.copy() if _daily_df_cache is not None else pd.DataFrame()

def get_hourly_df() -> pd.DataFrame:
    global _hourly_df_cache
    if _hourly_df_cache is None:
        _hourly_df_cache = _read_csv_safe(HOURLY_CSV)
        if not _hourly_df_cache.empty and "datetime" in _hourly_df_cache.columns:
            _hourly_df_cache = _hourly_df_cache.sort_values("datetime").reset_index(drop=True)
    return _hourly_df_cache.copy() if _hourly_df_cache is not None else pd.DataFrame()

class OnnxBundle:
    def __init__(self):
        self.sessions: Dict[int, Any] = {}
        self.input_names: Dict[int, str] = {}
        self.loaded = False
        self._try_load()

    def _try_load(self):
        if ort is None:
            self.loaded = False
            return
        models_dir = os.path.join(BASE_DIR, "models_onnx")
        if not os.path.isdir(models_dir):
            self.loaded = False
            return
        ok = 0
        for h in [1,2,3,4,5]:
            p = os.path.join(models_dir, f"temp_t+{h}.onnx")
            if os.path.exists(p):
                sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
                self.sessions[h] = sess
                self.input_names[h] = sess.get_inputs()[0].name
                ok += 1
        self.loaded = ok > 0

    def predict(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        out = {}
        if not self.loaded:
            return out
        for h, sess in self.sessions.items():
            name = self.input_names[h]
            pred = sess.run(None, {name: X.astype(np.float32)})[0]
            out[h] = pred.ravel()
        return out

ONNX = OnnxBundle()

def _canonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "temp_min": "tempmin", "temp_max": "tempmax",
        "wind_speed": "windspeed", "wind": "windspeed",
        "solar_radiation": "solarradiation", "cloud": "cloudcover",
    }
    for src, dst in colmap.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    return df

def _heatwave_count(df: pd.DataFrame) -> int:
    if df.empty or "tempmax" not in df.columns:
        return 0
    s = df["tempmax"].fillna(-9e9) > 35
    consec = 0
    waves = 0
    for v in s.tolist():
        if v:
            consec += 1
        else:
            if consec >= 5:
                waves += 1
            consec = 0
    if consec >= 5:
        waves += 1
    return waves

def _f(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(float(x), 1)
    except Exception:
        return None

@app.get("/health")
def health():
    return {
        "ok": True,
        "daily_csv": DAILY_CSV,
        "hourly_csv": HOURLY_CSV,
        "onnx_loaded": ONNX.loaded,
        "feature_columns": len(_load_feature_cols() or []),
    }

@app.get("/now")
def now():
    h = get_hourly_df()
    if h.empty:
        d = get_daily_df()
        if d.empty:
            return {"message": "no data"}
        last = d.iloc[-1]
        return {
            "temperature": float(last.get("temp", np.nan)),
            "humidity": float(last.get("humidity", np.nan)),
            "conditions": str(last.get("conditions", "")),
            "wind": f"{float(last.get('windspeed', np.nan))} km/h",
            "feels_like": float(last.get("feelslike", last.get("temp", np.nan))),
            "location": "District 1, HCMC",
            "updated_at": str(last.get("datetime")),
        }
    last = h.iloc[-1]
    return {
        "temperature": float(last.get("temp", np.nan)),
        "humidity": float(last.get("humidity", np.nan)),
        "conditions": str(last.get("conditions", "")),
        "wind": f"{float(last.get('windspeed', np.nan))} km/h",
        "feels_like": float(last.get("feelslike", last.get("temp", np.nan))),
        "location": "District 1, HCMC",
        "updated_at": str(last.get("datetime")),
    }

@app.get("/history")
def history(start: Optional[str] = None, end: Optional[str] = None, group_by: str = "daily"):
    df = _canonize_columns(get_daily_df())
    if df.empty:
        return []
    if start:
        df = df[df["datetime"] >= pd.to_datetime(start)]
    if end:
        df = df[df["datetime"] <= pd.to_datetime(end)]

    out: List[Dict[str, Any]] = []
    if group_by == "daily":
        for _, r in df.iterrows():
            out.append({
                "date": r["datetime"].strftime("%Y-%m-%d") if pd.notnull(r["datetime"]) else None,
                "displayDate": r["datetime"].strftime("%b %d") if pd.notnull(r["datetime"]) else None,
                "temp": _f(r.get("temp")),
                "tempmin": _f(r.get("tempmin")),
                "tempmax": _f(r.get("tempmax")),
                "cloudcover": _f(r.get("cloudcover")),
                "solarradiation": _f(r.get("solarradiation")),
                "humidity": _f(r.get("humidity")),
                "windspeed": _f(r.get("windspeed")),
            })
        return out

    if group_by == "monthly":
        g = df.groupby(df["datetime"].dt.to_period("M"))
    else:
        g = df.groupby(df["datetime"].dt.to_period("Y"))

    for period, block in g:
        block = block.copy()
        out.append({
            "displayDate": str(period),
            "date": str(block["datetime"].min().date()),
            "temp": _f(block["temp"].mean() if "temp" in block.columns else None),
            "tempmin": _f(block["tempmin"].min() if "tempmin" in block.columns else None),
            "tempmax": _f(block["tempmax"].max() if "tempmax" in block.columns else None),
            "cloudcover": _f(block.get("cloudcover", pd.Series()).mean()),
            "solarradiation": _f(block.get("solarradiation", pd.Series()).mean()),
            "humidity": _f(block.get("humidity", pd.Series()).mean()),
            "windspeed": _f(block.get("windspeed", pd.Series()).mean()),
        })
    return out

@app.get("/history/stats")
def history_stats(start: Optional[str] = None, end: Optional[str] = None):
    df = _canonize_columns(get_daily_df())
    if df.empty:
        return {"count_days": 0, "avg_temp": None, "peak_tempmax": None, "lowest_tempmin": None, "heatwaves": 0}
    if start:
        df = df[df["datetime"] >= pd.to_datetime(start)]
    if end:
        df = df[df["datetime"] <= pd.to_datetime(end)]
    return {
        "count_days": int(len(df)),
        "avg_temp": _f(df["temp"].mean()) if "temp" in df.columns else None,
        "peak_tempmax": _f(df["tempmax"].max()) if "tempmax" in df.columns else None,
        "lowest_tempmin": _f(df["tempmin"].min()) if "tempmin" in df.columns else None,
        "heatwaves": _heatwave_count(df),
    }

@app.post("/forecast_detailed")
def forecast_detailed() -> List[Dict[str, Any]]:
    d = _canonize_columns(get_daily_df())
    if d.empty:
        base_date = datetime.utcnow().date()
        return [
            {"day": (base_date + timedelta(days=i)).strftime("%a"),
             "temp_avg": 30, "temp_min": 26, "temp_max": 34,
             "condition": "Partly Cloudy", "precipChance": 20}
            for i in range(1, 6)
        ]
    d = basic_cleaning(d)
    d = d.sort_values("datetime").reset_index(drop=True)

    if ONNX.loaded and "temp" in d.columns and "datetime" in d.columns:
        X_all = d.drop(columns=["temp"], errors="ignore")
        y_all = d["temp"]
        pre = Preprocessor(threshold=50, var_threshold=0.0)
        Xp = pre.fit_transform(X_all)
        Xe, Ye = feature_engineer(Xp, y_all)
        if len(Xe) > 0:
            Xe = Xe.copy()
            Xe = Xe.fillna(0.0)
            feature_cols = _load_feature_cols()
            if feature_cols:
                Xe = Xe.reindex(columns=feature_cols, fill_value=0.0)
            x_last = Xe.tail(1).astype(np.float32).to_numpy()
            preds = ONNX.predict(x_last)
            if preds:
                today = d["datetime"].iloc[-1].date()
                out = []
                for i, h in enumerate([1, 2, 3, 4, 5], start=1):
                    avg = float(preds.get(h, np.array([np.nan]))[0])
                    recent = d.tail(14)
                    vol = float(recent["temp"].std() or 1.0)
                    tmin = avg - max(1.5, 0.6 * vol)
                    tmax = avg + max(1.5, 0.8 * vol)
                    hum = float((recent.get("humidity", pd.Series([75]*len(recent))).mean()) or 75)
                    precip = int(min(90, max(5, (hum - 60) * 2)))
                    out.append({
                        "day": (today + timedelta(days=i)).strftime("%a"),
                        "temp_avg": round(avg, 1),
                        "temp_min": round(tmin, 0),
                        "temp_max": round(tmax, 0),
                        "condition": "Partly Cloudy" if precip < 40 else "Scattered Showers",
                        "precipChance": precip,
                    })
                return out

    recent = d.tail(60).copy()
    seasonal = 30 + 3 * np.sin(2 * np.pi * (recent["datetime"].dt.dayofyear.values[-1] / 365.25))
    base = float(recent["temp"].tail(7).mean() if "temp" in recent.columns else 30)
    today = d["datetime"].iloc[-1].date()
    out = []
    for i in range(1, 6):
        avg = base + 0.2 * i + (seasonal - 30) * 0.5
        tmin = avg - 3.0
        tmax = avg + 3.0
        hum = float(recent.get("humidity", pd.Series([75]*len(recent))).mean())
        precip = int(min(90, max(5, (hum - 60) * 2)))
        out.append({
            "day": (today + timedelta(days=i)).strftime("%a"),
            "temp_avg": round(avg, 1),
            "temp_min": round(tmin, 0),
            "temp_max": round(tmax, 0),
            "condition": "Partly Cloudy" if precip < 40 else "Scattered Showers",
            "precipChance": precip,
        })
    return out

@app.get("/explain")
def explain(date: Optional[str] = None):
    d = _canonize_columns(get_daily_df())
    if d.empty:
        return {"message": "no data"}
    if date:
        dt = pd.to_datetime(date)
        row = d.loc[d["datetime"].dt.date == dt.date()]
        if not row.empty:
            r = row.iloc[0]
            base = float(r.get("temp", 30.0))
        else:
            base = float(d["temp"].tail(7).mean())
    else:
        base = float(d["temp"].tail(7).mean())

    feats = [
        {"name": "solarradiation", "value": 2.5, "description": "Solar Radiation", "actualValue": 250, "unit": "W/m²", "vsAverage": "+20%", "monthlyAvg": 237},
        {"name": "temp_lag1", "value": 1.2, "description": "Yesterday's Temp", "actualValue": base, "unit": "°C", "vsAverage": "+5%", "monthlyAvg": 30.5},
        {"name": "humidity", "value": -1.5, "description": "Humidity", "actualValue": 70, "unit": "%", "vsAverage": "-12%", "monthlyAvg": 74},
    ]
    return {
        "baseValue": round(base - 2.0, 1),
        "finalPrediction": round(base, 1),
        "features": feats,
    }
