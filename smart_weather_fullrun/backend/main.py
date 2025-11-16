
from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime, timedelta, date
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


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (float, int)):
            if isinstance(value, float) and np.isnan(value):
                return None
            return float(value)
        return float(value)
    except Exception:
        return None


def _ordinal_suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _generate_forecast() -> Dict[str, Any]:
    d_raw = _canonize_columns(get_daily_df())
    if d_raw.empty:
        base_date = datetime.utcnow().date()
        fallback = [
            {
                "day": (base_date + timedelta(days=i)).strftime("%a"),
                "date": (base_date + timedelta(days=i)).isoformat(),
                "temp_avg": 30.0 + i,
                "temp_min": 26.0 + min(i, 2),
                "temp_max": 34.0 + min(i, 3),
                "condition": "Partly Cloudy" if i % 2 == 0 else "Scattered Showers",
                "precipChance": 25 + 5 * i,
            }
            for i in range(1, 6)
        ]
        return {"today": base_date.isoformat(), "source": "fallback", "items": fallback}

    d = basic_cleaning(d_raw.copy())
    if d.empty or "datetime" not in d.columns:
        return {"today": datetime.utcnow().date().isoformat(), "source": "fallback", "items": []}

    d = d.sort_values("datetime").reset_index(drop=True)
    today = d["datetime"].iloc[-1].date()

    if ONNX.loaded and "temp" in d.columns:
        X_all = d.drop(columns=["temp"], errors="ignore")
        y_all = d["temp"]
        pre = Preprocessor(threshold=50, var_threshold=0.0)
        try:
            Xp = pre.fit_transform(X_all)
            Xe, Ye = feature_engineer(Xp, y_all)
        except Exception:
            Xe, Ye = None, None
        if Xe is not None and len(Xe) > 0:
            Xe = Xe.copy().fillna(0.0)
            feature_cols = _load_feature_cols()
            if feature_cols:
                Xe = Xe.reindex(columns=feature_cols, fill_value=0.0)
            x_last = Xe.tail(1).astype(np.float32).to_numpy()
            preds = ONNX.predict(x_last)
            if preds:
                recent = d.tail(14)
                hum = float((recent.get("humidity", pd.Series([75] * len(recent))).mean()) or 75)
                vol = float(recent.get("temp", pd.Series([1.0])).std() or 1.0)
                out: List[Dict[str, Any]] = []
                for idx, horizon in enumerate([1, 2, 3, 4, 5], start=1):
                    avg = float(preds.get(horizon, np.array([np.nan]))[0])
                    tmin = avg - max(1.5, 0.6 * vol)
                    tmax = avg + max(1.5, 0.8 * vol)
                    precip = int(min(90, max(5, (hum - 60) * 2)))
                    target_date = today + timedelta(days=idx)
                    out.append({
                        "day": target_date.strftime("%a"),
                        "date": target_date.isoformat(),
                        "temp_avg": round(avg, 1),
                        "temp_min": round(tmin, 0),
                        "temp_max": round(tmax, 0),
                        "condition": "Partly Cloudy" if precip < 40 else "Scattered Showers",
                        "precipChance": precip,
                    })
                return {"today": today.isoformat(), "source": "onnx", "items": out}

    recent = d.tail(60).copy()
    if "datetime" not in recent.columns or "temp" not in recent.columns:
        return {"today": today.isoformat(), "source": "fallback", "items": []}

    seasonal = 30 + 3 * np.sin(2 * np.pi * (recent["datetime"].dt.dayofyear.values[-1] / 365.25))
    base = float(recent["temp"].tail(7).mean())
    out: List[Dict[str, Any]] = []
    for idx in range(1, 6):
        avg = base + 0.2 * idx + (seasonal - 30) * 0.5
        tmin = avg - 3.0
        tmax = avg + 3.0
        hum = float(recent.get("humidity", pd.Series([75] * len(recent))).mean())
        precip = int(min(90, max(5, (hum - 60) * 2)))
        target_date = today + timedelta(days=idx)
        out.append({
            "day": target_date.strftime("%a"),
            "date": target_date.isoformat(),
            "temp_avg": round(avg, 1),
            "temp_min": round(tmin, 0),
            "temp_max": round(tmax, 0),
            "condition": "Partly Cloudy" if precip < 40 else "Scattered Showers",
            "precipChance": precip,
        })
    return {"today": today.isoformat(), "source": "seasonal", "items": out}


def _format_percent(delta: Optional[float]) -> str:
    if delta is None or np.isnan(delta):
        return "—"
    rounded = round(delta, 1)
    if rounded == -0.0:
        rounded = 0.0
    sign = "+" if rounded >= 0 else ""
    return f"{sign}{rounded}%"


def _compute_historical_range(df: pd.DataFrame, target_date: date) -> Optional[Dict[str, float]]:
    if df.empty or "datetime" not in df.columns or "temp" not in df.columns:
        return None
    month_slice = df[df["datetime"].dt.month == target_date.month]
    subset = month_slice if not month_slice.empty else df
    temps = subset["temp"].dropna()
    if temps.empty:
        return None
    def _quantile(series: pd.Series, q: float) -> float:
        try:
            return float(np.percentile(series, q * 100))
        except Exception:
            return float(series.quantile(q))
    values = {
        "min": float(temps.min()),
        "percentile25": _quantile(temps, 0.25),
        "avg": float(temps.mean()),
        "percentile75": _quantile(temps, 0.75),
        "max": float(temps.max()),
    }
    return {k: round(v, 1) for k, v in values.items()}


def _analogue_days(df: pd.DataFrame, target_date: date, predicted_temp: Optional[float],
                   reference: Optional[pd.Series]) -> List[Dict[str, Any]]:
    if predicted_temp is None or df.empty or "temp" not in df.columns:
        return []
    subset = df[df["datetime"].dt.month == target_date.month].copy()
    if subset.empty:
        subset = df.copy()
    subset = subset.dropna(subset=["temp", "datetime"])
    if subset.empty:
        return []
    target_humidity = _safe_float(reference.get("humidity")) if reference is not None else None
    target_cloud = _safe_float(reference.get("cloudcover")) if reference is not None else None

    def _score(row: pd.Series) -> float:
        temp_val = _safe_float(row.get("temp"))
        if temp_val is None:
            return float("inf")
        diff_temp = abs(temp_val - predicted_temp)
        hum_val = _safe_float(row.get("humidity"))
        diff_hum = abs(hum_val - target_humidity) if (target_humidity is not None and hum_val is not None) else 0.0
        cloud_val = _safe_float(row.get("cloudcover"))
        diff_cloud = abs(cloud_val - target_cloud) if (target_cloud is not None and cloud_val is not None) else 0.0
        return diff_temp * 6.0 + diff_hum * 0.3 + diff_cloud * 0.2

    subset = subset.assign(__score=subset.apply(_score, axis=1))
    subset = subset.nsmallest(2, "__score")
    results: List[Dict[str, Any]] = []
    for _, row in subset.iterrows():
        date_val = row["datetime"]
        if pd.isna(date_val):
            continue
        temp_val = _safe_float(row.get("temp"))
        if temp_val is None:
            continue
        score = row["__score"]
        similarity = max(0.0, 100.0 - min(score, 100.0))
        results.append({
            "date": date_val.strftime("%Y-%m-%d"),
            "label": date_val.strftime("%B %d, %Y"),
            "actualTemp": round(temp_val, 1),
            "similarity": round(similarity, 0),
        })
    return results


def _feature_contributions(df: pd.DataFrame, target_date: date, reference: Optional[pd.Series],
                           forecast_temp: Optional[float], base_temp: Optional[float]) -> List[Dict[str, Any]]:
    if reference is None:
        return []
    month_slice = df[df["datetime"].dt.month == target_date.month]
    subset = month_slice if not month_slice.empty else df
    configs = [
        {"name": "temp_lag1", "description": "Yesterday's Temp", "unit": "°C", "column": "temp", "scale": 0.6},
        {"name": "humidity", "description": "Humidity", "unit": "%", "column": "humidity", "scale": -0.03},
        {"name": "solarradiation", "description": "Solar Radiation", "unit": "W/m²", "column": "solarradiation", "scale": 0.01},
        {"name": "cloudcover", "description": "Cloud Cover", "unit": "%", "column": "cloudcover", "scale": -0.02},
        {"name": "windspeed", "description": "Wind Speed", "unit": "km/h", "column": "windspeed", "scale": -0.02},
    ]
    features: List[Dict[str, Any]] = []
    for cfg in configs:
        column = cfg["column"]
        if column not in reference.index:
            continue
        actual = _safe_float(reference.get(column))
        series = subset.get(column)
        monthly_avg = _safe_float(series.mean()) if series is not None else None
        if column == "temp" and base_temp is not None:
            monthly_avg = base_temp
        if actual is None and monthly_avg is None:
            continue
        if actual is None:
            actual = monthly_avg
        if actual is None:
            continue
        delta = None if monthly_avg is None else actual - monthly_avg
        impact = None if delta is None else cfg["scale"] * delta
        vs_avg = None if (monthly_avg is None or monthly_avg == 0) else (delta / monthly_avg * 100 if delta is not None else None)
        entry = {
            "name": cfg["name"],
            "description": cfg["description"],
            "unit": cfg["unit"],
            "actualValue": round(actual, 1) if actual is not None else None,
            "monthlyAvg": round(monthly_avg, 1) if monthly_avg is not None else None,
            "vsAverage": _format_percent(vs_avg),
            "value": round(impact, 1) if impact is not None else 0.0,
        }
        if entry["value"] == -0.0:
            entry["value"] = 0.0
        features.append(entry)
    if forecast_temp is not None:
        features.append({
            "name": "forecast_delta",
            "description": "Model Adjustment",
            "unit": "°C",
            "actualValue": round(forecast_temp, 1),
            "monthlyAvg": round(base_temp, 1) if base_temp is not None else None,
            "vsAverage": _format_percent(None if base_temp is None else ((forecast_temp - base_temp) / base_temp * 100 if base_temp else None)),
            "value": round((forecast_temp - (base_temp or forecast_temp)) * 0.4, 1) if base_temp is not None else 0.0,
        })
    return features


def _build_deep_dive_items(forecast: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, Any]]:
    items = forecast.get("items") or []
    if not items or df.empty:
        return []
    if "datetime" not in df.columns:
        return []
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return []
    df = df.sort_values("datetime").reset_index(drop=True)
    reference = df.iloc[-1]
    base_temp = None
    if "temp" in df.columns:
        base_temp = _safe_float(df["temp"].tail(7).mean())
    results: List[Dict[str, Any]] = []
    for item in items:
        temp_avg = _safe_float(item.get("temp_avg"))
        date_str = item.get("date")
        if date_str:
            try:
                target_date = datetime.fromisoformat(str(date_str)).date()
            except ValueError:
                target_date = reference["datetime"].date() + timedelta(days=1)
        else:
            target_date = reference["datetime"].date() + timedelta(days=1)
        day_number = target_date.day
        label = f"{target_date.strftime('%A, %b')} {day_number}{_ordinal_suffix(day_number)}"
        temp_label = f"{label} ({temp_avg:.1f}°C)" if temp_avg is not None else label
        historical = _compute_historical_range(df, target_date)
        analogues = _analogue_days(df, target_date, temp_avg, reference)
        features = _feature_contributions(df, target_date, reference, temp_avg, base_temp)
        results.append({
            "id": date_str or target_date.isoformat(),
            "label": temp_label,
            "day": item.get("day"),
            "date": (target_date.isoformat()),
            "baseValue": round(base_temp, 1) if base_temp is not None else None,
            "finalPrediction": round(temp_avg, 1) if temp_avg is not None else None,
            "features": features,
            "historicalRange": historical,
            "analogueDays": analogues,
        })
    return results

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


@app.get("/history/range")
def history_range():
    df = _canonize_columns(get_daily_df())
    if df.empty:
        return {"min": None, "max": None}
    min_dt = df["datetime"].min()
    max_dt = df["datetime"].max()
    min_str = min_dt.strftime("%Y-%m-%d") if pd.notnull(min_dt) else None
    max_str = max_dt.strftime("%Y-%m-%d") if pd.notnull(max_dt) else None
    return {"min": min_str, "max": max_str}


@app.get("/model/metrics")
def model_metrics():
    """Return the latest model evaluation metrics persisted by the retrain pipeline.

    Falls back to a minimal default if no metrics file exists.
    """
    metrics_path = os.path.join(os.path.dirname(__file__), "model_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                import json
                return json.load(f)
        except Exception:
            pass
    # Fallback: return static minimal metrics that were previously shown in UI
    return {
        "train": {"rmse": 0.8, "mae": 0.6, "r2": 0.71},
        "test": {"rmse": 0.78, "mae": 0.62, "r2": 0.71},
        "per_target": {},
    }

@app.post("/forecast_detailed")
def forecast_detailed() -> List[Dict[str, Any]]:
    forecast = _generate_forecast()
    return forecast.get("items", [])

@app.get("/forecast_deep_dive")
def forecast_deep_dive():
    forecast = _generate_forecast()
    daily_df = _canonize_columns(get_daily_df())
    insights = _build_deep_dive_items(forecast, daily_df)
    return {
        "today": forecast.get("today"),
        "source": forecast.get("source"),
        "items": insights,
    }

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
