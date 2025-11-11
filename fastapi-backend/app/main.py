import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import onnxruntime as ort
from final import feature_engineer

APP_NAME = "HCMC Temperature Forecast API"
MODEL_DIR = os.getenv("MODEL_DIR", "models")
SCHEMA_DIR = os.getenv("SCHEMA_DIR", "schemas")
DATA_CSV_DAILY = os.getenv("DATA_CSV_DAILY", "data/weather_hcm_daily.csv")

HORIZONS = [1,2,3,4,5]

class PredictRequest(BaseModel):
    # Option A: Provide full feature dict (already engineered) -> keys must match schema feature_names
    features: Optional[Dict[str, float]] = None
    # Option B: Let backend build features from latest CSV row (quick demo). This is a simplified baseline.
    use_latest_csv_row: bool = True

class PredictResponse(BaseModel):
    predictions: Dict[str, float] = Field(..., description="Mapping horizon 'y+H' -> predicted temperature")
    used_features: Optional[Dict[str, float]] = None
    info: Dict[str, Any]

app = FastAPI(title=APP_NAME)

def load_session(path: str) -> ort.InferenceSession:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    so = ort.SessionOptions()
    return ort.InferenceSession(path, sess_options=so, providers=['CPUExecutionProvider'])

def load_schema(h: int) -> List[str]:
    schema_path = os.path.join(SCHEMA_DIR, f"features_y+{h}.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file missing: {schema_path}")
    with open(schema_path, "r") as f:
        data = json.load(f)
    feats = data.get("feature_names", [])
    if not feats:
        raise ValueError(f"Empty feature_names in schema {schema_path}. Please fill with training feature order.")
    return feats

# Load all sessions at startup
SESSIONS = {}
SCHEMAS = {}
@app.on_event("startup")
def _startup():
    global SESSIONS, SCHEMAS
    for h in HORIZONS:
        model_path = os.path.join(MODEL_DIR, f"temp_t+{h}.onnx")
        try:
            SESSIONS[h] = load_session(model_path)
        except Exception as e:
            # Allow startup with partial availability; raise errors at call time
            SESSIONS[h] = None
        try:
            SCHEMAS[h] = load_schema(h)
        except Exception:
            SCHEMAS[h] = None

def _baseline_features_from_csv() -> Dict[str, float]:
    """
    Gọi lại feature_engineer từ final.py để build đầy đủ feature
    và lấy dòng mới nhất làm input cho mô hình.
    """
    if not os.path.exists(DATA_CSV_DAILY):
        raise FileNotFoundError(f"CSV not found: {DATA_CSV_DAILY}")

    df = pd.read_csv(DATA_CSV_DAILY)
    if len(df) < 40:
        raise ValueError("CSV too short to compute rolling features (need at least ~40 days).")

    # Dữ liệu đầu vào cho feature_engineer giống như lúc train
    X = df.drop(columns=["temp"], errors="ignore")
    y = df["temp"] if "temp" in df.columns else pd.Series([0]*len(df))

    # Gọi lại feature_engineer
    X_final, Y_final = feature_engineer(X, y)

    # Lấy dòng cuối cùng (mới nhất)
    latest = X_final.iloc[-1].to_dict()
    feats = {k: float(v) if pd.notnull(v) else 0.0 for k, v in latest.items() if k != "datetime"}

    print(f"✅ Imported {len(feats)} engineered features from feature_engineer() in final.py")
    return feats

def _predict_one(h: int, feat_map: Dict[str, float]) -> float:
    sess = SESSIONS.get(h)
    schema = SCHEMAS.get(h)
    if sess is None:
        raise HTTPException(500, f"Model for horizon y+{h} not loaded. Make sure models/temp_t+{h}.onnx exists.")
    if not schema:
        raise HTTPException(500, f"Feature schema for y+{h} not available. Fill schemas/features_y+{h}.json.")
    # Arrange features in the exact order
    try:
        x = np.array([[float(feat_map[k]) for k in schema]], dtype=np.float32)
    except KeyError as e:
        raise HTTPException(400, f"Missing feature in request for y+{h}: {e.args[0]}")
    input_name = sess.get_inputs()[0].name  # e.g., 'float_input'
    y = sess.run(None, {input_name: x})[0].ravel()[0]
    return float(y)

@app.get("/health")
def health():
    ready_models = [h for h,s in SESSIONS.items() if s is not None]
    ready_schemas = [h for h,s in SCHEMAS.items() if s]
    return {
        "status": "ok",
        "models_loaded": ready_models,
        "schemas_loaded": ready_schemas,
        "requires_schema_note": "ONNX needs exact feature order per horizon in schemas/features_y+H.json"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest = Body(...)):
    if body.features is None and not body.use_latest_csv_row:
        raise HTTPException(400, "Either provide 'features' or set 'use_latest_csv_row' = true.")

    feats = body.features or _baseline_features_from_csv()
    preds = {}
    for h in HORIZONS:
        preds[f"y+{h}"] = _predict_one(h, feats)

    return PredictResponse(
        predictions=preds,
        used_features=feats if body.features else None,
        info={
            "horizons": HORIZONS,
            "model_dir": MODEL_DIR,
            "schema_dir": SCHEMA_DIR
        }
    )

@app.get("/forecast", response_model=PredictResponse)
def forecast():
    import traceback
    try:
        feats = _baseline_features_from_csv()
        # (khuyến nghị) so khớp schema y+1 để phát hiện thiếu/nhầm cột
        schema = SCHEMAS.get(1) or []
        missing = [k for k in schema if k not in feats]
        extra = [k for k in feats if k not in schema]
        if missing:
            return HTTPException(
                status_code=400,
                detail=f"Missing {len(missing)} features (e.g. {missing[:10]}). Extra: {extra[:5]}"
            )

        preds = {f"y+{h}": _predict_one(h, feats) for h in HORIZONS}
        return PredictResponse(predictions=preds, used_features=None, info={"source": "csv"})
    except Exception as e:
        tb = traceback.format_exc()
        # ⚠️ chỉ bật khi debug local
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{tb}")