# Weather Forecast API (FastAPI + ONNX)

This backend serves RandomForest ONNX models for HCMC temperature forecasting (t+1..t+5).

## Endpoints
- `GET /health` -> service status
- `POST /predict` -> compute features from input (or latest CSV row) and return predictions for horizons 1..5
- `GET /forecast` -> convenience wrapper that uses latest available row from the CSV and returns 5-step forecast

## IMPORTANT: Feature schema
ONNX models do **not** preserve column names. You **must** put the exact feature order used during training into `schemas/features_y+H.json` for H=1..5.

If you trained with `final.py`, ensure you dump `model_info['feature_names']` per horizon into those JSON files.

## Deploy on Render (free)
1. Create a new repo with this folder content.
2. Push models (`models/temp_t+1.onnx`..`temp_t+5.onnx`) and CSV into `data/`.
3. Connect repo to Render -> New Web Service -> Use `render.yaml`.
