# smart_weather_ml/export.py
from pathlib import Path
import os
import numpy as np
import traceback
import json

# --- Optional imports (có thể vắng mặt, ta sẽ fallback) ---
try:
    from skl2onnx import convert_sklearn as _convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except Exception:
    _convert_sklearn = None
    FloatTensorType = None

try:
    from hummingbird.ml import convert as hb_convert
except Exception:
    hb_convert = None

try:
    # onnxmltools có converter riêng cho LightGBM
    from onnxmltools import convert_lightgbm as _convert_lightgbm
except Exception:
    _convert_lightgbm = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    # fallback placeholder để isinstance() không lỗi nếu lightgbm chưa cài
    class LGBMRegressor:  # type: ignore
        pass


def _log(logger, msg: str) -> None:
    """In ra console và đẩy text lên logger (nếu có)."""
    print(msg)
    if logger is not None:
        try:
            logger.report_text(msg)
        except Exception:
            pass


def save_onnx_atomic(onnx_obj, out_path: Path) -> None:
    """
    Ghi file ONNX theo kiểu 'atomic replace' để tránh lỗi ghi đè/lock file trên Windows.
    Chấp nhận:
      - ONNX ModelProto (có .SerializeToString)
      - bytes/bytearray đã là nội dung onnx
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    if hasattr(onnx_obj, "SerializeToString"):
        data = onnx_obj.SerializeToString()
    elif isinstance(onnx_obj, (bytes, bytearray)):
        data = onnx_obj
    else:
        raise TypeError("save_onnx_atomic: onnx_obj phải là ModelProto hoặc bytes.")

    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_path)  # ghi đè an toàn


def _unwrap_model(obj):
    """
    Trả về tuple (kind, model)
      - kind = "onnx" nếu obj là ModelProto
      - kind = "sklearn" nếu là estimator/pipeline sklearn (hoặc wrapper dict)
    Hỗ trợ dict wrapper: {'onnx'| 'model'| 'estimator'| 'sk_model'| 'lgbm'| 'pipeline': ...}
    """
    if hasattr(obj, "SerializeToString"):
        return "onnx", obj

    if isinstance(obj, dict):
        for k in ("onnx", "model", "estimator", "sk_model", "lgbm", "pipeline"):
            if k in obj and obj[k] is not None:
                v = obj[k]
                if hasattr(v, "SerializeToString"):
                    return "onnx", v
                return "sklearn", v

    return "sklearn", obj


def _infer_input_dim(pipeline) -> int:
    """
    Tìm số chiều input cho ONNX:
      - pipeline['input_dim'] hoặc pipeline['feature_dim']
      - pipeline['X_example'].shape[1] nếu có
      - lấy từ model.n_features_in_ của bất kỳ estimator nào
            - fallback từ wrapper metadata như 'n_features'
      - cuối cùng fallback = 20
    """
    dim = (
        pipeline.get("input_dim")
        or pipeline.get("feature_dim")
        or (np.array(pipeline.get("X_example")).shape[1] if pipeline.get("X_example") is not None else None)
    )
    if dim is not None:
        return int(dim)

    for m in (pipeline.get("models") or {}).values():
        # Ưu tiên metadata nếu wrapper là dict
        if isinstance(m, dict):
            n_features = m.get("n_features") or m.get("feature_dim")
            if n_features:
                return int(n_features)

        kind, est = _unwrap_model(m)
        if kind == "sklearn":
            if hasattr(est, "n_features_in_"):
                return int(est.n_features_in_)
            if hasattr(est, "n_features_"):
                return int(est.n_features_)

    return 20  # fallback an toàn


def save_models_to_onnx(pipeline, save_dir="models_onnx", logger=None, overwrite=False):
    """
    Xuất tất cả model trong pipeline['models'] sang ONNX.

    pipeline['models'] có thể chứa:
      - trực tiếp ONNX ModelProto (có .SerializeToString)
      - sklearn estimator/pipeline
      - dict wrapper: {'onnx':..., 'model':..., 'estimator':..., 'sk_model':..., 'lgbm':...}

    Chiến lược convert:
      1) Nếu là LightGBM (LGBMRegressor):
         - Ưu tiên Hummingbird -> ONNX
         - Fallback: onnxmltools.convert_lightgbm
      2) Nếu là sklearn (không phải LightGBM):
         - Thử skl2onnx
         - Fallback: Hummingbird -> ONNX
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    feature_columns = pipeline.get("feature_columns") if isinstance(pipeline, dict) else None
    default_input_dim = _infer_input_dim(pipeline)

    models = pipeline.get("models") or {}
    if not isinstance(models, dict) or not models:
        _log(logger, "[export] Không tìm thấy pipeline['models']. Không có gì để xuất.")
        return {"written": written, "skipped": skipped}

    for target_name, model in models.items():
        out_path = save_dir / f"{target_name}.onnx"

        if out_path.exists() and not overwrite:
            _log(logger, f"[export] Skip existing: {out_path}")
            skipped.append(str(out_path))
            continue

        kind, est = _unwrap_model(model)

        try:
            # Thu thập thông tin số chiều đầu vào từ nhiều nguồn
            wrapper_n_features = None
            if isinstance(model, dict):
                wrapper_n_features = model.get("n_features") or model.get("feature_dim")

            attr_n_features = (
                getattr(est, "n_features_in_", None)
                or getattr(est, "n_features_", None)
            )
            input_dim = int(attr_n_features or wrapper_n_features or default_input_dim)

            if kind == "onnx":
                # Đã là ModelProto -> ghi thẳng
                save_onnx_atomic(est, out_path)
                _log(logger, f"[export] Wrote (passthrough ONNX): {out_path}")
                written.append(str(out_path))
                continue

            # sklearn estimator/pipeline
            exported = False

            # --- Nhánh LightGBM: ưu tiên Hummingbird, rồi onnxmltools ---
            if isinstance(est, LGBMRegressor):
                # 1) Hummingbird
                if hb_convert is not None:
                    try:
                        X_dummy = np.random.rand(2, input_dim).astype("float32")
                        hb_model = hb_convert(est, backend="onnx", X=X_dummy)
                        onnx_model = hb_model.model  # ONNX ModelProto
                        save_onnx_atomic(onnx_model, out_path)
                        _log(logger, f"[export] Wrote (hummingbird, LightGBM): {out_path}")
                        exported = True
                    except Exception as e:
                        _log(logger, f"[export] Hummingbird failed for {target_name}: {e}\n{traceback.format_exc()}")

                # 2) onnxmltools (convert_lightgbm)
                if not exported and _convert_lightgbm is not None and FloatTensorType is not None:
                    try:
                        onnx_model = _convert_lightgbm(
                            est, initial_types=[("input", FloatTensorType([None, input_dim]))]
                        )
                        save_onnx_atomic(onnx_model, out_path)
                        _log(logger, f"[export] Wrote (onnxmltools.lightgbm): {out_path}")
                        exported = True
                    except Exception as e:
                        _log(logger, f"[export] onnxmltools LightGBM failed for {target_name}: {e}\n{traceback.format_exc()}")

            else:
                # --- Nhánh sklearn thường ---
                # 1) skl2onnx
                if _convert_sklearn is not None and FloatTensorType is not None:
                    try:
                        onnx_model = _convert_sklearn(
                            est,
                            initial_types=[("input", FloatTensorType([None, input_dim]))],
                        )
                        save_onnx_atomic(onnx_model, out_path)
                        _log(logger, f"[export] Wrote (skl2onnx): {out_path}")
                        exported = True
                    except Exception as e:
                        _log(logger, f"[export] skl2onnx failed for {target_name}: {e}\n{traceback.format_exc()}")

                # 2) Fallback: Hummingbird
                if not exported and hb_convert is not None:
                    try:
                        X_dummy = np.random.rand(2, input_dim).astype("float32")
                        hb_model = hb_convert(est, backend="onnx", X=X_dummy)
                        onnx_model = hb_model.model  # ONNX ModelProto
                        save_onnx_atomic(onnx_model, out_path)
                        _log(logger, f"[export] Wrote (hummingbird): {out_path}")
                        exported = True
                    except Exception as e:
                        _log(logger, f"[export] Hummingbird fallback failed for {target_name}: {e}\n{traceback.format_exc()}")

            if not exported:
                raise RuntimeError(
                    f"Could not export model for target '{target_name}'. "
                    f"Thiếu converter phù hợp hoặc gặp lỗi khi convert."
                )

            written.append(str(out_path))

        except Exception as e:
            _log(logger, f"[export] ERROR for target '{target_name}': {e}\n{traceback.format_exc()}")

    if feature_columns:
        feature_cols_path = save_dir / "feature_cols.json"
        try:
            feature_cols_path.write_text(json.dumps(feature_columns, ensure_ascii=False, indent=2))
            _log(logger, f"[export] Wrote feature column list: {feature_cols_path}")
        except Exception as exc:
            _log(logger, f"[export] Failed to write feature column metadata: {exc}")

    return {"written": written, "skipped": skipped}
