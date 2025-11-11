
import os, sys, tempfile
import numpy as np
def save_models_to_onnx(pipeline, save_dir="models_onnx/", logger=None):
    print(f"Python: {sys.version}")
    try:
        import onnx, lightgbm as lgb
    except Exception as e:
        print(f"Missing onnx/lightgbm: {e}")
        return {"success": 0, "failed": len(pipeline['models'])}
    os.makedirs(save_dir, exist_ok=True)
    results = {"success": 0, "failed": 0}
    for target_name, model_info in pipeline["models"].items():
        model = model_info["model"]
        n_features = len(model_info["feature_names"])
        try:
            booster = model.booster_
            temp_model_path = tempfile.mktemp(suffix=".txt")
            booster.save_model(temp_model_path)
            try:
                from hummingbird.ml import convert as hb_convert
            except ImportError:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "hummingbird-ml[onnx]"])
                from hummingbird.ml import convert as hb_convert
            test_input = np.random.randn(1, n_features).astype(np.float32)
            onnx_model = hb_convert(model, "onnx", test_input=test_input, extra_config={"onnx_target_opset": 12}).model
            file_path = os.path.join(save_dir, f"{target_name}.onnx")
            with open(file_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            results["success"] += 1
            try:
                import onnx
                onnx_model_check = onnx.load(file_path)
                onnx.checker.check_model(onnx_model_check)
            except Exception:
                pass
            if logger:
                try:
                    size_kb = os.path.getsize(file_path) / 1024
                    logger.report_scalar(title="Model Export", series=f"{target_name}_Size_KB", value=size_kb, iteration=0)
                except Exception:
                    pass
        except Exception as e:
            results["failed"] += 1
        finally:
            try: os.remove(temp_model_path)
            except Exception: pass
    return results
