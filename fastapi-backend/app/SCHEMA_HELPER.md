# Helper script doc (not executable): How to dump feature_names during training
#
# In your training loop (final.py), after training each model:
#   feature_names = list(X_train_final.columns)
#   with open(f"schemas/features_y+{h}.json","w") as f:
#       json.dump({"feature_names": feature_names}, f, indent=2)
#
# Commit these JSON files alongside the ONNX models.
