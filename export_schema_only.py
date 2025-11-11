import os
import json
import pandas as pd

# 1️⃣ Đường dẫn file
X_TRAIN_PATH = "X_train.csv"   # file X_train bạn đã có
SCHEMA_DIR = "schemas"

# 2️⃣ Tạo thư mục schemas nếu chưa có
os.makedirs(SCHEMA_DIR, exist_ok=True)

# 3️⃣ Đọc dữ liệu
print(f"📂 Loading {X_TRAIN_PATH} ...")
df = pd.read_csv(X_TRAIN_PATH)

# 4️⃣ Loại bỏ các cột không cần (nếu có)
drop_cols = ["datetime", "sunrise", "sunset"]
feature_names = [c for c in df.columns if c not in drop_cols]

# 5️⃣ Xuất 5 file JSON (1 → 5)
for h in [1, 2, 3, 4, 5]:
    out_path = os.path.join(SCHEMA_DIR, f"features_y+{h}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names}, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved schema: {out_path} ({len(feature_names)} features)")

print("\n🎉 All 5 schema files exported successfully!")
