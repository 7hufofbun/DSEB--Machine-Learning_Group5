# 1) Backend (FastAPI)

**Thư mục dự án backend:**

```
D:\ML--06_11_25\DSEB--Machine-Learning_Group5\smart_weather_fullrun\backend\smart_weather_ml

```

**Các bước:**

```powershell
# 0) Mở PowerShell mới
cd D:\ML--06_11_25\DSEB--Machine-Learning_Group5\smart_weather_fullrun\backend

# 1) Tạo & kích hoạt venv (chỉ cần 1 lần, sau đó chỉ Activate)
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Cài thư viện
pip install -r requirements.txt

# (Tuỳ chọn) 3) Train lại & export ONNX nếu thư mục models_onnx đang rỗng
#   → chạy lệnh sau sẽ tự train + export .onnx
python -m smart_weather_ml.model.train --save_dir models_onnx --overwrite

# Kiểm tra đã có model ONNX chưa
Get-ChildItem .\models_onnx\*.onnx | Select Name,Length,LastWriteTime

# 4) Chạy server FastAPI (chạy từ đúng thư mục backend)
uvicorn main:app --reload --host 127.0.0.1 --port 8000

```

**Kỳ vọng log OK (ví dụ):**

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.

```

**Test nhanh:**

- Sức khoẻ: http://127.0.0.1:8000/health → `{"status":"ok"}`
- OpenAPI: http://127.0.0.1:8000/docs

> ⚠️ Nếu thấy lỗi “Could not import module 'main'”, gần như bạn đang không đứng trong thư mục backend. Hãy cd đúng đường dẫn ở trên rồi chạy lại uvicorn main:app --reload.
> 

---

# 2) Frontend (Vite + React)

**Thư mục frontend:**

```
D:\ML--06_11_25\DSEB--Machine-Learning_Group5\smart_weather_fullrun\frontend

```

**Các bước:**

```powershell
# 0) Mở PowerShell mới (backend vẫn chạy ở cửa sổ kia)
cd D:\ML--06_11_25\DSEB--Machine-Learning_Group5\smart_weather_fullrun\frontend

# 1) Tạo file cấu hình API (nếu chưa có)
ni .env.local -ItemType File -Force
'VITE_API_BASE=http://127.0.0.1:8000' | Out-File -FilePath .env.local -Encoding utf8 -Force

# 2) (Nếu mạng chặn npmjs) chuyển registry sang mirror ổn định
npm config set registry https://registry.npmmirror.com/

# 3) Cài packages
npm install
# nếu bị peer-deps lỗi:
# npm install --legacy-peer-deps

# 4) Chạy dev server
npm run dev
# nếu báo 'vite' không nhận: npx vite

```

**Kỳ vọng:** Vite báo địa chỉ như:

```
Local:   http://localhost:5173/

```

Mở trình duyệt vào `http://localhost:5173/`.

Frontend sẽ gọi API tới `http://127.0.0.1:8000` (nhờ `.env.local`).

---

## Ghi chú quan trọng

- **Thứ tự chạy:** Luôn bật **backend trước**, rồi mới **frontend**.
- **CORS:** Backend đã cấu hình CORS cho `http://localhost:5173`. Nếu bạn đổi port/frontend origin khác, thêm vào `allow_origins` trong `main.py`.
- **Model ONNX:** Nếu `models_onnx` đã có file `.onnx`, **không cần** train lại. Chỉ train khi bạn muốn cập nhật model.
- **Port đang bận:**
    - Backend (8000) bận → đổi `-port 8001` hoặc tắt tiến trình chiếm port.
    - Frontend (5173) bận → Vite sẽ hỏi đổi port; chọn yes.
- **Lỗi mạng npm (ECONNRESET/E404)** → giữ mirror `npmmirror`, thử `npm cache clean --force`, rồi `npm install` lại.
- **‘vite’ không nhận** → `npm i -D vite` hoặc dùng `npx vite`.
- **Kiểm tra nhanh endpoint từ PowerShell:**
    
    ```powershell
    # Tránh alias của curl trên Windows, dùng curl.exe
    curl.exe http://127.0.0.1:8000/health
    
    ```
    

---

## Tóm tắt lệnh “chạy ngay”

**Backend (cửa sổ 1):**

```powershell
cd D:\ML--06_11_25\DSEB--Machine-Learning_Group5\smart_weather_fullrun\backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 127.0.0.1 --port 8000

```

**Frontend (cửa sổ 2):**

```powershell
cd D:\ML--06_11_25\DSEB--Machine-Learning_Group5\smart_weather_fullrun\frontend
npm run dev

```

Nếu muốn mình kiểm tra nhanh log khi bạn chạy 2 khối lệnh trên, bạn dán output là mình “soi” tiếp giúp ngay nhé.
