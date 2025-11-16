# Smart Weather — Full Stack Pack (Unzip & Run)

> Bộ này gộp **backend (FastAPI + ML)** và **frontend (React + Vite)**. 
---

---

## Tổng quan dự án

Smart Weather là một dự án end-to-end gọn nhẹ dùng để dự báo nhiệt độ cho **TP. Hồ Chí Minh** và giải thích *vì sao* một ngày lại nóng hơn hoặc mát hơn bình thường.

- Về **phía ML**, chúng em huấn luyện các mô hình chuỗi thời gian LightGBM trên dữ liệu thời tiết lịch sử (theo ngày + theo giờ), export sang chuẩn **ONNX**, và phục vụ qua backend **FastAPI**. Backend cũng cung cấp các endpoint tiện dụng cho “thời tiết hiện tại”, dự báo nhiều ngày và thống kê lịch sử.
- Về **phía sản phẩm**, frontend **React + Vite** biến các dự báo đó thành một giao diện giải thích thân thiện: các thẻ tóm tắt (“How today compares”), các yếu tố chính (“What’s driving today’s temperature”), và biểu đồ cho thấy mỗi yếu tố đang đẩy nhiệt độ lên hay kéo xuống như thế nào.
- Toàn bộ pack này đã được nối sẵn, nên bạn chỉ cần **giải nén, chạy backend + frontend là có ngay một Smart Weather assistant hoạt động trên máy local**.

_(Tóm tắt nhanh: dự án dự báo và giải thích nhiệt độ TP.HCM, gồm đủ backend ML + frontend UI, chỉ cần unzip là chạy được.)_

---


## 1) Cấu trúc thư mục

```text
smart_weather_full_project_fullrun/
├─ backend/
│  ├─ main.py
│  ├─ requirements.txt
│  ├─ data/
│  │  ├─ weather_hcm_daily.csv
│  │  └─ weather_hcm_hourly.csv
│  ├─ models_onnx/
│  │  └─ temp_t+{1..5}.onnx           
│  └─ smart_weather_ml/               (mã ML để train & export ONNX )
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
│     ├─ .env.local                   (đã trỏ: VITE_API_BASE=http://localhost:8000)
│     └─ src/
│        ├─ lib/api.ts                (helper gọi API)
│        └─ (các component UI  – đã gắn API)
├─ scripts/
│  ├─ setup_backend.ps1               (tạo venv, cài dep, chạy server)
│  └─ fix_frontend_imports.ps1        (sửa import @version & cài các UI deps)
└─ README_RUN.md                      (hướng dẫn ngắn – có trong gói)
```

---

## 2) Yêu cầu môi trường

- **Windows + PowerShell**
- **Python 3.10+** (khuyên dùng)
- **Node.js LTS** (18 hoặc 20), npm đi kèm

---

## 3) Chạy Backend (FastAPI)

```powershell
cd D:\ML--06_11_25\DSEB--Machine-Learning_Group5\smart_weather_fullrun\backend
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Kiểm tra: mở trình duyệt tới `http://127.0.0.1:8000/health` → thấy `{ "ok": true, ... }` là ổn.

### Tuỳ chỉnh dữ liệu thật

- Thay file trong `backend/data/` **hoặc** đặt biến môi trường trước khi chạy:

  ```powershell
  $env:WEATHER_DAILY_CSV="C:\path\to\your\weather_daily.csv"
  $env:WEATHER_HOURLY_CSV="C:\path\to\your\weather_hourly.csv"
  uvicorn main:app --reload --port 8000
  ```


### Dùng mô hình ONNX (nếu có)

- Đặt file `temp_t+1.onnx … temp_t+5.onnx` vào `backend/models_onnx/`.
- Nếu **không có**, API sẽ dùng baseline theo thống kê.

### Các endpoint chính

- `GET /health` – tình trạng server.
- `GET /now` – điều kiện hiện tại (ưu tiên hourly; fallback sang daily).
- `GET /history?start=YYYY-MM-DD&end=YYYY-MM-DD&group_by=daily|monthly|yearly`
- `GET /history/stats` – thống kê nhanh.
- `POST /forecast_detailed` – dự báo 5 ngày (dùng ONNX nếu có).
- `GET /explain?date=YYYY-MM-DD` – giải thích dự báo (có dữ liệu mẫu).

---

## 4) Chạy Frontend (React + Vite)

```powershell
cd [Folder link path](VD: D:\ML--06_11_25\DSEB--Machine-Learning_Group5\smart_weather_fullrun\frontend\Smartweatherassistantuidesign-main)
# Lần đầu cài đặt: Sửa import UI kiểu @radix-ui/...@version và cài deps
powershell -ExecutionPolicy Bypass -File ..\..\scripts\fix_frontend_imports.ps1

# Cài thư viện & chạy
npm install
npm run dev
```

Mở URL Vite hiển thị (thường là `http://localhost:5173`).

> **Ghi chú**
>
> - File `.env.local` đã có sẵn: `VITE_API_BASE=http://localhost:8000`.
> - Nếu mạng npm chậm: `npm config set registry https://registry.npmmirror.com/` rồi `npm install` lại.

### Các tab trong UI (đã nối API)

- **Current Conditions** → gọi `GET /now`
- **The Week Ahead** → gọi `POST /forecast_detailed`
- **Historical Explorer** → gọi `GET /history`

---

## 5) Lỗi thường gặp & cách xử

- **Vite báo “Failed to resolve import '@radix-ui/...@x.y.z'”**  
  → Chạy `fix_frontend_imports.ps1` trong thư mục UI (như hướng dẫn trên).

- **`vite` not recognized / `npm` lỗi mạng**  
  → Cài Node.js LTS; nếu mạng yếu dùng mirror:  
  `npm config set registry https://registry.npmmirror.com/`

- **Backend: “Error loading ASGI app” / import lỗi**  
  → Kiểm tra đang chạy trong thư mục `backend`, venv đã kích hoạt, và `uvicorn main:app` đúng tên module.

- **Backend không thấy CSV**  
  → Đặt file đúng vào `backend/data/` hoặc dùng biến môi trường `WEATHER_..._CSV`.

- **Push GitHub bị chặn vì file >100MB**  
  → Dùng **Git LFS**: `git lfs install` rồi `git lfs track "*.onnx" "*.csv"` trước khi `git add`.

---

## 6) (Tuỳ chọn) Train & Export ONNX

Trong `backend` (đã kích hoạt venv):

```powershell
python -m smart_weather_ml.train
```

- Pipeline sẽ train LightGBM cho các horizon và **export ONNX** vào `backend/models_onnx/`.
- Khởi động lại backend để API dùng mô hình mới.

---

