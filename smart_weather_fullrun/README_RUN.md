# Smart Weather – Full Stack (Unzip & Run)
## Backend
- Location: `backend/`
- Start:
```
cd backend
python -m venv .venv
. ./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
## Frontend
- Location: `/mnt/data/smart_weather_full_project_fullrun/frontend/Smartweatherassistantuidesign-main`
- Already has `.env.local` pointing to `http://localhost:8000`.
- Before first run (Windows PowerShell):
```
cd "/mnt/data/smart_weather_full_project_fullrun/frontend/Smartweatherassistantuidesign-main"
powershell -ExecutionPolicy Bypass -File ..\..\scripts\fix_frontend_imports.ps1
npm install
npm run dev
```
