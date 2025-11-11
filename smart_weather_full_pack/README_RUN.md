# How to run
## Backend
cd backend
python -m venv .venv; . .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

## Frontend
Copy frontend/.env.local into your UI repo root.
If imports contain @version suffixes, run scripts/fix_frontend_imports.ps1 from your UI folder.
npm install
npm run dev
