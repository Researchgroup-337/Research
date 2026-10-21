@echo off
cd /d "%~dp0\.."
set PYTHONUTF8=1
echo Moderator dashboard: open http://127.0.0.1:8000/dashboard  (API key: dev-key-change-me)
echo In a SECOND window run 4_demo_traffic.bat to fill it with chat activity.
echo Press Ctrl+C here to stop the server.
python -m uvicorn bae.api:app --app-dir src --host 127.0.0.1 --port 8000
pause
