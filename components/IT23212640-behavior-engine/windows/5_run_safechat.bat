@echo off
cd /d "%~dp0\.."
set PYTHONUTF8=1
set BAE_RESEARCHER_KEY=lab-key-change-me
echo SafeChat Lab:
echo   Researcher console: http://127.0.0.1:8100/researcher   (key: lab-key-change-me)
echo   Participants:       http://YOUR-LAPTOP-IP:8100
echo In a SECOND window run 6_safechat_bots.bat to add bot users.
echo Press Ctrl+C here to stop the server.
python -m uvicorn bae.chatapp.server:app --app-dir src --host 0.0.0.0 --port 8100
pause
