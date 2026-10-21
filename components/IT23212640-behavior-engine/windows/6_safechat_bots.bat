@echo off
cd /d "%~dp0\.."
echo Bots will JOIN the sessions you start in the researcher console (room "Lab Stream").
echo To use another room:  python scripts\run_chat_bots.py --follow --room "your room name"
echo Press Ctrl+C to stop the bots.
python scripts\run_chat_bots.py --follow
pause
