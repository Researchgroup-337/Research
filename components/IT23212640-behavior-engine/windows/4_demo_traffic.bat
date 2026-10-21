@echo off
cd /d "%~dp0\.."
set PYTHONUTF8=1
python scripts\demo_stream.py --days 14 --api http://127.0.0.1:8000
pause
